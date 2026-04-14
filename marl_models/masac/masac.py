from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.masac.agents import ActorNetwork, AttentionActorNetwork, AttentionCriticNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, masked_mean
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class MASAC(MARLModel):
    """MADDPG + SAC style MASAC implementation"""

    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self._validate_hyperparameters()
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim
        self.checkpoint_path = "masac.pt"
        actor_cls = AttentionActorNetwork if config.USE_ATTENTION else ActorNetwork
        critic_cls = AttentionCriticNetwork if config.USE_ATTENTION else CriticNetwork

        self.actors = nn.ModuleList(actor_cls(obs_dim, action_dim) for _ in range(num_agents)).to(device)
        if critic_cls is AttentionCriticNetwork:
            critic_factory = lambda: critic_cls(num_agents, obs_dim, action_dim)
        else:
            critic_factory = lambda: critic_cls(self.total_obs_dim, self.total_action_dim)

        self.critics_1 = nn.ModuleList(critic_factory() for _ in range(num_agents)).to(device)
        self.critics_2 = nn.ModuleList(critic_factory() for _ in range(num_agents)).to(device)
        self.target_critics_1 = nn.ModuleList(critic_factory() for _ in range(num_agents)).to(device)
        self.target_critics_2 = nn.ModuleList(critic_factory() for _ in range(num_agents)).to(device)
        self.log_alphas = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, device=self.device)) for _ in range(num_agents)]
        )
        self._init_target_networks()
        self._set_target_critic_grads(enabled=False)

        self.actor_optimizers: list[torch.optim.Adam] = [
            torch.optim.Adam(actor.parameters(), lr=config.ACTOR_LR, weight_decay=0.0)
            for actor in self.actors
        ]
        self.critic_1_optimizers: list[torch.optim.Adam] = [
            torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR, weight_decay=0.0)
            for critic in self.critics_1
        ]
        self.critic_2_optimizers: list[torch.optim.Adam] = [
            torch.optim.Adam(critic.parameters(), lr=config.CRITIC_LR, weight_decay=0.0)
            for critic in self.critics_2
        ]

        self.target_entropy: float = -float(action_dim) * config.TARGET_ENTROPY_SCALE
        self.alpha_optimizers: list[torch.optim.Adam] = [
            torch.optim.Adam([log_alpha], lr=config.ALPHA_LR, weight_decay=0.0)
            for log_alpha in self.log_alphas
        ]

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        # Batch observations for better GPU utilization
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)

        actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        with torch.no_grad():
            for i in range(self.num_agents):
                if obs_array[i, config.OWN_STATE_DIM - 1] < 0.5:
                    continue
                if exploration:
                    action, _ = self.actors[i].sample(obs_tensor[i:i+1])
                else:
                    # For testing, use the mean of the distribution (deterministic policy)
                    # Note: sample() already applies tanh, so for consistency we do the same
                    mean, log_std = self.actors[i](obs_tensor[i:i+1])
                    # Use mean of pre-tanh distribution, then apply tanh (consistent with training)
                    action = torch.tanh(mean)
                actions[i] = action.squeeze(0).cpu().numpy()
        return actions

    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, dict), "MASAC expects OffPolicyExperienceBatch (dict)"
        obs_batch = batch["obs"]
        actions_batch = batch["actions"]
        rewards_batch = batch["rewards"]
        next_obs_batch = batch["next_obs"]
        active_mask_batch = batch["active_mask"]
        bootstrap_mask_batch = batch["bootstrap_mask"]
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        active_mask_tensor = torch.as_tensor(active_mask_batch, dtype=torch.float32, device=self.device)
        bootstrap_mask_tensor = torch.as_tensor(bootstrap_mask_batch, dtype=torch.float32, device=self.device)

        batch_size: int = obs_tensor.shape[0]
        obs_flat: torch.Tensor = obs_tensor.reshape(batch_size, -1)
        next_obs_flat: torch.Tensor = next_obs_tensor.reshape(batch_size, -1)
        actions_flat: torch.Tensor = actions_tensor.reshape(batch_size, -1)

        total_actor_loss: float = 0.0
        total_critic_loss: float = 0.0
        total_alpha_loss: float = 0.0
        total_actor_grad_norm: float = 0.0
        total_critic_grad_norm: float = 0.0
        q_values: list[float] = []
        updated_agents: int = 0

        self._clamp_log_alphas()
        alpha_tensors: list[torch.Tensor] = [log_alpha.exp() for log_alpha in self.log_alphas]

        with torch.no_grad():
            next_actions_list: list[torch.Tensor] = []
            next_log_probs_list: list[torch.Tensor] = []
            for i in range(self.num_agents):
                next_action, next_log_prob = self.actors[i].sample(next_obs_tensor[:, i, :])
                next_actions_list.append(next_action * bootstrap_mask_tensor[:, i:i + 1])
                next_log_probs_list.append(next_log_prob)

            next_actions_tensor: torch.Tensor = torch.cat(next_actions_list, dim=1)

        for agent_idx in range(self.num_agents):
            valid_mask: torch.Tensor = active_mask_tensor[:, agent_idx:agent_idx + 1]
            if valid_mask.sum().item() == 0.0:
                continue
            alpha: torch.Tensor = alpha_tensors[agent_idx]

            with torch.no_grad():
                target_q1: torch.Tensor = self.target_critics_1[agent_idx](next_obs_flat, next_actions_tensor)
                target_q2: torch.Tensor = self.target_critics_2[agent_idx](next_obs_flat, next_actions_tensor)
                min_target_q: torch.Tensor = torch.min(target_q1, target_q2)
                agent_next_log_prob: torch.Tensor = next_log_probs_list[agent_idx]
                target_q: torch.Tensor = min_target_q - alpha * agent_next_log_prob
                agent_reward: torch.Tensor = rewards_tensor[:, agent_idx].unsqueeze(1)
                agent_bootstrap: torch.Tensor = bootstrap_mask_tensor[:, agent_idx].unsqueeze(1)
                y: torch.Tensor = agent_reward + config.DISCOUNT_FACTOR * target_q * agent_bootstrap

            current_q1: torch.Tensor = self.critics_1[agent_idx](obs_flat, actions_flat)
            current_q2: torch.Tensor = self.critics_2[agent_idx](obs_flat, actions_flat)
            critic_1_loss: torch.Tensor = masked_mean(F.smooth_l1_loss(current_q1, y, reduction="none"), valid_mask)
            critic_2_loss: torch.Tensor = masked_mean(F.smooth_l1_loss(current_q2, y, reduction="none"), valid_mask)

            self.critic_1_optimizers[agent_idx].zero_grad(set_to_none=True)
            self.critic_2_optimizers[agent_idx].zero_grad(set_to_none=True)
            combined_critic_loss: torch.Tensor = critic_1_loss + critic_2_loss
            combined_critic_loss.backward()
            c1_grad = torch.nn.utils.clip_grad_norm_(self.critics_1[agent_idx].parameters(), config.MAX_GRAD_NORM)
            c2_grad = torch.nn.utils.clip_grad_norm_(self.critics_2[agent_idx].parameters(), config.MAX_GRAD_NORM)
            self.critic_1_optimizers[agent_idx].step()
            self.critic_2_optimizers[agent_idx].step()

            total_critic_loss += combined_critic_loss.item()
            total_critic_grad_norm += float(max(c1_grad, c2_grad))
            q_values.extend(current_q1[valid_mask.squeeze(1).bool()].detach().cpu().numpy().flatten().tolist())

            actor_loss, agent_log_prob, a_grad = self._optimize_actor(
                agent_idx=agent_idx,
                obs_tensor=obs_tensor,
                obs_flat=obs_flat,
                active_mask_tensor=active_mask_tensor,
                alpha=alpha,
            )
            total_actor_loss += actor_loss.item()
            total_actor_grad_norm += float(a_grad)

            alpha_loss: torch.Tensor = -masked_mean(
                self.log_alphas[agent_idx] * (agent_log_prob + self.target_entropy).detach(),
                valid_mask,
            )
            self.alpha_optimizers[agent_idx].zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizers[agent_idx].step()
            self._clamp_log_alpha(self.log_alphas[agent_idx])

            total_alpha_loss += alpha_loss.item()
            updated_agents += 1

            soft_update(self.target_critics_1[agent_idx], self.critics_1[agent_idx], config.UPDATE_FACTOR)
            soft_update(self.target_critics_2[agent_idx], self.critics_2[agent_idx], config.UPDATE_FACTOR)

        normalizer = max(1, updated_agents)
        alpha_mean = float(torch.stack([log_alpha.detach().exp() for log_alpha in self.log_alphas]).mean().item())
        return {
            "actor_loss": total_actor_loss / normalizer,
            "critic_loss": total_critic_loss / normalizer,
            "alpha_loss": total_alpha_loss / normalizer,
            "alpha_mean": alpha_mean,
            "actor_grad_norm": total_actor_grad_norm / normalizer,
            "critic_grad_norm": total_critic_grad_norm / normalizer,
            "q_value_mean": float(np.mean(q_values)) if q_values else 0.0,
        }

    def _validate_hyperparameters(self) -> None:
        if config.ALPHA_MIN <= 0.0:
            raise ValueError(f"ALPHA_MIN must be positive, got {config.ALPHA_MIN}")
        if config.TARGET_ENTROPY_SCALE <= 0.0:
            raise ValueError(
                f"TARGET_ENTROPY_SCALE must be positive, got {config.TARGET_ENTROPY_SCALE}"
            )

    def _min_log_alpha(self) -> float:
        return float(np.log(config.ALPHA_MIN))

    def _clamp_log_alpha(self, log_alpha: torch.Tensor) -> None:
        with torch.no_grad():
            log_alpha.clamp_(min=self._min_log_alpha())

    def _clamp_log_alphas(self) -> None:
        for log_alpha in self.log_alphas:
            self._clamp_log_alpha(log_alpha)

    def _init_target_networks(self) -> None:
        for critic1, target_critic1 in zip(self.critics_1, self.target_critics_1):
            target_critic1.load_state_dict(critic1.state_dict())
        for critic2, target_critic2 in zip(self.critics_2, self.target_critics_2):
            target_critic2.load_state_dict(critic2.state_dict())

    def reset(self) -> None:
        pass

    def save(self, directory: str) -> None:
        torch.save(
            {
                "model": self.state_dict(),
                "actor_optimizers": [optimizer.state_dict() for optimizer in self.actor_optimizers],
                "critic_1_optimizers": [optimizer.state_dict() for optimizer in self.critic_1_optimizers],
                "critic_2_optimizers": [optimizer.state_dict() for optimizer in self.critic_2_optimizers],
                "alpha_optimizers": [optimizer.state_dict() for optimizer in self.alpha_optimizers],
            },
            os.path.join(directory, self.checkpoint_path),
        )

    def load(self, directory: str) -> None:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"❌ Model directory not found: {directory}")

        checkpoint_file = os.path.join(directory, self.checkpoint_path)
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"❌ Model file not found: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.load_state_dict(checkpoint["model"])
        for optimizer, state_dict in zip(self.actor_optimizers, checkpoint["actor_optimizers"]):
            optimizer.load_state_dict(state_dict)
        for optimizer, state_dict in zip(self.critic_1_optimizers, checkpoint["critic_1_optimizers"]):
            optimizer.load_state_dict(state_dict)
        for optimizer, state_dict in zip(self.critic_2_optimizers, checkpoint["critic_2_optimizers"]):
            optimizer.load_state_dict(state_dict)
        for optimizer, state_dict in zip(self.alpha_optimizers, checkpoint["alpha_optimizers"]):
            optimizer.load_state_dict(state_dict)

    def _set_target_critic_grads(self, enabled: bool) -> None:
        for critic in self.target_critics_1:
            critic.requires_grad_(enabled)
        for critic in self.target_critics_2:
            critic.requires_grad_(enabled)

    def _set_critic_grads(self, agent_idx: int, enabled: bool) -> None:
        self.critics_1[agent_idx].requires_grad_(enabled)
        self.critics_2[agent_idx].requires_grad_(enabled)

    def _optimize_actor(
        self,
        agent_idx: int,
        obs_tensor: torch.Tensor,
        obs_flat: torch.Tensor,
        active_mask_tensor: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        valid_mask: torch.Tensor = active_mask_tensor[:, agent_idx:agent_idx + 1]
        pred_actions_list: list[torch.Tensor] = []
        agent_log_prob: torch.Tensor | None = None
        for i in range(self.num_agents):
            agent_active = active_mask_tensor[:, i:i + 1]
            if i == agent_idx:
                pred_action, agent_log_prob = self.actors[i].sample(obs_tensor[:, i, :])
                pred_actions_list.append(pred_action * agent_active)
            else:
                with torch.no_grad():
                    other_action, _ = self.actors[i].sample(obs_tensor[:, i, :])
                pred_actions_list.append((other_action * agent_active).detach())

        pred_actions_flat: torch.Tensor = torch.cat(pred_actions_list, dim=1)
        assert agent_log_prob is not None
        self._set_critic_grads(agent_idx, enabled=False)
        try:
            q1_pred: torch.Tensor = self.critics_1[agent_idx](obs_flat, pred_actions_flat)
            q2_pred: torch.Tensor = self.critics_2[agent_idx](obs_flat, pred_actions_flat)
            min_q_pred: torch.Tensor = torch.min(q1_pred, q2_pred)
            actor_loss: torch.Tensor = masked_mean(agent_log_prob * alpha.detach() - min_q_pred, valid_mask)
            self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), config.MAX_GRAD_NORM)
            )
            self.actor_optimizers[agent_idx].step()
        finally:
            self._set_critic_grads(agent_idx, enabled=True)
        return actor_loss, agent_log_prob, actor_grad_norm
