from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.mappo.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import masked_mean
import config
import numpy as np
import os
import torch
from torch.distributions import Normal


class MAPPO(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, state_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.state_dim: int = state_dim
        self.active_flag_index: int = config.OWN_STATE_DIM - 1

        # Create networks
        self.actors: ActorNetwork = ActorNetwork(obs_dim, action_dim).to(device)
        self.critics: CriticNetwork = CriticNetwork(state_dim).to(device)  # Removed num_agents parameter

        # Create optimizers
        self.actor_optimizer: torch.optim.AdamW = torch.optim.AdamW(self.actors.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer: torch.optim.AdamW = torch.optim.AdamW(self.critics.parameters(), lr=config.CRITIC_LR)

    def _get_active_mask(self, obs: np.ndarray) -> torch.Tensor:
        active_mask_np = (obs[:, self.active_flag_index] >= 0.5).astype(np.float32)
        return torch.from_numpy(active_mask_np).to(self.device, non_blocking=True)

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)
        action_mask: torch.Tensor = self._get_active_mask(obs_array).unsqueeze(1)
        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            if exploration:
                actions: torch.Tensor = dist.sample()
            else:
                actions: torch.Tensor = dist.mean
            actions = torch.clamp(actions, -1.0, 1.0) * action_mask

        return actions.cpu().numpy()

    def get_action_and_value(self, obs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Sample actions and compute log-probs and values for PPO."""
        obs_tensor: torch.Tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device, non_blocking=True)
        state_tensor: torch.Tensor = torch.from_numpy(state.astype(np.float32)).to(self.device, non_blocking=True)
        active_mask: torch.Tensor = self._get_active_mask(obs)
        action_mask: torch.Tensor = active_mask.unsqueeze(1)

        with torch.no_grad():
            dist: Normal = self.actors(obs_tensor)
            actions: torch.Tensor = dist.sample()
            actions = torch.clamp(actions, -1.0, 1.0) * action_mask
            log_probs: torch.Tensor = dist.log_prob(actions).sum(dim=-1) * active_mask

            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            values: torch.Tensor = self.critics(state_tensor).squeeze(-1)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy().item()

    def update(self, batch: ExperienceBatch, entropy_coef: float) -> dict:
        assert isinstance(batch, dict), "MAPPO expects OnPolicyExperienceBatch (dict)"
        obs_batch: torch.Tensor = batch["obs"]
        actions_batch: torch.Tensor = batch["actions"]
        old_log_probs_batch: torch.Tensor = batch["old_log_probs"]
        advantages_batch: torch.Tensor = batch["advantages"]
        returns_batch: torch.Tensor = batch["returns"]
        states_batch: torch.Tensor = batch["states"]
        old_values_batch: torch.Tensor = batch["old_values"]
        active_mask_batch: torch.Tensor = batch["active_mask"]

        actor_mask: torch.Tensor = active_mask_batch.float()
        actor_mask_bool: torch.Tensor = actor_mask.bool()
        valid_count = int(actor_mask.sum().item())

        # Early return if no active agents
        if valid_count == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "ratio_mean": 0.0,
                "clip_fraction": 0.0,
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
                "value_mean": 0.0,
                "valid_samples": 0.0,
                "log_std_mean": 0.0,
                "log_std_std": 0.0,
            }

        # Advantage normalization (only on active agents)
        valid_advantages: torch.Tensor = advantages_batch[actor_mask_bool]
        if valid_count > 1:
            adv_mean: torch.Tensor = valid_advantages.mean()
            adv_std: torch.Tensor = valid_advantages.std(unbiased=False)
            valid_advantages = (valid_advantages - adv_mean) / (adv_std + 1e-8)
        valid_advantages = torch.clamp(valid_advantages, -10.0, 10.0)
        advantages_batch = torch.zeros_like(advantages_batch)
        advantages_batch[actor_mask_bool] = valid_advantages

        # Actor Loss
        dist: Normal = self.actors(obs_batch)
        new_log_probs: torch.Tensor = dist.log_prob(actions_batch).sum(dim=-1)
        log_ratio: torch.Tensor = torch.clamp(
            new_log_probs - old_log_probs_batch,
            -config.PPO_MAX_LOG_RATIO,
            config.PPO_MAX_LOG_RATIO,
        )
        ratio: torch.Tensor = torch.exp(log_ratio)

        surr1: torch.Tensor = ratio * advantages_batch
        surr2: torch.Tensor = torch.clamp(ratio, 1.0 - config.PPO_CLIP_EPS, 1.0 + config.PPO_CLIP_EPS) * advantages_batch
        actor_loss: torch.Tensor = -masked_mean(torch.min(surr1, surr2), actor_mask)

        # Entropy bonus
        entropy: torch.Tensor = masked_mean(dist.entropy().sum(dim=-1), actor_mask)
        actor_loss = actor_loss - entropy_coef * entropy

        # Critic Loss (masked)
        # Critic outputs single value per state
        values_all: torch.Tensor = self.critics(states_batch)  # (batch, 1)
        values: torch.Tensor = values_all.squeeze(-1)  # (batch,) - direct use, no indexing needed
        values_clipped: torch.Tensor = old_values_batch + torch.clamp(values - old_values_batch, -config.PPO_VALUE_CLIP_EPS, config.PPO_VALUE_CLIP_EPS)
        vf_loss1: torch.Tensor = (values - returns_batch).pow(2)
        vf_loss2: torch.Tensor = (values_clipped - returns_batch).pow(2)
        critic_loss: torch.Tensor = 0.5 * masked_mean(torch.max(vf_loss1, vf_loss2), actor_mask)

        # Actor update
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actors.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        # Critic update (separate)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critics.parameters(), config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        # Metrics
        log_std_data: torch.Tensor = self.actors.log_std.data.squeeze(0)
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "ratio_mean": masked_mean(ratio, actor_mask).item(),
            "clip_fraction": masked_mean((torch.abs(ratio - 1.0) > config.PPO_CLIP_EPS).float(), actor_mask).item(),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
            "value_mean": masked_mean(values, actor_mask).item(),
            "valid_samples": float(valid_count),
            "log_std_mean": log_std_data.mean().item(),
            "log_std_std": log_std_data.std().item(),
        }

    def reset(self) -> None:
        pass  # Nothing to reset

    def save(self, directory: str) -> None:
        torch.save(
            {
                "actor": self.actors.state_dict(),
                "critic": self.critics.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            os.path.join(directory, "mappo.pth"),
        )

    def load(self, directory: str) -> None:
        path: str = os.path.join(directory, "mappo.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")
        checkpoint: dict = torch.load(path, map_location=self.device)
        self.actors.load_state_dict(checkpoint["actor"])
        self.critics.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
