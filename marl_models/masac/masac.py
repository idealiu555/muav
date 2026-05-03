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
    CHECKPOINT_FORMAT = "shared_masac_vector_critic"
    CHECKPOINT_VERSION = 1

    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self._validate_hyperparameters()
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim
        self.checkpoint_path = "masac.pt"
        actor_cls = AttentionActorNetwork if config.USE_ATTENTION else ActorNetwork
        critic_cls = AttentionCriticNetwork if config.USE_ATTENTION else CriticNetwork
        self.use_attention: bool = actor_cls is AttentionActorNetwork

        self.actor = actor_cls(obs_dim, action_dim).to(device)
        if critic_cls is AttentionCriticNetwork:
            self.critic_1 = critic_cls(num_agents, obs_dim, action_dim).to(device)
            self.critic_2 = critic_cls(num_agents, obs_dim, action_dim).to(device)
            self.target_critic_1 = critic_cls(num_agents, obs_dim, action_dim).to(device)
            self.target_critic_2 = critic_cls(num_agents, obs_dim, action_dim).to(device)
        else:
            self.critic_1 = critic_cls(self.total_obs_dim, self.total_action_dim, num_agents).to(device)
            self.critic_2 = critic_cls(self.total_obs_dim, self.total_action_dim, num_agents).to(device)
            self.target_critic_1 = critic_cls(self.total_obs_dim, self.total_action_dim, num_agents).to(device)
            self.target_critic_2 = critic_cls(self.total_obs_dim, self.total_action_dim, num_agents).to(device)

        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self._init_target_networks()
        self._set_target_critic_grads(enabled=False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.MASAC_ACTOR_LR, weight_decay=0.0)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=config.CRITIC_LR, weight_decay=0.0)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=config.CRITIC_LR, weight_decay=0.0)
        self.target_entropy_per_agent: float = -float(action_dim) * config.TARGET_ENTROPY_SCALE
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.ALPHA_LR, weight_decay=0.0)

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)

        actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        with torch.no_grad():
            for i in range(self.num_agents):
                if obs_array[i, config.OWN_STATE_DIM - 1] < 0.5:
                    continue
                if exploration:
                    action, _ = self.actor.sample(obs_tensor[i:i + 1])
                else:
                    mean, _ = self.actor(obs_tensor[i:i + 1])
                    action = torch.tanh(mean)
                actions[i] = action.squeeze(0).cpu().numpy()
        return actions

    def _per_agent_bootstrap_mask(self, bootstrap_mask_tensor: torch.Tensor) -> torch.Tensor:
        return bootstrap_mask_tensor

    def _per_agent_targets(
        self,
        rewards_tensor: torch.Tensor,
        target_q_value: torch.Tensor,
        bootstrap_mask_tensor: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_per_agent_tensor("rewards_tensor", rewards_tensor)
        self._validate_per_agent_tensor("target_q_value", target_q_value)
        self._validate_per_agent_tensor("bootstrap_mask_tensor", bootstrap_mask_tensor)
        return rewards_tensor + config.DISCOUNT_FACTOR * target_q_value * bootstrap_mask_tensor

    def _masked_log_probs(self, log_probs: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        self._validate_per_agent_tensor("log_probs", log_probs)
        self._validate_per_agent_tensor("mask_tensor", mask_tensor)
        return log_probs * mask_tensor

    def _validate_per_agent_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.dim() != 2 or tensor.shape[1] != self.num_agents:
            raise ValueError(f"{name} must have shape [batch, {self.num_agents}], got {tuple(tensor.shape)}.")

    def _require_same_shape(self, left_name: str, left: torch.Tensor, right_name: str, right: torch.Tensor) -> None:
        if left.shape != right.shape:
            raise ValueError(
                f"Incompatible MASAC tensor shapes for {left_name} and {right_name}: "
                f"expected matching shapes, got {tuple(left.shape)} and {tuple(right.shape)}"
            )

    def _validate_batch_tensors(
        self,
        obs_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_obs_tensor: torch.Tensor,
        active_mask_tensor: torch.Tensor,
        bootstrap_mask_tensor: torch.Tensor,
    ) -> None:
        if obs_tensor.dim() != 3 or obs_tensor.shape[1:] != (self.num_agents, self.obs_dim):
            raise ValueError(
                f"obs_tensor must have shape [batch, {self.num_agents}, {self.obs_dim}], got {tuple(obs_tensor.shape)}."
            )
        if next_obs_tensor.dim() != 3 or next_obs_tensor.shape[1:] != (self.num_agents, self.obs_dim):
            raise ValueError(
                f"next_obs_tensor must have shape [batch, {self.num_agents}, {self.obs_dim}], got {tuple(next_obs_tensor.shape)}."
            )
        if actions_tensor.dim() != 3 or actions_tensor.shape[1:] != (self.num_agents, self.action_dim):
            raise ValueError(
                f"actions_tensor must have shape [batch, {self.num_agents}, {self.action_dim}], got {tuple(actions_tensor.shape)}."
            )
        self._validate_per_agent_tensor("rewards_tensor", rewards_tensor)
        self._validate_per_agent_tensor("active_mask_tensor", active_mask_tensor)
        self._validate_per_agent_tensor("bootstrap_mask_tensor", bootstrap_mask_tensor)

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
        self._validate_batch_tensors(
            obs_tensor,
            actions_tensor,
            rewards_tensor,
            next_obs_tensor,
            active_mask_tensor,
            bootstrap_mask_tensor,
        )

        batch_size: int = obs_tensor.shape[0]
        obs_flat: torch.Tensor = obs_tensor.reshape(batch_size, -1)
        next_obs_flat: torch.Tensor = next_obs_tensor.reshape(batch_size, -1)
        actions_flat: torch.Tensor = actions_tensor.reshape(batch_size, -1)

        per_agent_bootstrap_mask = self._per_agent_bootstrap_mask(bootstrap_mask_tensor)
        self._validate_per_agent_tensor("per_agent_bootstrap_mask", per_agent_bootstrap_mask)

        self._clamp_log_alpha()
        alpha: torch.Tensor = self.log_alpha.exp()

        with torch.no_grad():
            next_actions_list: list[torch.Tensor] = []
            next_log_probs_list: list[torch.Tensor] = []
            for i in range(self.num_agents):
                next_action, next_log_prob = self.actor.sample(next_obs_tensor[:, i, :])
                next_actions_list.append(next_action * per_agent_bootstrap_mask[:, i:i + 1])
                next_log_probs_list.append(next_log_prob)

            next_actions_tensor: torch.Tensor = torch.cat(next_actions_list, dim=1)
            next_log_probs: torch.Tensor = self._masked_log_probs(
                torch.cat(next_log_probs_list, dim=1),
                per_agent_bootstrap_mask,
            )
            target_q1: torch.Tensor = self.target_critic_1(next_obs_flat, next_actions_tensor)
            target_q2: torch.Tensor = self.target_critic_2(next_obs_flat, next_actions_tensor)
            self._validate_per_agent_tensor("target_critic_1", target_q1)
            self._validate_per_agent_tensor("target_critic_2", target_q2)
            self._require_same_shape("target_critic_1", target_q1, "next_log_probs", next_log_probs)
            self._require_same_shape("target_critic_2", target_q2, "next_log_probs", next_log_probs)
            min_target_q: torch.Tensor = torch.min(target_q1, target_q2)
            target_q: torch.Tensor = min_target_q - alpha.detach() * next_log_probs
            y: torch.Tensor = self._per_agent_targets(rewards_tensor, target_q, per_agent_bootstrap_mask)

        current_q1: torch.Tensor = self.critic_1(obs_flat, actions_flat)
        current_q2: torch.Tensor = self.critic_2(obs_flat, actions_flat)
        self._validate_per_agent_tensor("critic_1", current_q1)
        self._validate_per_agent_tensor("critic_2", current_q2)
        self._require_same_shape("critic_1", current_q1, "y", y)
        self._require_same_shape("critic_2", current_q2, "y", y)
        critic_1_loss: torch.Tensor = masked_mean(
            F.smooth_l1_loss(current_q1, y, reduction="none"),
            active_mask_tensor,
        )
        critic_2_loss: torch.Tensor = masked_mean(
            F.smooth_l1_loss(current_q2, y, reduction="none"),
            active_mask_tensor,
        )

        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        combined_critic_loss: torch.Tensor = critic_1_loss + critic_2_loss
        combined_critic_loss.backward()
        critic_1_grad = torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), config.MAX_GRAD_NORM)
        critic_2_grad = torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), config.MAX_GRAD_NORM)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        actor_loss, masked_log_probs, actor_grad_norm = self._optimize_actor(
            obs_tensor=obs_tensor,
            obs_flat=obs_flat,
            active_mask_tensor=active_mask_tensor,
            alpha=alpha,
        )

        alpha_loss: torch.Tensor = -masked_mean(
            self.log_alpha * (masked_log_probs + self.target_entropy_per_agent).detach(),
            active_mask_tensor,
        )
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self._clamp_log_alpha()

        soft_update(self.target_critic_1, self.critic_1, config.UPDATE_FACTOR)
        soft_update(self.target_critic_2, self.critic_2, config.UPDATE_FACTOR)

        valid_q = current_q1[active_mask_tensor.bool()]
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(combined_critic_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha_mean": float(self.log_alpha.detach().exp().item()),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(max(critic_1_grad, critic_2_grad)),
            "q_value_mean": float(valid_q.mean().item()) if valid_q.numel() else 0.0,
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

    def _clamp_log_alpha(self) -> None:
        with torch.no_grad():
            self.log_alpha.clamp_(min=self._min_log_alpha())

    def _init_target_networks(self) -> None:
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def reset(self) -> None:
        pass

    def save(self, directory: str) -> None:
        torch.save(
            {
                "checkpoint_format": self.CHECKPOINT_FORMAT,
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "num_agents": self.num_agents,
                "actor_type": "shared",
                "critic_type": "shared_vector",
                "uses_attention": self.use_attention,
                "model": self.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            os.path.join(directory, self.checkpoint_path),
        )

    def load(self, directory: str) -> None:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Shared MASAC directory not found: {directory}")

        checkpoint_file = os.path.join(directory, self.checkpoint_path)
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Shared MASAC checkpoint not found: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        checkpoint_format = checkpoint.get("checkpoint_format")
        checkpoint_version = checkpoint.get("checkpoint_version")
        checkpoint_num_agents = checkpoint.get("num_agents")
        actor_type = checkpoint.get("actor_type")
        critic_type = checkpoint.get("critic_type")
        uses_attention = checkpoint.get("uses_attention")

        if checkpoint_format != self.CHECKPOINT_FORMAT:
            raise ValueError(
                "Incompatible MASAC checkpoint format: "
                f"expected '{self.CHECKPOINT_FORMAT}', got {checkpoint_format!r}"
            )
        if checkpoint_version != self.CHECKPOINT_VERSION:
            raise ValueError(
                "Incompatible MASAC checkpoint version: "
                f"expected {self.CHECKPOINT_VERSION}, got {checkpoint_version!r}"
            )
        if checkpoint_num_agents != self.num_agents:
            raise ValueError(
                "Incompatible MASAC checkpoint agent count: "
                f"expected {self.num_agents}, got {checkpoint_num_agents!r}"
            )
        if actor_type != "shared" or critic_type != "shared_vector":
            raise ValueError(
                "Incompatible MASAC checkpoint architecture: "
                f"expected shared actor/shared vector critic, got actor_type={actor_type!r}, "
                f"critic_type={critic_type!r}"
            )
        if uses_attention != self.use_attention:
            raise ValueError(
                "Incompatible MASAC checkpoint attention architecture: "
                f"expected uses_attention={self.use_attention}, got {uses_attention!r}"
            )
        self.load_state_dict(checkpoint["model"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

    def _set_target_critic_grads(self, enabled: bool) -> None:
        self.target_critic_1.requires_grad_(enabled)
        self.target_critic_2.requires_grad_(enabled)

    def _set_critic_grads(self, enabled: bool) -> None:
        self.critic_1.requires_grad_(enabled)
        self.critic_2.requires_grad_(enabled)

    def _optimize_actor(
        self,
        obs_tensor: torch.Tensor,
        obs_flat: torch.Tensor,
        active_mask_tensor: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        pred_actions_list: list[torch.Tensor] = []
        log_probs_list: list[torch.Tensor] = []
        for i in range(self.num_agents):
            pred_action, log_prob = self.actor.sample(obs_tensor[:, i, :])
            pred_actions_list.append(pred_action * active_mask_tensor[:, i:i + 1])
            log_probs_list.append(log_prob)

        pred_actions_flat: torch.Tensor = torch.cat(pred_actions_list, dim=1)
        masked_log_probs: torch.Tensor = self._masked_log_probs(
            torch.cat(log_probs_list, dim=1),
            active_mask_tensor,
        )
        self._set_critic_grads(enabled=False)
        try:
            q1_pred: torch.Tensor = self.critic_1(obs_flat, pred_actions_flat)
            q2_pred: torch.Tensor = self.critic_2(obs_flat, pred_actions_flat)
            self._validate_per_agent_tensor("critic_1", q1_pred)
            self._validate_per_agent_tensor("critic_2", q2_pred)
            self._require_same_shape("critic_1", q1_pred, "masked_log_probs", masked_log_probs)
            self._require_same_shape("critic_2", q2_pred, "masked_log_probs", masked_log_probs)
            min_q_pred: torch.Tensor = torch.min(q1_pred, q2_pred)
            actor_loss: torch.Tensor = masked_mean(alpha.detach() * masked_log_probs - min_q_pred, active_mask_tensor)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM))
            self.actor_optimizer.step()
        finally:
            self._set_critic_grads(enabled=True)
        return actor_loss, masked_log_probs, actor_grad_norm
