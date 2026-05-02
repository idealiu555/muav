from marl_models.base_model import MARLModel, ExperienceBatch
from marl_models.maddpg.agents import ActorNetwork, CriticNetwork
from marl_models.buffer_and_helpers import soft_update, GaussianNoise, masked_mean
import config
import torch
import numpy as np
import os


class MADDPG(MARLModel):
    CHECKPOINT_FORMAT = "shared_maddpg_vector_critic"
    CHECKPOINT_VERSION = 1

    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.total_obs_dim: int = num_agents * obs_dim
        self.total_action_dim: int = num_agents * action_dim

        self.actor = ActorNetwork(obs_dim, action_dim).to(device)
        self.critic = CriticNetwork(self.total_obs_dim, self.total_action_dim, num_agents).to(device)
        self.target_actor = ActorNetwork(obs_dim, action_dim).to(device)
        self.target_critic = CriticNetwork(self.total_obs_dim, self.total_action_dim, num_agents).to(device)
        self._init_target_networks()

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config.CRITIC_LR)

        self.noise: list[GaussianNoise] = [GaussianNoise() for _ in range(num_agents)]

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        obs_array: np.ndarray = np.array(observations, dtype=np.float32)
        obs_tensor: torch.Tensor = torch.from_numpy(obs_array).to(self.device, non_blocking=True)

        actions = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        with torch.no_grad():
            for i in range(self.num_agents):
                if obs_array[i, config.OWN_STATE_DIM - 1] < 0.5:
                    continue
                action: torch.Tensor = self.actor(obs_tensor[i:i + 1]).squeeze(0)
                action_np = action.cpu().numpy()
                if exploration:
                    action_np = action_np + self.noise[i].sample()
                actions[i] = np.clip(action_np, -1.0, 1.0)

        return actions

    def _per_agent_bootstrap_mask(self, bootstrap_mask_tensor: torch.Tensor) -> torch.Tensor:
        return bootstrap_mask_tensor

    def _require_shape(self, tensor_name: str, tensor: torch.Tensor, expected_shape: torch.Size) -> None:
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Incompatible MADDPG tensor shape for {tensor_name}: "
                f"expected {tuple(expected_shape)}, got {tuple(tensor.shape)}"
            )

    def _per_agent_targets(
        self,
        rewards_tensor: torch.Tensor,
        target_q_value: torch.Tensor,
        bootstrap_mask_tensor: torch.Tensor,
    ) -> torch.Tensor:
        self._require_shape("target_q_value", target_q_value, rewards_tensor.shape)
        self._require_shape("bootstrap_mask_tensor", bootstrap_mask_tensor, rewards_tensor.shape)
        return rewards_tensor + config.DISCOUNT_FACTOR * target_q_value * bootstrap_mask_tensor

    def update(self, batch: ExperienceBatch) -> dict:
        assert isinstance(batch, dict), "MADDPG expects OffPolicyExperienceBatch (dict)"
        obs_batch = batch["obs"]
        actions_batch = batch["actions"]
        rewards_batch = batch["rewards"]
        next_obs_batch = batch["next_obs"]
        active_mask_batch = batch["active_mask"]
        bootstrap_mask_batch = batch["bootstrap_mask"]

        obs_tensor: torch.Tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.as_tensor(actions_batch, dtype=torch.float32, device=self.device)
        rewards_tensor: torch.Tensor = torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device)
        next_obs_tensor: torch.Tensor = torch.as_tensor(next_obs_batch, dtype=torch.float32, device=self.device)
        active_mask_tensor: torch.Tensor = torch.as_tensor(active_mask_batch, dtype=torch.float32, device=self.device)
        bootstrap_mask_tensor: torch.Tensor = torch.as_tensor(bootstrap_mask_batch, dtype=torch.float32, device=self.device)

        batch_size: int = obs_tensor.shape[0]
        obs_flat: torch.Tensor = obs_tensor.reshape(batch_size, -1)
        next_obs_flat: torch.Tensor = next_obs_tensor.reshape(batch_size, -1)
        actions_flat: torch.Tensor = actions_tensor.reshape(batch_size, -1)

        expected_q_shape = rewards_tensor.shape
        self._require_shape("active_mask_tensor", active_mask_tensor, expected_q_shape)
        self._require_shape("bootstrap_mask_tensor", bootstrap_mask_tensor, expected_q_shape)

        per_agent_bootstrap_mask = self._per_agent_bootstrap_mask(bootstrap_mask_tensor)

        with torch.no_grad():
            next_actions = [
                self.target_actor(next_obs_tensor[:, i, :]) * per_agent_bootstrap_mask[:, i:i + 1]
                for i in range(self.num_agents)
            ]
            next_actions_tensor: torch.Tensor = torch.cat(next_actions, dim=1)
            target_q_value = self.target_critic(next_obs_flat, next_actions_tensor)
            self._require_shape("target_q_value", target_q_value, expected_q_shape)
            y = self._per_agent_targets(rewards_tensor, target_q_value, per_agent_bootstrap_mask)
            self._require_shape("y", y, expected_q_shape)

        current_q_value = self.critic(obs_flat, actions_flat)
        self._require_shape("current_q_value", current_q_value, expected_q_shape)
        critic_loss = masked_mean((current_q_value - y).pow(2), active_mask_tensor)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        pred_actions = [
            self.actor(obs_tensor[:, i, :]) * active_mask_tensor[:, i:i + 1]
            for i in range(self.num_agents)
        ]
        pred_actions_flat = torch.cat(pred_actions, dim=1)

        for param in self.critic.parameters():
            param.requires_grad_(False)
        actor_q = self.critic(obs_flat, pred_actions_flat)
        self._require_shape("actor_q", actor_q, expected_q_shape)
        actor_loss = -masked_mean(actor_q, active_mask_tensor)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
        self.actor_optimizer.step()
        for param in self.critic.parameters():
            param.requires_grad_(True)

        soft_update(self.target_actor, self.actor, config.UPDATE_FACTOR)
        soft_update(self.target_critic, self.critic, config.UPDATE_FACTOR)
        valid_q = current_q_value[active_mask_tensor.bool()]
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
            "q_value_mean": float(valid_q.mean().item()) if valid_q.numel() else 0.0,
            "q_value_std": float(valid_q.std(unbiased=False).item()) if valid_q.numel() else 0.0,
            "noise_scale": self.noise[0].scale,
        }

    def _init_target_networks(self) -> None:
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def reset(self) -> None:
        for n in self.noise:
            n.reset()

    def save(self, directory: str) -> None:
        torch.save(
            {
                "checkpoint_format": self.CHECKPOINT_FORMAT,
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "num_agents": self.num_agents,
                "actor_type": "shared",
                "critic_type": "shared_vector",
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "noise_scales": [noise.scale for noise in self.noise],
            },
            os.path.join(directory, "maddpg.pt"),
        )

    def load(self, directory: str) -> None:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Model directory not found: {directory}")

        checkpoint_path = os.path.join(directory, "maddpg.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Shared MADDPG checkpoint not found: {checkpoint_path}")

        checkpoint: dict = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_format = checkpoint.get("checkpoint_format")
        checkpoint_version = checkpoint.get("checkpoint_version")
        checkpoint_num_agents = checkpoint.get("num_agents")
        actor_type = checkpoint.get("actor_type")
        critic_type = checkpoint.get("critic_type")

        if checkpoint_format != self.CHECKPOINT_FORMAT:
            raise ValueError(
                "Incompatible MADDPG checkpoint format: "
                f"expected '{self.CHECKPOINT_FORMAT}', got {checkpoint_format!r}"
            )
        if checkpoint_version != self.CHECKPOINT_VERSION:
            raise ValueError(
                "Incompatible MADDPG checkpoint version: "
                f"expected {self.CHECKPOINT_VERSION}, got {checkpoint_version!r}"
            )
        if checkpoint_num_agents != self.num_agents:
            raise ValueError(
                "Incompatible MADDPG checkpoint agent count: "
                f"expected {self.num_agents}, got {checkpoint_num_agents!r}"
            )
        if actor_type != "shared" or critic_type != "shared_vector":
            raise ValueError(
                "Incompatible MADDPG checkpoint architecture: "
                f"expected shared actor/shared vector critic, got actor_type={actor_type!r}, "
                f"critic_type={critic_type!r}"
            )

        noise_scales = checkpoint.get("noise_scales")
        if noise_scales is None:
            raise ValueError("Incompatible MADDPG checkpoint: missing noise_scales")
        if len(noise_scales) != self.num_agents:
            raise ValueError(
                "Incompatible MADDPG checkpoint noise_scales length: "
                f"expected {self.num_agents}, got {len(noise_scales)}"
            )

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        for noise, scale in zip(self.noise, noise_scales):
            noise.scale = scale
