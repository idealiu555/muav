from collections.abc import Generator

import numpy as np
import torch


class MAPPORolloutBuffer:
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str,
    ) -> None:
        self.num_agents: int = num_agents
        self.obs_dim: int = obs_dim
        self.action_dim: int = action_dim
        self.buffer_size: int = buffer_size
        self.storage_device: torch.device = torch.device("cpu")
        self.train_device: torch.device = torch.device(device)
        self.use_pinned_memory: bool = self.train_device.type == "cuda"

        self.observations: torch.Tensor = self._allocate_float_storage(buffer_size, num_agents, obs_dim)
        self.share_obs: torch.Tensor = self._allocate_float_storage(buffer_size, num_agents * obs_dim)
        self.raw_actions: torch.Tensor = self._allocate_float_storage(buffer_size, num_agents, action_dim)
        self.log_probs: torch.Tensor = self._allocate_float_storage(buffer_size, num_agents)
        self.rewards: torch.Tensor = self._allocate_float_storage(buffer_size, num_agents)
        self.values: torch.Tensor = self._allocate_float_storage(buffer_size, num_agents)
        self.active_masks: torch.Tensor = self._allocate_float_storage(buffer_size, num_agents)
        self.advantages: torch.Tensor = self._allocate_float_storage(buffer_size, num_agents)
        self.returns: torch.Tensor = self._allocate_float_storage(buffer_size, num_agents)

        self.step: int = 0

    def _allocate_float_storage(self, *shape: int) -> torch.Tensor:
        return torch.zeros(
            shape,
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=self.use_pinned_memory,
        )

    def _coerce_float32_array(self, value: np.ndarray | list[float], expected_shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != expected_shape:
            raise ValueError(f"Expected {name} shape {expected_shape}, got {array.shape}")
        return np.ascontiguousarray(array)

    def _copy_into_step(
        self,
        dest: torch.Tensor,
        value: np.ndarray | list[float],
        expected_shape: tuple[int, ...],
        name: str,
    ) -> None:
        src = torch.from_numpy(self._coerce_float32_array(value, expected_shape, name))
        dest.copy_(src)

    def _to_train_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.train_device, non_blocking=self.use_pinned_memory)

    def add(
        self,
        obs: np.ndarray,
        share_obs: np.ndarray,
        raw_actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: list[float],
        values: np.ndarray,
        active_mask: np.ndarray,
    ) -> None:
        if self.step >= self.buffer_size:
            raise ValueError("Rollout buffer overflow")

        self._copy_into_step(self.observations[self.step], obs, (self.num_agents, self.obs_dim), "obs")
        self._copy_into_step(self.share_obs[self.step], share_obs, (self.num_agents * self.obs_dim,), "share_obs")
        self._copy_into_step(self.raw_actions[self.step], raw_actions, (self.num_agents, self.action_dim), "raw_actions")
        self._copy_into_step(self.log_probs[self.step], log_probs, (self.num_agents,), "log_probs")
        self._copy_into_step(self.rewards[self.step], rewards, (self.num_agents,), "rewards")
        self._copy_into_step(self.values[self.step], values, (self.num_agents,), "values")
        self._copy_into_step(self.active_masks[self.step], active_mask, (self.num_agents,), "active_mask")

        self.step += 1

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float) -> None:
        num_steps = self.step
        if num_steps == 0:
            return

        last_gae_lam = torch.zeros(self.num_agents, dtype=torch.float32, device=self.storage_device)
        zero_vec = torch.zeros(self.num_agents, dtype=torch.float32, device=self.storage_device)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_values = zero_vec
                next_non_terminal = zero_vec
            else:
                next_values = self.values[t + 1]
                next_non_terminal = self.active_masks[t] * self.active_masks[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * last_gae_lam * next_non_terminal
            self.advantages[t].copy_(last_gae_lam)

        self.returns[:num_steps].copy_(self.advantages[:num_steps] + self.values[:num_steps])

    def normalize_advantages(self) -> None:
        num_steps = self.step
        if num_steps == 0:
            return

        advantages_slice = self.advantages[:num_steps]
        active_mask = self.active_masks[:num_steps] > 0.5
        normalized_advantages = torch.zeros_like(advantages_slice)

        if not active_mask.any():
            self.advantages[:num_steps].copy_(normalized_advantages)
            return

        valid_advantages = advantages_slice[active_mask]
        mean_adv = valid_advantages.mean()
        std_adv = valid_advantages.std(unbiased=False)

        if torch.isfinite(mean_adv) and torch.isfinite(std_adv):
            normalized_advantages[active_mask] = (valid_advantages - mean_adv) / (std_adv + 1e-5)

        self.advantages[:num_steps].copy_(normalized_advantages)

    def get_batches(self, batch_size: int) -> Generator[dict[str, torch.Tensor], None, None]:
        num_steps = self.step
        num_samples = num_steps * self.num_agents
        if num_samples == 0:
            return

        obs_flat = self.observations[:num_steps].reshape(num_samples, self.obs_dim)
        share_obs_flat = self.share_obs[:num_steps].repeat_interleave(self.num_agents, dim=0)
        raw_actions_flat = self.raw_actions[:num_steps].reshape(num_samples, self.action_dim)
        log_probs_flat = self.log_probs[:num_steps].reshape(num_samples)
        advantages_flat = self.advantages[:num_steps].reshape(num_samples)
        returns_flat = self.returns[:num_steps].reshape(num_samples)
        values_flat = self.values[:num_steps].reshape(num_samples)
        active_masks_flat = self.active_masks[:num_steps].reshape(num_samples)
        agent_indices_flat = torch.arange(self.num_agents, device=self.storage_device).repeat(num_steps)
        permuted_indices = torch.randperm(num_samples, device=self.storage_device)

        for start in range(0, num_samples, batch_size):
            batch_indices = permuted_indices[start : start + batch_size]
            yield {
                "obs": self._to_train_device(obs_flat[batch_indices]),
                "share_obs": self._to_train_device(share_obs_flat[batch_indices]),
                "raw_actions": self._to_train_device(raw_actions_flat[batch_indices]),
                "old_log_probs": self._to_train_device(log_probs_flat[batch_indices]),
                "advantages": self._to_train_device(advantages_flat[batch_indices]),
                "returns": self._to_train_device(returns_flat[batch_indices]),
                "old_values": self._to_train_device(values_flat[batch_indices]),
                "active_mask": self._to_train_device(active_masks_flat[batch_indices]),
                "agent_index": self._to_train_device(agent_indices_flat[batch_indices]),
            }

    def clear(self) -> None:
        self.step = 0
