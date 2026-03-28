from marl_models.base_model import OffPolicyExperienceBatch
import config
import torch
import numpy as np
from collections import deque
from collections.abc import Generator


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.buffer: deque[OffPolicyExperienceBatch] = deque(maxlen=max_size)

    def add(
        self,
        obs: list[np.ndarray],
        actions: np.ndarray,
        rewards: list[float],
        next_obs: list[np.ndarray],
        active_mask: np.ndarray,
        bootstrap_mask: np.ndarray,
    ) -> None:
        """Store one transition tuple with explicit agent activity/bootstrap masks."""
        obs_arr: np.ndarray = np.array(obs)
        next_obs_arr: np.ndarray = np.array(next_obs)
        rewards_arr: np.ndarray = np.array(rewards)
        self.buffer.append(
            {
                "obs": obs_arr,
                "actions": actions,
                "rewards": rewards_arr,
                "next_obs": next_obs_arr,
                "active_mask": active_mask.astype(np.float32, copy=False),
                "bootstrap_mask": bootstrap_mask.astype(np.float32, copy=False),
            }
        )

    def sample(self, batch_size: int) -> OffPolicyExperienceBatch:
        """Sample a batch of experiences."""
        indices: np.ndarray = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch: list[OffPolicyExperienceBatch] = [self.buffer[i] for i in indices]
        keys = batch[0].keys()
        return {key: np.array([experience[key] for experience in batch]) for key in keys}

    def __len__(self) -> int:
        return len(self.buffer)


class RolloutBuffer:
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, state_dim: int, buffer_size: int, device: str) -> None:
        self.num_agents: int = num_agents
        self.obs_dim: int = obs_dim
        self.state_dim: int = state_dim
        self.buffer_size: int = buffer_size
        self.device: str = device

        # Initialize storage
        self.states: np.ndarray = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.observations: np.ndarray = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.pre_tanh_actions: np.ndarray = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.log_probs: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.active_masks: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)

        # For on-policy return/advantage calculation
        self.advantages: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns: np.ndarray = np.zeros((buffer_size, num_agents), dtype=np.float32)

        self.step: int = 0

    def add(
        self,
        state: np.ndarray,
        obs: np.ndarray,
        pre_tanh_actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: list[float],
        values: np.ndarray,
        active_mask: np.ndarray,
    ) -> None:
        if self.step >= self.buffer_size:
            raise ValueError("Rollout buffer overflow")
        self.states[self.step] = state
        self.observations[self.step] = obs
        self.pre_tanh_actions[self.step] = pre_tanh_actions
        self.log_probs[self.step] = log_probs
        self.rewards[self.step] = np.array(rewards)
        self.values[self.step] = values
        self.active_masks[self.step] = active_mask.astype(np.float32, copy=False)

        self.step += 1

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float) -> None:
        """Compute finite-horizon GAE targets for the collected rollout."""
        num_steps = self.step
        last_gae_lam: np.ndarray = np.zeros(self.num_agents, dtype=np.float32)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_values: np.ndarray = np.zeros(self.num_agents, dtype=np.float32)
            else:
                next_values = self.values[t + 1]

            delta: np.ndarray = self.rewards[t] + gamma * next_values - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
            self.advantages[t] = last_gae_lam

        self.returns[:num_steps] = self.advantages[:num_steps] + self.values[:num_steps]

    def get_batches(self, batch_size: int) -> Generator[dict[str, torch.Tensor], None, None]:
        """A generator that yields mini-batches from the buffer."""
        num_steps = self.step
        num_samples: int = num_steps * self.num_agents

        states: np.ndarray = np.repeat(self.states[:num_steps], self.num_agents, axis=0)
        obs: np.ndarray = self.observations[:num_steps].reshape(-1, self.obs_dim)
        pre_tanh_actions: np.ndarray = self.pre_tanh_actions[:num_steps].reshape(-1, self.pre_tanh_actions.shape[-1])
        log_probs: np.ndarray = self.log_probs[:num_steps].reshape(-1)
        advantages: np.ndarray = self.advantages[:num_steps].reshape(-1)
        returns: np.ndarray = self.returns[:num_steps].reshape(-1)
        values: np.ndarray = self.values[:num_steps].reshape(-1)
        active_masks: np.ndarray = self.active_masks[:num_steps].reshape(-1)
        # Create agent indices array to track which agent each sample belongs to
        # When we reshape (buffer_size, num_agents) to (buffer_size * num_agents,)
        # the pattern is: [agent0, agent1, ..., agentN, agent0, agent1, ..., agentN, ...]
        agent_indices: np.ndarray = np.tile(np.arange(self.num_agents), num_steps)

        indices: np.ndarray = np.random.permutation(num_samples)
        
        # Pre-convert all data to tensors on GPU for faster batching
        states_tensor: torch.Tensor = torch.from_numpy(states).to(self.device, non_blocking=True)
        obs_tensor: torch.Tensor = torch.from_numpy(obs).to(self.device, non_blocking=True)
        pre_tanh_actions_tensor: torch.Tensor = torch.from_numpy(pre_tanh_actions).to(self.device, non_blocking=True)
        log_probs_tensor: torch.Tensor = torch.from_numpy(log_probs).to(self.device, non_blocking=True)
        advantages_tensor: torch.Tensor = torch.from_numpy(advantages).to(self.device, non_blocking=True)
        returns_tensor: torch.Tensor = torch.from_numpy(returns).to(self.device, non_blocking=True)
        values_tensor: torch.Tensor = torch.from_numpy(values).to(self.device, non_blocking=True)
        active_masks_tensor: torch.Tensor = torch.from_numpy(active_masks).to(self.device, non_blocking=True)
        agent_indices_tensor: torch.Tensor = torch.from_numpy(agent_indices).to(self.device, non_blocking=True)

        for start in range(0, num_samples, batch_size):
            end: int = start + batch_size
            batch_indices: np.ndarray = indices[start:end]
            batch_idx_tensor: torch.Tensor = torch.from_numpy(batch_indices).to(self.device, non_blocking=True)

            yield {
                "states": states_tensor[batch_idx_tensor],
                "obs": obs_tensor[batch_idx_tensor],
                "pre_tanh_actions": pre_tanh_actions_tensor[batch_idx_tensor],
                "old_log_probs": log_probs_tensor[batch_idx_tensor],
                "advantages": advantages_tensor[batch_idx_tensor],
                "returns": returns_tensor[batch_idx_tensor],
                "old_values": values_tensor[batch_idx_tensor],
                "active_mask": active_masks_tensor[batch_idx_tensor],
                "agent_indices": agent_indices_tensor[batch_idx_tensor],
            }

    def clear(self) -> None:
        self.step = 0


def soft_update(target_net: torch.nn.Module, source_net: torch.nn.Module, tau: float):
    """Performs a soft update of the target network's parameters."""
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.copy_(tau * param + (1.0 - tau) * target_param)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute a numerically stable mean over masked samples."""
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


class GaussianNoise:
    """Gaussian noise with decay for exploration.
    
    支持分层噪声：
    - 3D位移动作(dx, dy, dz)使用正常噪声
    - 波束控制动作使用较小的噪声，因为offset模式下已有质心作为基准
    """

    def __init__(self) -> None:
        self.scale: float = config.INITIAL_NOISE_SCALE

    def sample(self) -> np.ndarray:
        if config.BEAM_CONTROL_ENABLED and config.ACTION_DIM == 5:
            # 分层噪声：3D位移用正常噪声，波束用较小噪声
            movement_noise = np.random.normal(0, self.scale, 3)  # dx, dy, dz
            beam_noise = np.random.normal(0, self.scale * config.BEAM_NOISE_RATIO, 2)  # beam_theta, beam_phi
            return np.concatenate([movement_noise, beam_noise])
        else:
            return np.random.normal(0, self.scale, config.ACTION_DIM)

    def decay(self) -> None:
        self.scale = max(config.MIN_NOISE_SCALE, self.scale * config.NOISE_DECAY_RATE)

    def reset(self) -> None:
        self.scale = config.INITIAL_NOISE_SCALE
