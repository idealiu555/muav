from marl_models.base_model import OffPolicyExperienceBatch
import config
import torch
import numpy as np
from collections import deque


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
