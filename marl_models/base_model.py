from abc import ABC, abstractmethod
import numpy as np
import torch

OffPolicyExperienceBatch = dict[str, np.ndarray]
OnPolicyExperienceBatch = dict[str, torch.Tensor]
ExperienceBatch = OffPolicyExperienceBatch | OnPolicyExperienceBatch


class MARLModel(ABC):
    """
    Abstract Base Class for Multi-Agent Reinforcement Learning models.
    This class defines the essential methods that any MARL algorithm implementation
    must have to be compatible with the training framework.
    """

    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        self.model_name = model_name
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    def get_action_and_value(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets environment actions, raw policy actions, log probabilities, and per-agent value predictions.
        Essential for on-policy algorithms like PPO/MAPPO.
        """
        raise NotImplementedError("This method is required for on-policy algorithms.")

    def train_on_rollout(self, rollout_buffer: object, *, current_update: int, total_updates: int) -> dict:
        """Optional high-level on-policy training entry point for a full rollout."""
        _ = rollout_buffer
        _ = current_update
        _ = total_updates
        raise NotImplementedError("This model does not support on-policy rollout training.")

    @abstractmethod
    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        """
        Selects actions for all agents based on their observations.
        """
        pass

    @abstractmethod
    def update(self, batch: ExperienceBatch) -> dict:
        """
        Performs a learning update on the model's networks using a batch of experiences.

        Args:
            batch (ExperienceBatch): A dictionary containing either
                on-policy rollout tensors or off-policy replay arrays.
        
        Returns:
            dict: Training statistics (loss, grad norms, etc.). Empty dict if not applicable.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the model's internal state (if any) for a new episode.
        """
        pass

    @abstractmethod
    def save(self, directory: str) -> None:
        pass

    @abstractmethod
    def load(self, directory: str) -> None:
        pass
