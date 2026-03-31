import config
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# Added layer normalization and orthogonal initialization for better training stability


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.mean: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)
        # Fixed log_std (not learnable) - prevents entropy explosion
        # Registered as buffer so it moves with device and saves/loads properly
        self.register_buffer("log_std", torch.zeros(1, action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        x: torch.Tensor = torch.relu(self.ln1(self.fc1(obs)))
        x = torch.relu(self.ln2(self.fc2(x)))
        # Output the mean of the distribution without tanh - let PPO clip handle bounds
        # This prevents gradient issues from double squashing (tanh + clip)
        mean: torch.Tensor = self.mean(x)
        log_std: torch.Tensor = torch.clamp(self.log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        std: torch.Tensor = torch.exp(log_std)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, num_agents: int) -> None:
        super(CriticNetwork, self).__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(state_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        # Output independent value for each agent based on global state
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, num_agents))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.out(x)
