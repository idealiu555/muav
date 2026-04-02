import config
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        x: torch.Tensor = F.silu(self.ln1(self.fc1(obs)))
        x = F.silu(self.ln2(self.fc2(x)))
        mean: torch.Tensor = self.mean(x)
        log_std: torch.Tensor = torch.clamp(self.log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        std: torch.Tensor = torch.exp(log_std)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        # First layer: input projection (no residual, dimension change)
        self.fc1: nn.Linear = layer_init(nn.Linear(state_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        # Hidden layers with residual connections (same dimension)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc3: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln3: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        # Output layer
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = state
        # First layer (no residual due to dimension change)
        x = F.silu(self.ln1(self.fc1(x)))
        # Second layer with residual connection
        residual = x
        x = F.silu(self.ln2(self.fc2(x)))
        x = x + residual
        # Third layer with residual connection
        residual = x
        x = F.silu(self.ln3(self.fc3(x)))
        x = x + residual
        return self.out(x)
