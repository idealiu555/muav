import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

import config

# Added layer normalization and orthogonal initialization for better training stability


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def _validate_share_obs(share_obs: torch.Tensor, num_agents: int, obs_dim: int) -> None:
    if share_obs.ndim != 2:
        raise ValueError(f"Expected share_obs rank 2, got shape {tuple(share_obs.shape)}")
    expected_dim = num_agents * obs_dim
    if share_obs.shape[1] != expected_dim:
        raise ValueError(f"Expected share_obs trailing dim {expected_dim}, got {share_obs.shape[1]}")


class _GaussianPolicyHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_norm: nn.LayerNorm = nn.LayerNorm(input_dim)
        self.fc1: nn.Linear = layer_init(nn.Linear(input_dim, hidden_dim))
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.fc2: nn.Linear = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.activation: nn.SiLU = nn.SiLU()
        self.mean: nn.Linear = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, features: torch.Tensor) -> Normal:
        x: torch.Tensor = self.input_norm(features)
        x = self.ln1(self.activation(self.fc1(x)))
        x = self.ln2(self.activation(self.fc2(x)))
        mean: torch.Tensor = self.mean(x)
        log_std: torch.Tensor = torch.clamp(self.log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        std: torch.Tensor = torch.exp(log_std)
        return Normal(mean, std)


class _VectorValueHead(nn.Module):
    def __init__(self, context_dim: int, num_agents: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_norm: nn.LayerNorm = nn.LayerNorm(context_dim)
        self.fc1: nn.Linear = layer_init(nn.Linear(context_dim, hidden_dim))
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.fc2: nn.Linear = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.fc3: nn.Linear = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln3: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.activation: nn.SiLU = nn.SiLU()
        self.out: nn.Linear = layer_init(nn.Linear(hidden_dim, num_agents), std=1.0)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.input_norm(context)
        x = self.ln1(self.activation(self.fc1(x)))
        x = self.ln2(self.activation(self.fc2(x)))
        x = self.ln3(self.activation(self.fc3(x)))
        return self.out(x)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.policy = _GaussianPolicyHead(obs_dim, action_dim, config.BASE_ACTOR_HIDDEN_DIM)

    def forward(self, obs: torch.Tensor) -> Normal:
        return self.policy(obs)


class CriticNetwork(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.share_obs_dim = num_agents * obs_dim
        self.value_head = _VectorValueHead(self.share_obs_dim, num_agents, config.BASE_CRITIC_HIDDEN_DIM)

    def forward(self, share_obs: torch.Tensor) -> torch.Tensor:
        _validate_share_obs(share_obs, self.num_agents, self.obs_dim)
        return self.value_head(share_obs)

