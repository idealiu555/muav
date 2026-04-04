import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config
from marl_models.attention import AttentionEncoder

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


def _validate_attention_obs_dim(obs_dim: int) -> None:
    if obs_dim != config.OBS_DIM_SINGLE:
        raise ValueError(
            f"Attention MAPPO expects obs_dim {config.OBS_DIM_SINGLE}, got {obs_dim}"
        )


def _reshape_share_obs(share_obs: torch.Tensor, num_agents: int, obs_dim: int) -> torch.Tensor:
    _validate_share_obs(share_obs, num_agents, obs_dim)
    return share_obs.reshape(share_obs.shape[0], num_agents, obs_dim)


class _GaussianPolicyHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(input_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.mean: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, features: torch.Tensor) -> Normal:
        x: torch.Tensor = F.silu(self.ln1(self.fc1(features)))
        x = F.silu(self.ln2(self.fc2(x)))
        mean: torch.Tensor = self.mean(x)
        log_std: torch.Tensor = torch.clamp(self.log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        std: torch.Tensor = torch.exp(log_std)
        return Normal(mean, std)


class _ScalarValueHead(nn.Module):
    def __init__(self, context_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(context_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc3: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln3: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = F.silu(self.ln1(self.fc1(context)))
        residual = x
        x = F.silu(self.ln2(self.fc2(x)))
        x = x + residual
        residual = x
        x = F.silu(self.ln3(self.fc3(x)))
        x = x + residual
        return self.out(x).squeeze(-1)


class _TeamContextConditioner(nn.Module):
    def __init__(self, agent_dim: int, context_dim: int) -> None:
        super().__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(agent_dim, config.MLP_HIDDEN_DIM))
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, context_dim * 2), std=0.01)

    def forward(self, agent_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conditioned = F.silu(self.fc1(agent_features))
        gamma_beta = self.fc2(conditioned)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return torch.tanh(gamma), beta


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.policy = _GaussianPolicyHead(obs_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Normal:
        return self.policy(obs)


class AttentionActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        _validate_attention_obs_dim(obs_dim)
        self.encoder = AttentionEncoder()
        self.policy = _GaussianPolicyHead(self.encoder.output_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Normal:
        encoded = self.encoder(obs)
        return self.policy(encoded)


class CriticNetwork(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.share_obs_dim = num_agents * obs_dim
        self.value_head = _ScalarValueHead(self.share_obs_dim)

    def forward(self, share_obs: torch.Tensor) -> torch.Tensor:
        _validate_share_obs(share_obs, self.num_agents, self.obs_dim)
        return self.value_head(share_obs)


class AttentionCriticNetwork(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int) -> None:
        super().__init__()
        _validate_attention_obs_dim(obs_dim)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.encoder = AttentionEncoder()
        self.encoded_share_obs_dim = num_agents * self.encoder.output_dim
        self.context_norm: nn.LayerNorm = nn.LayerNorm(self.encoded_share_obs_dim)
        self.conditioner = _TeamContextConditioner(
            agent_dim=self.encoder.output_dim,
            context_dim=self.encoded_share_obs_dim,
        )
        self.value_head = _ScalarValueHead(self.encoded_share_obs_dim)

    def forward(self, share_obs: torch.Tensor) -> torch.Tensor:
        joint_obs = _reshape_share_obs(share_obs, self.num_agents, self.obs_dim)
        batch_size = joint_obs.shape[0]
        encoded = self.encoder(joint_obs.reshape(batch_size * self.num_agents, self.obs_dim))
        agent_encodings = encoded.reshape(batch_size, self.num_agents, self.encoder.output_dim)
        team_context = agent_encodings.reshape(batch_size, self.encoded_share_obs_dim)
        normalized_context = self.context_norm(team_context).unsqueeze(1).expand(-1, self.num_agents, -1)
        gamma, beta = self.conditioner(agent_encodings)
        modulated_context = (1.0 + gamma) * normalized_context + beta
        per_agent_values = self.value_head(modulated_context.reshape(batch_size * self.num_agents, self.encoded_share_obs_dim))
        return per_agent_values.reshape(batch_size, self.num_agents)
