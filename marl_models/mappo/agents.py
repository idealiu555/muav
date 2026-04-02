import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config
from marl_models.attention import AttentionEncoder
from marl_models.attention import AgentPoolingValue
from marl_models.attention import MeanPoolingEncoder

# Added layer normalization and orthogonal initialization for better training stability


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def _validate_joint_obs(joint_obs: torch.Tensor, num_agents: int, obs_dim: int) -> None:
    if joint_obs.ndim != 3:
        raise ValueError(f"Expected joint_obs rank 3, got shape {tuple(joint_obs.shape)}")
    expected_shape = (joint_obs.shape[0], num_agents, obs_dim)
    if joint_obs.shape[1:] != expected_shape[1:]:
        raise ValueError(f"Expected joint_obs shape {expected_shape}, got {tuple(joint_obs.shape)}")


def _validate_attention_obs_dim(obs_dim: int) -> None:
    if obs_dim != config.OBS_DIM_SINGLE:
        raise ValueError(
            f"Attention MAPPO expects obs_dim {config.OBS_DIM_SINGLE}, got {obs_dim}"
        )


def _masked_mean(tokens: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    if active_mask.shape[:2] != tokens.shape[:2]:
        raise ValueError(
            f"Expected active_mask leading shape {tuple(tokens.shape[:2])}, got {tuple(active_mask.shape)}"
        )
    if active_mask.dtype != torch.bool:
        raise TypeError(f"active_mask must use dtype torch.bool, got {active_mask.dtype}")

    weights = active_mask.unsqueeze(-1).to(dtype=tokens.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


def _agent_one_hot(agent_index: torch.Tensor, num_agents: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if agent_index.ndim != 1:
        raise ValueError(f"Expected agent_index rank 1, got shape {tuple(agent_index.shape)}")
    if agent_index.dtype != torch.long:
        agent_index = agent_index.long()
    if agent_index.numel() == 0:
        return torch.zeros((0, num_agents), device=device, dtype=dtype)
    if agent_index.min().item() < 0 or agent_index.max().item() >= num_agents:
        raise ValueError(f"agent_index must be in [0, {num_agents}), got values {agent_index.tolist()}")
    return F.one_hot(agent_index, num_classes=num_agents).to(device=device, dtype=dtype)


def _resolve_sample_index(
    num_contexts: int,
    agent_index: torch.Tensor,
    sample_index: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if agent_index.ndim != 1:
        raise ValueError(f"Expected agent_index rank 1, got shape {tuple(agent_index.shape)}")

    if sample_index is None:
        if agent_index.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        if agent_index.numel() == num_contexts:
            return torch.arange(num_contexts, dtype=torch.long, device=device)
        if num_contexts == 1:
            return torch.zeros(agent_index.shape[0], dtype=torch.long, device=device)
        raise ValueError(
            "sample_index is required when the number of agent queries does not match the number of joint contexts"
        )

    if sample_index.ndim != 1:
        raise ValueError(f"Expected sample_index rank 1, got shape {tuple(sample_index.shape)}")
    if sample_index.shape != agent_index.shape:
        raise ValueError(
            f"Expected sample_index shape {tuple(agent_index.shape)}, got {tuple(sample_index.shape)}"
        )
    sample_index = sample_index.to(device=device, dtype=torch.long)
    if sample_index.numel() == 0:
        return sample_index
    if sample_index.min().item() < 0 or sample_index.max().item() >= num_contexts:
        raise ValueError(f"sample_index must be in [0, {num_contexts}), got values {sample_index.tolist()}")
    return sample_index


def _resolve_sample_active_mask(
    active_mask: torch.Tensor,
    resolved_sample_index: torch.Tensor,
    agent_index: torch.Tensor,
) -> torch.Tensor:
    if active_mask.ndim != 2:
        raise ValueError(f"Expected active_mask rank 2, got shape {tuple(active_mask.shape)}")
    if active_mask.dtype != torch.bool:
        raise TypeError(f"active_mask must use dtype torch.bool, got {active_mask.dtype}")
    if resolved_sample_index.shape != agent_index.shape:
        raise ValueError(
            f"Expected resolved_sample_index shape {tuple(agent_index.shape)}, got {tuple(resolved_sample_index.shape)}"
        )

    agent_index = agent_index.to(device=active_mask.device, dtype=torch.long)
    if agent_index.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=active_mask.device)
    if agent_index.min().item() < 0 or agent_index.max().item() >= active_mask.shape[1]:
        raise ValueError(f"agent_index must be in [0, {active_mask.shape[1]}), got values {agent_index.tolist()}")
    return active_mask[resolved_sample_index, agent_index]


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


class _AgentConditionedValueHead(nn.Module):
    def __init__(self, context_dim: int, num_agents: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.fc1: nn.Linear = layer_init(nn.Linear(context_dim + num_agents, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc3: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln3: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))

    def forward(self, context: torch.Tensor, agent_index: torch.Tensor) -> torch.Tensor:
        agent_one_hot: torch.Tensor = _agent_one_hot(
            agent_index,
            self.num_agents,
            device=context.device,
            dtype=context.dtype,
        )
        x: torch.Tensor = torch.cat([context, agent_one_hot], dim=-1)
        x = F.silu(self.ln1(self.fc1(x)))
        residual = x
        x = F.silu(self.ln2(self.fc2(x)))
        x = x + residual
        residual = x
        x = F.silu(self.ln3(self.fc3(x)))
        x = x + residual
        return self.out(x).squeeze(-1)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()
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
        super(CriticNetwork, self).__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.encoder = MeanPoolingEncoder()
        self.value_head = _AgentConditionedValueHead(self.encoder.output_dim, num_agents)

    def _encode_joint_obs(self, joint_obs: torch.Tensor) -> torch.Tensor:
        _validate_joint_obs(joint_obs, self.num_agents, self.obs_dim)
        batch_size = joint_obs.shape[0]
        encoded = self.encoder(joint_obs.reshape(batch_size * self.num_agents, self.obs_dim))
        return encoded.view(batch_size, self.num_agents, -1)

    def forward(
        self,
        joint_obs: torch.Tensor,
        agent_index: torch.Tensor,
        active_mask: torch.Tensor,
        sample_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self._encode_joint_obs(joint_obs)
        context = _masked_mean(encoded, active_mask)
        resolved_sample_index = _resolve_sample_index(
            context.shape[0],
            agent_index,
            sample_index,
            device=context.device,
        )
        sample_context = context[resolved_sample_index]
        sample_valid = _resolve_sample_active_mask(active_mask, resolved_sample_index, agent_index)
        values = sample_context.new_zeros(agent_index.shape[0])
        if sample_valid.any():
            values[sample_valid] = self.value_head(sample_context[sample_valid], agent_index[sample_valid])
        return values


class AttentionCriticNetwork(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int) -> None:
        super().__init__()
        _validate_attention_obs_dim(obs_dim)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.encoder = AttentionEncoder()
        self.agent_pooling = AgentPoolingValue(encoder_dim=self.encoder.output_dim)
        self.value_head = _AgentConditionedValueHead(self.encoder.output_dim, num_agents)

    def _encode_joint_obs(self, joint_obs: torch.Tensor) -> torch.Tensor:
        _validate_joint_obs(joint_obs, self.num_agents, self.obs_dim)
        batch_size = joint_obs.shape[0]
        encoded = self.encoder(joint_obs.reshape(batch_size * self.num_agents, self.obs_dim))
        return encoded.view(batch_size, self.num_agents, -1)

    def forward(
        self,
        joint_obs: torch.Tensor,
        agent_index: torch.Tensor,
        active_mask: torch.Tensor,
        sample_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self._encode_joint_obs(joint_obs)
        context = self.agent_pooling(encoded, active_mask)
        resolved_sample_index = _resolve_sample_index(
            context.shape[0],
            agent_index,
            sample_index,
            device=context.device,
        )
        sample_context = context[resolved_sample_index]
        sample_valid = _resolve_sample_active_mask(active_mask, resolved_sample_index, agent_index)
        values = sample_context.new_zeros(agent_index.shape[0])
        if sample_valid.any():
            values[sample_valid] = self.value_head(sample_context[sample_valid], agent_index[sample_valid])
        return values
