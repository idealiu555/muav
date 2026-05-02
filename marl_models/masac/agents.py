import config
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from marl_models.attention import AttentionEncoder

# Added layer normalization and orthogonal initialization for better training stability


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def _validate_attention_obs_dim(obs_dim: int) -> None:
    if obs_dim != config.OBS_DIM_SINGLE:
        raise ValueError(
            f"Attention MASAC requires obs_dim == config.OBS_DIM_SINGLE ({config.OBS_DIM_SINGLE}), got {obs_dim}."
        )


def _validate_rank(name: str, tensor: torch.Tensor, expected_rank: int) -> None:
    if tensor.dim() != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got shape {tuple(tensor.shape)}.")


def _validate_trailing_dim(name: str, tensor: torch.Tensor, expected_dim: int) -> None:
    if tensor.shape[1] != expected_dim:
        raise ValueError(f"{name} trailing dim must be {expected_dim}, got {tensor.shape[1]}.")


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(obs_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act1: nn.SiLU = nn.SiLU()
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act2: nn.SiLU = nn.SiLU()
        self.mean: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)
        self.log_std: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.act1(self.ln1(self.fc1(obs)))
        x = self.act2(self.ln2(self.fc2(x)))
        mean: torch.Tensor = self.mean(x)
        log_std: torch.Tensor = torch.clamp(self.log_std(x), min=config.LOG_STD_MIN, max=config.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std: torch.Tensor = log_std.exp()
        dist: Normal = Normal(mean, std)

        # Reparameterization for backpropagation
        x_t: torch.Tensor = dist.rsample()
        y_t: torch.Tensor = torch.tanh(x_t)  # Squash action to be in [-1, 1]
        action: torch.Tensor = y_t

        # Calculate log probability, correcting for the tanh squashing
        # This correction is a key part of the SAC algorithm
        log_prob: torch.Tensor = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + config.EPSILON)
        # Sum over action dimensions to get scalar log_prob per sample
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class AttentionActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        _validate_attention_obs_dim(obs_dim)
        self.obs_dim = obs_dim
        self.encoder = AttentionEncoder()
        self.fc1: nn.Linear = layer_init(nn.Linear(self.encoder.output_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act1: nn.SiLU = nn.SiLU()
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act2: nn.SiLU = nn.SiLU()
        self.mean: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)
        self.log_std: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_rank("obs", obs, 2)
        _validate_trailing_dim("obs", obs, self.obs_dim)
        x: torch.Tensor = self.encoder(obs)
        x = self.act1(self.ln1(self.fc1(x)))
        x = self.act2(self.ln2(self.fc2(x)))
        mean: torch.Tensor = self.mean(x)
        log_std: torch.Tensor = torch.clamp(self.log_std(x), min=config.LOG_STD_MIN, max=config.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std: torch.Tensor = log_std.exp()
        dist: Normal = Normal(mean, std)
        x_t: torch.Tensor = dist.rsample()
        y_t: torch.Tensor = torch.tanh(x_t)
        action: torch.Tensor = y_t
        log_prob: torch.Tensor = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + config.EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim: int, total_action_dim: int, num_agents: int) -> None:
        super(CriticNetwork, self).__init__()
        self.total_obs_dim = total_obs_dim
        self.total_action_dim = total_action_dim
        self.fc1: nn.Linear = layer_init(nn.Linear(total_obs_dim + total_action_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act1: nn.SiLU = nn.SiLU()
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act2: nn.SiLU = nn.SiLU()
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, num_agents))

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        _validate_rank("joint_obs", joint_obs, 2)
        _validate_rank("joint_action", joint_action, 2)
        _validate_trailing_dim("joint_obs", joint_obs, self.total_obs_dim)
        _validate_trailing_dim("joint_action", joint_action, self.total_action_dim)
        x: torch.Tensor = torch.cat([joint_obs, joint_action], dim=1)
        x = self.act1(self.ln1(self.fc1(x)))
        x = self.act2(self.ln2(self.fc2(x)))
        return self.out(x)


class AttentionCriticNetwork(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        _validate_attention_obs_dim(obs_dim)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.total_obs_dim = num_agents * obs_dim
        self.total_action_dim = num_agents * action_dim
        self.encoder = AttentionEncoder()
        critic_input_dim = num_agents * self.encoder.output_dim + self.total_action_dim
        self.fc1: nn.Linear = layer_init(nn.Linear(critic_input_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act1: nn.SiLU = nn.SiLU()
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act2: nn.SiLU = nn.SiLU()
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, num_agents))

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        _validate_rank("joint_obs", joint_obs, 2)
        _validate_rank("joint_action", joint_action, 2)
        _validate_trailing_dim("joint_obs", joint_obs, self.total_obs_dim)
        _validate_trailing_dim("joint_action", joint_action, self.total_action_dim)

        batch_size = joint_obs.shape[0]
        obs_per_agent = joint_obs.reshape(batch_size * self.num_agents, self.obs_dim)
        encoded_obs = self.encoder(obs_per_agent).reshape(batch_size, -1)

        x: torch.Tensor = torch.cat([encoded_obs, joint_action], dim=1)
        x = self.act1(self.ln1(self.fc1(x)))
        x = self.act2(self.ln2(self.fc2(x)))
        return self.out(x)
