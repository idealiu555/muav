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


class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim: int, total_action_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        self.fc1: nn.Linear = layer_init(nn.Linear(total_obs_dim + total_action_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act1: nn.SiLU = nn.SiLU()
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.act2: nn.SiLU = nn.SiLU()
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.cat([joint_obs, joint_action], dim=1)
        x = self.act1(self.ln1(self.fc1(x)))
        x = self.act2(self.ln2(self.fc2(x)))
        return self.out(x)
