import config
import torch
import torch.nn as nn
import numpy as np


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        hidden_dim = config.BASE_ACTOR_HIDDEN_DIM
        self.fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.out = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.ln1(self.fc1(obs)))
        x = self.act2(self.ln2(self.fc2(x)))
        return torch.tanh(self.out(x))


class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim: int, total_action_dim: int, num_agents: int) -> None:
        super().__init__()
        hidden_dim = config.BASE_CRITIC_HIDDEN_DIM
        self.fc1 = layer_init(nn.Linear(total_obs_dim + total_action_dim, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.out = layer_init(nn.Linear(hidden_dim, num_agents))

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([joint_obs, joint_action], dim=1)
        x = self.act1(self.ln1(self.fc1(x)))
        x = self.act2(self.ln2(self.fc2(x)))
        return self.out(x)
