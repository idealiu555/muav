import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Network architecture optimizations for training stability:
# 1. Orthogonal initialization - better gradient flow at init
# 2. LayerNorm - stable for RL (batch-independent normalization)
# 3. SiLU - smooth nonlinearity used consistently across local MARL models
# 4. Residual connections (Critic) - improves gradient flow in deep networks

from marl_models.attention import AttentionEncoder, AgentPoolingAttention, MeanPoolingEncoder


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()
        self.encoder = MeanPoolingEncoder()
        self.fc1: nn.Linear = layer_init(nn.Linear(self.encoder.output_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)
        
        self.activation = nn.SiLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input)
        x: torch.Tensor = self.activation(self.ln1(self.fc1(encoded)))
        x = self.activation(self.ln2(self.fc2(x)))
        return torch.tanh(self.out(x))


class CriticNetwork(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = MeanPoolingEncoder()
        total_encoded_obs_dim = num_agents * self.encoder.output_dim
        total_action_dim = num_agents * action_dim

        # First layer: input projection (no residual, dimension change)
        self.fc1: nn.Linear = layer_init(nn.Linear(total_encoded_obs_dim + total_action_dim, config.MLP_HIDDEN_DIM))
        self.ln1: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        
        # Hidden layers with residual connections (same dimension)
        self.fc2: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc3: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln3: nn.LayerNorm = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        
        # Output layer
        self.out: nn.Linear = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))
        
        self.activation = nn.SiLU()

    def encode_observations(self, joint_obs: torch.Tensor) -> torch.Tensor:
        batch_size = joint_obs.shape[0]
        obs_reshaped = joint_obs.view(batch_size * self.num_agents, self.obs_dim)
        encoded = self.encoder(obs_reshaped)
        return encoded.view(batch_size, self.num_agents * self.encoder.output_dim)

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor, joint_encoded: torch.Tensor | None = None) -> torch.Tensor:
        if joint_encoded is None:
            joint_encoded = self.encode_observations(joint_obs)

        x: torch.Tensor = torch.cat([joint_encoded, joint_action], dim=1)
        
        # First layer (no residual due to dimension change)
        x = self.activation(self.ln1(self.fc1(x)))
        
        # Second layer with residual connection
        residual = x
        x = self.activation(self.ln2(self.fc2(x)))
        x = x + residual  # Residual connection
        
        # Third layer with residual connection
        residual = x
        x = self.activation(self.ln3(self.fc3(x)))
        x = x + residual  # Residual connection
        
        return self.out(x)


class ActorNetworkWithAttention(nn.Module):
    """使用注意力机制的 Actor 网络"""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        # 注意力编码器
        self.encoder = AttentionEncoder()
        encoder_output_dim = self.encoder.output_dim  # 256

        # MLP 层
        self.fc1 = layer_init(nn.Linear(encoder_output_dim, config.MLP_HIDDEN_DIM))
        self.ln1 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2 = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, action_dim), std=0.01)

        self.activation = nn.SiLU()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 注意力编码
        encoded = self.encoder(obs)
        # MLP 处理
        x = self.activation(self.ln1(self.fc1(encoded)))
        x = self.activation(self.ln2(self.fc2(x)))
        return torch.tanh(self.out(x))


class CriticNetworkWithAttention(nn.Module):
    """使用注意力机制的 Critic 网络（Permutation-Invariant 版本）
    
    使用 AgentPoolingAttention 实现置换不变聚合：
    - 输入维度与 agent 数量无关
    - 支持通过复制权重扩展 agent 数量
    - 所有 agent 共享同一个编码器
    """

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, shared_encoder: "AttentionEncoder") -> None:
        super().__init__()
        self.num_agents = num_agents  # 仅用于 encode_observations，不影响网络结构
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 共享编码器
        self.encoder = shared_encoder
        encoder_output_dim = self.encoder.output_dim  # 256

        # Permutation-Invariant 聚合模块（与 N 无关）
        self.agent_pooling = AgentPoolingAttention(
            encoder_dim=encoder_output_dim,
            action_dim=action_dim,
            action_embed_dim=64,
            num_heads=4
        )
        pooled_dim = self.agent_pooling.output_dim  # 256

        # MLP 层（输入维度固定，与 N 无关）
        self.fc1 = layer_init(nn.Linear(pooled_dim, config.MLP_HIDDEN_DIM))
        self.ln1 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc2 = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln2 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.fc3 = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM))
        self.ln3 = nn.LayerNorm(config.MLP_HIDDEN_DIM)
        self.out = layer_init(nn.Linear(config.MLP_HIDDEN_DIM, 1))

        self.activation = nn.SiLU()

    def encode_observations(self, joint_obs: torch.Tensor, num_agents: int | None = None) -> torch.Tensor:
        """编码所有 agent 的观测，返回 [batch, N, encoder_dim] 格式。
        
        Args:
            joint_obs: [batch, N * obs_dim] 所有 agent 的观测（flat）
            num_agents: agent 数量（可选，用于扩展场景）
        
        Returns:
            encodings: [batch, N, encoder_dim] 编码结果（保持 agent 维度）
        """
        if num_agents is None:
            num_agents = self.num_agents
        
        batch_size = joint_obs.shape[0]
        # 批量化编码：将 [batch, N * obs_dim] 重塑为 [batch * N, obs_dim]，一次前向传播
        reshaped = joint_obs.view(batch_size, num_agents, self.obs_dim).reshape(batch_size * num_agents, self.obs_dim)
        encoded = self.encoder(reshaped)  # [batch * N, encoder_dim]
        return encoded.view(batch_size, num_agents, -1)  # [batch, N, encoder_dim]

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor, 
                joint_encoded: torch.Tensor | None = None,
                num_agents: int | None = None) -> torch.Tensor:
        """
        Args:
            joint_obs: [batch, N * obs_dim] 所有 agent 的观测（仅当 joint_encoded=None 时使用）
            joint_action: [batch, N * action_dim] 所有 agent 的动作
            joint_encoded: [batch, N, encoder_dim] 预计算的编码（可选）
            num_agents: agent 数量（可选，用于扩展场景）
        
        Returns:
            q_value: [batch, 1] Q 值
        """
        if num_agents is None:
            num_agents = self.num_agents
        
        # 获取编码
        if joint_encoded is None:
            joint_encoded = self.encode_observations(joint_obs, num_agents)
        
        # 重塑动作为 [batch, N, action_dim]
        batch_size = joint_encoded.shape[0]
        actions_reshaped = joint_action.view(batch_size, num_agents, self.action_dim)
        
        # Permutation-Invariant 聚合
        pooled = self.agent_pooling(joint_encoded, actions_reshaped)  # [batch, pooled_dim]

        # MLP 处理（带残差连接）
        x = self.activation(self.ln1(self.fc1(pooled)))
        residual = x
        x = self.activation(self.ln2(self.fc2(x)))
        x = x + residual
        residual = x
        x = self.activation(self.ln3(self.fc3(x)))
        x = x + residual

        return self.out(x)
    
    def mlp_parameters(self):
        """返回仅属于 MLP 和 pooling 的参数（不包括共享编码器）"""
        params = []
        # AgentPoolingAttention 参数
        params.extend(self.agent_pooling.parameters())
        # MLP 参数
        params.extend(self.fc1.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.fc3.parameters())
        params.extend(self.ln3.parameters())
        params.extend(self.out.parameters())
        return params
