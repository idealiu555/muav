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
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"{name} trailing dim must be {expected_dim}, got {tensor.shape[-1]}.")


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()
        hidden_dim = config.BASE_ACTOR_HIDDEN_DIM
        self.fc1: nn.Linear = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.act1: nn.SiLU = nn.SiLU()
        self.fc2: nn.Linear = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.act2: nn.SiLU = nn.SiLU()
        self.mean: nn.Linear = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std: nn.Linear = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

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
        hidden_dim = config.ATTENTION_ACTOR_HIDDEN_DIM
        self.fc1: nn.Linear = layer_init(nn.Linear(self.encoder.output_dim, hidden_dim))
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.act1: nn.SiLU = nn.SiLU()
        self.fc2: nn.Linear = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.act2: nn.SiLU = nn.SiLU()
        self.mean: nn.Linear = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std: nn.Linear = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

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
        hidden_dim = config.BASE_CRITIC_HIDDEN_DIM
        self.fc1: nn.Linear = layer_init(nn.Linear(total_obs_dim + total_action_dim, hidden_dim))
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.act1: nn.SiLU = nn.SiLU()
        self.fc2: nn.Linear = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.act2: nn.SiLU = nn.SiLU()
        self.out: nn.Linear = layer_init(nn.Linear(hidden_dim, num_agents))

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        _validate_rank("joint_obs", joint_obs, 2)
        _validate_rank("joint_action", joint_action, 2)
        _validate_trailing_dim("joint_obs", joint_obs, self.total_obs_dim)
        _validate_trailing_dim("joint_action", joint_action, self.total_action_dim)
        x: torch.Tensor = torch.cat([joint_obs, joint_action], dim=1)
        x = self.act1(self.ln1(self.fc1(x)))
        x = self.act2(self.ln2(self.fc2(x)))
        return self.out(x)


class LocalAttentionCriticNetwork(nn.Module):
    """Critic with per-agent local entity encoding, flattened concat, and MLP — no agent self-attention."""

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
        hidden_dim = config.ATTENTION_CRITIC_HIDDEN_DIM
        self.fc1: nn.Linear = layer_init(nn.Linear(critic_input_dim, hidden_dim))
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.act1: nn.SiLU = nn.SiLU()
        self.fc2: nn.Linear = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.act2: nn.SiLU = nn.SiLU()
        self.out: nn.Linear = layer_init(nn.Linear(hidden_dim, num_agents))

    def forward(
        self,
        joint_obs: torch.Tensor,
        joint_action: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if joint_obs.dim() == 2:
            _validate_rank("joint_action", joint_action, 2)
            _validate_trailing_dim("joint_obs", joint_obs, self.total_obs_dim)
            _validate_trailing_dim("joint_action", joint_action, self.total_action_dim)
            batch_size = joint_obs.shape[0]
            obs_per_agent = joint_obs.reshape(batch_size * self.num_agents, self.obs_dim)
            encoded_obs = self.encoder(obs_per_agent).reshape(
                batch_size,
                self.num_agents,
                -1,
            )
            actions_per_agent = joint_action.reshape(batch_size, self.num_agents, self.action_dim)
        elif joint_obs.dim() == 3:
            _validate_rank("joint_action", joint_action, 3)
            if joint_obs.shape[1] != self.num_agents or joint_action.shape[1] != self.num_agents:
                raise ValueError(
                    f"Agent count mismatch: joint_obs={joint_obs.shape[1]}, "
                    f"joint_action={joint_action.shape[1]}, expected {self.num_agents}"
                )
            if joint_obs.shape[0] != joint_action.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: joint_obs={joint_obs.shape[0]}, joint_action={joint_action.shape[0]}"
                )
            _validate_trailing_dim("joint_obs", joint_obs, self.obs_dim)
            _validate_trailing_dim("joint_action", joint_action, self.action_dim)
            batch_size = joint_obs.shape[0]
            encoded_obs = self.encoder(joint_obs.reshape(batch_size * self.num_agents, self.obs_dim))
            encoded_obs = encoded_obs.reshape(batch_size, self.num_agents, -1)
            actions_per_agent = joint_action
        else:
            raise ValueError(f"joint_obs must have rank 2 or 3, got shape {tuple(joint_obs.shape)}.")

        if agent_mask is not None:
            _validate_rank("agent_mask", agent_mask, 2)
            if agent_mask.shape != (batch_size, self.num_agents):
                raise ValueError(
                    f"agent_mask must have shape [{batch_size}, {self.num_agents}], "
                    f"got {tuple(agent_mask.shape)}."
                )
            mask = (agent_mask > 0).to(encoded_obs.dtype).unsqueeze(-1)
            encoded_obs = encoded_obs * mask
            actions_per_agent = actions_per_agent * mask

        x: torch.Tensor = torch.cat(
            [
                encoded_obs.reshape(batch_size, -1),
                actions_per_agent.reshape(batch_size, -1),
            ],
            dim=1,
        )
        x = self.act1(self.ln1(self.fc1(x)))
        x = self.act2(self.ln2(self.fc2(x)))
        return self.out(x)


class AgentSelfAttentionBlock(nn.Module):
    """Pre-LN residual self-attention block for agent-level attention."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            layer_init(nn.Linear(dim, dim * ffn_mult)),
            nn.SiLU(),
            layer_init(nn.Linear(dim * ffn_mult, dim)),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        *,
        need_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        y = self.norm1(x)
        y, weights = self.attn(
            y,
            y,
            y,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + y
        z = self.norm2(x)
        x = x + self.ffn(z)
        if need_weights:
            return x, weights
        return x


class AgentSelfAttentionCriticNetwork(nn.Module):
    """MASAC critic with agent-level self-attention over per-UAV tokens.

    Encodes each agent's local observation via a shared AttentionEncoder, builds
    per-agent tokens from local encoding + action + learned agent-id embedding, applies
    masked agent-level self-attention, and outputs per-agent Q values through a shared
    Q head over the attended context and direct action input.
    """

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        _validate_attention_obs_dim(obs_dim)
        if config.MASAC_AGENT_ATTENTION_DIM <= 0:
            raise ValueError(
                f"MASAC_AGENT_ATTENTION_DIM must be > 0, got {config.MASAC_AGENT_ATTENTION_DIM}"
            )
        if config.MASAC_AGENT_ATTENTION_HEADS <= 0:
            raise ValueError(
                f"MASAC_AGENT_ATTENTION_HEADS must be > 0, got {config.MASAC_AGENT_ATTENTION_HEADS}"
            )
        if config.MASAC_AGENT_ATTENTION_DIM % config.MASAC_AGENT_ATTENTION_HEADS != 0:
            raise ValueError(
                f"MASAC_AGENT_ATTENTION_DIM ({config.MASAC_AGENT_ATTENTION_DIM}) "
                f"must be divisible by MASAC_AGENT_ATTENTION_HEADS ({config.MASAC_AGENT_ATTENTION_HEADS})"
            )
        if config.MASAC_AGENT_ID_DIM <= 0:
            raise ValueError(f"MASAC_AGENT_ID_DIM must be > 0, got {config.MASAC_AGENT_ID_DIM}")
        if config.MASAC_AGENT_ATTENTION_LAYERS < 1:
            raise ValueError(f"MASAC_AGENT_ATTENTION_LAYERS must be >= 1, got {config.MASAC_AGENT_ATTENTION_LAYERS}")
        if config.MASAC_AGENT_ATTENTION_FFN_MULT <= 0:
            raise ValueError(
                f"MASAC_AGENT_ATTENTION_FFN_MULT must be > 0, got {config.MASAC_AGENT_ATTENTION_FFN_MULT}"
            )

        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.encoder = AttentionEncoder()
        self.agent_id_embedding = nn.Embedding(num_agents, config.MASAC_AGENT_ID_DIM)

        token_input_dim = self.encoder.output_dim + action_dim + config.MASAC_AGENT_ID_DIM
        self.token_projection = nn.Sequential(
            layer_init(nn.Linear(token_input_dim, config.MASAC_AGENT_ATTENTION_DIM)),
            nn.LayerNorm(config.MASAC_AGENT_ATTENTION_DIM),
        )

        self.attention_blocks = nn.ModuleList([
            AgentSelfAttentionBlock(
                config.MASAC_AGENT_ATTENTION_DIM,
                config.MASAC_AGENT_ATTENTION_HEADS,
                config.MASAC_AGENT_ATTENTION_FFN_MULT,
            )
            for _ in range(config.MASAC_AGENT_ATTENTION_LAYERS)
        ])

        q_input_dim = config.MASAC_AGENT_ATTENTION_DIM + action_dim
        hidden_dim = config.ATTENTION_CRITIC_HIDDEN_DIM
        self.q_head = nn.Sequential(
            layer_init(nn.Linear(q_input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(obs, actions, agent_mask)
        x, valid_query_mask, safe_key_padding_mask = self._build_tokens(obs, actions, agent_mask)
        for block in self.attention_blocks:
            x = block(x, safe_key_padding_mask)

        context = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.where(valid_query_mask.unsqueeze(-1), context, torch.zeros_like(context))

        q_input = torch.cat([context, actions], dim=-1)
        q_values = self.q_head(q_input).squeeze(-1)
        return torch.where(valid_query_mask, q_values, torch.zeros_like(q_values))

    def _validate_inputs(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> None:
        _validate_rank("obs", obs, 3)
        _validate_rank("actions", actions, 3)
        _validate_rank("agent_mask", agent_mask, 2)
        if obs.shape[1] != self.num_agents or actions.shape[1] != self.num_agents or agent_mask.shape[1] != self.num_agents:
            raise ValueError(
                f"Agent count mismatch: obs={obs.shape[1]}, actions={actions.shape[1]}, "
                f"mask={agent_mask.shape[1]}, expected {self.num_agents}"
            )
        if obs.shape[0] != actions.shape[0] or obs.shape[0] != agent_mask.shape[0]:
            raise ValueError(
                f"Batch size mismatch: obs={obs.shape[0]}, actions={actions.shape[0]}, "
                f"mask={agent_mask.shape[0]}"
            )
        _validate_trailing_dim("obs", obs, self.obs_dim)
        _validate_trailing_dim("actions", actions, self.action_dim)

    def _build_tokens(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = obs.shape[0]
        obs_flat = obs.reshape(batch_size * self.num_agents, self.obs_dim)
        encoded_obs = self.encoder(obs_flat).reshape(batch_size, self.num_agents, -1)
        agent_ids = torch.arange(self.num_agents, device=obs.device)
        agent_id_emb = self.agent_id_embedding(agent_ids).unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat([encoded_obs, actions, agent_id_emb], dim=-1)
        tokens = self.token_projection(tokens)
        valid_query_mask = agent_mask > 0
        key_padding_mask = ~valid_query_mask
        safe_key_padding_mask = key_padding_mask.clone()
        safe_key_padding_mask[key_padding_mask.all(dim=1)] = False
        return tokens, valid_query_mask, safe_key_padding_mask

    @torch.no_grad()
    def attention_diagnostics(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> dict[str, float]:
        self._validate_inputs(obs, actions, agent_mask)
        x, valid_query_mask, safe_key_padding_mask = self._build_tokens(obs, actions, agent_mask)
        for block in self.attention_blocks[:-1]:
            x = block(x, safe_key_padding_mask)
        _, weights = self.attention_blocks[-1](x, safe_key_padding_mask, need_weights=True)

        valid_key_mask = valid_query_mask[:, None, None, :]
        expanded_query_mask = valid_query_mask[:, None, :, None]
        valid_weights = weights.masked_fill(~(expanded_query_mask & valid_key_mask), 0.0)
        valid_query_head_mask = expanded_query_mask.squeeze(-1).expand(-1, weights.shape[1], -1)
        if not valid_query_head_mask.any():
            return {"agent_attention_entropy": 0.0, "agent_attention_max_weight": 0.0}
        entropy_per_query = -(
            valid_weights * valid_weights.clamp_min(config.EPSILON).log()
        ).sum(dim=-1)
        return {
            "agent_attention_entropy": float(entropy_per_query[valid_query_head_mask].mean().item()),
            "agent_attention_max_weight": float(valid_weights.max().item()),
        }
