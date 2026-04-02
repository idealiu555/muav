from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from marl_models.attention import AgentPoolingAttention
from marl_models.attention import AttentionEncoder
from marl_models.maddpg.agents import CriticNetworkWithAttention


def test_agent_pooling_returns_zero_for_all_inactive_rows() -> None:
    torch.manual_seed(0)
    module = AgentPoolingAttention(encoder_dim=32, action_dim=3, action_embed_dim=8, num_heads=4)

    agent_encodings = torch.randn(2, 4, 32)
    agent_actions = torch.randn(2, 4, 3)
    active_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )

    pooled = module(agent_encodings, agent_actions, active_mask)

    assert pooled.shape == (2, 32)
    assert torch.allclose(pooled[1], torch.zeros(32, dtype=pooled.dtype), atol=1e-6)


def test_critic_with_attention_ignores_inactive_agents() -> None:
    torch.manual_seed(1)
    shared_encoder = AttentionEncoder()
    critic = CriticNetworkWithAttention(
        num_agents=3,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        shared_encoder=shared_encoder,
    )

    batch_size = 2
    num_agents = 3
    joint_obs = torch.zeros(batch_size, num_agents * config.OBS_DIM_SINGLE)
    joint_encoded = torch.randn(batch_size, num_agents, shared_encoder.output_dim)
    joint_action = torch.randn(batch_size, num_agents * config.ACTION_DIM)
    active_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )

    masked_output = critic(
        joint_obs,
        joint_action,
        joint_encoded=joint_encoded,
        active_mask=active_mask,
    )

    joint_encoded_modified = joint_encoded.clone()
    joint_encoded_modified[:, 2, :] = torch.randn_like(joint_encoded_modified[:, 2, :]) * 100.0
    joint_action_modified = joint_action.clone().view(batch_size, num_agents, config.ACTION_DIM)
    joint_action_modified[:, 2, :] = torch.randn_like(joint_action_modified[:, 2, :]) * 100.0

    masked_output_modified = critic(
        joint_obs,
        joint_action_modified.view(batch_size, -1),
        joint_encoded=joint_encoded_modified,
        active_mask=active_mask,
    )

    assert torch.allclose(masked_output, masked_output_modified, atol=1e-5, rtol=1e-5)


def test_agent_pooling_requires_bool_active_mask() -> None:
    torch.manual_seed(2)
    module = AgentPoolingAttention(encoder_dim=16, action_dim=3, action_embed_dim=8, num_heads=4)

    agent_encodings = torch.randn(1, 3, 16)
    agent_actions = torch.randn(1, 3, 3)
    float_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    try:
        module(agent_encodings, agent_actions, float_mask)
    except TypeError as exc:
        assert "torch.bool" in str(exc)
    else:
        raise AssertionError("Expected AgentPoolingAttention to reject non-bool active_mask")


def test_agent_pooling_value_ignores_inactive_agents() -> None:
    from marl_models.attention import AgentPoolingValue

    torch.manual_seed(4)
    module = AgentPoolingValue(encoder_dim=32, num_heads=4)

    agent_encodings = torch.randn(2, 3, 32)
    active_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )

    pooled = module(agent_encodings, active_mask)

    modified = agent_encodings.clone()
    modified[:, 2, :] = torch.randn_like(modified[:, 2, :]) * 100.0
    pooled_modified = module(modified, active_mask)

    assert pooled.shape == (2, 32)
    assert torch.allclose(pooled, pooled_modified, atol=1e-5, rtol=1e-5)


def test_agent_pooling_value_returns_zero_for_all_inactive_rows() -> None:
    from marl_models.attention import AgentPoolingValue

    torch.manual_seed(5)
    module = AgentPoolingValue(encoder_dim=16, num_heads=4)

    agent_encodings = torch.randn(2, 4, 16)
    active_mask = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
        ],
        dtype=torch.bool,
    )

    pooled = module(agent_encodings, active_mask)

    assert pooled.shape == (2, 16)
    assert torch.allclose(pooled[0], torch.zeros(16, dtype=pooled.dtype), atol=1e-6)
