from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import marl_models.attention as attention_module
from marl_models.maddpg.agents import ActorNetwork, CriticNetwork
from marl_models.maddpg.maddpg import MADDPG


def _make_batch(batch_size: int, num_agents: int, obs_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "obs": rng.standard_normal((batch_size, num_agents, obs_dim), dtype=np.float32),
        "actions": rng.uniform(-1.0, 1.0, size=(batch_size, num_agents, action_dim)).astype(np.float32),
        "rewards": rng.standard_normal((batch_size, num_agents), dtype=np.float32),
        "next_obs": rng.standard_normal((batch_size, num_agents, obs_dim), dtype=np.float32),
        "active_mask": np.ones((batch_size, num_agents), dtype=np.float32),
        "bootstrap_mask": np.ones((batch_size, num_agents), dtype=np.float32),
    }


def test_standard_maddpg_networks_use_relu_and_expected_shapes() -> None:
    actor = ActorNetwork(obs_dim=12, action_dim=3)
    critic = CriticNetwork(total_obs_dim=36, total_action_dim=9)

    actor_output = actor(torch.randn(4, 12))
    critic_output = critic(torch.randn(4, 36), torch.randn(4, 9))

    assert actor.fc1.in_features == 12
    assert critic.fc1.in_features == 45
    assert actor_output.shape == (4, 3)
    assert critic_output.shape == (4, 1)
    assert torch.all(actor_output <= 1.0)
    assert torch.all(actor_output >= -1.0)
    assert not any(isinstance(module, torch.nn.SiLU) for module in actor.modules())
    assert not any(isinstance(module, torch.nn.SiLU) for module in critic.modules())


def test_obsolete_maddpg_attention_pooling_symbols_are_removed() -> None:
    assert not hasattr(attention_module, "MeanPoolingEncoder")
    assert not hasattr(attention_module, "AgentPoolingAttention")
    assert not hasattr(attention_module, "AgentPoolingValue")


def test_maddpg_ignores_attention_flag_and_uses_standard_networks(monkeypatch) -> None:
    for use_attention in (False, True):
        monkeypatch.setattr(config, "USE_ATTENTION", use_attention)
        model = MADDPG("maddpg", num_agents=3, obs_dim=8, action_dim=2, device="cpu")

        assert all(isinstance(actor, ActorNetwork) for actor in model.actors)
        assert all(isinstance(actor, ActorNetwork) for actor in model.target_actors)
        assert all(isinstance(critic, CriticNetwork) for critic in model.critics)
        assert all(isinstance(critic, CriticNetwork) for critic in model.target_critics)
        assert not hasattr(model, "shared_encoder")
        assert not hasattr(model, "target_shared_encoder")
        assert not hasattr(model, "shared_encoder_optimizer")


def test_maddpg_registers_modules_in_state_dict() -> None:
    model = MADDPG(
        "maddpg",
        num_agents=2,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )

    assert isinstance(model.actors, torch.nn.ModuleList)
    assert isinstance(model.critics, torch.nn.ModuleList)
    assert isinstance(model.target_actors, torch.nn.ModuleList)
    assert isinstance(model.target_critics, torch.nn.ModuleList)
    assert "actors.0.fc1.weight" in model.state_dict()
    assert "critics.0.fc1.weight" in model.state_dict()


def test_maddpg_update_runs_without_attention_branch(monkeypatch) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)
    model = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")

    stats = model.update(_make_batch(batch_size=4, num_agents=2, obs_dim=8, action_dim=2))

    assert {
        "actor_loss",
        "critic_loss",
        "actor_grad_norm",
        "critic_grad_norm",
        "q_value_mean",
        "q_value_std",
        "noise_scale",
    } <= stats.keys()
    assert all(np.isfinite(float(value)) for value in stats.values())
