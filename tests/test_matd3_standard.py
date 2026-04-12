from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from marl_models.matd3.matd3 import MATD3


def _make_batch(batch_size: int, num_agents: int, obs_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2)
    return {
        "obs": rng.standard_normal((batch_size, num_agents, obs_dim), dtype=np.float32),
        "actions": rng.uniform(-1.0, 1.0, size=(batch_size, num_agents, action_dim)).astype(np.float32),
        "rewards": rng.standard_normal((batch_size, num_agents), dtype=np.float32),
        "next_obs": rng.standard_normal((batch_size, num_agents, obs_dim), dtype=np.float32),
        "active_mask": np.ones((batch_size, num_agents), dtype=np.float32),
        "bootstrap_mask": np.ones((batch_size, num_agents), dtype=np.float32),
    }


def test_matd3_registers_modules_in_state_dict() -> None:
    model = MATD3(
        "matd3",
        num_agents=2,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )

    assert isinstance(model.actors, torch.nn.ModuleList)
    assert isinstance(model.critics_1, torch.nn.ModuleList)
    assert isinstance(model.critics_2, torch.nn.ModuleList)
    assert isinstance(model.target_actors, torch.nn.ModuleList)
    assert isinstance(model.target_critics_1, torch.nn.ModuleList)
    assert isinstance(model.target_critics_2, torch.nn.ModuleList)
    assert "actors.0.fc1.weight" in model.state_dict()
    assert "critics_1.0.fc1.weight" in model.state_dict()


def test_matd3_update_runs_with_registered_modules() -> None:
    model = MATD3(
        "matd3",
        num_agents=2,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )

    stats = model.update(_make_batch(4, 2, config.OBS_DIM_SINGLE, config.ACTION_DIM))

    assert {
        "critic_loss",
        "critic_grad_norm",
        "q_value_mean",
        "noise_scale",
        "actor_loss",
        "actor_grad_norm",
    } <= stats.keys()
