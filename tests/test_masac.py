from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from marl_models.masac.agents import ActorNetwork, CriticNetwork
from marl_models.masac.masac import MASAC


def _make_batch(batch_size: int, num_agents: int, obs_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    obs = rng.standard_normal((batch_size, num_agents, obs_dim), dtype=np.float32)
    next_obs = rng.standard_normal((batch_size, num_agents, obs_dim), dtype=np.float32)
    obs[:, :, config.OWN_STATE_DIM - 1] = 1.0
    next_obs[:, :, config.OWN_STATE_DIM - 1] = 1.0
    return {
        "obs": obs,
        "actions": rng.uniform(-1.0, 1.0, size=(batch_size, num_agents, action_dim)).astype(np.float32),
        "rewards": rng.standard_normal((batch_size, num_agents), dtype=np.float32),
        "next_obs": next_obs,
        "active_mask": np.ones((batch_size, num_agents), dtype=np.float32),
        "bootstrap_mask": np.ones((batch_size, num_agents), dtype=np.float32),
    }


def test_masac_registers_modules_and_alpha_parameters() -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.actors, torch.nn.ModuleList)
    assert isinstance(model.critics_1, torch.nn.ModuleList)
    assert isinstance(model.critics_2, torch.nn.ModuleList)
    assert isinstance(model.target_critics_1, torch.nn.ModuleList)
    assert isinstance(model.target_critics_2, torch.nn.ModuleList)
    assert isinstance(model.log_alphas, torch.nn.ParameterList)

    state_dict_keys = model.state_dict().keys()
    assert "actors.0.fc1.weight" in state_dict_keys
    assert "log_alphas.0" in state_dict_keys


def test_masac_networks_use_silu_not_relu() -> None:
    actor = ActorNetwork(obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)
    critic = CriticNetwork(total_obs_dim=2 * config.OBS_DIM_SINGLE, total_action_dim=2 * config.ACTION_DIM)

    actor_output = actor(torch.randn(4, config.OBS_DIM_SINGLE))
    critic_output = critic(torch.randn(4, 2 * config.OBS_DIM_SINGLE), torch.randn(4, 2 * config.ACTION_DIM))

    assert actor_output[0].shape == (4, config.ACTION_DIM)
    assert actor_output[1].shape == (4, config.ACTION_DIM)
    assert critic_output.shape == (4, 1)
    assert any(isinstance(module, torch.nn.SiLU) for module in actor.modules())
    assert any(isinstance(module, torch.nn.SiLU) for module in critic.modules())
    assert not any(isinstance(module, torch.nn.ReLU) for module in actor.modules())
    assert not any(isinstance(module, torch.nn.ReLU) for module in critic.modules())


def test_masac_uses_adam_without_weight_decay() -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    optimizers = [
        model.actor_optimizers[0],
        model.critic_1_optimizers[0],
        model.critic_2_optimizers[0],
        model.alpha_optimizers[0],
    ]
    assert all(isinstance(optimizer, torch.optim.Adam) for optimizer in optimizers)
    assert all(optimizer.defaults["weight_decay"] == 0.0 for optimizer in optimizers)


def test_masac_update_reports_current_alpha_mean() -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    stats = model.update(batch)
    actual_alpha_mean = float(torch.stack([log_alpha.detach().exp() for log_alpha in model.log_alphas]).mean().item())

    assert np.isclose(stats["alpha_mean"], actual_alpha_mean)


def test_masac_actor_step_does_not_accumulate_critic_gradients() -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    obs_tensor = torch.as_tensor(batch["obs"], dtype=torch.float32, device=model.device)
    active_mask_tensor = torch.as_tensor(batch["active_mask"], dtype=torch.float32, device=model.device)
    obs_flat = obs_tensor.reshape(obs_tensor.shape[0], -1)

    model.actor_optimizers[0].zero_grad(set_to_none=True)
    model.critic_1_optimizers[0].zero_grad(set_to_none=True)
    model.critic_2_optimizers[0].zero_grad(set_to_none=True)
    model._optimize_actor(
        agent_idx=0,
        obs_tensor=obs_tensor,
        obs_flat=obs_flat,
        active_mask_tensor=active_mask_tensor,
        alpha=model.log_alphas[0].exp(),
    )

    actor_grads = [param.grad for param in model.actors[0].parameters() if param.requires_grad]
    critic_grads = [param.grad for param in model.critics_1[0].parameters()] + [param.grad for param in model.critics_2[0].parameters()]

    assert any(grad is not None for grad in actor_grads)
    assert all(grad is None for grad in critic_grads)


def test_masac_save_and_load_round_trip(tmp_path: Path) -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)
    model.update(batch)

    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()
    model.save(str(save_dir))

    restored = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    restored.load(str(save_dir))

    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
