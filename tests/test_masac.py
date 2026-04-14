from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from marl_models.masac.agents import (
    ActorNetwork,
    AttentionActorNetwork,
    AttentionCriticNetwork,
    CriticNetwork,
)
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


def test_masac_attention_networks_forward_with_expected_shapes() -> None:
    actor = AttentionActorNetwork(obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)
    critic = AttentionCriticNetwork(num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    actions, log_prob = actor.sample(torch.randn(4, config.OBS_DIM_SINGLE))
    q_values = critic(
        torch.randn(4, 2 * config.OBS_DIM_SINGLE),
        torch.randn(4, 2 * config.ACTION_DIM),
    )

    assert actions.shape == (4, config.ACTION_DIM)
    assert log_prob.shape == (4, 1)
    assert torch.isfinite(log_prob).all()
    assert q_values.shape == (4, 1)


def test_masac_attention_actor_rejects_invalid_obs_dim() -> None:
    with pytest.raises(ValueError, match="OBS_DIM_SINGLE"):
        AttentionActorNetwork(obs_dim=config.OBS_DIM_SINGLE - 1, action_dim=config.ACTION_DIM)


def test_masac_attention_critic_rejects_invalid_action_width() -> None:
    critic = AttentionCriticNetwork(num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    with pytest.raises(ValueError, match="joint_action"):
        critic(
            torch.randn(4, 2 * config.OBS_DIM_SINGLE),
            torch.randn(4, 2 * config.ACTION_DIM - 1),
        )


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


def test_masac_scales_target_entropy_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "TARGET_ENTROPY_SCALE", 0.5, raising=False)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    assert model.target_entropy == -float(config.ACTION_DIM) * config.TARGET_ENTROPY_SCALE


def test_masac_clamps_alpha_to_configured_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ALPHA_MIN", 1e-3, raising=False)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    with torch.no_grad():
        for log_alpha in model.log_alphas:
            log_alpha.fill_(np.log(1e-6))

    model.update(batch)

    assert all(log_alpha.detach().exp().item() >= config.ALPHA_MIN - 1e-7 for log_alpha in model.log_alphas)


def test_masac_clamps_alpha_before_using_it_in_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ALPHA_MIN", 1e-3, raising=False)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    with torch.no_grad():
        for log_alpha in model.log_alphas:
            log_alpha.fill_(np.log(1e-6))

    observed_alphas: list[float] = []

    def fake_optimize_actor(*, alpha: torch.Tensor, **_: object) -> tuple[torch.Tensor, torch.Tensor, float]:
        observed_alphas.append(float(alpha.detach().item()))
        return torch.tensor(0.0, device=model.device), torch.zeros((4, 1), device=model.device), 0.0

    monkeypatch.setattr(model, "_optimize_actor", fake_optimize_actor)

    model.update(batch)

    assert observed_alphas
    assert min(observed_alphas) >= config.ALPHA_MIN - 1e-7


def test_masac_rejects_non_positive_alpha_min(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ALPHA_MIN", 0.0, raising=False)

    with pytest.raises(ValueError, match="ALPHA_MIN"):
        MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")


def test_masac_rejects_non_positive_target_entropy_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "TARGET_ENTROPY_SCALE", 0.0, raising=False)

    with pytest.raises(ValueError, match="TARGET_ENTROPY_SCALE"):
        MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")


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


def test_masac_builds_attention_networks_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    assert isinstance(model.actors[0], AttentionActorNetwork)
    assert isinstance(model.critics_1[0], AttentionCriticNetwork)
    assert isinstance(model.critics_2[0], AttentionCriticNetwork)
    assert isinstance(model.target_critics_1[0], AttentionCriticNetwork)
    assert isinstance(model.target_critics_2[0], AttentionCriticNetwork)


def test_masac_attention_targets_remain_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    assert all(not param.requires_grad for param in model.target_critics_1.parameters())
    assert all(not param.requires_grad for param in model.target_critics_2.parameters())


def test_masac_attention_update_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    stats = model.update(_make_batch(4, 2, config.OBS_DIM_SINGLE, config.ACTION_DIM))

    assert np.isfinite(stats["actor_loss"])
    assert np.isfinite(stats["critic_loss"])
