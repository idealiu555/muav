from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

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


def _assert_nested_tensors_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        assert torch.equal(left, right)
        return

    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_tensors_equal(left[key], right[key])
        return

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_tensors_equal(left_item, right_item)
        return

    assert left == right


def test_masac_registers_shared_modules_and_alpha_parameter() -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model.actor, ActorNetwork)
    assert isinstance(model.critic_1, CriticNetwork)
    assert isinstance(model.critic_2, CriticNetwork)
    assert isinstance(model.target_critic_1, CriticNetwork)
    assert isinstance(model.target_critic_2, CriticNetwork)
    assert isinstance(model.log_alpha, torch.nn.Parameter)

    state_dict_keys = model.state_dict().keys()
    assert "actor.fc1.weight" in state_dict_keys
    assert "critic_1.fc1.weight" in state_dict_keys
    assert "log_alpha" in state_dict_keys


def test_masac_networks_use_silu_not_relu() -> None:
    actor = ActorNetwork(obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)
    critic = CriticNetwork(
        total_obs_dim=2 * config.OBS_DIM_SINGLE,
        total_action_dim=2 * config.ACTION_DIM,
        num_agents=2,
    )

    actor_output = actor(torch.randn(4, config.OBS_DIM_SINGLE))
    critic_output = critic(torch.randn(4, 2 * config.OBS_DIM_SINGLE), torch.randn(4, 2 * config.ACTION_DIM))

    assert actor_output[0].shape == (4, config.ACTION_DIM)
    assert actor_output[1].shape == (4, config.ACTION_DIM)
    assert critic_output.shape == (4, 2)
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
    assert q_values.shape == (4, 2)


def test_masac_shared_critic_outputs_one_value_per_agent() -> None:
    critic = CriticNetwork(
        total_obs_dim=3 * config.OBS_DIM_SINGLE,
        total_action_dim=3 * config.ACTION_DIM,
        num_agents=3,
    )

    q_values = critic(
        torch.randn(5, 3 * config.OBS_DIM_SINGLE),
        torch.randn(5, 3 * config.ACTION_DIM),
    )

    assert q_values.shape == (5, 3)


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
        model.actor_optimizer,
        model.critic_1_optimizer,
        model.critic_2_optimizer,
        model.alpha_optimizer,
    ]
    assert all(isinstance(optimizer, torch.optim.Adam) for optimizer in optimizers)
    assert all(optimizer.defaults["weight_decay"] == 0.0 for optimizer in optimizers)
    assert model.actor_optimizer.defaults["lr"] == config.MASAC_ACTOR_LR


def test_masac_update_reports_current_alpha_mean() -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    stats = model.update(batch)
    actual_alpha_mean = float(model.log_alpha.detach().exp().item())

    assert np.isclose(stats["alpha_mean"], actual_alpha_mean)


def test_masac_per_agent_target_entropy_scales_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "TARGET_ENTROPY_SCALE", 0.5, raising=False)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    assert model.target_entropy_per_agent == -float(config.ACTION_DIM) * config.TARGET_ENTROPY_SCALE


def test_masac_preserves_per_agent_reward_columns() -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    rewards_tensor = torch.tensor(
        [
            [1.0, 3.0],
            [2.0, -2.0],
        ],
        dtype=torch.float32,
    )
    target_q = torch.tensor([[5.0, 11.0], [7.0, 13.0]], dtype=torch.float32)
    bootstrap_mask = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    targets = model._per_agent_targets(rewards_tensor, target_q, bootstrap_mask)

    expected = rewards_tensor + config.DISCOUNT_FACTOR * target_q * bootstrap_mask
    assert torch.equal(targets, expected)


def test_masac_preserves_per_agent_bootstrap_mask() -> None:
    model = MASAC("masac", num_agents=3, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    bootstrap_mask_tensor = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    per_agent_bootstrap_mask = model._per_agent_bootstrap_mask(bootstrap_mask_tensor)

    assert torch.equal(per_agent_bootstrap_mask, bootstrap_mask_tensor)


def test_masac_masked_inactive_agent_log_probs_per_agent() -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    log_probs = torch.tensor([[1.5, 10.0], [2.0, 20.0]], device=model.device)
    active_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=model.device)

    masked_log_probs = model._masked_log_probs(log_probs, active_mask)

    expected = torch.tensor([[1.5, 0.0], [2.0, 20.0]], device=model.device)
    assert torch.equal(masked_log_probs, expected)


def test_masac_update_rejects_scalar_critic_outputs() -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    class ScalarCritic(torch.nn.Module):
        def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
            del joint_obs, joint_action
            return torch.zeros((4, 1), dtype=torch.float32, device=model.device)

    model.critic_1 = ScalarCritic()

    with pytest.raises(ValueError, match="critic_1"):
        model.update(batch)


def test_masac_update_uses_per_agent_masks_for_vector_entropy_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=2, device="cpu")
    batch = {
        "obs": np.zeros((2, 2, config.OBS_DIM_SINGLE), dtype=np.float32),
        "actions": np.zeros((2, 2, 2), dtype=np.float32),
        "rewards": np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
        "next_obs": np.zeros((2, 2, config.OBS_DIM_SINGLE), dtype=np.float32),
        "active_mask": np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "bootstrap_mask": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    }
    batch["obs"][:, :, config.OWN_STATE_DIM - 1] = 1.0
    batch["next_obs"][:, :, config.OWN_STATE_DIM - 1] = 1.0

    sample_calls = {"count": 0}

    def fake_sample(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = (
            torch.tensor([[1.0], [3.0]], dtype=torch.float32)
            if sample_calls["count"] % 2 == 0
            else torch.tensor([[5.0], [7.0]], dtype=torch.float32)
        )
        sample_calls["count"] += 1
        return torch.zeros((obs.shape[0], 2), dtype=torch.float32, device=obs.device), log_probs.to(obs.device)

    monkeypatch.setattr(model.actor, "sample", fake_sample)

    target_q1 = torch.tensor([[4.0, 40.0], [6.0, 60.0]], dtype=torch.float32)
    target_q2 = torch.tensor([[2.0, 20.0], [8.0, 80.0]], dtype=torch.float32)
    current_q1 = torch.tensor([[5.0, 50.0], [7.0, 70.0]], dtype=torch.float32)
    current_q2 = torch.tensor([[9.0, 90.0], [11.0, 110.0]], dtype=torch.float32)

    monkeypatch.setattr(model.target_critic_1, "forward", lambda joint_obs, joint_action: target_q1.clone())
    monkeypatch.setattr(model.target_critic_2, "forward", lambda joint_obs, joint_action: target_q2.clone())
    monkeypatch.setattr(model.critic_1, "forward", lambda joint_obs, joint_action: current_q1.clone().requires_grad_())
    monkeypatch.setattr(model.critic_2, "forward", lambda joint_obs, joint_action: current_q2.clone().requires_grad_())

    with torch.no_grad():
        model.log_alpha.fill_(0.0)
    pre_update_log_alpha = model.log_alpha.detach().clone()

    stats = model.update(batch)

    bootstrap_mask = torch.tensor(batch["bootstrap_mask"], dtype=torch.float32)
    active_mask = torch.tensor(batch["active_mask"], dtype=torch.float32)
    masked_next_log_probs = torch.tensor([[1.0, 0.0], [0.0, 7.0]], dtype=torch.float32)
    min_target_q = torch.minimum(target_q1, target_q2)
    expected_targets = torch.tensor(batch["rewards"], dtype=torch.float32) + config.DISCOUNT_FACTOR * (
        min_target_q - masked_next_log_probs
    ) * bootstrap_mask
    expected_critic_loss = (
        F.smooth_l1_loss(current_q1, expected_targets, reduction="none") * active_mask
    ).sum() / active_mask.sum() + (
        F.smooth_l1_loss(current_q2, expected_targets, reduction="none") * active_mask
    ).sum() / active_mask.sum()
    actor_log_probs = torch.tensor([[1.0, 0.0], [3.0, 7.0]], dtype=torch.float32)
    actor_min_q = torch.minimum(current_q1, current_q2)
    expected_actor_loss = ((actor_log_probs - actor_min_q) * active_mask).sum() / active_mask.sum()
    expected_alpha_loss = (
        -(pre_update_log_alpha * (actor_log_probs + model.target_entropy_per_agent) * active_mask).sum()
        / active_mask.sum()
    )

    assert stats["critic_loss"] == pytest.approx(expected_critic_loss.item())
    assert stats["actor_loss"] == pytest.approx(expected_actor_loss.item())
    assert stats["alpha_loss"] == pytest.approx(expected_alpha_loss.item())


def test_masac_update_rejects_target_log_prob_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=2, device="cpu")
    batch = _make_batch(batch_size=2, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=2)

    sample_calls = {"count": 0}

    def fake_sample(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sample_calls["count"] += 1
        width = 1 if sample_calls["count"] == 1 else 2
        return (
            torch.zeros((obs.shape[0], 2), dtype=torch.float32, device=obs.device),
            torch.zeros((obs.shape[0], width), dtype=torch.float32, device=obs.device),
        )

    monkeypatch.setattr(model.actor, "sample", fake_sample)

    with pytest.raises(ValueError, match="log_probs"):
        model.update(batch)


def test_masac_actor_step_rejects_log_prob_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=2, device="cpu")
    batch = _make_batch(batch_size=2, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=2)
    obs_tensor = torch.as_tensor(batch["obs"], dtype=torch.float32, device=model.device)
    active_mask_tensor = torch.as_tensor(batch["active_mask"], dtype=torch.float32, device=model.device)
    obs_flat = obs_tensor.reshape(obs_tensor.shape[0], -1)

    sample_calls = {"count": 0}

    def fake_sample(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sample_calls["count"] += 1
        width = 1 if sample_calls["count"] == 1 else 2
        return (
            torch.zeros((obs.shape[0], 2), dtype=torch.float32, device=obs.device),
            torch.zeros((obs.shape[0], width), dtype=torch.float32, device=obs.device),
        )

    monkeypatch.setattr(model.actor, "sample", fake_sample)

    with pytest.raises(ValueError, match="log_probs"):
        model._optimize_actor(
            obs_tensor=obs_tensor,
            obs_flat=obs_flat,
            active_mask_tensor=active_mask_tensor,
            alpha=model.log_alpha.exp(),
        )


def test_masac_clamps_alpha_to_configured_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ALPHA_MIN", 1e-3, raising=False)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    with torch.no_grad():
        model.log_alpha.fill_(np.log(1e-6))

    model.update(batch)

    assert model.log_alpha.detach().exp().item() >= config.ALPHA_MIN - 1e-7


def test_masac_clamps_alpha_before_using_it_in_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ALPHA_MIN", 1e-3, raising=False)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM)

    with torch.no_grad():
        model.log_alpha.fill_(np.log(1e-6))

    observed_alphas: list[float] = []

    def fake_optimize_actor(*, alpha: torch.Tensor, **_: object) -> tuple[torch.Tensor, torch.Tensor, float]:
        observed_alphas.append(float(alpha.detach().item()))
        return torch.tensor(0.0, device=model.device), torch.zeros((4, 2), device=model.device), 0.0

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

    model.actor_optimizer.zero_grad(set_to_none=True)
    model.critic_1_optimizer.zero_grad(set_to_none=True)
    model.critic_2_optimizer.zero_grad(set_to_none=True)
    model._optimize_actor(
        obs_tensor=obs_tensor,
        obs_flat=obs_flat,
        active_mask_tensor=active_mask_tensor,
        alpha=model.log_alpha.exp(),
    )

    actor_grads = [param.grad for param in model.actor.parameters() if param.requires_grad]
    critic_grads = [param.grad for param in model.critic_1.parameters()] + [param.grad for param in model.critic_2.parameters()]

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
    _assert_nested_tensors_equal(model.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())
    _assert_nested_tensors_equal(model.critic_1_optimizer.state_dict(), restored.critic_1_optimizer.state_dict())
    _assert_nested_tensors_equal(model.critic_2_optimizer.state_dict(), restored.critic_2_optimizer.state_dict())
    _assert_nested_tensors_equal(model.alpha_optimizer.state_dict(), restored.alpha_optimizer.state_dict())


def test_masac_save_writes_vector_critic_checkpoint_metadata(tmp_path: Path) -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    original_use_attention = config.USE_ATTENTION

    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()
    try:
        config.USE_ATTENTION = not original_use_attention
        model.save(str(save_dir))
    finally:
        config.USE_ATTENTION = original_use_attention

    checkpoint = torch.load(save_dir / "masac.pt", map_location="cpu")

    assert checkpoint["checkpoint_format"] == "shared_masac_vector_critic"
    assert checkpoint["checkpoint_version"] == 1
    assert checkpoint["num_agents"] == 2
    assert checkpoint["actor_type"] == "shared"
    assert checkpoint["critic_type"] == "shared_vector"
    assert checkpoint["uses_attention"] is model.use_attention


@pytest.mark.parametrize(
    ("checkpoint_format", "checkpoint_version"),
    [
        ("legacy_masac", 1),
        ("shared_masac_vector_critic", 2),
    ],
)
def test_masac_load_rejects_incompatible_checkpoint_metadata(
    tmp_path: Path,
    checkpoint_format: str,
    checkpoint_version: int,
) -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()

    torch.save(
        {
            "checkpoint_format": checkpoint_format,
            "checkpoint_version": checkpoint_version,
            "num_agents": 2,
            "actor_type": "shared",
            "critic_type": "shared_vector",
            "uses_attention": False,
            "model": model.state_dict(),
            "actor_optimizer": model.actor_optimizer.state_dict(),
            "critic_1_optimizer": model.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": model.critic_2_optimizer.state_dict(),
            "alpha_optimizer": model.alpha_optimizer.state_dict(),
        },
        save_dir / "masac.pt",
    )

    with pytest.raises(ValueError, match="Incompatible|version"):
        model.load(str(save_dir))


def test_masac_load_rejects_attention_architecture_mismatch(tmp_path: Path) -> None:
    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()

    torch.save(
        {
            "checkpoint_format": "shared_masac_vector_critic",
            "checkpoint_version": 1,
            "num_agents": 2,
            "actor_type": "shared",
            "critic_type": "shared_vector",
            "uses_attention": True,
            "model": model.state_dict(),
            "actor_optimizer": model.actor_optimizer.state_dict(),
            "critic_1_optimizer": model.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": model.critic_2_optimizer.state_dict(),
            "alpha_optimizer": model.alpha_optimizer.state_dict(),
        },
        save_dir / "masac.pt",
    )

    with pytest.raises(ValueError, match="architecture|attention"):
        model.load(str(save_dir))


def test_masac_load_uses_instance_attention_mode_not_mutable_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", False)
    source = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()

    monkeypatch.setattr(config, "USE_ATTENTION", True)
    source.save(str(save_dir))
    attention_model = MASAC(
        "masac",
        num_agents=2,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )

    with pytest.raises(ValueError, match="attention"):
        attention_model.load(str(save_dir))


def test_masac_builds_attention_networks_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    assert isinstance(model.actor, AttentionActorNetwork)
    assert isinstance(model.critic_1, AttentionCriticNetwork)
    assert isinstance(model.critic_2, AttentionCriticNetwork)
    assert isinstance(model.target_critic_1, AttentionCriticNetwork)
    assert isinstance(model.target_critic_2, AttentionCriticNetwork)


def test_masac_attention_targets_remain_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")

    assert all(not param.requires_grad for param in model.target_critic_1.parameters())
    assert all(not param.requires_grad for param in model.target_critic_2.parameters())


def test_masac_attention_update_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)

    model = MASAC("masac", num_agents=2, obs_dim=config.OBS_DIM_SINGLE, action_dim=config.ACTION_DIM, device="cpu")
    stats = model.update(_make_batch(4, 2, config.OBS_DIM_SINGLE, config.ACTION_DIM))

    assert np.isfinite(stats["actor_loss"])
    assert np.isfinite(stats["critic_loss"])
