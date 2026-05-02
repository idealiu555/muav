from pathlib import Path
import sys

import numpy as np
import torch
import pytest

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


def test_maddpg_critic_outputs_one_value_per_agent() -> None:
    actor = ActorNetwork(obs_dim=12, action_dim=3)
    critic = CriticNetwork(total_obs_dim=36, total_action_dim=9, num_agents=3)

    actor_output = actor(torch.randn(4, 12))
    critic_output = critic(torch.randn(4, 36), torch.randn(4, 9))

    assert actor.fc1.in_features == 12
    assert critic.fc1.in_features == 45
    assert actor_output.shape == (4, 3)
    assert critic_output.shape == (4, 3)
    assert torch.all(actor_output <= 1.0)
    assert torch.all(actor_output >= -1.0)
    assert any(isinstance(module, torch.nn.SiLU) for module in actor.modules())
    assert any(isinstance(module, torch.nn.SiLU) for module in critic.modules())
    assert not any(isinstance(module, torch.nn.ReLU) for module in actor.modules())
    assert not any(isinstance(module, torch.nn.ReLU) for module in critic.modules())


def test_obsolete_maddpg_attention_pooling_symbols_are_removed() -> None:
    assert not hasattr(attention_module, "MeanPoolingEncoder")
    assert not hasattr(attention_module, "AgentPoolingAttention")
    assert not hasattr(attention_module, "AgentPoolingValue")


def test_maddpg_ignores_attention_flag_and_uses_standard_networks(monkeypatch) -> None:
    for use_attention in (False, True):
        monkeypatch.setattr(config, "USE_ATTENTION", use_attention)
        model = MADDPG("maddpg", num_agents=3, obs_dim=8, action_dim=2, device="cpu")

        assert isinstance(model.actor, ActorNetwork)
        assert isinstance(model.target_actor, ActorNetwork)
        assert isinstance(model.critic, CriticNetwork)
        assert isinstance(model.target_critic, CriticNetwork)
        assert not hasattr(model, "shared_encoder")
        assert not hasattr(model, "target_shared_encoder")
        assert not hasattr(model, "shared_encoder_optimizer")


def test_maddpg_registers_shared_modules_in_state_dict() -> None:
    model = MADDPG(
        "maddpg",
        num_agents=2,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )

    assert isinstance(model.actor, ActorNetwork)
    assert isinstance(model.critic, CriticNetwork)
    assert isinstance(model.target_actor, ActorNetwork)
    assert isinstance(model.target_critic, CriticNetwork)
    state_dict_keys = model.state_dict().keys()
    assert "actor.fc1.weight" in state_dict_keys
    assert "critic.fc1.weight" in state_dict_keys
    assert "target_actor.fc1.weight" in state_dict_keys
    assert "target_critic.fc1.weight" in state_dict_keys


def test_maddpg_preserves_per_agent_reward_columns() -> None:
    model = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")
    rewards_tensor = torch.tensor(
        [
            [1.0, 3.0],
            [2.0, -2.0],
        ],
        dtype=torch.float32,
    )
    target_q = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
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


def test_maddpg_preserves_per_agent_bootstrap_mask() -> None:
    model = MADDPG("maddpg", num_agents=3, obs_dim=8, action_dim=2, device="cpu")
    bootstrap_mask_tensor = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    per_agent_bootstrap_mask = model._per_agent_bootstrap_mask(bootstrap_mask_tensor)

    assert torch.equal(per_agent_bootstrap_mask, bootstrap_mask_tensor)


def test_maddpg_update_uses_per_agent_masks_for_vector_critic(monkeypatch) -> None:
    model = MADDPG("maddpg", num_agents=3, obs_dim=4, action_dim=2, device="cpu")
    batch = {
        "obs": np.zeros((2, 3, 4), dtype=np.float32),
        "actions": np.zeros((2, 3, 2), dtype=np.float32),
        "rewards": np.array(
            [
                [0.5, 1.0, -1.5],
                [2.0, -2.0, 0.25],
            ],
            dtype=np.float32,
        ),
        "next_obs": np.zeros((2, 3, 4), dtype=np.float32),
        "active_mask": np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "bootstrap_mask": np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    monkeypatch.setattr(model.target_actor, "forward", lambda obs: torch.zeros((obs.shape[0], 2), requires_grad=False))
    monkeypatch.setattr(model.actor, "forward", lambda obs: torch.zeros((obs.shape[0], 2), requires_grad=True))

    target_q_value = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    current_q_value = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ],
        dtype=torch.float32,
    )

    monkeypatch.setattr(model.target_critic, "forward", lambda joint_obs, joint_action: target_q_value.clone())
    monkeypatch.setattr(model.critic, "forward", lambda joint_obs, joint_action: current_q_value.clone().requires_grad_())

    stats = model.update(batch)

    expected_targets = torch.tensor(
        [
            [0.5, 1.0, -1.5],
            [2.0, -2.0, 0.25],
        ],
        dtype=torch.float32,
    ) + config.DISCOUNT_FACTOR * target_q_value * torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_critic_loss = ((current_q_value - expected_targets).pow(2) * torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )).sum() / 3.0
    expected_actor_loss = -torch.tensor([10.0, 30.0, 50.0], dtype=torch.float32).mean()

    assert stats["critic_loss"] == pytest.approx(expected_critic_loss.item())
    assert stats["actor_loss"] == pytest.approx(expected_actor_loss.item())


def test_maddpg_update_rejects_current_q_shape_mismatch(monkeypatch) -> None:
    model = MADDPG("maddpg", num_agents=3, obs_dim=4, action_dim=2, device="cpu")
    batch = {
        "obs": np.zeros((2, 3, 4), dtype=np.float32),
        "actions": np.zeros((2, 3, 2), dtype=np.float32),
        "rewards": np.zeros((2, 3), dtype=np.float32),
        "next_obs": np.zeros((2, 3, 4), dtype=np.float32),
        "active_mask": np.ones((2, 3), dtype=np.float32),
        "bootstrap_mask": np.ones((2, 3), dtype=np.float32),
    }

    monkeypatch.setattr(model.actor, "forward", lambda obs: torch.zeros((obs.shape[0], 2), requires_grad=True))
    monkeypatch.setattr(model.target_actor, "forward", lambda obs: torch.zeros((obs.shape[0], 2), requires_grad=False))
    monkeypatch.setattr(model.target_critic, "forward", lambda joint_obs, joint_action: torch.zeros((2, 3)))
    monkeypatch.setattr(model.critic, "forward", lambda joint_obs, joint_action: torch.zeros((2, 4), requires_grad=True))

    with pytest.raises(ValueError, match="current_q_value"):
        model.update(batch)


def test_maddpg_update_rejects_target_q_shape_mismatch(monkeypatch) -> None:
    model = MADDPG("maddpg", num_agents=3, obs_dim=4, action_dim=2, device="cpu")
    batch = {
        "obs": np.zeros((2, 3, 4), dtype=np.float32),
        "actions": np.zeros((2, 3, 2), dtype=np.float32),
        "rewards": np.zeros((2, 3), dtype=np.float32),
        "next_obs": np.zeros((2, 3, 4), dtype=np.float32),
        "active_mask": np.ones((2, 3), dtype=np.float32),
        "bootstrap_mask": np.ones((2, 3), dtype=np.float32),
    }

    monkeypatch.setattr(model.actor, "forward", lambda obs: torch.zeros((obs.shape[0], 2), requires_grad=True))
    monkeypatch.setattr(model.target_actor, "forward", lambda obs: torch.zeros((obs.shape[0], 2), requires_grad=False))
    monkeypatch.setattr(model.critic, "forward", lambda joint_obs, joint_action: torch.zeros((2, 3), requires_grad=True))
    monkeypatch.setattr(model.target_critic, "forward", lambda joint_obs, joint_action: torch.zeros((2, 4)))

    with pytest.raises(ValueError, match="target_q_value"):
        model.update(batch)


def test_maddpg_update_rejects_active_mask_shape_mismatch() -> None:
    model = MADDPG("maddpg", num_agents=3, obs_dim=4, action_dim=2, device="cpu")
    batch = {
        "obs": np.zeros((2, 3, 4), dtype=np.float32),
        "actions": np.zeros((2, 3, 2), dtype=np.float32),
        "rewards": np.zeros((2, 3), dtype=np.float32),
        "next_obs": np.zeros((2, 3, 4), dtype=np.float32),
        "active_mask": np.ones((2, 2), dtype=np.float32),
        "bootstrap_mask": np.ones((2, 3), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="active_mask"):
        model.update(batch)


def test_maddpg_save_and_load_round_trip(tmp_path: Path) -> None:
    model = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=8, action_dim=2)
    model.update(batch)
    model.noise[0].scale = 0.125
    model.noise[1].scale = 0.75

    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()
    model.save(str(save_dir))

    restored = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")
    restored.load(str(save_dir))

    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    _assert_nested_tensors_equal(model.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())
    _assert_nested_tensors_equal(model.critic_optimizer.state_dict(), restored.critic_optimizer.state_dict())
    assert [noise.scale for noise in restored.noise] == [0.125, 0.75]


def test_maddpg_save_writes_vector_critic_checkpoint_metadata(tmp_path: Path) -> None:
    model = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")

    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()
    model.save(str(save_dir))

    checkpoint = torch.load(save_dir / "maddpg.pt", map_location="cpu")

    assert checkpoint["checkpoint_format"] == "shared_maddpg_vector_critic"
    assert checkpoint["checkpoint_version"] == 1
    assert checkpoint["num_agents"] == 2
    assert checkpoint["actor_type"] == "shared"
    assert checkpoint["critic_type"] == "shared_vector"


@pytest.mark.parametrize(
    ("checkpoint_format", "checkpoint_version"),
    [
        ("legacy_maddpg", 1),
        ("shared_maddpg_vector_critic", 2),
    ],
)
def test_maddpg_load_rejects_incompatible_checkpoint_metadata(
    tmp_path: Path,
    checkpoint_format: str,
    checkpoint_version: int,
) -> None:
    model = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")
    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()

    torch.save(
        {
            "checkpoint_format": checkpoint_format,
            "checkpoint_version": checkpoint_version,
            "num_agents": 2,
            "actor_type": "shared",
            "critic_type": "shared_vector",
            "actor": model.actor.state_dict(),
            "critic": model.critic.state_dict(),
            "target_actor": model.target_actor.state_dict(),
            "target_critic": model.target_critic.state_dict(),
            "actor_optimizer": model.actor_optimizer.state_dict(),
            "critic_optimizer": model.critic_optimizer.state_dict(),
            "noise_scales": [noise.scale for noise in model.noise],
        },
        save_dir / "maddpg.pt",
    )

    with pytest.raises(ValueError, match="Incompatible|version"):
        model.load(str(save_dir))


def test_maddpg_load_rejects_noise_scale_length_mismatch(tmp_path: Path) -> None:
    model = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")
    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()

    torch.save(
        {
            "checkpoint_format": "shared_maddpg_vector_critic",
            "checkpoint_version": 1,
            "num_agents": 2,
            "actor_type": "shared",
            "critic_type": "shared_vector",
            "actor": model.actor.state_dict(),
            "critic": model.critic.state_dict(),
            "target_actor": model.target_actor.state_dict(),
            "target_critic": model.target_critic.state_dict(),
            "actor_optimizer": model.actor_optimizer.state_dict(),
            "critic_optimizer": model.critic_optimizer.state_dict(),
            "noise_scales": [0.5],
        },
        save_dir / "maddpg.pt",
    )

    with pytest.raises(ValueError, match="noise_scales"):
        model.load(str(save_dir))


def test_maddpg_load_does_not_mutate_state_when_noise_scale_validation_fails(tmp_path: Path) -> None:
    source = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")
    source_batch = _make_batch(batch_size=4, num_agents=2, obs_dim=8, action_dim=2)
    source.update(source_batch)
    source.noise[0].scale = 0.125
    source.noise[1].scale = 0.75

    target = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")
    target_batch = _make_batch(batch_size=4, num_agents=2, obs_dim=8, action_dim=2)
    target.update(target_batch)
    original_state_dict = {key: value.detach().clone() for key, value in target.state_dict().items()}
    original_actor_optimizer = target.actor_optimizer.state_dict()
    original_critic_optimizer = target.critic_optimizer.state_dict()
    original_noise_scales = [noise.scale for noise in target.noise]

    save_dir = tmp_path / "checkpoint"
    save_dir.mkdir()
    checkpoint = {
        "checkpoint_format": "shared_maddpg_vector_critic",
        "checkpoint_version": 1,
        "num_agents": 2,
        "actor_type": "shared",
        "critic_type": "shared_vector",
        "actor": source.actor.state_dict(),
        "critic": source.critic.state_dict(),
        "target_actor": source.target_actor.state_dict(),
        "target_critic": source.target_critic.state_dict(),
        "actor_optimizer": source.actor_optimizer.state_dict(),
        "critic_optimizer": source.critic_optimizer.state_dict(),
        "noise_scales": [0.5],
    }
    torch.save(checkpoint, save_dir / "maddpg.pt")

    with pytest.raises(ValueError, match="noise_scales"):
        target.load(str(save_dir))

    for key, value in original_state_dict.items():
        assert torch.equal(value, target.state_dict()[key])
    _assert_nested_tensors_equal(original_actor_optimizer, target.actor_optimizer.state_dict())
    _assert_nested_tensors_equal(original_critic_optimizer, target.critic_optimizer.state_dict())
    assert [noise.scale for noise in target.noise] == original_noise_scales


def test_maddpg_update_runs_without_attention_branch(monkeypatch) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)
    model = MADDPG("maddpg", num_agents=2, obs_dim=8, action_dim=2, device="cpu")
    batch = _make_batch(batch_size=4, num_agents=2, obs_dim=8, action_dim=2)

    stats = model.update(batch)

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
