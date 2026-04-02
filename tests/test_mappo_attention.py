from pathlib import Path
import copy
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from marl_models.mappo.rollout_buffer import MAPPORolloutBuffer
from marl_models.mappo.mappo import MAPPO


def _fake_obs(active_mask: np.ndarray | None = None) -> np.ndarray:
    obs = np.zeros((config.NUM_UAVS, config.OBS_DIM_SINGLE), dtype=np.float32)
    obs[:, config.OWN_STATE_DIM - 1] = 1.0
    if active_mask is not None:
        obs[:, config.OWN_STATE_DIM - 1] = active_mask
    return obs


def _make_distinct_joint_obs(step: int = 0) -> np.ndarray:
    obs = np.zeros((config.NUM_UAVS, config.OBS_DIM_SINGLE), dtype=np.float32)
    for agent in range(config.NUM_UAVS):
        obs[agent] = (step * 1000.0) + (agent * 10.0) + np.arange(config.OBS_DIM_SINGLE, dtype=np.float32)
    obs[:, config.OWN_STATE_DIM - 1] = 1.0
    return obs


def _make_training_obs() -> np.ndarray:
    obs = np.zeros((config.NUM_UAVS, config.OBS_DIM_SINGLE), dtype=np.float32)
    obs[:, config.OWN_STATE_DIM - 1] = 1.0
    return obs


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in state_dict.items()}


def test_attention_actor_outputs_correct_distribution_shape() -> None:
    from marl_models.mappo.agents import AttentionActorNetwork

    actor = AttentionActorNetwork(config.OBS_DIM_SINGLE, config.ACTION_DIM)
    obs = torch.from_numpy(_fake_obs())
    dist = actor(obs)

    assert tuple(dist.mean.shape) == (config.NUM_UAVS, config.ACTION_DIM)
    assert tuple(dist.stddev.shape) == (config.NUM_UAVS, config.ACTION_DIM)
    assert tuple(dist.sample().shape) == (config.NUM_UAVS, config.ACTION_DIM)


def test_attention_critic_outputs_one_value_per_agent_sample() -> None:
    from marl_models.mappo.agents import AttentionCriticNetwork

    critic = AttentionCriticNetwork(config.NUM_UAVS, config.OBS_DIM_SINGLE)
    base_joint_obs = torch.from_numpy(_make_distinct_joint_obs()).unsqueeze(0)
    joint_obs = base_joint_obs.repeat(2, 1, 1)
    agent_index = torch.tensor([0, 9], dtype=torch.long)

    active_mask_a = torch.zeros((2, config.NUM_UAVS), dtype=torch.bool)
    active_mask_a[:, :2] = True
    active_mask_b = torch.zeros((2, config.NUM_UAVS), dtype=torch.bool)
    active_mask_b[:, :8] = True

    values_a = critic(joint_obs, agent_index, active_mask_a)
    values_b = critic(joint_obs, agent_index, active_mask_b)
    values_swapped = critic(joint_obs, torch.tensor([9, 0], dtype=torch.long), active_mask_a)

    assert tuple(values_a.shape) == (2,)
    assert tuple(values_b.shape) == (2,)
    assert not torch.allclose(values_a, values_b, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(values_a, values_swapped, atol=1e-5, rtol=1e-5)


def test_mappo_constructor_selects_attention_and_non_attention_branches(monkeypatch) -> None:
    from marl_models.mappo.agents import ActorNetwork
    from marl_models.mappo.agents import AttentionActorNetwork
    from marl_models.mappo.agents import AttentionCriticNetwork
    from marl_models.mappo.agents import CriticNetwork

    obs = _fake_obs()

    monkeypatch.setattr(config, "USE_ATTENTION", False)
    model = MAPPO(
        model_name="mappo",
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )
    assert isinstance(model.actors, ActorNetwork)
    assert isinstance(model.critics, CriticNetwork)
    env_actions, raw_actions, log_probs, values = model.get_action_and_value(obs)
    assert tuple(env_actions.shape) == (config.NUM_UAVS, config.ACTION_DIM)
    assert tuple(raw_actions.shape) == (config.NUM_UAVS, config.ACTION_DIM)
    assert tuple(log_probs.shape) == (config.NUM_UAVS,)
    assert tuple(values.shape) == (config.NUM_UAVS,)

    monkeypatch.setattr(config, "USE_ATTENTION", True)
    model = MAPPO(
        model_name="mappo",
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )
    assert isinstance(model.actors, AttentionActorNetwork)
    assert isinstance(model.critics, AttentionCriticNetwork)
    env_actions, raw_actions, log_probs, values = model.get_action_and_value(obs)
    assert tuple(env_actions.shape) == (config.NUM_UAVS, config.ACTION_DIM)
    assert tuple(raw_actions.shape) == (config.NUM_UAVS, config.ACTION_DIM)
    assert tuple(log_probs.shape) == (config.NUM_UAVS,)
    assert tuple(values.shape) == (config.NUM_UAVS,)


def test_mappo_get_action_and_value_evaluates_single_joint_obs_per_step(monkeypatch) -> None:
    obs = _fake_obs()

    for use_attention in (False, True):
        monkeypatch.setattr(config, "USE_ATTENTION", use_attention)
        model = MAPPO(
            model_name="mappo",
            num_agents=config.NUM_UAVS,
            obs_dim=config.OBS_DIM_SINGLE,
            action_dim=config.ACTION_DIM,
            device="cpu",
        )

        original_forward = model.critics.forward
        captured: dict[str, tuple[int, ...] | None] = {
            "joint_obs_shape": None,
            "active_mask_shape": None,
        }

        def wrapped_forward(
            joint_obs: torch.Tensor,
            agent_index: torch.Tensor,
            active_mask: torch.Tensor,
            sample_index: torch.Tensor | None = None,
        ) -> torch.Tensor:
            captured["joint_obs_shape"] = tuple(joint_obs.shape)
            captured["active_mask_shape"] = tuple(active_mask.shape)
            assert sample_index is None
            return original_forward(joint_obs, agent_index, active_mask, sample_index=sample_index)

        model.critics.forward = wrapped_forward  # type: ignore[method-assign]
        _env_actions, _raw_actions, _log_probs, values = model.get_action_and_value(obs)

        assert tuple(values.shape) == (config.NUM_UAVS,)
        assert captured["joint_obs_shape"] == (1, config.NUM_UAVS, config.OBS_DIM_SINGLE)
        assert captured["active_mask_shape"] == (1, config.NUM_UAVS)


def test_critic_supports_single_joint_obs_for_multiple_agent_queries() -> None:
    from marl_models.mappo.agents import AttentionCriticNetwork
    from marl_models.mappo.agents import CriticNetwork

    base_joint_obs = torch.from_numpy(_make_distinct_joint_obs()).unsqueeze(0)
    expanded_joint_obs = base_joint_obs.expand(config.NUM_UAVS, -1, -1).clone()
    active_mask_single = torch.ones((1, config.NUM_UAVS), dtype=torch.bool)
    active_mask_expanded = active_mask_single.expand(config.NUM_UAVS, -1).clone()
    agent_index = torch.arange(config.NUM_UAVS, dtype=torch.long)

    for critic_cls in (CriticNetwork, AttentionCriticNetwork):
        critic = critic_cls(config.NUM_UAVS, config.OBS_DIM_SINGLE)
        values_single = critic(base_joint_obs, agent_index, active_mask_single)
        values_expanded = critic(expanded_joint_obs, agent_index, active_mask_expanded)

        assert tuple(values_single.shape) == (config.NUM_UAVS,)
        assert torch.allclose(values_single, values_expanded, atol=1e-5, rtol=1e-5)


def test_critics_return_zero_for_inactive_queried_agents() -> None:
    from marl_models.mappo.agents import AttentionCriticNetwork
    from marl_models.mappo.agents import CriticNetwork

    joint_obs = torch.from_numpy(_make_distinct_joint_obs()).unsqueeze(0)
    active_mask = torch.zeros((1, config.NUM_UAVS), dtype=torch.bool)
    active_mask[0, 0] = True
    active_mask[0, 1] = True
    agent_index = torch.tensor([0, 3], dtype=torch.long)

    for critic_cls in (CriticNetwork, AttentionCriticNetwork):
        critic = critic_cls(config.NUM_UAVS, config.OBS_DIM_SINGLE)
        values = critic(joint_obs, agent_index, active_mask)

        assert tuple(values.shape) == (2,)
        assert not torch.allclose(values[0], torch.zeros((), dtype=values.dtype), atol=1e-6, rtol=1e-6)
        assert torch.allclose(values[1], torch.zeros((), dtype=values.dtype), atol=1e-6, rtol=1e-6)


def test_rollout_buffer_emits_joint_obs_per_flat_sample() -> None:
    buffer = MAPPORolloutBuffer(
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        buffer_size=2,
        device="cpu",
    )

    def make_obs(step: int) -> np.ndarray:
        obs = np.zeros((config.NUM_UAVS, config.OBS_DIM_SINGLE), dtype=np.float32)
        for agent in range(config.NUM_UAVS):
            obs[agent] = (step * 1000.0) + (agent * 10.0) + np.arange(config.OBS_DIM_SINGLE, dtype=np.float32)
        obs[:, config.OWN_STATE_DIM - 1] = 1.0
        return obs

    raw_actions = np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)
    log_probs = np.zeros(config.NUM_UAVS, dtype=np.float32)
    rewards = np.zeros(config.NUM_UAVS, dtype=np.float32)
    values = np.zeros(config.NUM_UAVS, dtype=np.float32)
    active = np.ones(config.NUM_UAVS, dtype=np.float32)

    obs0 = make_obs(0)
    obs1 = make_obs(1)
    expected_joint_obs = [torch.from_numpy(obs0), torch.from_numpy(obs1)]
    buffer.add(obs0, raw_actions, log_probs, rewards, values, active)
    buffer.add(obs1, raw_actions, log_probs, rewards, values, active)

    seen_pairs: set[tuple[int, int]] = set()
    for batch in buffer.get_batches(4):
        assert "joint_obs" in batch
        assert "joint_active_mask" in batch
        assert "joint_obs_index" in batch
        assert "agent_index" in batch
        assert "obs" in batch
        assert batch["joint_obs"].shape[0] <= batch["obs"].shape[0]
        assert tuple(batch["joint_active_mask"].shape) == (batch["joint_obs"].shape[0], config.NUM_UAVS)
        for sample_idx in range(batch["obs"].shape[0]):
            joint_obs_idx = int(batch["joint_obs_index"][sample_idx].item())
            joint_obs = batch["joint_obs"][joint_obs_idx]
            joint_active_mask = batch["joint_active_mask"][joint_obs_idx]
            agent_index = int(batch["agent_index"][sample_idx].item())
            flat_obs = batch["obs"][sample_idx]

            assert tuple(joint_obs.shape) == (config.NUM_UAVS, config.OBS_DIM_SINGLE)
            assert torch.all(joint_active_mask)
            assert torch.allclose(flat_obs, joint_obs[agent_index], atol=1e-6, rtol=1e-6)

            if torch.allclose(joint_obs, expected_joint_obs[0], atol=1e-6, rtol=1e-6):
                expected_step = 0
            elif torch.allclose(joint_obs, expected_joint_obs[1], atol=1e-6, rtol=1e-6):
                expected_step = 1
            else:
                raise AssertionError("joint_obs did not match either stored timestep")

            seen_pairs.add((expected_step, agent_index))

    assert len(seen_pairs) == 2 * config.NUM_UAVS


def test_mappo_update_minibatch_uses_joint_active_mask(monkeypatch) -> None:
    torch.manual_seed(7)
    obs = _make_training_obs()

    for use_attention in (False, True):
        monkeypatch.setattr(config, "USE_ATTENTION", use_attention)
        model = MAPPO(
            model_name="mappo",
            num_agents=config.NUM_UAVS,
            obs_dim=config.OBS_DIM_SINGLE,
            action_dim=config.ACTION_DIM,
            device="cpu",
        )

        env_actions, raw_actions, log_probs, values = model.get_action_and_value(obs)
        batch_size = config.NUM_UAVS
        batch = {
            "obs": torch.from_numpy(obs),
            "raw_actions": torch.from_numpy(raw_actions),
            "old_log_probs": torch.from_numpy(log_probs),
            "advantages": torch.full((batch_size,), 0.25, dtype=torch.float32),
            "returns": torch.from_numpy(values).float() + 0.5,
            "joint_obs": torch.from_numpy(obs).unsqueeze(0).clone(),
            "joint_active_mask": torch.ones((1, config.NUM_UAVS), dtype=torch.bool),
            "joint_obs_index": torch.zeros(batch_size, dtype=torch.long),
            "agent_index": torch.arange(config.NUM_UAVS, dtype=torch.long),
            "old_values": torch.from_numpy(values).float(),
            "active_mask": torch.ones(batch_size, dtype=torch.float32),
        }

        stats = model._update_minibatch(batch)

        assert tuple(env_actions.shape) == (config.NUM_UAVS, config.ACTION_DIM)
        assert stats["valid_samples"] == float(config.NUM_UAVS)
        assert stats["critic_loss"] >= 0.0
        assert torch.isfinite(torch.tensor(stats["actor_loss"]))
        assert torch.isfinite(torch.tensor(stats["critic_loss"]))


def test_mappo_load_round_trips_checkpoint_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)
    model = MAPPO(
        model_name="mappo",
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )
    saved_actor_state = _clone_state_dict(model.actors.state_dict())
    saved_critic_state = _clone_state_dict(model.critics.state_dict())

    monkeypatch.setattr(config, "USE_ATTENTION", False)
    model.save(str(tmp_path))

    with torch.no_grad():
        for parameter in model.actors.parameters():
            parameter.add_(1.0)
        for parameter in model.critics.parameters():
            parameter.sub_(1.0)

    monkeypatch.setattr(config, "USE_ATTENTION", True)
    model.load(str(tmp_path))

    for name, tensor in model.actors.state_dict().items():
        assert torch.allclose(tensor, saved_actor_state[name])
    for name, tensor in model.critics.state_dict().items():
        assert torch.allclose(tensor, saved_critic_state[name])


def test_mappo_load_rejects_incompatible_checkpoint_metadata(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", True)
    source_model = MAPPO(
        model_name="mappo",
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )
    source_model.save(str(tmp_path))

    monkeypatch.setattr(config, "USE_ATTENTION", False)
    target_model = MAPPO(
        model_name="mappo",
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )

    try:
        target_model.load(str(tmp_path))
    except ValueError as exc:
        assert "checkpoint" in str(exc).lower()
        assert "attention" in str(exc).lower()
    else:
        raise AssertionError("Expected load() to reject incompatible checkpoint metadata")


def test_mappo_load_rolls_back_when_critic_load_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "USE_ATTENTION", False)
    model = MAPPO(
        model_name="mappo",
        num_agents=config.NUM_UAVS,
        obs_dim=config.OBS_DIM_SINGLE,
        action_dim=config.ACTION_DIM,
        device="cpu",
    )
    original_actor_state = _clone_state_dict(model.actors.state_dict())
    original_critic_state = _clone_state_dict(model.critics.state_dict())

    checkpoint = {
        "actor": copy.deepcopy(model.actors.state_dict()),
        "critic": copy.deepcopy(model.critics.state_dict()),
    }
    first_critic_key = next(iter(checkpoint["critic"]))
    checkpoint["critic"][first_critic_key] = checkpoint["critic"][first_critic_key][1:].clone()
    torch.save(checkpoint, tmp_path / "mappo.pth")

    try:
        model.load(str(tmp_path))
    except ValueError as exc:
        assert "checkpoint" in str(exc).lower()
        assert "critic" in str(exc).lower()
    else:
        raise AssertionError("Expected load() to fail on incompatible critic state")

    for name, tensor in model.actors.state_dict().items():
        assert torch.allclose(tensor, original_actor_state[name])
    for name, tensor in model.critics.state_dict().items():
        assert torch.allclose(tensor, original_critic_state[name])
