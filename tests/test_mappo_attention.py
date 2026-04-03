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


def _flatten_share_obs(obs: np.ndarray) -> np.ndarray:
    return obs.reshape(-1).astype(np.float32, copy=False)


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
    share_obs = torch.stack(
        [
            torch.from_numpy(_flatten_share_obs(_make_distinct_joint_obs(0))),
            torch.from_numpy(_flatten_share_obs(_make_distinct_joint_obs(1))),
        ],
        dim=0,
    )

    values = critic(share_obs)

    assert tuple(values.shape) == (2,)


def test_critic_accepts_flattened_share_obs_and_returns_scalar_values() -> None:
    from marl_models.mappo.agents import CriticNetwork

    critic = CriticNetwork(config.NUM_UAVS, config.OBS_DIM_SINGLE)
    share_obs = torch.randn(4, config.NUM_UAVS * config.OBS_DIM_SINGLE)

    values = critic(share_obs)

    assert tuple(values.shape) == (4,)


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


def test_mappo_get_action_and_value_flattens_share_obs_for_critic(monkeypatch) -> None:
    obs = _fake_obs()
    expected_shape = (1, config.NUM_UAVS * config.OBS_DIM_SINGLE)

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
            "share_obs_shape": None,
        }

        def wrapped_forward(share_obs: torch.Tensor) -> torch.Tensor:
            captured["share_obs_shape"] = tuple(share_obs.shape)
            return original_forward(share_obs)

        model.critics.forward = wrapped_forward  # type: ignore[method-assign]
        _env_actions, _raw_actions, _log_probs, values = model.get_action_and_value(obs)

        assert tuple(values.shape) == (config.NUM_UAVS,)
        assert captured["share_obs_shape"] == expected_shape


def test_critics_accept_flattened_share_obs_for_both_branches() -> None:
    from marl_models.mappo.agents import AttentionCriticNetwork
    from marl_models.mappo.agents import CriticNetwork

    share_obs = torch.stack(
        [
            torch.from_numpy(_flatten_share_obs(_make_distinct_joint_obs(0))),
            torch.from_numpy(_flatten_share_obs(_make_distinct_joint_obs(1))),
            torch.from_numpy(_flatten_share_obs(_make_distinct_joint_obs(2))),
        ],
        dim=0,
    )

    for critic_cls in (CriticNetwork, AttentionCriticNetwork):
        critic = critic_cls(config.NUM_UAVS, config.OBS_DIM_SINGLE)
        values = critic(share_obs)

        assert tuple(values.shape) == (3,)


def test_rollout_buffer_emits_share_obs_per_flat_sample() -> None:
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
    expected_share_obs = [
        torch.from_numpy(_flatten_share_obs(obs0)),
        torch.from_numpy(_flatten_share_obs(obs1)),
    ]
    buffer.add(obs0, _flatten_share_obs(obs0), raw_actions, log_probs, rewards, values, active)
    buffer.add(obs1, _flatten_share_obs(obs1), raw_actions, log_probs, rewards, values, active)

    seen_pairs: set[tuple[int, int]] = set()
    for batch in buffer.get_batches(4):
        assert "share_obs" in batch
        assert "obs" in batch
        assert tuple(batch["share_obs"].shape) == (
            batch["obs"].shape[0],
            config.NUM_UAVS * config.OBS_DIM_SINGLE,
        )
        for sample_idx in range(batch["obs"].shape[0]):
            share_obs = batch["share_obs"][sample_idx]
            flat_obs = batch["obs"][sample_idx]
            agent_index = None
            expected_step = None
            for step, expected in enumerate(expected_share_obs):
                if torch.allclose(share_obs, expected, atol=1e-6, rtol=1e-6):
                    expected_step = step
                    start = step * 0  # keep branch explicit for readability
                    del start
                    for idx in range(config.NUM_UAVS):
                        expected_obs = expected.view(config.NUM_UAVS, config.OBS_DIM_SINGLE)[idx]
                        if torch.allclose(flat_obs, expected_obs, atol=1e-6, rtol=1e-6):
                            agent_index = idx
                            break
                    break
            if expected_step is None or agent_index is None:
                raise AssertionError("share_obs or obs did not match either stored timestep")
            else:
                seen_pairs.add((expected_step, agent_index))

    assert len(seen_pairs) == 2 * config.NUM_UAVS


def test_mappo_rollout_replicates_scalar_value_across_agents(monkeypatch) -> None:
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

        def wrapped_forward(share_obs: torch.Tensor) -> torch.Tensor:
            assert tuple(share_obs.shape) == (1, config.NUM_UAVS * config.OBS_DIM_SINGLE)
            return torch.tensor([7.5], dtype=torch.float32, device=share_obs.device)

        model.critics.forward = wrapped_forward  # type: ignore[method-assign]
        _env_actions, _raw_actions, _log_probs, values = model.get_action_and_value(obs)
        model.critics.forward = original_forward  # type: ignore[method-assign]

        assert tuple(values.shape) == (config.NUM_UAVS,)
        assert np.allclose(values, 7.5)


def test_mappo_update_minibatch_uses_share_obs(monkeypatch) -> None:
    torch.manual_seed(7)
    obs = _make_training_obs()
    share_obs = _flatten_share_obs(obs)

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
            "share_obs": torch.from_numpy(np.repeat(share_obs[None, :], batch_size, axis=0)),
            "raw_actions": torch.from_numpy(raw_actions),
            "old_log_probs": torch.from_numpy(log_probs),
            "advantages": torch.full((batch_size,), 0.25, dtype=torch.float32),
            "returns": torch.from_numpy(values).float() + 0.5,
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
