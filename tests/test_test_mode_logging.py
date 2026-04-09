from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import test as test_module
from utils.logger import Logger


class _StubEnv:
    def reset(self) -> list[np.ndarray]:
        obs = np.zeros((config.NUM_UAVS, config.OBS_DIM_SINGLE), dtype=np.float32)
        obs[:, config.OWN_STATE_DIM - 1] = 1.0
        return [agent_obs.copy() for agent_obs in obs]

    def step(self, actions: np.ndarray):
        _ = actions
        next_obs = self.reset()
        rewards = np.zeros(config.NUM_UAVS, dtype=np.float32)
        metrics = (0.0, 0.0, 0.0, 0.0, {}, 0, 0)
        step_info = {"active_mask": np.ones(config.NUM_UAVS, dtype=np.float32)}
        return next_obs, rewards, metrics, step_info


class _StubModel:
    def reset(self) -> None:
        pass

    def select_actions(self, observations: list[np.ndarray], exploration: bool) -> np.ndarray:
        _ = observations
        _ = exploration
        return np.zeros((config.NUM_UAVS, config.ACTION_DIM), dtype=np.float32)


def test_test_mode_persists_episode_metrics_to_main_log(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "STEPS_PER_EPISODE", 1)
    monkeypatch.setattr(config, "TEST_LOG_FREQ", 1)
    monkeypatch.setattr(test_module, "_should_capture_artifacts", lambda _episode: False)

    logger = Logger(str(tmp_path), "unit-test")

    test_module.test_model(_StubEnv(), _StubModel(), logger, num_episodes=1)

    log_path = Path(logger.json_file_path)
    assert log_path.exists()

    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["episode"] == 1
    assert {"reward", "latency", "energy", "fairness", "rate", "collisions", "boundaries"} <= entries[0].keys()

    summary_path = tmp_path / "test_summary_unit-test.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["num_episodes"] == 1
    assert summary["averages"] == {
        "reward": 0.0,
        "latency": 0.0,
        "energy": 0.0,
        "fairness": 0.0,
        "rate": 0.0,
        "collisions": 0.0,
        "boundaries": 0.0,
    }
