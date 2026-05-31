from collections import deque
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import _rolling_reward_metrics


def test_rolling_reward_metrics_use_available_history_up_to_each_window() -> None:
    rewards = deque([1.0, 2.0, 3.0], maxlen=200)

    assert _rolling_reward_metrics(rewards) == {
        "reward_mean_50": 2.0,
        "reward_mean_100": 2.0,
        "reward_mean_200": 2.0,
    }


def test_rolling_reward_metrics_use_requested_tail_windows() -> None:
    rewards = deque(range(1, 201), maxlen=200)

    assert _rolling_reward_metrics(rewards) == {
        "reward_mean_50": 175.5,
        "reward_mean_100": 150.5,
        "reward_mean_200": 100.5,
    }
