from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_comparison import plot_algorithm_comparison
from utils import plot_logs


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries),
        encoding="utf-8",
    )


def _make_entry(step_key: str, step_value: int, reward: float) -> dict:
    return {
        step_key: step_value,
        "reward": reward,
        "latency": reward + 0.1,
        "energy": reward + 0.2,
        "fairness": reward + 0.3,
        "rate": reward + 0.4,
    }


def test_plot_algorithm_comparison_accepts_mixed_episode_and_update_logs(tmp_path, capsys) -> None:
    episode_log = tmp_path / "episode.jsonl"
    update_log = tmp_path / "update.jsonl"
    output_path = tmp_path / "comparison.png"

    _write_jsonl(
        episode_log,
        [
            _make_entry("episode", 1, 1.0),
            _make_entry("episode", 2, 2.0),
        ],
    )
    _write_jsonl(
        update_log,
        [
            _make_entry("update", 1, 1.5),
            _make_entry("update", 2, 2.5),
        ],
    )

    plot_algorithm_comparison(
        [str(episode_log), str(update_log)],
        ["episode-based", "update-based"],
        str(output_path),
        metric="reward",
        smoothing=0.0,
    )

    captured = capsys.readouterr()
    assert "混用的 x 轴类型" not in captured.out
    assert output_path.exists()


def test_generate_plots_labels_update_logs_as_episode(tmp_path, monkeypatch) -> None:
    update_log = tmp_path / "update.jsonl"
    output_dir = tmp_path / "plots"
    labels_seen: list[str] = []

    _write_jsonl(
        update_log,
        [
            _make_entry("update", 1, 1.0),
            _make_entry("update", 2, 2.0),
        ],
    )

    def _capture_plot_metric(x, y, xlabel, ylabel, title, output_path, **kwargs) -> None:
        _ = (x, y, ylabel, title, output_path, kwargs)
        labels_seen.append(xlabel)

    def _capture_plot_metric_comparison(x, y1, y2, xlabel, ylabel1, ylabel2, title, output_path, **kwargs) -> None:
        _ = (x, y1, y2, ylabel1, ylabel2, title, output_path, kwargs)
        labels_seen.append(xlabel)

    monkeypatch.setattr(plot_logs, "plot_metric", _capture_plot_metric)
    monkeypatch.setattr(plot_logs, "plot_metric_comparison", _capture_plot_metric_comparison)

    plot_logs.generate_plots(str(update_log), str(output_dir), "train", "stamp", smoothing=0.0)

    assert labels_seen
    assert set(labels_seen) == {"Episode"}
