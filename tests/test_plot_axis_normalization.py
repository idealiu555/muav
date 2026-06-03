from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_comparison


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries),
        encoding="utf-8",
    )


def _write_summary(path: Path, base_value: float) -> None:
    path.write_text(
        json.dumps(
            {
                "num_episodes": 50,
                "averages": {
                    "reward": base_value,
                    "latency": base_value + 0.1,
                    "energy": base_value + 0.2,
                    "fairness": base_value + 0.3,
                    "rate": base_value + 0.4,
                    "collisions": base_value + 0.5,
                    "boundaries": base_value + 0.6,
                },
            }
        ),
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


def test_plot_algorithm_comparison_generates_summary_bar_svgs(tmp_path, monkeypatch) -> None:
    masac_summary = tmp_path / "summary_masac.json"
    amasac_summary = tmp_path / "summary_amasac.json"
    output_dir = tmp_path / "plots"
    calls: list[tuple[list[str], list[float], str, str]] = []

    _write_summary(masac_summary, 1.0)
    _write_summary(amasac_summary, 2.0)

    def _capture_plot_metric_bar(labels, values, metric, output_path) -> None:
        calls.append((list(labels), list(values), metric, output_path))

    monkeypatch.setattr(plot_comparison, "plot_metric_bar", _capture_plot_metric_bar)

    plot_comparison.plot_algorithm_comparison(
        [str(masac_summary), str(amasac_summary)],
        ["MASAC", "AMASAC"],
        str(output_dir),
    )

    assert len(calls) == len(plot_comparison.METRIC_CONFIG)
    assert {metric for _, _, metric, _ in calls} == set(plot_comparison.METRIC_CONFIG)
    assert calls[0][0] == ["MASAC", "AMASAC"]
    assert calls[0][1] == [1.0, 2.0]
    assert {
        Path(output_path).name
        for _, _, _, output_path in calls
    } == {
        f"MASACvsAMASAC_{metric}.svg"
        for metric in plot_comparison.METRIC_CONFIG
    }
    assert all(Path(output_path).parent == output_dir for _, _, _, output_path in calls)


def test_generate_plots_labels_update_logs_as_episode(tmp_path, monkeypatch) -> None:
    from utils import plot_logs

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
