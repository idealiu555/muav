"""
python plot_comparison.py \
   --files model_test/comparison_data/summary_masac.json model_test/comparison_data/summary_amasac.json \
   --labels MASAC AMASAC

   默认输出到 model_test/comparison_plots 目录下，自动生成命名为算法标签和指标名称的 SVG 图片。
"""
import argparse
import json
import os
import re

import numpy as np


METRIC_CONFIG: dict[str, dict[str, str]] = {
    "reward": {"ylabel": "Average Reward", "title": "Reward"},
    "latency": {"ylabel": "Average Latency (s)", "title": "Latency"},
    "energy": {"ylabel": "Average Energy Consumption (J)", "title": "Energy"},
    "fairness": {"ylabel": "Jain's Fairness Index", "title": "Fairness"},
    "rate": {"ylabel": "Average System Throughput (bps)", "title": "Throughput"},
    "collisions": {"ylabel": "Average Collision Count", "title": "Collisions"},
    "boundaries": {"ylabel": "Average Boundary Violation Count", "title": "Boundaries"},
}

COLORS: tuple[str, ...] = (
    "#C73E1D",
    "#3580B8",
    "#3A7D44",
    "#F18F01",
    "#6F4E9B",
    "#4C6A6D",
    "#A23E48",
)


def _load_summary_averages(file_path: str) -> dict[str, float]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    averages = data.get("averages")
    if not isinstance(averages, dict):
        raise ValueError(f"{file_path} does not contain an 'averages' object")

    result: dict[str, float] = {}
    for metric in METRIC_CONFIG:
        if metric not in averages:
            raise ValueError(f"{file_path} is missing averages.{metric}")
        result[metric] = float(averages[metric])
    return result


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("_.")
    return cleaned or "algorithm"


def _comparison_prefix(labels: list[str]) -> str:
    safe_labels = [_safe_filename_part(label) for label in labels]
    return "vs".join(safe_labels)


def _format_bar_label(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e6 or abs(value) < 1e-3:
        return f"{value:.2e}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def plot_metric_bar(
    labels: list[str],
    values: list[float],
    metric: str,
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    cfg = METRIC_CONFIG[metric]
    x = np.arange(len(labels))
    width = 0.62 if len(labels) <= 3 else 0.72
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]

    fig_width = max(5.6, 1.35 * len(labels) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    bars = ax.bar(x, values, width=width, color=colors, edgecolor="#222222", linewidth=0.8)

    ax.set_title(f"{cfg['title']} Comparison", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel(cfg["ylabel"], fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_value = max(values) if values else 0.0
    min_value = min(values) if values else 0.0
    if min_value >= 0:
        ax.set_ylim(bottom=0)
    if max_value > 0:
        ax.margins(y=0.16)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        offset = 4 if height >= 0 else -12
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            _format_bar_label(value),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_algorithm_comparison(
    summary_files: list[str],
    labels: list[str],
    output_dir: str = "model_test/comparison_plots",
) -> None:
    if len(summary_files) != len(labels):
        raise ValueError("The number of files must match the number of labels")
    if not summary_files:
        raise ValueError("At least one summary file is required")

    summaries = [_load_summary_averages(file_path) for file_path in summary_files]
    prefix = _comparison_prefix(labels)

    for metric in METRIC_CONFIG:
        values = [summary[metric] for summary in summaries]
        output_path = os.path.join(output_dir, f"{prefix}_{metric}.svg")
        plot_metric_bar(labels, values, metric, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制多算法测试 summary 指标柱形对比图。")
    parser.add_argument("--files", nargs="+", required=True, help="测试 summary JSON 文件路径列表")
    parser.add_argument("--labels", nargs="+", required=True, help="对应算法标签，数量必须与文件一致")
    parser.add_argument("--output_dir", type=str, default="model_test/comparison_plots", help="SVG 输出目录")

    args = parser.parse_args()
    plot_algorithm_comparison(args.files, args.labels, args.output_dir)
