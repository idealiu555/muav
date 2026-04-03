"""
训练日志可视化模块
===================
生成符合顶级学术会议/期刊标准的训练曲线图。

特性：
- 指数移动平均(EMA)平滑曲线
- 原始值细线 + EMA 趋势线
- 学术论文风格排版
- 支持从保存的日志文件独立绘图
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Optional

# 学术论文风格配置
ACADEMIC_STYLE = {
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.0,
    'lines.markersize': 4,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# 学术配色方案 (colorblind-friendly)
COLORS = {
    'primary': '#2E86AB',      # 深蓝
    'secondary': '#A23B72',    # 紫红
    'tertiary': '#F18F01',     # 橙色
    'quaternary': '#C73E1D',   # 红色
    'success': '#3A7D44',      # 绿色
}


def smooth_curve(values: np.ndarray, smoothing_weight: float = 0.9) -> np.ndarray:
    """
    使用指数移动平均(EMA)平滑曲线。

    Args:
        values: 原始数据序列
        smoothing_weight: 平滑权重，越大越平滑 (0-1)

    Returns:
        平滑后的数据序列
    """
    if len(values) == 0:
        return values
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = smoothing_weight * smoothed[i-1] + (1 - smoothing_weight) * values[i]
    return smoothed


def _plot_raw_and_ema(
    ax,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    color: str,
    smoothing: float,
    label: str | None = None,
    linestyle: str = "-",
    show_raw: bool = True,
) -> None:
    if show_raw and len(y_arr) > 1:
        ax.plot(
            x_arr,
            y_arr,
            color=color,
            linewidth=1.0,
            alpha=0.25,
            linestyle=linestyle,
            label="_nolegend_",
        )

    if len(y_arr) > 1:
        smoothed_y = smooth_curve(y_arr, smoothing)
        ax.plot(
            x_arr,
            smoothed_y,
            color=color,
            linewidth=2.5,
            alpha=1.0,
            linestyle=linestyle,
            label=label,
        )
    else:
        ax.plot(
            x_arr,
            y_arr,
            color=color,
            linewidth=2.5,
            alpha=1.0,
            linestyle=linestyle,
            label=label,
        )


def plot_metric(
    x: list,
    y: list,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
    color: str = COLORS['primary'],
    smoothing: float = 0.9,
    show_raw: bool = True,
) -> None:
    """
    绘制单个指标的学术风格曲线图。

    Args:
        x: x轴数据
        y: y轴数据
        xlabel: x轴标签
        ylabel: y轴标签
        title: 图标题
        output_path: 输出文件路径
        color: 主曲线颜色
        smoothing: EMA平滑权重
        show_raw: 是否显示原始数据细线
    """
    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.rcParams.update(ACADEMIC_STYLE)

        fig, ax = plt.subplots()
        x_arr = np.array(x)
        y_arr = np.array(y)

        _plot_raw_and_ema(ax, x_arr, y_arr, color, smoothing, label="EMA", show_raw=show_raw)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold', pad=10)

        # x轴使用整数刻度
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 添加图例
        if len(y_arr) > 1:
            ax.legend(loc='best', fancybox=True, shadow=False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
        plt.close()


def plot_metric_comparison(
    x: list,
    y1: list,
    y2: list,
    xlabel: str,
    ylabel1: str,
    ylabel2: str,
    title: str,
    output_path: str,
    smoothing: float = 0.9,
) -> None:
    """
    绘制两个指标的对比图（双y轴，原始值 + EMA）。

    Args:
        x: x轴数据
        y1: 第一个指标的y轴数据
        y2: 第二个指标的y轴数据
        xlabel: x轴标签
        ylabel1: 第一个y轴标签
        ylabel2: 第二个y轴标签
        title: 图标题
        output_path: 输出文件路径
        smoothing: EMA平滑权重
    """
    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.rcParams.update(ACADEMIC_STYLE)

        fig, ax1 = plt.subplots()
        x_arr = np.array(x)
        y1_arr = np.array(y1)
        y2_arr = np.array(y2)

        # 第一个指标
        color1 = COLORS['primary']
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel1, color=color1)
        _plot_raw_and_ema(ax1, x_arr, y1_arr, color1, smoothing, label=ylabel1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # 第二个指标（共享x轴）
        ax2 = ax1.twinx()
        color2 = COLORS['secondary']
        ax2.set_ylabel(ylabel2, color=color2)
        _plot_raw_and_ema(ax2, x_arr, y2_arr, color2, smoothing, label=ylabel2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.set_title(title, fontweight='bold', pad=10)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        fig.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
        plt.close()


def generate_plots(
    log_file: str, 
    output_dir: str, 
    output_file_prefix: str, 
    timestamp: str,
    smoothing: float = 0.9
) -> None:
    """
    从日志文件生成学术风格的训练曲线图。
    
    Args:
        log_file: 日志JSON文件路径
        output_dir: 输出目录
        output_file_prefix: 输出文件前缀
        timestamp: 时间戳
        smoothing: EMA平滑权重 (0-1, 越大越平滑)
    """
    log_data: list[dict] = []
    try:
        with open(log_file, "r", encoding="utf-8") as file:
            # Peek to determine if it's a JSON array or JSON Lines
            content_start = file.read(1)
            file.seek(0)
            
            if content_start == '[':
                # Old style: Single JSON array
                log_data = json.load(file)
            else:
                # New style: JSON Lines (one JSON object per line)
                for line in file:
                    line = line.strip()
                    if line:
                        try:
                            log_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
    except Exception as e:
        print(f"❌ Error reading {log_file}: {e}")
    
    if not log_data:
        print(f"❌ No valid data found in {log_file}")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    # 确定x轴类型
    if "update" in log_data[0]:
        x_axis_key: str = "update"
        x_label: str = "Training Update"
    elif "episode" in log_data[0]:
        x_axis_key = "episode"
        x_label = "Episode"
    else:
        print("❌ Log file does not contain 'episode' or 'update' keys.")
        return
    
    # 提取数据
    x_data = [entry[x_axis_key] for entry in log_data]
    metrics_data = {
        "reward": [entry["reward"] for entry in log_data],
        "latency": [entry["latency"] for entry in log_data],
        "energy": [entry["energy"] for entry in log_data],
        "fairness": [entry["fairness"] for entry in log_data],
        "rate": [entry["rate"] for entry in log_data],
    }
    
    # 指标配置
    metric_config = {
        "reward": {"ylabel": "Average Team Reward (per step)", "color": COLORS['primary']},
        "latency": {"ylabel": "Average Latency (s)", "color": COLORS['secondary']},
        "energy": {"ylabel": "Average Energy Consumption (J)", "color": COLORS['tertiary']},
        "fairness": {"ylabel": "Jain's Fairness Index", "color": COLORS['success']},
        "rate": {"ylabel": "Average System Throughput (bps)", "color": COLORS['quaternary']},
    }
    
    # 绘制单指标曲线
    for metric, cfg in metric_config.items():
        title = f"{cfg['ylabel']} vs {x_label}"
        output_path = os.path.join(output_dir, f"{output_file_prefix}_{metric}_{timestamp}.png")
        plot_metric(
            x_data, metrics_data[metric], 
            x_label, cfg['ylabel'], title, output_path,
            color=cfg['color'], smoothing=smoothing
        )
    
    # 绘制对比图：Reward + Fairness
    plot_metric_comparison(
        x_data, metrics_data["reward"], metrics_data["fairness"],
        x_label, "Average Team Reward (per step)", "Fairness Index",
        "Reward and Fairness Convergence",
        os.path.join(output_dir, f"{output_file_prefix}_reward_fairness_{timestamp}.png"),
        smoothing=smoothing
    )
    
    # 绘制对比图：Latency + Energy
    plot_metric_comparison(
        x_data, metrics_data["latency"], metrics_data["energy"],
        x_label, "Latency (s)", "Energy (J)",
        "Latency-Energy Trade-off",
        os.path.join(output_dir, f"{output_file_prefix}_latency_energy_{timestamp}.png"),
        smoothing=smoothing
    )

    # 绘制对比图：Rate + Fairness
    plot_metric_comparison(
        x_data, metrics_data["rate"], metrics_data["fairness"],
        x_label, "System Throughput (bps)", "Fairness Index",
        "Throughput-Fairness Trade-off",
        os.path.join(output_dir, f"{output_file_prefix}_rate_fairness_{timestamp}.png"),
        smoothing=smoothing
    )
    
    print(f"✅ Academic-style plots saved to {output_dir}\n")


def generate_plots_if_available(
    log_file: str,
    output_dir: str,
    output_file_prefix: str,
    timestamp: str,
    smoothing: float = 0.9,
) -> bool:
    """Generate plots only when a non-empty log file exists."""
    if not os.path.exists(log_file):
        print(f"ℹ️ Skipping plot generation: no log data at {log_file}")
        return False
    if os.path.getsize(log_file) == 0:
        print(f"ℹ️ Skipping plot generation: log file is empty at {log_file}")
        return False

    generate_plots(log_file, output_dir, output_file_prefix, timestamp, smoothing)
    return True


def generate_plots_from_file(log_file: str, output_dir: Optional[str] = None, smoothing: float = 0.9) -> None:
    """
    便捷函数：直接从日志文件生成图表。
    
    Args:
        log_file: 日志文件路径 (如 'log_data_2025-10-20_10-11-10.json')
        output_dir: 输出目录，默认与日志文件同目录
        smoothing: 平滑权重
    
    Example:
        >>> from utils.plot_logs import generate_plots_from_file
        >>> generate_plots_from_file('sample_logs/sample_train_logs/log_data_2025-10-20_10-11-10.json')
    """
    import re
    
    # 提取时间戳
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', log_file)
    timestamp = match.group(1) if match else "output"
    
    if output_dir is None:
        output_dir = os.path.dirname(log_file) or "."
    
    generate_plots(log_file, output_dir, "train", timestamp, smoothing)
