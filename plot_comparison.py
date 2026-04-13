# python plot_comparison.py \
#     --files train_logs/amappo.json train_logs/mappo.json \
#     --labels "AMAPPO" "MAPPO" \
#     --output comparison_reward.png \
#     --metric reward

# python plot_comparison.py --files train_logs/amappo.json train_logs/mappo.json --labels "AMAPPO" "MAPPO" --output comparison_reward.png --metric reward

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils.plot_logs import ACADEMIC_STYLE, COLORS, resolve_x_axis, smooth_curve

def load_log_data(file_path: str) -> list[dict]:
    """支持读取标准 JSON 格式或新版 JSONL 格式的日志文件。"""
    data = []
    with open(file_path, "r") as f:
        # 尝试读取第一行判断格式
        first_line = f.readline().strip()
        if not first_line:
            return []
        
        f.seek(0)
        if first_line.startswith("["):
            # 标准 JSON 数组格式
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
        else:
            # JSONL 格式
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return data

def plot_algorithm_comparison(
    log_files: list[str],
    labels: list[str],
    output_path: str,
    metric: str = "reward",
    smoothing: float = 0.9
) -> None:
    """
    绘制多个算法的指标对比图。

    Args:
        log_files: 日志文件路径列表
        labels: 对应的算法标签列表
        output_path: 输出图片路径
        metric: 要对比的指标 (reward, latency, energy, fairness, rate)
        smoothing: 平滑系数
    """
    
    # 验证输入
    if len(log_files) != len(labels):
        print("❌ 错误: 日志文件数量与标签数量不一致。")
        return

    # 指标配置
    metric_config = {
        "reward": "Average Reward",
        "latency": "Average Latency (s)",
        "energy": "Average Energy Consumption (J)",
        "fairness": "Jain's Fairness Index",
        "rate": "Average System Throughput (bps)",
    }
    
    if metric not in metric_config:
        print(f"❌ 错误: 不支持的指标 '{metric}'。支持的指标: {list(metric_config.keys())}")
        return

    ylabel = metric_config[metric]
    
    # 第一步：读取并校验日志；episode 是标准字段，update 兼容后统一显示为 Episode
    x_label = "Episode"
    file_data_cache = {}  # 缓存文件数据，避免重复读取
    
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"⚠️ 警告: 文件未找到，跳过: {log_file}")
            continue
            
        try:
            data = load_log_data(log_file)
            if not data:
                print(f"⚠️ 警告: 文件为空或格式错误，跳过: {log_file}")
                continue
            
            file_data_cache[log_file] = data  # 缓存数据

            try:
                resolve_x_axis(data[0])
            except KeyError:
                print(f"❌ 错误: 文件 {log_file} 缺少 'episode' 或 'update' 键。")
                print(f"   可用键: {list(data[0].keys())}")
                continue

        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析错误 {log_file}: {e}")
        except Exception as e:
            print(f"❌ 读取文件出错 {log_file}: {e}")

    if len(file_data_cache) == 0:
        print("❌ 错误: 没有找到任何有效的日志文件。")
        return

    # 第二步：每个文件独立提取x轴；update兼容映射为Episode
    all_data = []
    for log_file in log_files:
        if log_file not in file_data_cache:
            all_data.append(None)
            continue
            
        try:
            data = file_data_cache[log_file]

            x_axis_key, _ = resolve_x_axis(data[0])

            if metric not in data[0]:
                print(f"❌ 错误: 文件 {log_file} 缺少指标键 '{metric}'。")
                print(f"   可用键: {list(data[0].keys())}")
                all_data.append(None)
                continue
            
            # 提取数据；对外统一显示为Episode
            x = [entry[x_axis_key] for entry in data]
            y = [entry[metric] for entry in data]
            all_data.append((x, y))
            
        except KeyError as e:
            print(f"❌ 键错误 {log_file}: {e}")
            print(f"   请检查文件格式是否正确。")
            all_data.append(None)
        except Exception as e:
            print(f"❌ 提取数据出错 {log_file}: {e}")
            all_data.append(None)

    # 开始绘图
    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.rcParams.update(ACADEMIC_STYLE)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = list(COLORS.values()) # 使用预定义颜色
        
        has_plotted = False
        for i, (log_file, label) in enumerate(zip(log_files, labels)):
            if all_data[i] is None:
                continue
                
            x, y = all_data[i]
            color = colors[i % len(colors)]
            
            # 绘制平滑曲线
            if len(y) > 1:
                smoothed_y = smooth_curve(np.array(y), smoothing)
                ax.plot(x, smoothed_y, label=label, color=color, linewidth=2.5)
                # 绘制原始数据的浅色背景（可选，为了清晰度这里只画平滑线，或者可以画非常淡的）
                ax.plot(x, y, color=color, alpha=0.5, linewidth=1) 
            else:
                ax.plot(x, y, label=label, color=color, linewidth=2.5)
                
            has_plotted = True

        if not has_plotted:
            print("❌ 没有有效数据可绘制。")
            return

        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Performance Comparison: {ylabel}", fontweight='bold', pad=10)
        ax.legend(loc='best', frameon=True, framealpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ 对比图已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制多算法训练对比图。")
    
    # 使用 nargs='+' 允许传入列表
    parser.add_argument("--files", nargs='+', required=True, help="日志文件路径列表 (空格分隔)")
    parser.add_argument("--labels", nargs='+', required=True, help="对应的算法标签列表 (空格分隔，数量必须与文件一致)")
    parser.add_argument("--output", type=str, default="comparison_plot.png", help="输出图片路径")
    parser.add_argument("--metric", type=str, default="reward", choices=["reward", "latency", "energy", "fairness", "rate"], help="要对比的指标")
    parser.add_argument("--smoothing", type=float, default=0.90, help="平滑系数")

    args = parser.parse_args()
    
    plot_algorithm_comparison(args.files, args.labels, args.output, args.metric, args.smoothing)
