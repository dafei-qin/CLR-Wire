#!/usr/bin/env python3
"""
分析JSON文件中曲面数量的分布。

该脚本遍历指定文件夹下的所有JSON文件，统计每个文件中的曲面数量，
并绘制CDF图以及去掉5%极端case的CDF图。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def is_surface_type(feature_type):
    """检查feature类型是否为曲面"""
    surface_types = {
        'plane', 'cylinder', 'cone', 'sphere', 'torus', 
        'bezier_surface', 'bspline_surface'
    }
    return feature_type in surface_types


def find_json_files(directory):
    """查找目录及子目录下的所有JSON文件"""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def count_surfaces_in_json(json_path):
    """统计单个JSON文件中的曲面数量"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return 0
        
        surface_count = 0
        for feature in data:
            if isinstance(feature, dict) and 'type' in feature:
                if is_surface_type(feature['type']):
                    surface_count += 1
        
        return surface_count
    
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return 0


def plot_cdf(surface_counts, title, output_path):
    """绘制CDF图"""
    sorted_counts = np.sort(surface_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_counts, cumulative, linewidth=2, color='steelblue')
    plt.xlabel('曲面数量 (Number of Surfaces)', fontsize=12)
    plt.ylabel('累积概率 (Cumulative Probability)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3)
    
    # 添加统计信息
    stats_text = (
        f'样本数 (n): {len(surface_counts)}\n'
        f'最小值 (Min): {sorted_counts.min()}\n'
        f'最大值 (Max): {sorted_counts.max()}\n'
        f'平均值 (Mean): {surface_counts.mean():.2f}\n'
        f'中位数 (Median): {np.median(surface_counts):.2f}\n'
        f'标准差 (Std): {surface_counts.std():.2f}'
    )
    plt.text(0.65, 0.15, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")
    plt.close()


def analyze_surface_distribution(input_dir, output_dir='assets'):
    """
    分析曲面数量分布并生成可视化图表
    
    Args:
        input_dir (str): 包含JSON文件的输入目录
        output_dir (str): 输出图表的目录
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JSON文件
    print(f"正在查找 {input_dir} 中的JSON文件...")
    json_files = find_json_files(input_dir)
    
    if not json_files:
        print(f"错误: 在 {input_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 统计每个JSON文件的曲面数量
    print("正在统计曲面数量...")
    surface_counts = []
    for json_file in tqdm(json_files):
        count = count_surfaces_in_json(json_file)
        surface_counts.append(count)
    
    surface_counts = np.array(surface_counts)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("曲面数量统计")
    print("="*60)
    print(f"JSON文件总数: {len(json_files)}")
    print(f"曲面数量最小值: {surface_counts.min()}")
    print(f"曲面数量最大值: {surface_counts.max()}")
    print(f"曲面数量平均值: {surface_counts.mean():.2f}")
    print(f"曲面数量中位数: {np.median(surface_counts):.2f}")
    print(f"曲面数量标准差: {surface_counts.std():.2f}")
    
    # 显示分布直方图
    print("\n曲面数量分布 (前10个最常见的值):")
    counter = Counter(surface_counts)
    for count, freq in counter.most_common(10):
        print(f"  {count} 个曲面: {freq} 个文件 ({freq/len(json_files)*100:.2f}%)")
    
    # 绘制完整数据的CDF图
    print("\n正在生成完整数据的CDF图...")
    plot_cdf(
        surface_counts,
        f'曲面数量累积分布函数 (CDF)\n(n={len(surface_counts)} 个JSON文件)',
        output_dir / 'surface_count_cdf_full.png'
    )
    
    # 去掉5%极端值 (最低2.5%和最高2.5%)
    print("正在生成去掉5%极端值的CDF图...")
    lower_percentile = np.percentile(surface_counts, 2.5)
    upper_percentile = np.percentile(surface_counts, 97.5)
    
    filtered_counts = surface_counts[
        (surface_counts >= lower_percentile) & 
        (surface_counts <= upper_percentile)
    ]
    
    print(f"\n去掉极端值后:")
    print(f"  剩余样本数: {len(filtered_counts)} ({len(filtered_counts)/len(surface_counts)*100:.2f}%)")
    print(f"  去掉的样本数: {len(surface_counts) - len(filtered_counts)}")
    print(f"  范围: [{lower_percentile:.0f}, {upper_percentile:.0f}]")
    print(f"  平均值: {filtered_counts.mean():.2f}")
    print(f"  中位数: {np.median(filtered_counts):.2f}")
    
    plot_cdf(
        filtered_counts,
        f'曲面数量累积分布函数 (CDF) - 去掉5%极端值\n(n={len(filtered_counts)} 个JSON文件, 范围: [{lower_percentile:.0f}, {upper_percentile:.0f}])',
        output_dir / 'surface_count_cdf_filtered.png'
    )
    
    # 绘制对比图
    print("正在生成对比图...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: 完整数据
    sorted_counts_full = np.sort(surface_counts)
    cumulative_full = np.arange(1, len(sorted_counts_full) + 1) / len(sorted_counts_full)
    axes[0].plot(sorted_counts_full, cumulative_full, linewidth=2, color='steelblue')
    axes[0].set_xlabel('曲面数量 (Number of Surfaces)', fontsize=12)
    axes[0].set_ylabel('累积概率 (Cumulative Probability)', fontsize=12)
    axes[0].set_title(f'完整数据 (n={len(surface_counts)})', fontsize=14)
    axes[0].grid(alpha=0.3)
    
    # 右图: 去掉5%极端值
    sorted_counts_filtered = np.sort(filtered_counts)
    cumulative_filtered = np.arange(1, len(sorted_counts_filtered) + 1) / len(sorted_counts_filtered)
    axes[1].plot(sorted_counts_filtered, cumulative_filtered, linewidth=2, color='forestgreen')
    axes[1].set_xlabel('曲面数量 (Number of Surfaces)', fontsize=12)
    axes[1].set_ylabel('累积概率 (Cumulative Probability)', fontsize=12)
    axes[1].set_title(f'去掉5%极端值 (n={len(filtered_counts)})', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'surface_count_cdf_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_path}")
    plt.close()
    
    # 保存统计信息到文本文件
    stats_file = output_dir / 'surface_count_statistics.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("曲面数量统计报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("完整数据统计:\n")
        f.write(f"  JSON文件总数: {len(json_files)}\n")
        f.write(f"  曲面数量最小值: {surface_counts.min()}\n")
        f.write(f"  曲面数量最大值: {surface_counts.max()}\n")
        f.write(f"  曲面数量平均值: {surface_counts.mean():.2f}\n")
        f.write(f"  曲面数量中位数: {np.median(surface_counts):.2f}\n")
        f.write(f"  曲面数量标准差: {surface_counts.std():.2f}\n\n")
        
        f.write("去掉5%极端值后的统计:\n")
        f.write(f"  剩余样本数: {len(filtered_counts)} ({len(filtered_counts)/len(surface_counts)*100:.2f}%)\n")
        f.write(f"  去掉的样本数: {len(surface_counts) - len(filtered_counts)}\n")
        f.write(f"  范围: [{lower_percentile:.0f}, {upper_percentile:.0f}]\n")
        f.write(f"  平均值: {filtered_counts.mean():.2f}\n")
        f.write(f"  中位数: {np.median(filtered_counts):.2f}\n")
        f.write(f"  标准差: {filtered_counts.std():.2f}\n\n")
        
        f.write("曲面数量分布 (前20个最常见的值):\n")
        for count, freq in counter.most_common(20):
            f.write(f"  {count} 个曲面: {freq} 个文件 ({freq/len(json_files)*100:.2f}%)\n")
    
    print(f"已保存统计信息: {stats_file}")
    
    print("\n" + "="*60)
    print(f"分析完成！所有图表已保存到: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="分析JSON文件中曲面数量的分布"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="包含JSON文件的输入目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets",
        help="输出图表的目录 (默认: assets)"
    )
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    analyze_surface_distribution(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()


