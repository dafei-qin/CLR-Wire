"""对比GT和Pred JSON文件的曲面类型分布"""
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import argparse


def analyze_surface_distribution_single(json_files: list, name: str = "Dataset"):
    """
    分析指定JSON文件列表的曲面类型分布
    
    Args:
        json_files: JSON文件路径列表
        name: 数据集名称（用于显示）
    
    Returns:
        dict: 包含统计信息的字典
    """
    # 统计数据
    surface_type_counter = Counter()  # 总体曲面类型计数
    file_surface_counts = []  # 每个文件的曲面数量
    total_surfaces = 0
    error_files = []
    
    # 遍历所有JSON文件
    for json_file in tqdm(json_files, desc=f"处理{name} JSON文件"):
        try:
            with open(json_file, 'r') as f:
                surfaces = json.load(f)
            
            # 统计当前文件的曲面类型
            file_surface_count = len(surfaces)
            file_surface_counts.append(file_surface_count)
            total_surfaces += file_surface_count
            
            for surface in surfaces:
                surface_type = surface.get('type', 'unknown')
                surface_type_counter[surface_type] += 1
        
        except Exception as e:
            error_files.append((str(json_file), str(e)))
    
    return {
        'name': name,
        'total_files': len(json_files),
        'success_files': len(json_files) - len(error_files),
        'error_files': error_files,
        'total_surfaces': total_surfaces,
        'file_surface_counts': file_surface_counts,
        'surface_type_counter': surface_type_counter
    }


def format_statistics(stats: dict) -> list:
    """格式化统计信息为输出行"""
    output_lines = []
    
    output_lines.append(f"数据集: {stats['name']}")
    output_lines.append(f"总文件数: {stats['total_files']}")
    output_lines.append(f"成功处理: {stats['success_files']}")
    output_lines.append(f"处理失败: {len(stats['error_files'])}")
    output_lines.append("")
    
    # 曲面总数统计
    output_lines.append(f"曲面总数: {stats['total_surfaces']:,}")
    if stats['file_surface_counts']:
        avg_surfaces = sum(stats['file_surface_counts']) / len(stats['file_surface_counts'])
        output_lines.append(f"每个文件平均曲面数: {avg_surfaces:.2f}")
        output_lines.append(f"每个文件最少曲面数: {min(stats['file_surface_counts'])}")
        output_lines.append(f"每个文件最多曲面数: {max(stats['file_surface_counts'])}")
    output_lines.append("")
    
    # 曲面类型分布
    output_lines.append("-" * 80)
    output_lines.append("曲面类型详细分布:")
    output_lines.append("-" * 80)
    output_lines.append(f"{'曲面类型':<30} {'数量':>15} {'占比':>15}")
    output_lines.append("-" * 80)
    
    # 按数量降序排序
    sorted_types = sorted(stats['surface_type_counter'].items(), key=lambda x: x[1], reverse=True)
    
    for surface_type, count in sorted_types:
        percentage = (count / stats['total_surfaces'] * 100) if stats['total_surfaces'] > 0 else 0
        output_lines.append(f"{surface_type:<30} {count:>15,} {percentage:>14.2f}%")
    
    output_lines.append("-" * 80)
    
    return output_lines


def format_comparison(gt_stats: dict, pred_stats: dict) -> list:
    """格式化GT和Pred的对比信息"""
    output_lines = []
    
    output_lines.append("=" * 100)
    output_lines.append("GT vs PRED 曲面类型对比")
    output_lines.append("=" * 100)
    output_lines.append("")
    
    # 获取所有曲面类型
    all_types = set(gt_stats['surface_type_counter'].keys()) | set(pred_stats['surface_type_counter'].keys())
    
    output_lines.append(f"{'曲面类型':<25} {'GT数量':>15} {'GT占比':>12} {'Pred数量':>15} {'Pred占比':>12} {'差异':>12}")
    output_lines.append("=" * 100)
    
    # 按GT数量降序排序
    sorted_types = sorted(
        all_types,
        key=lambda x: gt_stats['surface_type_counter'].get(x, 0),
        reverse=True
    )
    
    for surface_type in sorted_types:
        gt_count = gt_stats['surface_type_counter'].get(surface_type, 0)
        pred_count = pred_stats['surface_type_counter'].get(surface_type, 0)
        
        gt_percentage = (gt_count / gt_stats['total_surfaces'] * 100) if gt_stats['total_surfaces'] > 0 else 0
        pred_percentage = (pred_count / pred_stats['total_surfaces'] * 100) if pred_stats['total_surfaces'] > 0 else 0
        
        diff = pred_percentage - gt_percentage
        diff_str = f"{diff:+.2f}%"
        
        output_lines.append(
            f"{surface_type:<25} {gt_count:>15,} {gt_percentage:>11.2f}% "
            f"{pred_count:>15,} {pred_percentage:>11.2f}% {diff_str:>12}"
        )
    
    output_lines.append("=" * 100)
    output_lines.append("")
    
    # 总体统计对比
    output_lines.append("总体统计对比:")
    output_lines.append("-" * 80)
    output_lines.append(f"{'指标':<40} {'GT':>20} {'Pred':>20}")
    output_lines.append("-" * 80)
    output_lines.append(f"{'总文件数':<40} {gt_stats['total_files']:>20} {pred_stats['total_files']:>20}")
    output_lines.append(f"{'曲面总数':<40} {gt_stats['total_surfaces']:>20,} {pred_stats['total_surfaces']:>20,}")
    
    if gt_stats['file_surface_counts'] and pred_stats['file_surface_counts']:
        gt_avg = sum(gt_stats['file_surface_counts']) / len(gt_stats['file_surface_counts'])
        pred_avg = sum(pred_stats['file_surface_counts']) / len(pred_stats['file_surface_counts'])
        output_lines.append(f"{'平均每文件曲面数':<40} {gt_avg:>20.2f} {pred_avg:>20.2f}")
    
    output_lines.append(f"{'曲面类型数':<40} {len(gt_stats['surface_type_counter']):>20} {len(pred_stats['surface_type_counter']):>20}")
    output_lines.append("-" * 80)
    
    return output_lines


def find_json_files(base_dir: str, pattern: str = "*.json") -> list:
    """
    查找指定目录下的JSON文件
    
    Args:
        base_dir: 基础目录路径
        pattern: 文件匹配模式
    
    Returns:
        list: JSON文件路径列表
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"警告: 目录不存在: {base_dir}")
        return []
    
    json_files = list(base_path.rglob(pattern))
    return json_files


def analyze_and_compare_experiment(
    gt_dir: str,
    pred_dir: str,
    output_file: str = None,
    gt_pattern: str = "*.json",
    pred_pattern: str = "*.json"
):
    """
    对比实验输出中GT和Pred的曲面分布
    
    Args:
        gt_dir: GT JSON文件目录
        pred_dir: Pred JSON文件目录
        output_file: 输出文件路径（可选）
        gt_pattern: GT文件匹配模式
        pred_pattern: Pred文件匹配模式
    """
    print("=" * 80)
    print("开始分析GT和Pred曲面分布对比")
    print("=" * 80)
    print(f"GT目录: {gt_dir}")
    print(f"Pred目录: {pred_dir}")
    print("")
    
    # 查找JSON文件
    print("查找JSON文件...")
    gt_files = find_json_files(gt_dir, gt_pattern)
    pred_files = find_json_files(pred_dir, pred_pattern)
    
    print(f"找到 {len(gt_files)} 个 GT JSON 文件")
    print(f"找到 {len(pred_files)} 个 Pred JSON 文件")
    print("")
    
    if not gt_files:
        print(f"错误: 未在 {gt_dir} 中找到任何 JSON 文件")
        return
    
    if not pred_files:
        print(f"错误: 未在 {pred_dir} 中找到任何 JSON 文件")
        return
    
    # 分析GT和Pred
    gt_stats = analyze_surface_distribution_single(gt_files, "GT")
    print("")
    pred_stats = analyze_surface_distribution_single(pred_files, "Pred")
    print("")
    
    # 准备输出内容
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("GT 和 Pred 曲面类型分布统计与对比")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.append(f"GT目录: {gt_dir}")
    output_lines.append(f"Pred目录: {pred_dir}")
    output_lines.append("")
    
    # GT统计
    output_lines.append("=" * 100)
    output_lines.append("GT 曲面分布统计")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.extend(format_statistics(gt_stats))
    output_lines.append("")
    
    # Pred统计
    output_lines.append("=" * 100)
    output_lines.append("Pred 曲面分布统计")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.extend(format_statistics(pred_stats))
    output_lines.append("")
    
    # 对比
    output_lines.extend(format_comparison(gt_stats, pred_stats))
    output_lines.append("")
    
    # 错误文件列表
    if gt_stats['error_files'] or pred_stats['error_files']:
        output_lines.append("=" * 100)
        output_lines.append("处理失败的文件:")
        output_lines.append("=" * 100)
        
        if gt_stats['error_files']:
            output_lines.append("")
            output_lines.append("GT 错误文件:")
            output_lines.append("-" * 80)
            for file_path, error in gt_stats['error_files'][:10]:
                output_lines.append(f"文件: {file_path}")
                output_lines.append(f"错误: {error}")
                output_lines.append("")
            if len(gt_stats['error_files']) > 10:
                output_lines.append(f"... 还有 {len(gt_stats['error_files']) - 10} 个错误文件未显示")
        
        if pred_stats['error_files']:
            output_lines.append("")
            output_lines.append("Pred 错误文件:")
            output_lines.append("-" * 80)
            for file_path, error in pred_stats['error_files'][:10]:
                output_lines.append(f"文件: {file_path}")
                output_lines.append(f"错误: {error}")
                output_lines.append("")
            if len(pred_stats['error_files']) > 10:
                output_lines.append(f"... 还有 {len(pred_stats['error_files']) - 10} 个错误文件未显示")
    
    output_lines.append("=" * 100)
    
    # 打印到控制台
    output_text = '\n'.join(output_lines)
    print(output_text)
    
    # 写入文件（如果指定了输出路径）
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='对比GT和Pred JSON文件的曲面类型分布'
    )
    parser.add_argument(
        '--gt_dir',
        type=str,
        required=True,
        help='GT JSON文件所在目录'
    )
    parser.add_argument(
        '--pred_dir',
        type=str,
        required=True,
        help='Pred JSON文件所在目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（可选，不指定则只打印到控制台）'
    )
    parser.add_argument(
        '--gt_pattern',
        type=str,
        default='*.json',
        help='GT文件匹配模式（默认: *.json）'
    )
    parser.add_argument(
        '--pred_pattern',
        type=str,
        default='*.json',
        help='Pred文件匹配模式（默认: *.json）'
    )
    
    args = parser.parse_args()
    
    analyze_and_compare_experiment(
        args.gt_dir,
        args.pred_dir,
        args.output,
        args.gt_pattern,
        args.pred_pattern
    )


if __name__ == '__main__':
    # 示例用法（如果不使用命令行参数）
    # gt_dir = '/path/to/gt/json/files'
    # pred_dir = '/path/to/pred/json/files'
    # output_file = '/path/to/output/surface_dist_comparison.txt'
    # analyze_and_compare_experiment(gt_dir, pred_dir, output_file)
    
    main()

