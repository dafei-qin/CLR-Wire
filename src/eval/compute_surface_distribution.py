"""统计JSON文件中的曲面类型分布"""
import json
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm


def analyze_surface_distribution(data_dir: str, output_file: str):
    """
    分析指定目录下所有JSON文件的曲面类型分布
    
    Args:
        data_dir: 数据目录路径
        output_file: 输出文件路径
    """
    data_path = Path(data_dir)
    
    # 查找所有JSON文件
    json_files = list(data_path.rglob("*.json"))
    
    if not json_files:
        print(f"未在 {data_dir} 中找到任何 JSON 文件")
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    # 统计数据
    surface_type_counter = Counter()  # 总体曲面类型计数
    file_surface_counts = []  # 每个文件的曲面数量
    total_surfaces = 0
    error_files = []
    
    # 遍历所有JSON文件
    for json_file in tqdm(json_files, desc="处理JSON文件"):
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
    
    # 准备输出内容
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("曲面类型分布统计")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # 文件统计信息
    output_lines.append(f"数据目录: {data_dir}")
    output_lines.append(f"总文件数: {len(json_files)}")
    output_lines.append(f"成功处理: {len(json_files) - len(error_files)}")
    output_lines.append(f"处理失败: {len(error_files)}")
    output_lines.append("")
    
    # 曲面总数统计
    output_lines.append(f"曲面总数: {total_surfaces:,}")
    if file_surface_counts:
        output_lines.append(f"每个文件平均曲面数: {sum(file_surface_counts) / len(file_surface_counts):.2f}")
        output_lines.append(f"每个文件最少曲面数: {min(file_surface_counts)}")
        output_lines.append(f"每个文件最多曲面数: {max(file_surface_counts)}")
    output_lines.append("")
    
    # 曲面类型分布
    output_lines.append("-" * 80)
    output_lines.append("曲面类型详细分布:")
    output_lines.append("-" * 80)
    output_lines.append(f"{'曲面类型':<30} {'数量':>15} {'占比':>15}")
    output_lines.append("-" * 80)
    
    # 按数量降序排序
    sorted_types = sorted(surface_type_counter.items(), key=lambda x: x[1], reverse=True)
    
    for surface_type, count in sorted_types:
        percentage = (count / total_surfaces * 100) if total_surfaces > 0 else 0
        output_lines.append(f"{surface_type:<30} {count:>15,} {percentage:>14.2f}%")
    
    output_lines.append("-" * 80)
    output_lines.append("")
    
    # 错误文件列表（如果有）
    if error_files:
        output_lines.append("-" * 80)
        output_lines.append("处理失败的文件:")
        output_lines.append("-" * 80)
        for file_path, error in error_files[:20]:  # 只显示前20个错误
            output_lines.append(f"文件: {file_path}")
            output_lines.append(f"错误: {error}")
            output_lines.append("")
        if len(error_files) > 20:
            output_lines.append(f"... 还有 {len(error_files) - 20} 个错误文件未显示")
        output_lines.append("")
    
    output_lines.append("=" * 80)
    
    # 写入文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    # 同时打印到控制台
    print('\n'.join(output_lines))
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    data_dir = '/deemos-research-area-d/meshgen/cad/data/abc_step_pc_correct_normal/0'
    output_file = '/deemos-research-area-d/meshgen/cad/CLR-Wire/src/eval/surface_dist.txt'
    
    analyze_surface_distribution(data_dir, output_file)

