import sys
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import re

def parse_stats_file(file_path):
    """解析单个统计文件，提取关键信息"""
    stats = {
        'config': {},
        'json_stats': {},
        'surface_stats': {},
        'type_distribution': Counter()
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取 JSON 统计信息
        match = re.search(r'Total JSON files processed:\s*(\d+)', content)
        if match:
            stats['json_stats']['total_jsons'] = int(match.group(1))
        
        match = re.search(r'JSON files with all valid surfaces:\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if match:
            stats['json_stats']['all_valid_jsons'] = int(match.group(1))
            stats['json_stats']['all_valid_percentage'] = float(match.group(2))
        
        match = re.search(r'JSON files with some invalid surfaces:\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if match:
            stats['json_stats']['some_invalid_jsons'] = int(match.group(1))
            stats['json_stats']['some_invalid_percentage'] = float(match.group(2))
        
        # 提取 Surface 统计信息
        match = re.search(r'Total original surfaces \(in JSON files\):\s*(\d+)', content)
        if match:
            stats['surface_stats']['total_original'] = int(match.group(1))
        
        match = re.search(r'Total valid surfaces \(after processing\):\s*(\d+)', content)
        if match:
            stats['surface_stats']['total_valid'] = int(match.group(1))
        
        match = re.search(r'Total surfaces saved to cache:\s*(\d+)', content)
        if match:
            stats['surface_stats']['total_saved'] = int(match.group(1))
        
        match = re.search(r'Success rate:\s*([0-9.]+)%', content)
        if match:
            stats['surface_stats']['success_rate'] = float(match.group(1))
        
        match = re.search(r'Invalid/dropped surfaces:\s*(\d+)\s*\(([0-9.]+)%\)', content)
        if match:
            stats['surface_stats']['invalid_count'] = int(match.group(1))
            stats['surface_stats']['invalid_percentage'] = float(match.group(2))
        
        # 提取类型分布信息
        type_section = re.search(r'\[Surface Type Distribution\].*?^=+', content, re.MULTILINE | re.DOTALL)
        if type_section:
            type_lines = type_section.group(0).split('\n')
            for line in type_lines:
                # 匹配格式: TypeName         Count      Percentage
                match = re.match(r'^([A-Za-z_]+(?:\([0-9]+\))?)\s+(\d+)\s+([0-9.]+)%', line.strip())
                if match:
                    type_name = match.group(1)
                    count = int(match.group(2))
                    stats['type_distribution'][type_name] = count
        
        # 提取配置信息
        match = re.search(r'Canonical:\s*(True|False)', content)
        if match:
            stats['config']['canonical'] = match.group(1) == 'True'
        
        match = re.search(r'Detect closed:\s*(True|False)', content)
        if match:
            stats['config']['detect_closed'] = match.group(1) == 'True'
        
        match = re.search(r'Bspline fit threshold:\s*([0-9.e-]+)', content)
        if match:
            stats['config']['bspline_fit_threshold'] = float(match.group(1))
        
        return stats
    except Exception as e:
        print(f"Warning: Failed to parse {file_path}: {e}")
        return None

def aggregate_stats(stats_list):
    """汇总所有统计信息"""
    aggregated = {
        'total_files': len(stats_list),
        'json_stats': {
            'total_jsons': 0,
            'all_valid_jsons': 0,
            'some_invalid_jsons': 0
        },
        'surface_stats': {
            'total_original': 0,
            'total_valid': 0,
            'total_saved': 0,
            'invalid_count': 0
        },
        'type_distribution': Counter(),
        'configs': defaultdict(list)
    }
    
    for stats in stats_list:
        if stats is None:
            continue
        
        # 汇总 JSON 统计
        for key in ['total_jsons', 'all_valid_jsons', 'some_invalid_jsons']:
            if key in stats['json_stats']:
                aggregated['json_stats'][key] += stats['json_stats'][key]
        
        # 汇总 Surface 统计
        for key in ['total_original', 'total_valid', 'total_saved', 'invalid_count']:
            if key in stats['surface_stats']:
                aggregated['surface_stats'][key] += stats['surface_stats'][key]
        
        # 汇总类型分布
        aggregated['type_distribution'].update(stats['type_distribution'])
        
        # 收集配置信息
        for key, value in stats['config'].items():
            aggregated['configs'][key].append(value)
    
    return aggregated

def print_aggregated_stats(aggregated, output_file=None):
    """打印汇总的统计信息"""
    lines = []
    lines.append('='*70)
    lines.append('AGGREGATED PROCESSING SUMMARY')
    lines.append('='*70)
    lines.append('')
    
    # 文件统计
    lines.append('[Files Processed]')
    lines.append(f'Total statistics files analyzed: {aggregated["total_files"]}')
    lines.append('')
    
    # 配置信息总结
    lines.append('[Configuration Summary]')
    for key, values in aggregated['configs'].items():
        unique_values = set(values)
        if len(unique_values) == 1:
            lines.append(f'{key}: {list(unique_values)[0]}')
        else:
            lines.append(f'{key}: {unique_values} (mixed)')
    lines.append('')
    
    # JSON 统计
    json_stats = aggregated['json_stats']
    lines.append('[Aggregated JSON Files Statistics]')
    lines.append(f'Total JSON files processed: {json_stats["total_jsons"]}')
    
    if json_stats['total_jsons'] > 0:
        all_valid_pct = 100 * json_stats['all_valid_jsons'] / json_stats['total_jsons']
        some_invalid_pct = 100 * json_stats['some_invalid_jsons'] / json_stats['total_jsons']
        lines.append(f'JSON files with all valid surfaces: {json_stats["all_valid_jsons"]} ({all_valid_pct:.2f}%)')
        lines.append(f'JSON files with some invalid surfaces: {json_stats["some_invalid_jsons"]} ({some_invalid_pct:.2f}%)')
    lines.append('')
    
    # Surface 统计
    surf_stats = aggregated['surface_stats']
    lines.append('[Aggregated Surface Statistics]')
    lines.append(f'Total original surfaces (in JSON files): {surf_stats["total_original"]}')
    lines.append(f'Total valid surfaces (after processing): {surf_stats["total_valid"]}')
    lines.append(f'Total surfaces saved to cache: {surf_stats["total_saved"]}')
    
    if surf_stats['total_original'] > 0:
        success_rate = 100 * surf_stats['total_valid'] / surf_stats['total_original']
        invalid_pct = 100 * surf_stats['invalid_count'] / surf_stats['total_original']
        lines.append(f'Overall success rate: {success_rate:.2f}%')
        lines.append(f'Invalid/dropped surfaces: {surf_stats["invalid_count"]} ({invalid_pct:.2f}%)')
    lines.append('')
    
    # 类型分布
    total_surfaces = sum(aggregated['type_distribution'].values())
    lines.append('[Aggregated Surface Type Distribution]')
    lines.append(f'{"Type":<20} {"Count":<10} {"Percentage":<10}')
    lines.append(f'{"-"*40}')
    
    # 按计数排序
    for type_name, count in sorted(aggregated['type_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
        percentage = 100 * count / total_surfaces if total_surfaces > 0 else 0
        lines.append(f'{type_name:<20} {count:<10} {percentage:>6.2f}%')
    
    lines.append('')
    lines.append(f'Total surfaces across all types: {total_surfaces}')
    lines.append('='*70)
    
    # 输出到控制台
    output_text = '\n'.join(lines)
    print(output_text)
    
    # 如果指定了输出文件，写入文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text + '\n')
        print(f'\nAggregated statistics saved to: {output_file}')

def main():
    parser = argparse.ArgumentParser(
        description='Analyze and aggregate statistics from multiple *_stats.txt files'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing *_stats.txt files')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path for aggregated statistics (optional)')
    parser.add_argument('--pattern', type=str, default='*_stats.txt',
                       help='File pattern to search for (default: *_stats.txt)')
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # 查找所有统计文件
    stats_files = list(input_path.glob(args.pattern))
    
    if not stats_files:
        print(f"Error: No files matching pattern '{args.pattern}' found in {args.input_dir}")
        return
    
    print(f"Found {len(stats_files)} statistics files")
    print("Parsing files...")
    
    # 解析所有统计文件
    stats_list = []
    for stats_file in stats_files:
        stats = parse_stats_file(stats_file)
        if stats:
            stats_list.append(stats)
    
    print(f"Successfully parsed {len(stats_list)} files")
    print()
    
    # 汇总统计信息
    aggregated = aggregate_stats(stats_list)
    
    # 打印汇总结果
    print_aggregated_stats(aggregated, args.output_file)

if __name__ == '__main__':
    main()









