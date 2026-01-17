import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
from utils.surface import visualize_json_interset, extract_mesh_from_face
from utils.surf_tools import surf_surf_interset,all_surf_interset,adjust_cylinders_to_planes
from utils.occ_build_face import cut_face_with_edges
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
import numpy as np
from utils.processply import filter_faces_by_coverage,estimate_optimal_radius,merge_faces_on_same_surface
from utils.processply import solve_maximum_score_manifold,estimate_point_cloud_normals
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator, TopoDS_Face, TopoDS_Wire, TopoDS_Vertex, TopoDS_Edge,topods, TopoDS_Shape

from OCC.Core.BRepTools import breptools
from plyfile import PlyData, PlyElement
from OCC.Core.gp import gp_Pnt,gp_Pnt2d,gp_Dir

import os
import traceback
import time
import argparse
from scipy.spatial import cKDTree

# 渲染相关导入
from PIL import Image
from OCC.Display.OCCViewer import OffscreenRenderer
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Extend.DataExchange import read_step_file


# ==================== 渲染相关函数 ====================
def load_step_file(filename):
    """读取 STEP/BREP 文件"""
    try:
        shape = TopoDS_Shape()
        builder = BRep_Builder()
        success = breptools.Read(shape, filename, builder)
        if not success:
            print(f"  ⚠ Failed to read file: {filename}")
            return None
        return shape
    except Exception as e:
        print(f"  ⚠ Exception reading file: {e}")
        return None

def render_views(shape, renderer, img_size=(800, 600)):
    """
    渲染四个视角的图像
    返回: List[PIL.Image] -> [Front, Left, Top, Iso]
    """
    BG_COLOR = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)  # 白色背景
    SHAPE_COLOR = Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB)  # 深灰色模型
    
    renderer.EraseAll()
    images = []
    temp_img_path = "temp_render.png"
    
    # Front view
    renderer.View_Front()
    renderer.View.FitAll()
    renderer.View.ZFitAll()
    renderer.View.SetScale(renderer.View.Scale() * 0.95)
    renderer.View.SetBgGradientColors(BG_COLOR, BG_COLOR)
    renderer.DisplayShape(shape, color=SHAPE_COLOR, dump_image_path=".", dump_image_filename=temp_img_path)
    img = Image.open(temp_img_path).convert("RGB")
    images.append(img)
    
    # Left view
    renderer.View_Left()
    renderer.View.FitAll()
    renderer.View.ZFitAll()
    renderer.View.SetScale(renderer.View.Scale() * 0.95)
    renderer.View.SetBgGradientColors(BG_COLOR, BG_COLOR)
    renderer.DisplayShape(shape, color=SHAPE_COLOR, dump_image_path=".", dump_image_filename=temp_img_path)
    img = Image.open(temp_img_path).convert("RGB")
    images.append(img)
    
    # Top view
    renderer.View_Top()
    renderer.View.FitAll()
    renderer.View.ZFitAll()
    renderer.View.SetScale(renderer.View.Scale() * 0.95)
    renderer.View.SetBgGradientColors(BG_COLOR, BG_COLOR)
    renderer.DisplayShape(shape, color=SHAPE_COLOR, dump_image_path=".", dump_image_filename=temp_img_path)
    img = Image.open(temp_img_path).convert("RGB")
    images.append(img)
    
    # Iso view
    renderer.View_Iso()
    renderer.View.FitAll()
    renderer.View.ZFitAll()
    renderer.View.SetScale(renderer.View.Scale() * 0.95)
    renderer.View.SetBgGradientColors(BG_COLOR, BG_COLOR)
    renderer.DisplayShape(shape, color=SHAPE_COLOR, dump_image_path=".", dump_image_filename=temp_img_path)
    img = Image.open(temp_img_path).convert("RGB")
    images.append(img)
    
    # Clean up temp file
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    return images

def create_grid_image(images):
    """
    将4张图片拼接成 2x2 网格
    布局:
    Top   | Iso
    ------+------
    Front | Left
    """
    w, h = images[0].size
    grid_img = Image.new('RGB', (w * 2, h * 2))
    
    grid_img.paste(images[2], (0, 0))      # Top-Left: Top
    grid_img.paste(images[3], (w, 0))      # Top-Right: Iso
    grid_img.paste(images[0], (0, h))      # Bottom-Left: Front
    grid_img.paste(images[1], (w, h))      # Bottom-Right: Left
    
    return grid_img

def render_step_to_jpg(step_file, jpg_file, renderer, img_size=(800, 600)):
    """
    将 STEP 文件渲染为 JPG 图像
    
    Args:
        step_file: STEP 文件路径
        jpg_file: 输出 JPG 文件路径
        renderer: OffscreenRenderer 实例
        img_size: 单张图片尺寸 (width, height)
    """
    try:
        # 读取 STEP 文件
        print(f"    Reading file: {os.path.basename(step_file)}", flush=True)
        shape = load_step_file(step_file)
        if shape is None:
            print(f"    ✗ Failed to load shape", flush=True)
            return False
        
        # 渲染 4 个视角
        print(f"    Rendering 4 views...", flush=True)
        images = render_views(shape, renderer, img_size)
        
        # 拼接成 2x2 网格
        print(f"    Creating grid image...", flush=True)
        final_img = create_grid_image(images)
        
        # 保存为 JPG
        print(f"    Saving JPG...", flush=True)
        final_img.save(jpg_file, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"    ✗ Render exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

# ==================== 原有函数 ====================
def find_matching_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return []
    all_files = os.listdir(folder_path)
    ply_files = [f for f in all_files if f.endswith('.ply')]
    json_files = [f for f in all_files if f.endswith('.json')]
    print(f"在 '{folder_path}' 中找到 {len(ply_files)} 个 ply 文件。")
    ls=[]
    for ply_file in ply_files:
        parts = ply_file.split('_')
        if len(parts) >= 4:
            s_string = "_".join(parts[:3])
            target_prefix = s_string + "_gt"
            found_gt = False
            for json_f in json_files:
                if json_f.startswith(target_prefix):
                    found_gt = json_f
                    break 
            target_prefix = s_string + "_pred"
            found_pred = False
            for json_f in json_files:
                if json_f.startswith(target_prefix):
                    found_pred = json_f
                    break 
            
            if found_gt and found_pred:
                ls.append((ply_file,found_gt,found_pred))
        else:
            print(f"[跳过]     PLY: {ply_file} (文件名中下划线不足3个)")
    return ls

def process_one(input_path, json_name, ply_name, output_path):
    """
    处理单个样本：从 JSON + PLY 生成 STEP 文件
    
    Args:
        input_path: 输入目录
        json_name: JSON 文件名
        ply_name: PLY 文件名
        output_path: 输出目录
    
    Returns:
        生成的 STEP 文件路径，失败返回 None
    """
    # 读取点云数据
    ply_data = PlyData.read(os.path.join(input_path, ply_name))
    vertex_element = ply_data['vertex']
    cloud_np = np.vstack([vertex_element['x'], vertex_element['y'], vertex_element['z']]).T
    cloud_normals = estimate_point_cloud_normals(cloud_np, k=10)
    tree = cKDTree(cloud_np)

    # 读取曲面数据
    surfaces = json.load(open(os.path.join(input_path, json_name), 'r'))

    # 构建曲面（不使用 polyscope 可视化）
    surfaces_vis = visualize_json_interset(surfaces, plot=False, plot_gui=False, tol=1e-5)

    # 提取所有面并调整
    all_faces = [face['surface'] for face in surfaces_vis.values()]
    all_faces = adjust_cylinders_to_planes(all_faces)

    # 面片切割与过滤
    all_split_faces = []
    all_scores = []
    for idx_m in range(len(all_faces)):
        print(f"  Processing Face-{idx_m}/{len(all_faces)}", flush=True)
        
        # 与其他面求交
        new_faces = all_surf_interset(all_faces[idx_m], all_faces)
        
        # 点云覆盖率过滤
        new_faces, scores = filter_faces_by_coverage(
            new_faces, cloud_normals, tree, 500,
            estimate_optimal_radius(cloud_np, 2000, 2.3), 0.6
        )
        all_split_faces = all_split_faces + new_faces
        all_scores = all_scores + scores
    
    # 流形约束优化
    all_split_faces = solve_maximum_score_manifold(all_split_faces, all_scores)
    
    # 合并同一曲面上的面
    all_split_faces = merge_faces_on_same_surface(all_split_faces)

    # 构建并保存 STEP 文件
    builder = BRep_Builder()
    new_compound = TopoDS_Compound()
    builder.MakeCompound(new_compound)
    for f in all_split_faces:
        builder.Add(new_compound, f)
    
    step_file = os.path.join(output_path, ply_name + ".step")
    breptools.Write(new_compound, step_file)
    
    return step_file

data_paths=[
    ("F:\\aa-test-data\\0-test\\","F:\\aa-test-data\\0-res\\"),
    ("F:\\aa-test-data\\1-results_23k_1e-2\\","F:\\aa-test-data\\1-brep_results\\"),
    ("F:\\aa-test-data\\2-results_23k_3e-2\\","F:\\aa-test-data\\2-brep_results\\"),
    ("F:\\aa-test-data\\3-test\\","F:\\aa-test-data\\3-res\\"),
    ("F:\\aa-test-data\\4-highres\\","F:\\aa-test-data\\4-res\\"),
    ("F:\\aa-test-data\\5-highres-part\\","F:\\aa-test-data\\5-res\\"),
    ("F:\\aa-test-data\\6-full\\","F:\\aa-test-data\\6-res\\"),
]


if __name__ == '__main__':
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Convert JSON + PLY to BREP/STEP files')
    parser.add_argument('--input_path', type=str, default=None,
                        help='Input directory containing PLY and JSON files')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output directory for BREP files')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (process first K samples)')
    parser.add_argument('--data_idx', type=int, default=6,
                        help='Index of data_paths to use (default: 6, ignored if --input_path is specified)')
    parser.add_argument('--cont', type=str, default='',
                        help='Continue from file starting with this prefix (for resuming)')
    
    args = parser.parse_args()
    
    # 初始化离屏渲染器
    print("\n" + "="*60)
    print("Initializing OffscreenRenderer...")
    print("="*60)
    renderer = None
    try:
        renderer = OffscreenRenderer((800, 600))
        print("  ✓ OffscreenRenderer initialized (800x600)")
    except Exception as e:
        print(f"  ✗ Failed to initialize renderer: {e}")
        print("  ⚠ Rendering will be skipped. STEP files will still be generated.")
        import traceback
        traceback.print_exc()
    
    # 确定输入输出路径
    if args.input_path is not None and args.output_path is not None:
        # 使用命令行指定的路径
        input_path = args.input_path
        output_path = args.output_path
        print(f"Using command line paths:")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
    else:
        # 使用预定义的路径
        data_idx = args.data_idx
        input_path, output_path = data_paths[data_idx]
        print(f"Using data_paths[{data_idx}]:")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 查找匹配的文件
    data = find_matching_files(input_path)
    
    # 限制处理的样本数
    if args.num_samples is not None:
        data = data[:args.num_samples]
        print(f"Processing first {args.num_samples} samples")
    
    print(f"Found {len(data)} file(s) to process\n")
    
    idx = 0
    cont = args.cont
    
    for ply, gt, pred in data:
        idx += 1
        
        # 断点续传逻辑
        if cont != '':
            if ply.startswith(cont):
                cont = ''
            else:
                continue
            
        print(f"[{idx}/{len(data)}] Processing: {pred}", flush=True)
        
        start_time = time.time()

        try:
            # 生成 STEP 文件
            step_file = process_one(input_path, pred, ply, output_path)
            if step_file and os.path.exists(step_file):
                print(f"  ✓ STEP saved to: {step_file}", flush=True)
                
                # 渲染 JPG 图像
                if renderer is not None:
                    jpg_file = step_file.replace('.step', '.jpg')
                    print(f"  🎨 Rendering to JPG...", flush=True)
                    if render_step_to_jpg(step_file, jpg_file, renderer):
                        print(f"  ✓ JPG saved to: {jpg_file}", flush=True)
                    else:
                        print(f"  ⚠ Rendering failed", flush=True)
            else:
                print(f"  ✗ Failed to generate STEP file", flush=True)
                
        except Exception as e:
            print(f"  ✗ Exception: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            
        end_time = time.time()
        print(f"  ⏱ Time: {end_time - start_time:.2f} seconds\n", flush=True)
    
    print(f"\n{'='*60}")
    print(f"Finished processing {len(data)} file(s)")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")


