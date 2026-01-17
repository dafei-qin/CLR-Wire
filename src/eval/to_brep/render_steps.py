#!/usr/bin/env python3
"""
独立的 STEP 文件渲染脚本
用于批量将 STEP 文件渲染为 JPG 预览图
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from PIL import Image

from OCC.Core.BRepTools import breptools
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRep import BRep_Builder
from OCC.Display.OCCViewer import OffscreenRenderer
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Extend.DataExchange import read_step_file


def load_step_file(filename):
    """读取 STEP/BREP 文件"""
    try:
        # shape = TopoDS_Shape()
        # builder = BRep_Builder()
        # success = breptools.Read(shape, filename, builder)
        # if not success:
        #     print(f"  ⚠ Failed to read file: {filename}")
        #     return None
        shape = read_step_file(filename)
        return shape
    except Exception as e:
        print(f"  ⚠ Exception reading file: {e}")
        return None


def render_views(shape, renderer, img_size=(800, 600)):
    """渲染四个视角的图像"""
    BG_COLOR = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
    SHAPE_COLOR = Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB)
    
    renderer.EraseAll()
    images = []
    temp_img_path = "temp_render.png"
    
    views = ['Front', 'Left', 'Top', 'Iso']
    for view_name in views:
        getattr(renderer, f'View_{view_name}')()
        renderer.View.FitAll()
        renderer.View.ZFitAll()
        renderer.View.SetScale(renderer.View.Scale() * 0.95)
        renderer.View.SetBgGradientColors(BG_COLOR, BG_COLOR)
        renderer.DisplayShape(shape, color=SHAPE_COLOR, dump_image_path=".", dump_image_filename=temp_img_path)
        img = Image.open(temp_img_path).convert("RGB")
        images.append(img)
    
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    return images


def create_grid_image(images):
    """将4张图片拼接成 2x2 网格"""
    w, h = images[0].size
    grid_img = Image.new('RGB', (w * 2, h * 2))
    
    grid_img.paste(images[2], (0, 0))      # Top
    grid_img.paste(images[3], (w, 0))      # Iso
    grid_img.paste(images[0], (0, h))      # Front
    grid_img.paste(images[1], (w, h))      # Left
    
    return grid_img


def render_step_to_jpg(step_file, jpg_file, renderer, verbose=True):
    """将 STEP 文件渲染为 JPG"""
    try:
        if verbose:
            print(f"  Reading: {os.path.basename(step_file)}")
        
        shape = load_step_file(step_file)
        if shape is None:
            return False
        
        if verbose:
            print(f"  Rendering...")
        
        images = render_views(shape, renderer)
        final_img = create_grid_image(images)
        final_img.save(jpg_file, 'JPEG', quality=95)
        
        if verbose:
            print(f"  ✓ Saved: {os.path.basename(jpg_file)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch render STEP files to JPG previews')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing STEP files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for JPG files (default: same as input)')
    parser.add_argument('--pattern', type=str, default='*.step',
                        help='File pattern to match (default: *.step)')
    parser.add_argument('--size', type=int, nargs=2, default=[800, 600],
                        help='Image size (width height), default: 800 600')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化渲染器
    print("="*60)
    print("Initializing OffscreenRenderer...")
    try:
        renderer = OffscreenRenderer(tuple(args.size))
        print(f"  ✓ Initialized ({args.size[0]}x{args.size[1]})")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 查找 STEP 文件
    step_files = glob.glob(os.path.join(input_dir, args.pattern))
    print(f"\nFound {len(step_files)} STEP file(s) in {input_dir}")
    print("="*60)
    
    if len(step_files) == 0:
        print("No files to process!")
        return
    
    # 批量渲染
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for i, step_file in enumerate(step_files, 1):
        filename = os.path.basename(step_file)
        jpg_file = os.path.join(output_dir, filename.replace('.step', '.jpg'))
        
        # 如果 JPG 已存在，跳过
        if os.path.exists(jpg_file):
            print(f"[{i}/{len(step_files)}] ⊙ Skipped: {filename} (already exists)")
            skipped_count += 1
            continue
        
        print(f"\n[{i}/{len(step_files)}] Processing: {filename}")
        
        if render_step_to_jpg(step_file, jpg_file, renderer):
            success_count += 1
        else:
            failed_count += 1
    
    print("\n" + "="*60)
    print(f"Rendering complete!")
    print(f"  Total files: {len(step_files)}")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

