import os
import glob
from PIL import Image

from OCC.Core.BRepTools import breptools
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRep import BRep_Builder
from OCC.Display.OCCViewer import OffscreenRenderer
from OCC.Core.V3d import V3d_Xneg, V3d_Yneg, V3d_Zpos, V3d_XposYnegZpos,V3d_View
from OCC.Core.Quantity import Quantity_Color,Quantity_TOC_RGB 

# =================配置区域=================
data_paths=[
    ("F:\\aa-test-data\\0-test\\","F:\\aa-test-data\\0-res\\"),
    ("F:\\aa-test-data\\1-results_23k_1e-2\\","F:\\aa-test-data\\1-brep_results\\"),
    ("F:\\aa-test-data\\2-results_23k_3e-2\\","F:\\aa-test-data\\2-brep_results\\"),
    ("F:\\aa-test-data\\3-test\\","F:\\aa-test-data\\3-res\\"),
    ("F:\\aa-test-data\\4-highres\\","F:\\aa-test-data\\4-res\\"),
    ("F:\\aa-test-data\\5-highres-part\\","F:\\aa-test-data\\5-res\\"),
]
data_idx=4
_,INPUT_DIR=data_paths[data_idx]
OUTPUT_DIR = INPUT_DIR

IMG_SIZE = (800, 600)  # 单张子图的分辨率 (宽, 高)
BG_COLOR1 = Quantity_Color(1.0, 1.0, 1.0,Quantity_TOC_RGB) # 背景渐变色顶部 (RGB)
BG_COLOR2 = Quantity_Color(1.0, 1.0, 1.0,Quantity_TOC_RGB) # 背景渐变色底部 (RGB)
SHAPE_COLOR = Quantity_Color(0.1,0.1,0.1,Quantity_TOC_RGB) # 模型颜色 (R, G, B, 0-1)
# =========================================

def load_brep(filename):
    """读取 BREP 文件"""
    shape = TopoDS_Shape()
    builder = BRep_Builder()
    success = breptools.Read(shape, filename, builder)
    if not success:
        return None
    return shape

def render_views(shape, renderer:OffscreenRenderer):
    """
    渲染四个视角的图像
    返回: List[PIL.Image] -> [Top, Front, Left, Iso]
    """
    renderer.EraseAll()
    images = []
    temp_img_path = "temp_render.png"
    
    renderer.View_Front()
    renderer.View.FitAll()
    renderer.View.ZFitAll()
    renderer.View.SetScale(renderer.View.Scale() * 0.95)
    renderer.View.SetBgGradientColors(BG_COLOR1,BG_COLOR1)
    renderer.DisplayShape(shape, color=SHAPE_COLOR, dump_image_path=".", dump_image_filename=temp_img_path) 
    img = Image.open(temp_img_path).convert("RGB")
    images.append(img)

    renderer.View_Left()
    renderer.View.FitAll()
    renderer.View.ZFitAll()
    renderer.View.SetScale(renderer.View.Scale() * 0.95)
    renderer.View.SetBgGradientColors(BG_COLOR1,BG_COLOR1)
    renderer.DisplayShape(shape, color=SHAPE_COLOR, dump_image_path=".", dump_image_filename=temp_img_path) 
    img = Image.open(temp_img_path).convert("RGB")
    images.append(img)

    renderer.View_Top()
    renderer.View.FitAll()
    renderer.View.ZFitAll()
    renderer.View.SetScale(renderer.View.Scale() * 0.95)
    renderer.View.SetBgGradientColors(BG_COLOR1,BG_COLOR1)
    renderer.DisplayShape(shape, color=SHAPE_COLOR, dump_image_path=".", dump_image_filename=temp_img_path) 
    img = Image.open(temp_img_path).convert("RGB")
    images.append(img)

    renderer.View_Iso()
    renderer.View.FitAll()
    renderer.View.ZFitAll()
    renderer.View.SetScale(renderer.View.Scale() * 0.95)
    renderer.View.SetBgGradientColors(BG_COLOR1,BG_COLOR1)
    renderer.DisplayShape(shape, color=SHAPE_COLOR, dump_image_path=".", dump_image_filename=temp_img_path) 
    img = Image.open(temp_img_path).convert("RGB")
    images.append(img)
        

    if os.path.exists("temp_render.png"):
        os.remove("temp_render.png")
        
    return images

def create_grid_image(images):
    """
    将4张图片拼接成 2x2 网格
    布局:
    Top   | Iso
    ------+------
    Front | Right
    """
    w, h = images[0].size
    grid_img = Image.new('RGB', (w * 2, h * 2))
    
    # 放置顺序
    # images[0]: Top   -> (0, 0)
    # images[1]: Front -> (0, h)
    # images[2]: Right -> (w, h)
    # images[3]: Iso   -> (w, 0)  <-- 放在右上角比较好看
    
    grid_img.paste(images[0], (0, 0))      # Top-Left: Top
    grid_img.paste(images[3], (w, 0))      # Top-Right: Iso
    grid_img.paste(images[1], (0, h))      # Bottom-Left: Front
    grid_img.paste(images[2], (w, h))      # Bottom-Right: Right
    
    return grid_img

def main():
    # 1. 确保目录存在
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"请将 .brep 文件放入 '{INPUT_DIR}' 文件夹中。")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 初始化离线渲染器
    # 宽, 高
    renderer = OffscreenRenderer((IMG_SIZE[0], IMG_SIZE[1]))

    brep_files = glob.glob(os.path.join(INPUT_DIR, "*.brep"))
    print(f"找到 {len(brep_files)} 个 brep 文件。")

    for file_path in brep_files:
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, name_no_ext + ".jpg")
        
        print(f"正在处理: {filename} ...")
        
        try:
            # 读取模型
            shape = load_brep(file_path)
            if not shape:
                print(f"  错误: 无法读取 {filename}")
                continue
            
            # 渲染4个视角
            images = render_views(shape, renderer)
            
            # 拼接图像
            final_img = create_grid_image(images)
            
            # 保存
            final_img.save(output_path, quality=95)
            print(f"  已保存: {output_path}")
            
        except Exception as e:
            print(f"  处理失败: {e}")

    print("全部完成。")

if __name__ == "__main__":
    main()