# sample_simple_surface.py - 使用说明

## 概述

该工具用于对简单曲面（plane, cylinder, cone, sphere, torus）进行UV参数空间的均匀采样。

## 关键特性

✅ **直接使用 `dataset_v1.py` 的 `_recover_surface` 方法** - 保证参数恢复的一致性  
✅ **复用 `utils/surface.py` 的曲面构建函数** - 使用 `build_plane_face` 和 `build_second_order_surface`  
✅ **无需网格化** - 直接从 OCC Geom_Surface 采样  
✅ **支持批量处理** - 高效处理多个曲面  

## 主要函数

### 1. `sample_surface_uniform()` - 单个曲面采样

```python
from src.tools.sample_simple_surface import sample_surface_uniform
from src.dataset.dataset_v1 import dataset_compound

# 加载数据集
dataset = dataset_compound("path/to/json/dir")
params, types, mask, _, _, _ = dataset[0]

# 采样第一个曲面
points = sample_surface_uniform(
    params=params[0],           # 参数向量
    surface_type_idx=types[0],  # 曲面类型索引
    num_u=32,                   # u方向采样点数
    num_v=32,                   # v方向采样点数
    flatten=True,               # 返回 (N, 3) 而非 (num_v, num_u, 3)
)
print(f"采样了 {points.shape[0]} 个点")
```

### 2. `sample_surface_batch()` - 批量曲面采样

```python
from src.tools.sample_simple_surface import sample_surface_batch

# 批量采样所有曲面
points_batch = sample_surface_batch(
    params_batch=params,        # shape: (batch_size, param_dim)
    surface_types_batch=types,  # shape: (batch_size,)
    mask_batch=mask,            # shape: (batch_size,)
    num_u=32,
    num_v=32,
    flatten=True,
)
# 返回 shape: (batch_size, num_u*num_v, 3)
```

### 3. 可视化采样结果

```python
import polyscope as ps

ps.init()
ps.register_point_cloud("sampled_points", points, radius=0.003)
ps.show()
```

或使用命令行：

```bash
python src/tools/sample_simple_surface.py \
    --json_dir path/to/json/dir \
    --index 0 \
    --num_u 32 \
    --num_v 32
```

## 技术实现

### 参数恢复
- 直接调用 `dataset_compound._recover_surface(params, surface_type_idx)`
- 自动处理参数后处理（如 radius 的 exp 变换）

### 曲面构建
- **Plane**: 使用 `utils.surface.build_plane_face(face_dict, meshify=False)`
- **其他曲面**: 使用 `utils.surface.build_second_order_surface(face_dict, meshify=False)`
- 从 `TopoDS_Face` 提取 `Geom_Surface`：`BRep_Tool.Surface(face_shape, location)`

### 采样方法
- 在UV参数空间均匀采样
- 使用 `occ_surface.Value(u, v)` 获取3D点坐标

## 修改记录

### 2024-11 更新
1. **使用 `dataset_v1._recover_surface`**: 不再自己实现参数恢复逻辑
2. **使用 `utils/surface.py` 函数**: 复用 `build_plane_face` 和 `build_second_order_surface`
3. **修复 `utils/surface.py` bug**: 修正 `build_plane_face` 中 `attr_str` 未定义的问题

## 参数格式

输入参数格式（与 `dataset_v1.py` 一致）：

```
[P(3), D(3), X(3), UV(8), scalar(0-2)]
```

- `P`: 位置 (3D)
- `D`: 主方向向量 (3D, normalized)
- `X`: U方向向量 (3D, normalized)
- `UV`: UV参数编码 (8D, 不同曲面类型编码不同)
- `scalar`: 标量参数（如 radius，已经过 log 处理）

## 曲面类型索引

```python
SURFACE_TYPE_MAP = {
    'plane': 0,
    'cylinder': 1,
    'cone': 2,
    'sphere': 3,
    'torus': 4,
}
```

## 性能

- 单个曲面采样 (32×32): ~1-5ms
- 批量采样 500 个曲面: ~1-2秒

## 依赖项

- `numpy`, `torch`
- `pythonOCC` (`OCC.Core.BRep`, `OCC.Core.TopLoc`)
- `src.dataset.dataset_v1`
- `utils.surface`

