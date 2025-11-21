# test_sample_surfaces.py - 交互式可视化测试工具

## 功能概述

交互式测试工具，用于对比可视化：
1. **原始曲面（mesh）**: 从 JSON 生成的网格化曲面（灰色）
2. **采样点云**: 使用 `sample_simple_surface` 采样的点云（蓝色）

**核心特性**:
- ✅ **实时切换**: 使用 GUI slider 实时切换不同的 JSON 文件
- ✅ **动态调整**: 实时修改采样密度（num_u, num_v）
- ✅ **灵活显示**: 可选择显示 mesh、点云或两者
- ✅ **同窗对比**: 两种可视化在同一窗口中便于对比

## 基本用法

```bash
python src/tests/test_sample_surfaces.py --dataset_path path/to/json/directory
```

运行后，使用 GUI 中的 slider 切换不同的 JSON 文件。

## 命令行参数

### 必需参数

```bash
--dataset_path PATH    # JSON 数据集目录路径
```

### 可选参数

```bash
--num_u INT           # 初始 u 方向采样点数（默认：32）
--num_v INT           # 初始 v 方向采样点数（默认：32）
--start_index INT     # 初始显示的 JSON 索引（默认：0）
--mesh_only           # 只显示 mesh（不显示采样点云）
--samples_only        # 只显示采样点云（不显示 mesh）
```

## 使用示例

### 1. 基础用法（推荐）

```bash
python src/tests/test_sample_surfaces.py \
    --dataset_path F:/WORK/CAD/CLR-Wire/data/abc_parsed/0000
```

启动后使用 slider 在 GUI 中切换 JSON 文件。

### 2. 指定初始索引和采样密度

```bash
python src/tests/test_sample_surfaces.py \
    --dataset_path F:/WORK/CAD/CLR-Wire/data/abc_parsed/0000 \
    --start_index 10 \
    --num_u 64 \
    --num_v 64
```

### 3. 只显示原始 mesh

```bash
python src/tests/test_sample_surfaces.py \
    --dataset_path F:/WORK/CAD/CLR-Wire/data/abc_parsed/0000 \
    --mesh_only
```

### 4. 只显示采样点云

```bash
python src/tests/test_sample_surfaces.py \
    --dataset_path F:/WORK/CAD/CLR-Wire/data/abc_parsed/0000 \
    --samples_only
```

### 5. 低密度快速浏览

```bash
python src/tests/test_sample_surfaces.py \
    --dataset_path F:/WORK/CAD/CLR-Wire/data/abc_parsed/0000 \
    --num_u 16 \
    --num_v 16
```

## GUI 交互控制

运行程序后，左侧面板提供以下交互控件：

### 1. JSON Index Slider
- **功能**: 实时切换显示不同的 JSON 文件
- **操作**: 拖动滑块或点击输入数字
- **范围**: 0 到 (总文件数 - 1)

### 2. num_u / num_v 输入框
- **功能**: 调整采样密度
- **操作**: 输入新值并按 Enter
- **效果**: 自动刷新采样点云
- **范围**: > 0 的整数

### 3. Show Mesh 复选框
- **功能**: 切换是否显示原始 mesh
- **效果**: 勾选显示灰色网格，取消隐藏

### 4. Show Samples 复选框
- **功能**: 切换是否显示采样点云
- **效果**: 勾选显示蓝色点云，取消隐藏

### 5. Refresh 按钮
- **功能**: 手动刷新当前可视化
- **用途**: 当显示异常时可以点击刷新

### 6. 状态信息
- 显示当前文件名、曲面数量、采样数量等信息

## Polyscope 视图控制

### 鼠标操作
- **旋转视图**: 左键拖动
- **平移视图**: 右键拖动 或 Shift + 左键拖动
- **缩放视图**: 滚轮

### 键盘快捷键
- **q**: 退出程序
- **r**: 重置相机视角
- **Space**: 显示/隐藏 GUI 面板

## 可视化说明

### 颜色编码
- **灰色半透明 mesh**: 原始曲面（使用 `build_*_face` 函数生成）
- **蓝色点云**: 采样点云（使用 `sample_surface_uniform` 生成）

### 命名规则
- **Mesh**: `mesh_{surface_index:03d}_{surface_type}`
  - 例如: `mesh_001_cylinder`
- **Samples**: `samples_{surface_index:03d}_{surface_type}`
  - 例如: `samples_001_cylinder`

### 验证方法
1. **重合度检查**: 蓝色点云应该密集分布在灰色 mesh 表面上
2. **完整性检查**: 点云应覆盖整个 mesh 的 UV 范围
3. **密度检查**: 增加 num_u/num_v 应使点云更密集

## 典型工作流程

### 快速浏览模式
1. 启动程序（使用默认参数）
2. 使用 slider 快速浏览所有 JSON 文件
3. 找到感兴趣的文件后调整采样密度仔细查看

### 详细对比模式
1. 使用 `--start_index` 直接跳到目标文件
2. 设置较高的 num_u/num_v（如 64×64）
3. 同时显示 mesh 和 samples 进行详细对比
4. 必要时切换显示选项（只显示 mesh 或只显示 samples）

### 批量检查模式
1. 使用低采样密度（16×16）快速加载
2. 使用 slider 逐个检查文件
3. 记录异常情况的索引
4. 针对异常索引使用高密度重新检查

## 输出示例

```
============================================================
Test: sample_simple_surface.py (Interactive)
============================================================
Dataset path: F:\WORK\CAD\CLR-Wire\data\abc_parsed\0000
Found 1000 JSON files
Initial sampling: num_u=32, num_v=32
Show mesh: True
Show samples: True
Starting at index: 0

Loading dataset...
✓ Dataset loaded with 1000 files

Initializing visualization...

============================================================
Visualization ready!
============================================================

Interactive controls:
  - Use slider to switch between JSON files
  - Adjust num_u/num_v to change sampling density
  - Toggle checkboxes to show/hide mesh and samples
  - Press 'q' to quit

[Polyscope window opens]
```

## 技术细节

### 数据流

```
命令行启动 → 加载数据集 → 初始化 Polyscope
    ↓
[GUI Slider] → _refresh_visualization()
    ↓
清除所有显示 → 加载新 JSON
    ↓
    ├─ [Show Mesh] → build_*_face → register_surface_mesh
    └─ [Show Samples] → sample_surface_uniform → register_point_cloud
```

### 实时更新机制
- 使用 `ps.set_user_callback(_polyscope_callback)` 注册回调
- 每帧检查 GUI 控件状态变化
- 状态改变时调用 `_refresh_visualization()` 刷新显示

### 性能优化
- 按需加载：只加载当前显示的 JSON
- 智能刷新：仅在参数改变时重新采样
- 结构清除：切换文件前清除所有旧结构

## 支持的曲面类型

| 类型 | Mesh 支持 | Samples 支持 |
|------|-----------|--------------|
| plane | ✅ | ✅ |
| cylinder | ✅ | ✅ |
| cone | ✅ | ✅ |
| sphere | ✅ | ✅ |
| torus | ✅ | ✅ |
| bspline_surface | ✅ | ❌ (跳过) |

注: bspline_surface 不支持采样，需使用 `sample_bspline_surface.py`。

## 常见问题

### Q: 为什么 slider 切换很慢？
A: 可能原因：
- 采样密度过高（降低 num_u/num_v）
- JSON 文件包含大量曲面
- bspline 曲面较多（mesh 生成较慢）

解决方案：使用较低采样密度（如 16×16）进行浏览。

### Q: 如何快速定位到特定索引？
A: 两种方法：
1. 使用 `--start_index` 参数启动
2. 在运行时点击 slider 旁边的数字输入框直接输入

### Q: 某些曲面的点云和 mesh 不匹配？
A: 可能原因：
1. 参数恢复有问题（检查 dataset_v1）
2. 曲面构建有差异（检查 build_*_face 函数）
3. 采样密度不够（增加 num_u/num_v）

### Q: 如何保存当前视图？
A: Polyscope 支持截图：
- 在窗口中按 `s` 键保存当前视图
- 或在 GUI 面板中找到 "Screenshot" 选项

## 相关文件

- `src/tools/sample_simple_surface.py`: 采样函数实现
- `utils/surface.py`: 曲面构建函数
- `src/dataset/dataset_v1.py`: 数据集加载
- `src/tests/test_sample_surfaces_quick.py`: 快速测试（无需数据集）

## 依赖项

- `numpy`, `torch`
- `polyscope` (带 ImGui 支持)
- `pythonOCC`
- 项目内部模块

## 更新日志

### 2024-11 交互式版本
- ✨ 新增 GUI slider 实时切换 JSON 文件
- ✨ 新增实时调整采样密度
- ✨ 新增 Show Mesh/Samples 复选框
- ✨ 移除命令行 `--json_index` 参数
- ✨ 新增 `--start_index` 指定初始索引

