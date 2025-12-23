"""
DC-AE 独立模块测试
测试精简版 DC-AE 的功能
"""

import torch
import torch.nn as nn
from dc_ae import DCAE, DCAEConfig, EncoderConfig, DecoderConfig, create_minimal_dcae


def test_minimal_config():
    """测试最小化配置：4x4x3 -> 2x2x3"""
    print("=" * 70)
    print("测试 1: 最小化配置 (4x4x3 -> 2x2x3)")
    print("=" * 70)
    
    # 方法1: 手动配置
    config = DCAEConfig(
        in_channels=3,
        latent_channels=3,
        encoder=EncoderConfig(
            in_channels=3,
            latent_channels=3,
            width_list=(64, 128),
            depth_list=(1, 1),
            block_type="ResBlock",
            norm="bn2d",
            act="relu",
            downsample_block_type="ConvPixelUnshuffle",
            downsample_match_channel=True,
            downsample_shortcut="averaging",
            out_norm="bn2d",
            out_act="relu",
            out_shortcut=None,  # 移除shortcut避免通道数不匹配问题
        ),
        decoder=DecoderConfig(
            in_channels=3,
            latent_channels=3,
            in_shortcut=None,        # 移除in_shortcut避免通道数不匹配问题
            width_list=(64, 128),
            depth_list=(1, 1),
            block_type="ResBlock",
            norm="bn2d",
            act="relu",
            upsample_block_type="ConvPixelShuffle",
            upsample_match_channel=True,
            upsample_shortcut=None,  # 移除shortcut避免通道数不匹配问题
            out_norm="bn2d",
            out_act="relu",
        ),
    )
    
    model = DCAE(config)
    model.eval()
    
    # 测试数据
    batch_size = 4
    x = torch.randn(batch_size, 3, 4, 4)
    
    print(f"\n输入形状: {tuple(x.shape)}")
    
    # 编码
    with torch.no_grad():
        latent = model.encode(x)
    print(f"Latent形状: {tuple(latent.shape)}")
    assert latent.shape == (batch_size, 3, 2, 2), f"Latent shape错误: {latent.shape}"
    
    # 解码
    with torch.no_grad():
        recon = model.decode(latent)
    print(f"重建形状: {tuple(recon.shape)}")
    assert recon.shape == x.shape, f"重建shape错误: {recon.shape}"
    
    # 计算损失
    mse = nn.functional.mse_loss(recon, x)
    print(f"\nMSE Loss (未训练): {mse.item():.4f}")
    
    print(f"✓ 测试通过!")
    print(f"  压缩比: {model.spatial_compression_ratio}x")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def test_helper_function():
    """测试便捷创建函数"""
    print("\n" + "=" * 70)
    print("测试 2: 使用便捷函数 create_minimal_dcae()")
    print("=" * 70)
    
    # 使用便捷函数
    model = create_minimal_dcae(input_size=4, in_channels=3, latent_channels=3, width_base=64)
    model.eval()
    
    x = torch.randn(2, 3, 4, 4)
    print(f"\n输入形状: {tuple(x.shape)}")
    
    with torch.no_grad():
        latent = model.encode(x)
        recon = model.decode(latent)
    
    print(f"Latent形状: {tuple(latent.shape)}")
    print(f"重建形状: {tuple(recon.shape)}")
    
    print(f"✓ 测试通过!")
    
    return model


def test_different_sizes():
    """测试不同输入尺寸"""
    print("\n" + "=" * 70)
    print("测试 3: 不同输入尺寸")
    print("=" * 70)
    
    model = create_minimal_dcae(input_size=8, in_channels=3, latent_channels=16, width_base=32)
    model.eval()
    
    test_cases = [
        (1, 3, 4, 4),
        (2, 3, 8, 8),
        (1, 3, 16, 16),
        (4, 3, 32, 32),
    ]
    
    for shape in test_cases:
        x = torch.randn(*shape)
        with torch.no_grad():
            latent = model.encode(x)
            recon = model.decode(latent)
        
        expected_latent_shape = (shape[0], 16, shape[2] // 2, shape[3] // 2)
        assert latent.shape == expected_latent_shape, f"Latent shape不匹配: {latent.shape} vs {expected_latent_shape}"
        assert recon.shape == shape, f"重建shape不匹配: {recon.shape} vs {shape}"
        
        print(f"  {tuple(x.shape)} -> {tuple(latent.shape)} -> {tuple(recon.shape)} ✓")
    
    print("✓ 所有尺寸测试通过!")
    
    return model


def test_latent_manipulation():
    """测试 latent 的不同展平方式"""
    print("\n" + "=" * 70)
    print("测试 4: Latent 展平与重塑")
    print("=" * 70)
    
    model = create_minimal_dcae(input_size=4, in_channels=3, latent_channels=3)
    model.eval()
    
    x = torch.randn(2, 3, 4, 4)
    
    with torch.no_grad():
        latent = model.encode(x)  # (2, 3, 2, 2)
    
    print(f"\n原始 Latent 形状: {tuple(latent.shape)}")
    
    # 不同的展平方式
    print("\n展平方式:")
    
    # 1. 完全展平
    latent_flat = latent.view(2, -1)
    print(f"  1. 完全展平:       {tuple(latent_flat.shape)} -> {latent_flat.shape[1]} 个值")
    
    # 2. 保留通道维度
    latent_spatial = latent.flatten(2)
    print(f"  2. 展平空间维度:   {tuple(latent_spatial.shape)}")
    
    # 3. 自定义形状 2x6
    latent_2x6 = latent.view(2, 2, 6)
    print(f"  3. 重塑为 2x6:     {tuple(latent_2x6.shape)}")
    
    # 4. 自定义形状 3x4
    latent_3x4 = latent.view(2, 3, 4)
    print(f"  4. 重塑为 3x4:     {tuple(latent_3x4.shape)}")
    
    # 重塑回原始形状并解码
    latent_reshaped = latent_flat.view(2, 3, 2, 2)
    with torch.no_grad():
        recon = model.decode(latent_reshaped)
    
    print(f"\n重塑后解码: {tuple(recon.shape)}")
    print("✓ Latent 操作测试通过!")
    
    return model


def test_forward_pass():
    """测试完整的前向传播"""
    print("\n" + "=" * 70)
    print("测试 5: 完整前向传播")
    print("=" * 70)
    
    model = create_minimal_dcae(input_size=4, in_channels=3, latent_channels=3)
    model.eval()
    
    x = torch.randn(2, 3, 4, 4)
    
    with torch.no_grad():
        recon, kl_loss, metrics = model(x, global_step=0)
    
    print(f"\n输入形状: {tuple(x.shape)}")
    print(f"重建形状: {tuple(recon.shape)}")
    print(f"KL Loss: {kl_loss.item()}")
    print(f"Metrics: {metrics}")
    
    # 计算重建误差
    mse = nn.functional.mse_loss(recon, x)
    mae = nn.functional.l1_loss(recon, x)
    
    print(f"\n重建误差 (未训练):")
    print(f"  MSE: {mse.item():.6f}")
    print(f"  MAE: {mae.item():.6f}")
    
    print("✓ 前向传播测试通过!")
    
    return model


def main():
    """运行所有测试"""
    print("\n" + "🚀" * 35)
    print(" " * 20 + "DC-AE 独立模块测试套件")
    print("🚀" * 35 + "\n")
    
    # 运行所有测试
    model1 = test_minimal_config()
    model2 = test_helper_function()
    model3 = test_different_sizes()
    model4 = test_latent_manipulation()
    model5 = test_forward_pass()
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)
    
    print("\n📋 使用示例:")
    print("""
from src.vae.dc_ae import create_minimal_dcae

# 创建模型
model = create_minimal_dcae(
    input_size=4,
    in_channels=3,
    latent_channels=3,
    width_base=64
)

# 使用模型
x = torch.randn(batch_size, 3, 4, 4)
latent = model.encode(x)          # (B, 3, 2, 2)
recon = model.decode(latent)      # (B, 3, 4, 4)

# 或者直接前向传播
recon, kl_loss, metrics = model(x, global_step=0)
""")


if __name__ == "__main__":
    main()

