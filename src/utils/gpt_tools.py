import torch
import einops

def tokenize_bspline_poles(vae, dataset, tokens, bspline_poles, bspline_valid_mask):
    with torch.no_grad():
        patches = bspline_poles[..., :3]
        bs = patches.shape[0]
        patches = patches.reshape(-1, 4, 4, 3)
        patches_valid = patches[bspline_valid_mask.reshape(-1)]
        patches_valid = einops.rearrange(patches_valid, "b h w c -> b c h w")
        
        # 🔥 FIX: 确保所有rank都调用vae.encode()以避免FSDP死锁
        # 如果没有有效patches，传入一个dummy tensor
        if patches_valid.shape[0] != 0:
            z_quantized, indices = vae.encode(patches_valid)
            tokens[tokens==-2] = indices.reshape(-1).long()
        else:
            # 创建一个dummy tensor确保所有rank都调用encode
            # 使用正确的设备
            device = bspline_poles.device
            dummy_patch = torch.zeros((1, 3, 4, 4), device=device, dtype=patches.dtype)
            _ = vae.encode(dummy_patch)  # 调用但不使用结果

    return tokens