import torch
import einops

def tokenize_bspline_poles(vae, dataset, tokens, bspline_poles, bspline_valid_mask):
    with torch.no_grad():
        patches = bspline_poles[..., :3]
        bs = patches.shape[0]
        patches = patches.reshape(-1, 4, 4, 3)
        patches_valid = patches[bspline_valid_mask.reshape(-1)]
        patches_valid = einops.rearrange(patches_valid, "b h w c -> b c h w")
        if patches_valid.shape[0] != 0:
            z_quantized, indices = vae.encode(patches_valid)
            tokens[tokens==-2] = indices.reshape(-1).long()

    return tokens