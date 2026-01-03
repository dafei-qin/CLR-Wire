# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import math
from typing import Any, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from lit_gpt.config import Config
from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func
from torch import Tensor
from .mamba_simple import Mamba
from functools import partial
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from .gla import GatedLinearAttention
from .multiscale_retention import MultiScaleRetention
from einops import rearrange
import torch.nn.functional as F

from causal_conv1d import causal_conv1d_fn
from .miche_conditioner import PointConditioner

from hy3dshape.models.autoencoders import ShapeVAE
from .mega import S6GatedAttention

import copy
from collections import namedtuple
CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = MBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

class GPT(nn.Module):
    def __init__(self, config: Config, build_conditioner: bool = True) -> None:
        super().__init__()
        factory_kwargs = { "dtype": torch.float32}
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        if config.mamba:
            if self.config.fused_add_norm:
                if layer_norm_fn is None or rms_norm_fn is None:
                    raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                    h=nn.ModuleList(
                        create_block(
                            config.n_embd,
                            ssm_cfg=None,
                            norm_epsilon=config.norm_eps,
                            rms_norm=config.rms_norm,
                            residual_in_fp32=config.residual_in_fp32,
                            fused_add_norm=config.fused_add_norm,
                            layer_idx=i,
                            **factory_kwargs,
                        )
                        for i in range(config.n_layer)),
                    ln_f= (nn.LayerNorm if not config.rms_norm else RMSNorm)(
                        config.n_embd, eps=config.norm_eps,
                        **factory_kwargs,
                    )
                )
            )            
            
        else:
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                    h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                    ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
                )
            )
        
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.max_len = self.config.block_size
        self.mamba_init = config.mamba or config.mamba_init
        if self.mamba_init:
            self.tie_weights()
        # self.conditioner = PointConditioner(model_name='miche-256-feature', freeze=True)
        # self.conditioner.eval()
        self.conditioner = None
        if build_conditioner:
            self.conditioner = ShapeVAE.from_pretrained(
                'tencent/Hunyuan3D-2.1',
                use_safetensors=False,
                variant='fp16',
            )
            for p in self.conditioner.parameters():
                p.requires_grad = False
            self.conditioner.eval()

        self.norm = nn.LayerNorm(config.n_embd)
        self.linear = nn.Linear(1024, config.n_embd)

        
    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            if self.mamba_init:
                torch.nn.init.normal_(module.weight, std=0.02)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, nn.Linear):
            if self.mamba_init:
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        nn.init.zeros_(module.bias)    
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        # GPT-NeoX       
        for name, p in module.named_parameters():
            if (name == "out_proj.weight" and isinstance(module, Mamba)) \
                or (name == "o_proj.weight") \
                    or (name == "proj.weight" and isinstance(module, LLaMAMLP)) \
                    or (name == "w3.weight" and isinstance(module, SwiGLU)) \
                    or (name=="proj.weight" and isinstance(module, CausalSelfAttention)):       
                    #if use xformer swiglu, fc2 layer will be renamed to w3       
                    # sclae fc2 is better than fc3 for moe, it is wierd
                if self.mamba_init:
                    if self.config.mamba or not self.config.mlp:
                        n_residuals_per_layer = 1  
                    elif self.config.mamba_swa_mlp:
                        n_residuals_per_layer = 3
                    else:
                        n_residuals_per_layer = 2
                    #nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)
                else:
                    nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)

    def tie_weights(self):
        self.lm_head.weight = self.transformer.wte.weight
        
    
    def reset_cache(self) -> None:
        self.max_len = self.config.block_size
        self.kv_caches.clear()
        # 重置inference_params状态
        if hasattr(self, 'inference_params') and self.inference_params is not None:
            self.inference_params.seqlen_offset = 0
            # 清空状态缓存
            if hasattr(self.inference_params, 'key_value_memory_dict'):
                self.inference_params.key_value_memory_dict.clear()
        if hasattr(self, '_last_seqlen_offset'):
            self._last_seqlen_offset = 0
        # 清空 mask_cache 和 rope_cache，让它们在需要时用正确的 max_seq_length 重建
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None
        else:
            # 对于非 XLA 设备，也清空 mask_cache，让它在需要时用正确的 max_seq_length 重建
            self.mask_cache = None

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None, pc=None
    ) -> torch.Tensor:
        if self.config.mamba:
            hidden_states = self.transformer.wte(idx)
            residual = None
            for block in self.transformer.h:
                hidden_states, residual = block(
                    hidden_states, residual, inference_params=None
                )
            norm_f = self.transformer.ln_f
            if not self.config.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = norm_f(residual.to(dtype= norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(norm_f, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    norm_f.weight,
                    norm_f.bias,
                    eps=norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.config.residual_in_fp32,
                )
            lm_logits = self.lm_head(hidden_states)
            return CausalLMOutput(logits=lm_logits)

        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        #assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        #assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"
        if not self.config.nope:
            if self.rope_cache is None:
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            elif T> self.max_len:
                self.max_len = T
                print(f"Warning: RoPE reset max_len to {self.max_len}")
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            cos, sin = self.rope_cache   
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache:
            # 检查 mask_cache 是否存在且大小是否足够
            if self.mask_cache is None:
                self.mask_cache = self.build_mask_cache(idx, max_seq_length)
            elif self.mask_cache.size(-1) < max_seq_length:
                # 如果 max_seq_length 大于当前 mask_cache 的大小，需要重建
                self.mask_cache = self.build_mask_cache(idx, max_seq_length)

        
        if use_kv_cache:
            if not self.config.nope:
                cos = cos.index_select(0, input_pos)
                sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            if not self.config.nope:
                cos = cos[:T]
                sin = sin[:T]
            mask = None
        if self.config.nope:
            rope = None
        else:
            rope = (cos, sin)
        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        
        # 处理点云条件输入
        if pc is not None and self.conditioner is not None:
            # cond_param = next(model.conditioner.parameters(), None) if hasattr(model, "conditioner") else None
            # target_dtype = cond_param.dtype if cond_param is not None else next(model.parameters()).dtype
            # pc = pc.to(device=device, dtype=target_dtype)

            cond_embeds = self.conditioner.encode(pc) # (bs,4096,64)
            # print(f"cond_embeds {cond_embeds.shape}")
            b, n, d = cond_embeds.shape  # b,4096,64
            # 先reshape为(b, 256, 1024)，保持排列信息不变
            cond_embeds = cond_embeds.reshape(b, -1, 1024)  # (b, 256, 1024)

            # 在进入主干 linear 前，显式对齐 dtype，避免 mat1/2 类型不一致
            cond_embeds = cond_embeds.to(self.linear.weight.dtype)
            # cond_embeds = self.conditioner(pc=pc)  # (bs, 257, 1024)
            cond_embeds = self.linear(cond_embeds) 
            cond_embeds = self.norm(cond_embeds)
            self.cond_embeds = cond_embeds
        else:
            cond_embeds = getattr(self, 'cond_embeds', None)
            
        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, rope, max_seq_length, pc=cond_embeds)
        else:
            if self.config.nope:
                self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, None )
            else:
                self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, cos.size(-1) * 2)
            
            # Create inference_params for mixed architecture layers (Mamba, RetNet, GLA)
            if not hasattr(self, 'inference_params') or self.inference_params is None:
                from flash_attn.utils.generation import InferenceParams
                self.inference_params = InferenceParams(max_seqlen=max_seq_length, max_batch_size=x.size(0))
            
            # 正确累积seqlen_offset
            if hasattr(self, '_last_seqlen_offset'):
                self.inference_params.seqlen_offset = self._last_seqlen_offset + 1
            else:
                self.inference_params.seqlen_offset = 0
            self._last_seqlen_offset = self.inference_params.seqlen_offset
            
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, rope, max_seq_length, mask, input_pos, self.kv_caches[i], self.inference_params, pc=cond_embeds)
        x = self.transformer.ln_f(x.to(dtype=self.transformer.ln_f.weight.dtype))
        lm_logits = self.lm_head(x)
        return CausalLMOutput(logits=lm_logits)
        # return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        n_elem = int(self.config.rotary_percentage * self.config.head_size)

        base = getattr(self.config, "rope_theta_base", 20000.0)
        condense_ratio = 1.0  # NTK scaling 不再用 condense_ratio 压位置

        # === NTK-aware scaling 部分 ===
        scaling_type = getattr(self.config, "rope_scaling_type", None)
        if scaling_type == "ntk":
            train_ctx = getattr(self.config, "rope_ntk_base_ctx", self.config.block_size)
            factor = getattr(self.config, "rope_ntk_factor", 1.0)  # 例如 2.0/4.0

            if factor != 1.0:
                # 目标上下文长度（比如 40k / 80k）
                max_position_embeddings = int(train_ctx * factor)

                # 确保我们不会去算超过目标的 cos/sin
                seq_len = min(seq_len, max_position_embeddings)

                dim = n_elem  # 相当于 RoPE 使用的总维度 d
                # bloc97 提出的 NTK-aware: 相当于把 base 换成一个放大版
                # base' = base * factor^(d / (d - 2))
                base = base * (factor ** (dim / (dim - 2)))
        # === 结束 ===

        return build_rope_cache(
            seq_len=seq_len,
            n_elem=n_elem,
            dtype=torch.bfloat16,
            device=idx.device,
            base=base,
            condense_ratio=condense_ratio,
        )


    def build_mask_cache(self, idx: torch.Tensor, max_seq_length: Optional[int] = None) -> torch.Tensor:
        if max_seq_length is None:
            max_seq_length = self.config.block_size
        ones = torch.ones((max_seq_length, max_seq_length), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups
        if rope_cache_length is not None:
            k_cache_shape = (
                B,
                max_seq_length,
                heads,
                rope_cache_length + self.config.head_size - int(self.config.rotary_percentage * self.config.head_size),
            )
        else:
            k_cache_shape = (
                B,
                max_seq_length,
                heads,
                self.config.head_size,
            )
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.abl_mamba = config.abl_mamba
        self.mamba_swa_mlp = config.mamba_swa_mlp
        factory_kwargs = { "dtype": torch.float32}
        if config.mamba_swa_mlp:
            self.norm_m = config.norm_class(config.n_embd, eps=config.norm_eps)
            self.norm_attn = config.norm_class(config.n_embd, eps=config.norm_eps)            
            self.mb = Mamba(config.n_embd, layer_idx=layer_idx, **factory_kwargs)
            self.attn = CausalSelfAttention(config, n_embd= config.n_embd, layer_idx= layer_idx, )
        elif config.use_mega:
            self.attn = S6GatedAttention(config.n_embd)
        else:
            self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
            self.use_retnet, self.use_gla = False, False
            if config.attn_layer_pos is not None:
                self.use_mamba = layer_idx not in eval(config.attn_layer_pos) 
            else:
                self.use_mamba = layer_idx % config.mb_per_layer == 0 if config.mb_per_layer >0 else False
                self.use_retnet = layer_idx % config.ret_per_layer == 0 if config.ret_per_layer >0 else False   
                self.use_gla = layer_idx % config.gla_per_layer == 0 if config.gla_per_layer >0 else False       
            if self.use_mamba:
                if self.abl_mamba:
                    config_temp = copy.deepcopy(config)
                    config_temp._mlp_class = "LLaMAMLP"
                    config_temp.intermediate_size = config_temp.n_embd * 2
                    self.attn = config.mlp_class(config_temp,)
                else:
                    self.attn = Mamba(config.n_embd, layer_idx=layer_idx, **factory_kwargs)
            elif self.use_retnet:
                self.attn = MultiScaleRetention(hidden_size=config.n_embd, num_heads=config.n_head // 2, expand_k=1, expand_v = 2, mode = 'fused_chunk', use_short_conv = False)
            elif self.use_gla:
                self.attn = GatedLinearAttention(hidden_size=config.n_embd, num_heads=config.n_embd // 384, expand_k = 0.5, expand_v = 1, mode = 'fused_chunk', use_short_conv = False)        
            else:
                self.attn = CausalSelfAttention(config, n_embd= config.n_embd, layer_idx= layer_idx, )
            
        # mlp
        if not config.shared_attention_norm and config.mlp and not config.parallel_residual:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if config.mlp:
            self.mlp = config.mlp_class(config,)
        self.config = config
        # 添加CrossAttention层用于点云条件
        self.cross_attn = CrossAttention(config.n_embd, config.n_embd, config.n_head)
        self.norm_cross = config.norm_class(config.n_embd, eps=config.norm_eps)
        
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        inference_params = None,
        pc = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        if self.mamba_swa_mlp:
            x = self.mb(self.norm_m(x.to(dtype=self.norm_m.weight.dtype)), inference_params=inference_params) + x.to(torch.float32)
            new_kv_cache = kv_cache # TODO 
            h, new_kv_cache = self.attn(self.norm_attn(x.to(dtype=self.norm_attn.weight.dtype)),  rope, max_seq_length, mask, input_pos, kv_cache) 
            x = h+x
        else:
            ox = x
            if self.config.use_mega:
                x = self.attn(x)
                new_kv_cache = None
            else:  
                n_1 = self.norm_1(x.to(dtype=self.norm_1.weight.dtype))
                if self.use_mamba:
                    if self.abl_mamba:
                        h  = self.attn(n_1)
                    else:
                        h  = self.attn(n_1, inference_params=inference_params)
                    new_kv_cache = kv_cache # TODO 
                    ox = ox.to(torch.float32)
                elif self.use_retnet or self.use_gla:
                    h, _ , new_kv_cache = self.attn(n_1)
                else:
                    h, new_kv_cache = self.attn(n_1, rope, max_seq_length, mask, input_pos, kv_cache)   
                x = ox + h
            
        # 添加点云cross-attention
        if pc is not None:
            x_skip = x
            x = self.norm_cross(x.to(dtype=self.norm_cross.weight.dtype))
            x = self.cross_attn(x, pc) + x_skip
            
        if self.config.mlp:
            ox = x
            n_2 = self.norm_2(x.to(dtype=self.norm_2.weight.dtype))
            h = self.mlp(n_2)
            x = ox + h
        return x, new_kv_cache


class MBlock(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int , n_embd: int, head_size = None) -> None:
        super().__init__()
        self.local = layer_idx % config.full_per_layer < config.full_per_layer-1
        if head_size is not None:
            self.head_size = head_size
            self.n_head =  n_embd // head_size
            self.n_query_groups = self.n_head 
        else:
            self.head_size = config.head_size
            self.n_head = config.n_head
            self.n_query_groups = config.n_query_groups
        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.config = config
        self.sc = config.sc_attn
        # 添加动态窗口控制
        self.wzy = 0
        self.use_dynamic_window = getattr(config, 'use_dynamic_window', True)
        self.window_thresholds = getattr(config, 'window_thresholds', [4501, 1501, 501])  # 可配置的阈值
        if self.sc:
            self.q_dim = self.n_head * self.head_size
            self.kv_dim = self.n_query_groups * self.head_size
            d_conv = 4
            self.q_conv1d = nn.Conv1d(
                in_channels=self.q_dim,
                out_channels=self.q_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.q_dim,
                padding=d_conv - 1,
            )
            self.k_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )
            self.v_conv1d = nn.Conv1d(
                in_channels= self.kv_dim,
                out_channels= self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups= self.kv_dim,
                padding=d_conv - 1,
            ) 

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)
        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B,  T, -1 )  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1 )  
        v = v.reshape(B,  T, -1 )  
        if self.sc:
            q = causal_conv1d_fn(
                        x = q.transpose(-1,-2),
                        weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                        bias=self.q_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            k = causal_conv1d_fn(
                        x = k.transpose(-1,-2),
                        weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                        bias=self.k_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            v = causal_conv1d_fn(
                        x = v.transpose(-1,-2),
                        weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                        bias=self.v_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2) 

        q = q.reshape(B,  T, -1, self.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.head_size)  
        v = v.reshape(B,  T, -1, self.head_size)

        if not self.config.nope:         
            cos, sin = rope
            # 对齐 dtype，避免 q/k 与 cos/sin 不一致导致的 RuntimeError
            if cos.dtype != q.dtype:
                cos = cos.to(q.dtype)
                sin = sin.to(q.dtype)
            # fused rope 期望 (B, T, H, D)
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            
            # 动态窗口控制逻辑
            if self.use_dynamic_window:
                # 检查是否达到窗口阈值
                if T in self.window_thresholds:
                    self.wzy = T - 1
                if self.wzy > 0:
                    self.wzy += 1
                    # 只使用最后wzy个位置的缓存
                    print(f"use dynamic window, wzy: {self.wzy}")
                    input_pos_index = torch.arange(max_seq_length, dtype=torch.long, device=cache_k.device)[-self.wzy:]
                else:
                    input_pos_index = input_pos
            else:
                input_pos_index = input_pos
            
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                # 保持input_pos为1D张量
                clamped_pos = torch.tensor([max_seq_length - 1], device=input_pos.device, dtype=input_pos.dtype)
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)
                input_pos = clamped_pos
            else:
                clamped_pos = input_pos

            k = cache_k.index_copy_(1, clamped_pos, k)
            v = cache_v.index_copy_(1, clamped_pos, v)
            kv_cache = k, v
            
            # 使用动态窗口选择实际的k, v
            if self.use_dynamic_window and hasattr(self, 'wzy') and self.wzy > 0:
                k = cache_k.index_select(1, input_pos_index)
                v = cache_v.index_select(1, input_pos_index)
                # 同步裁剪mask或置为None让attention使用is_causal
                if mask is not None:
                    mask = mask[..., :, input_pos_index]  # 裁剪mask到匹配的维度

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, -1)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y, kv_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.head_size)
        
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func
            if self.local and self.config.local_window > -1:
                win_tuple = (self.config.local_window-1, 0)
            else:
                win_tuple = (-1,-1)
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True, window_size=win_tuple)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config,) -> None:
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=config.bias, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.swiglu(x)
        return x
        
def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(context_dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # x: [batch, seq_len, dim], context: [batch, context_len, context_dim]
        B, N, C = x.shape
        _, M, _ = context.shape
        H = self.n_heads

        # Linear projections
        q = self.q_proj(x).view(B, N, H, C // H).transpose(1, 2)  # [B, H, seq_len, dim//H]
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = k.view(B, M, H, C // H).transpose(1, 2)  # [B, H, context_len, dim//H]
        v = v.view(B, M, H, C // H).transpose(1, 2)  # [B, H, context_len, dim//H]

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, seq_len, context_len]
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, C)  # [B, seq_len, dim]
        out = self.out_proj(attn_output)
        out = self.dropout(out)

        return out


    