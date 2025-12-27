# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import math
from typing import Any, List, Optional, Tuple
import copy
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self

from lit_gpt.config import Config
from xformers.ops import SwiGLU
from .fused_rotary_embedding import apply_rotary_emb_func
from torch import Tensor
from .mamba_simple import Mamba  # base Mamba implementation
from functools import partial

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .gla import GatedLinearAttention
from .multiscale_retention import MultiScaleRetention
from einops import rearrange

from causal_conv1d import causal_conv1d_fn
from .miche_conditioner import PointConditioner
from hy3dshape.models.autoencoders import ShapeVAE
from .mega import S6GatedAttention

CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


# ====================================================================
#                        LoRA 基础组件
# ====================================================================

class LinearLoRA(nn.Linear):
    """
    LoRA 改进版：直接继承 nn.Linear。
    结构更扁平，对 FSDP（特别是 NO_SHARD 模式）更友好，避免梯度 Write Back 错位。
    """
    def __init__(
        self,
        linear_layer: nn.Linear,  # 传入旧的 Linear 对象
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        # 1. 初始化父类 nn.Linear
        # 注意：这里我们先初始化一个新的，稍后会把旧的权重指过来
        super().__init__(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None,
            device=linear_layer.weight.device,
            dtype=linear_layer.weight.dtype
        )

        # 2. 复制权重数据到新创建的Parameter（FSDP兼容性关键）
        # 注意：不能直接赋值 self.weight = linear_layer.weight，这会破坏FSDP的参数视图
        # 必须复制数据到super().__init__()创建的Parameter中，保持Parameter对象不变
        with torch.no_grad():
            self.weight.data.copy_(linear_layer.weight.data)
            if linear_layer.bias is not None:
                if self.bias is not None:
                    self.bias.data.copy_(linear_layer.bias.data)
                else:
                    # 如果原层有bias但super().__init__创建时没有，需要添加
                    self.bias = nn.Parameter(linear_layer.bias.data.clone())

        # 冻结原始权重
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # 3. LoRA 参数初始化
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 确保 LoRA 参数和主权重 dtype/device 一致
        base_dtype = linear_layer.weight.dtype
        base_device = linear_layer.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, dtype=base_dtype, device=base_device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=base_dtype, device=base_device))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """初始化 LoRA 参数"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FSDP 兼容性修改：
        # 不要使用 self.linear(x)，而是直接调用父类 F.linear
        # 这样 FSDP 能够清晰地看到是对 self.weight 的操作
        # 1. 基础 Linear 计算
        result = F.linear(x, self.weight, self.bias)

        # 2. LoRA 分支计算
        # 显式转换 dtype，防止 AMP 混合精度训练时类型不匹配
        x_dtype = x.dtype
        lora_out = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

        return result + lora_out.to(x_dtype)

    def merge_weights(self):
        """把 LoRA 权重合并回 self.weight"""
        if self.r > 0:
            with torch.no_grad():
                delta_w = (self.lora_B @ self.lora_A) * self.scaling
                self.weight.data.add_(delta_w)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, r={self.r}, alpha={self.alpha}"



def _wrap_linear_with_lora(
    module: nn.Module,
    attr_name: str,
    r: int,
    alpha: float,
    dropout: float,
):
    """
    修改版 Wrap 函数：处理 module 替换逻辑
    """
    if not hasattr(module, attr_name):
        return
    old_layer = getattr(module, attr_name)

    # 防止重复 Wrap
    if isinstance(old_layer, LinearLoRA):
        return

    # 只 Wrap nn.Linear
    if isinstance(old_layer, nn.Linear):
        # 创建新 layer（继承式）
        new_layer = LinearLoRA(old_layer, r=r, alpha=alpha, dropout=dropout)
        # 替换掉 module 中的属性
        setattr(module, attr_name, new_layer)
        # 这一点很重要：
        # 如果旧 layer 在 GPU 上，新 layer 的 param 已经在 __init__ 里转过去了
        # 但我们最好显式把旧 layer 删掉引用（虽然 Python 会 GC）
        del old_layer


# ====================================================================
#                        原始模型结构 (未改动逻辑)
# ====================================================================

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
        factory_kwargs = {"dtype": torch.float32}
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
                        for i in range(config.n_layer)
                    ),
                    ln_f=(nn.LayerNorm if not config.rms_norm else RMSNorm)(
                        config.n_embd, eps=config.norm_eps, **factory_kwargs
                    ),
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

        self.conditioner = None
        if build_conditioner:
            self.conditioner = ShapeVAE.from_pretrained(
                "tencent/Hunyuan3D-2.1",
                use_safetensors=False,
                variant="fp16",
            )
            for p in self.conditioner.parameters():
                p.requires_grad = False
            self.conditioner.eval()

        self.norm = nn.LayerNorm(config.n_embd)
        self.linear = nn.Linear(1024, config.n_embd)

    # ---------------- weight init ----------------

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        if isinstance(module, nn.Embedding):
            if self.mamba_init:
                torch.nn.init.normal_(module.weight, std=0.02)
            else:
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=math.sqrt(2.0 / 5 / self.config.n_embd),
                )
        elif isinstance(module, nn.Linear):
            if self.mamba_init:
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        nn.init.zeros_(module.bias)
            else:
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=math.sqrt(2.0 / 5 / self.config.n_embd),
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        for name, p in module.named_parameters():
            if (
                (name == "out_proj.weight" and isinstance(module, Mamba))
                or (name == "o_proj.weight")
                or (name == "proj.weight" and isinstance(module, LLaMAMLP))
                or (name == "w3.weight" and isinstance(module, SwiGLU))
                or (name == "proj.weight" and isinstance(module, CausalSelfAttention))
            ):
                if self.mamba_init:
                    if self.config.mamba or not self.config.mlp:
                        n_residuals_per_layer = 1
                    elif self.config.mamba_swa_mlp:
                        n_residuals_per_layer = 3
                    else:
                        n_residuals_per_layer = 2
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)
                else:
                    nn.init.normal_(
                        p,
                        mean=0.0,
                        std=1 / math.sqrt(self.config.n_embd) / n_layer,
                    )

    def tie_weights(self):
        self.lm_head.weight = self.transformer.wte.weight

    # ---------------- cache management ----------------

    def reset_cache(self) -> None:
        self.max_len = self.config.block_size
        self.kv_caches.clear()
        if hasattr(self, "inference_params") and self.inference_params is not None:
            self.inference_params.seqlen_offset = 0
            if hasattr(self.inference_params, "key_value_memory_dict"):
                self.inference_params.key_value_memory_dict.clear()
        if hasattr(self, "_last_seqlen_offset"):
            self._last_seqlen_offset = 0
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            self.rope_cache = None
            self.mask_cache = None
        else:
            self.mask_cache = None

    # ---------------- forward ----------------

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
        pc=None,
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
                residual = (
                    hidden_states + residual if residual is not None else hidden_states
                )
                hidden_states = norm_f(residual.to(dtype=norm_f.weight.dtype))
            else:
                fused_add_norm_fn = (
                    rms_norm_fn if isinstance(norm_f, RMSNorm) else layer_norm_fn
                )
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
        if use_kv_cache:
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"

        if not self.config.nope:
            if self.rope_cache is None:
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            elif T > self.max_len:
                self.max_len = T
                self.rope_cache = self.build_rope_cache(idx, self.max_len)
            cos, sin = self.rope_cache

        if use_kv_cache:
            if self.mask_cache is None:
                self.mask_cache = self.build_mask_cache(idx, max_seq_length)
            elif self.mask_cache.size(-1) < max_seq_length:
                self.mask_cache = self.build_mask_cache(idx, max_seq_length)

        if use_kv_cache:
            if not self.config.nope:
                cos = cos.index_select(0, input_pos)
                sin = sin.index_select(0, input_pos)
            mask = None
        else:
            if not self.config.nope:
                cos = cos[:T]
                sin = sin[:T]
            mask = None

        if self.config.nope:
            rope = None
        else:
            rope = (cos, sin)

        x = self.transformer.wte(idx)

        if pc is not None and self.conditioner is not None:
            cond_embeds = self.conditioner.encode(pc)
            b, n, d = cond_embeds.shape
            cond_embeds = cond_embeds.reshape(b, -1, 1024)
            cond_embeds = cond_embeds.to(self.linear.weight.dtype)
            cond_embeds = self.linear(cond_embeds)
            cond_embeds = self.norm(cond_embeds)
            self.cond_embeds = cond_embeds
        else:
            cond_embeds = getattr(self, "cond_embeds", None)

        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, rope, max_seq_length, pc=cond_embeds)
        else:
            kv_max_len = (
                min(max_seq_length, self.config.local_window)
                if self.config.local_window > 0
                else max_seq_length
            )
            if self.config.nope:
                self.kv_caches = self.kv_caches or self.build_kv_caches(
                    x, kv_max_len, None
                )
            else:
                self.kv_caches = self.kv_caches or self.build_kv_caches(
                    x, kv_max_len, cos.size(-1) * 2
                )

            if not hasattr(self, "inference_params") or self.inference_params is None:
                from flash_attn.utils.generation import InferenceParams

                self.inference_params = InferenceParams(
                    max_seqlen=kv_max_len, max_batch_size=x.size(0)
                )

            if hasattr(self, "_last_seqlen_offset"):
                self.inference_params.seqlen_offset = self._last_seqlen_offset + 1
            else:
                self.inference_params.seqlen_offset = 0
            self._last_seqlen_offset = self.inference_params.seqlen_offset

            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x,
                    rope,
                    kv_max_len,
                    mask,
                    input_pos,
                    self.kv_caches[i],
                    self.inference_params,
                    pc=cond_embeds,
                )

        x = self.transformer.ln_f(x.to(dtype=self.transformer.ln_f.weight.dtype))
        lm_logits = self.lm_head(x)
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    # ---------------- RoPE & cache builders ----------------

    def build_rope_cache(self, idx: torch.Tensor, seq_len: int) -> RoPECache:
        n_elem = int(self.config.rotary_percentage * self.config.head_size)

        base = getattr(self.config, "rope_theta_base", 10000.0)
        condense_ratio = 1.0

        scaling_type = getattr(self.config, "rope_scaling_type", None)
        if scaling_type == "ntk":
            train_ctx = getattr(
                self.config, "rope_ntk_base_ctx", self.config.block_size
            )
            factor = getattr(self.config, "rope_ntk_factor", 1.0)

            if factor != 1.0:
                max_position_embeddings = int(train_ctx * factor)
                seq_len = min(seq_len, max_position_embeddings)
                dim = n_elem
                base = base * (factor ** (dim / (dim - 2)))

        return build_rope_cache(
            seq_len=seq_len,
            n_elem=n_elem,
            dtype=torch.bfloat16,
            device=idx.device,
            base=base,
            condense_ratio=condense_ratio,
        )

    def build_mask_cache(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None
    ) -> torch.Tensor:
        if max_seq_length is None:
            max_seq_length = self.config.block_size
        ones = torch.ones(
            (max_seq_length, max_seq_length), device=idx.device, dtype=torch.bool
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(
        self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int
    ) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups
        if rope_cache_length is not None:
            k_cache_shape = (
                B,
                max_seq_length,
                heads,
                rope_cache_length
                + self.config.head_size
                - int(self.config.rotary_percentage * self.config.head_size),
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
            (
                torch.zeros(k_cache_shape, device=device),
                torch.zeros(v_cache_shape, device=device),
            )
            for _ in range(self.config.n_layer)
        ]

    # ====================================================================
    #                              LoRA 接口
    # ====================================================================

    def enable_lora(
        self,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        enable_lm_head: bool = False,
    ):
        """
        在现有模型上挂 LoRA（不会破坏已加载的 ckpt）。

        使用方式示例：
            model = GPT(config)
            model.load_state_dict(base_ckpt)
            model.enable_lora(r=8, alpha=16.0, dropout=0.0)
        """
        if r <= 0:
            return

        # 1) Mamba 内部的 Linear
        for m in self.modules():
            if isinstance(m, Mamba):
                for name in ["in_proj", "out_proj", "x_proj"]:
                    _wrap_linear_with_lora(m, name, r, alpha, dropout)

        # 2) 自注意力投影
        for m in self.modules():
            if isinstance(m, CausalSelfAttention):
                _wrap_linear_with_lora(m, "attn", r, alpha, dropout)
                _wrap_linear_with_lora(m, "proj", r, alpha, dropout)

        # 3) CrossAttention
        for m in self.modules():
            if isinstance(m, CrossAttention):
                _wrap_linear_with_lora(m, "q_proj", r, alpha, dropout)
                _wrap_linear_with_lora(m, "kv_proj", r, alpha, dropout)
                _wrap_linear_with_lora(m, "out_proj", r, alpha, dropout)

        # 4) MLP 中的 SwiGLU 权重
        for m in self.modules():
            if isinstance(m, LLaMAMLP):
                sw = m.swiglu
                for name in ["w1", "w2", "w3"]:
                    _wrap_linear_with_lora(sw, name, r, alpha, dropout)

        # 5) 可选：输出 lm_head
        if enable_lm_head:
            _wrap_linear_with_lora(self, "lm_head", r, alpha, dropout)


class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.abl_mamba = config.abl_mamba
        self.mamba_swa_mlp = config.mamba_swa_mlp
        factory_kwargs = {"dtype": torch.float32}

        if config.mamba_swa_mlp:
            self.norm_m = config.norm_class(config.n_embd, eps=config.norm_eps)
            self.norm_attn = config.norm_class(config.n_embd, eps=config.norm_eps)
            self.mb = Mamba(config.n_embd, layer_idx=layer_idx, **factory_kwargs)
            self.attn = CausalSelfAttention(
                config, n_embd=config.n_embd, layer_idx=layer_idx
            )
        elif config.use_mega:
            self.attn = S6GatedAttention(config.n_embd)
        else:
            self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
            self.use_retnet, self.use_gla = False, False
            if config.attn_layer_pos is not None:
                self.use_mamba = layer_idx not in eval(config.attn_layer_pos)
            else:
                self.use_mamba = (
                    layer_idx % config.mb_per_layer == 0
                    if config.mb_per_layer > 0
                    else False
                )
                self.use_retnet = (
                    layer_idx % config.ret_per_layer == 0
                    if config.ret_per_layer > 0
                    else False
                )
                self.use_gla = (
                    layer_idx % config.gla_per_layer == 0
                    if config.gla_per_layer > 0
                    else False
                )

            if self.use_mamba:
                if self.abl_mamba:
                    config_temp = copy.deepcopy(config)
                    config_temp._mlp_class = "LLaMAMLP"
                    config_temp.intermediate_size = config_temp.n_embd * 2
                    self.attn = config.mlp_class(config_temp)
                else:
                    self.attn = Mamba(config.n_embd, layer_idx=layer_idx, **factory_kwargs)
            elif self.use_retnet:
                self.attn = MultiScaleRetention(
                    hidden_size=config.n_embd,
                    num_heads=config.n_head // 2,
                    expand_k=1,
                    expand_v=2,
                    mode="fused_chunk",
                    use_short_conv=False,
                )
            elif self.use_gla:
                self.attn = GatedLinearAttention(
                    hidden_size=config.n_embd,
                    num_heads=config.n_embd // 384,
                    expand_k=0.5,
                    expand_v=1,
                    mode="fused_chunk",
                    use_short_conv=False,
                )
            else:
                self.attn = CausalSelfAttention(
                    config, n_embd=config.n_embd, layer_idx=layer_idx
                )

        if not config.shared_attention_norm and config.mlp and not config.parallel_residual:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)

        if config.mlp:
            self.mlp = config.mlp_class(config)

        self.config = config
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
        inference_params=None,
        pc=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        if self.mamba_swa_mlp:
            x = self.mb(
                self.norm_m(x.to(dtype=self.norm_m.weight.dtype)),
                inference_params=inference_params,
            ) + x.to(torch.float32)
            new_kv_cache = kv_cache
            h, new_kv_cache = self.attn(
                self.norm_attn(x.to(dtype=self.norm_attn.weight.dtype)),
                rope,
                max_seq_length,
                mask,
                input_pos,
                kv_cache,
            )
            x = h + x
        else:
            ox = x
            if self.config.use_mega:
                x = self.attn(x)
                new_kv_cache = None
            else:
                n_1 = self.norm_1(x.to(dtype=self.norm_1.weight.dtype))
                if self.use_mamba:
                    if self.abl_mamba:
                        h = self.attn(n_1)
                    else:
                        h = self.attn(n_1, inference_params=inference_params)
                    new_kv_cache = kv_cache
                    ox = ox.to(torch.float32)
                elif self.use_retnet or self.use_gla:
                    h, _, new_kv_cache = self.attn(n_1)
                else:
                    h, new_kv_cache = self.attn(
                        n_1, rope, max_seq_length, mask, input_pos, kv_cache
                    )
                x = ox + h

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
        if not self.fused_add_norm:
            residual = (
                hidden_states + residual if residual is not None else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
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


# ==================== CausalSelfAttention ====================

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int, n_embd: int, head_size=None) -> None:
        super().__init__()
        self.local = layer_idx % config.full_per_layer < config.full_per_layer - 1
        if head_size is not None:
            self.head_size = head_size
            self.n_head = n_embd // head_size
            self.n_query_groups = self.n_head
        else:
            self.head_size = config.head_size
            self.n_head = config.n_head
            self.n_query_groups = config.n_query_groups

        shape = (self.n_head + 2 * self.n_query_groups) * self.head_size
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=config.bias)

        self.config = config
        self.sc = config.sc_attn
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
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
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
        B, T, C = x.size()

        qkv = self.attn(x)
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.head_size)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, T, -1)
        k = k.reshape(B, T, -1)
        v = v.reshape(B, T, -1)
        if self.sc:
            q = causal_conv1d_fn(
                x=q.transpose(-1, -2),
                weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                bias=self.q_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)
            k = causal_conv1d_fn(
                x=k.transpose(-1, -2),
                weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                bias=self.k_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)
            v = causal_conv1d_fn(
                x=v.transpose(-1, -2),
                weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                bias=self.v_conv1d.bias,
                activation="silu",
            ).transpose(-1, -2)

        q = q.reshape(B, T, -1, self.head_size)
        k = k.reshape(B, T, -1, self.head_size)
        v = v.reshape(B, T, -1, self.head_size)

        if not self.config.nope:
            cos, sin = rope
            if cos.dtype != q.dtype:
                cos = cos.to(q.dtype)
                sin = sin.to(q.dtype)
            q = apply_rotary_emb_func(q, cos, sin, False, True)
            k = apply_rotary_emb_func(k, cos, sin, False, True)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)

            local_window = self.config.local_window
            cache_len = cache_k.size(1)

            current_pos = input_pos[-1].item()
            write_idx = current_pos % cache_len
            cache_k[:, write_idx : write_idx + 1] = k
            cache_v[:, write_idx : write_idx + 1] = v

            if current_pos < cache_len:
                read_indices = torch.arange(0, current_pos + 1, device=cache_k.device)
            else:
                start = current_pos - cache_len + 1
                read_indices = torch.arange(start, current_pos + 1, device=cache_k.device) % cache_len

            k = cache_k.index_select(1, read_indices)
            v = cache_v.index_select(1, read_indices)
            kv_cache = (cache_k, cache_v)

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)
        y = y.reshape(B, T, -1)
        y = self.proj(y)
        return y, kv_cache

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
                win_tuple = (self.config.local_window - 1, 0)
            else:
                win_tuple = (-1, -1)
            return flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=scale,
                causal=True,
                window_size=win_tuple,
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.swiglu = SwiGLU(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            _pack_weights=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)
    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
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
        B, N, C = x.shape
        _, M, _ = context.shape
        H = self.n_heads
        q = self.q_proj(x).view(B, N, H, C // H).transpose(1, 2)
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        k = k.view(B, M, H, C // H).transpose(1, 2)
        v = v.view(B, M, H, C // H).transpose(1, 2)
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(attn_output)
        out = self.dropout(out)
        return out
