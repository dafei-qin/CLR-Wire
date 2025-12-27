#!/bin/bash
# Sample with trained conditioner
# Conditioner 现在直接从统一的 checkpoint 中加载（如果存在）

CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master_port=12346 sample_with_conditioner.py \
    --model_path "/deemos-research-area-d/meshgen/code/HG_DEEMOS/out/HY1024_tsz128x16k_100B_ScaleUp20k_unlockCondition_Diff_LLaMA_551M/Samba-DEEMOS-12-23-06/iter-062500-ckpt.pth" \
    --model_id 551 \
    --steps 50000 \
    --output_path mesh_output_conditioner \
    --repeat_num 4 \
    --temperature 0.5


