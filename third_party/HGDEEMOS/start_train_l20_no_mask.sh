#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cad
export WANDB_API_KEY=3c417f941b483432f09ba32cebabecae043cf11f
wandb login
cd /deemos-research-area-d/meshgen/cad/CLR-Wire
torchrun     --nnodes=1     --nproc_per_node=1     --master_addr=localhost     --master_port=29502     /deemos-research-area-d/meshgen/cad/CLR-Wire/third_party/HGDEEMOS/pretrain.py     --config_path /deemos-research-area-d/meshgen/cad/CLR-Wire/src/configs/gpt/gpt_l20_ablation_no_mask.yaml     --resume False
