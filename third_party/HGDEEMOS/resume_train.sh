#!/bin/bash

# 🔥 火山云 conda 环境激活（非交互式脚本）
# 方法：source conda.sh 以初始化 conda 命令
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cad

# 配置代理（如果需要）
if [ -f /root/clashctl/scripts/cmd/clashctl.sh ]; then
    . /root/clashctl/scripts/cmd/clashctl.sh
fi

# Wandb 登录
export WANDB_API_KEY=3c417f941b483432f09ba32cebabecae043cf11f   # ATTETNION:需要改成个人的key
wandb login

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    /deemos-research-area-d/meshgen/cad/CLR-Wire/third_party/HGDEEMOS/pretrain.py \
    --config_path /deemos-research-area-d/meshgen/cad/CLR-Wire/src/configs/gpt/gpt_0105_michel_a800.yaml --resume True


