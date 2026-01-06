#!/bin/bash

#!/bin/bash
# if [[ -z "${CONFIG}" ]]; then
#   MODEL_CONFIG=conf/model_configs/meshtron.yaml
#   echo ${MODEL_CONFIG}
# else
#   MODEL_CONFIG="${CONFIG}"
# fi

# DEBUG模式运行方式: bash start_train.sh DEBUG
# 配置不同运行模式特有的参数
# if [ "$1" == "DEBUG" ]  # DEBUG模式的参数配置
# then
#     echo -e "\033[32m$0 Running in DEBUG mode \033[0m"
#     export EXTRA_ARGS="--debug_mode"
#     export GRADIENT_ACCUMULATION_STEPS=1
# else  # 默认COS直连训练配置，运行方式: bash start_train.sh
#     echo -e "\033[32m$0 Running in TRAIN mode, run in DEBUG mode by ./start_train.sh DEBUG \033[0m"
#     export EXTRA_ARGS="--resume_from_checkpoint=latest"
#     export GRADIENT_ACCUMULATION_STEPS=16
# fi

# 其他公共参数配置
# export OFFLOAD_DEVICE=cpu
# export ZERO_STAGE=1
# export MIXED_PRECISION=bf16
# # export OMP_NUM_THREADS=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# echo -e "EXTRA_ARGS=$EXTRA_ARGS"

# 获取当前脚本的绝对路径
# script_path="$(dirname "$(realpath "$0")")"
# 切换到脚本所在的目录
# cd "$script_path"
# export PYTHONPATH=.

# 配置多节点训练
# export MACHINE_TYPE=normal
# h200_count=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -c -w "H200")
# h20_count=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep -c -w "H20")

# if [ "$h200_count" -eq 8 ]; then
#     export MACHINE_TYPE=h200

#     # 如果是H200,则
#     MODEL_CONFIG=conf/h200/meshtron.yaml

#     export NCCL_IB_GID_INDEX=3
#     export NCCL_IB_SL=3
#     export NCCL_CHECK_DISABLE=1
#     export NCCL_P2P_DISABLE=0
#     export NCCL_IB_DISABLE=0
#     export NCCL_LL_THRESHOLD=16384
#     export NCCL_IB_CUDA_SUPPORT=1
#     export NCCL_SOCKET_IFNAME=eth0
#     export UCX_NET_DEVICES=eth0
#     export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
#     export NCCL_COLLNET_ENABLE=0
#     export SHARP_COLL_ENABLE_SAT=0
#     export NCCL_NET_GDR_LEVEL=2
#     export NCCL_IB_QPS_PER_CONNECTION=4
#     export NCCL_IB_TC=160
#     export NCCL_PXN_DISABLE=1

#     export NCCL_TIMEOUT=1800
#     export FI_PROVIDER=efa
#     export OFI_NCCL_PROTOCOL=RDMA
#     export FI_EFA_USE_DEVICE_RDMA=1
#     export NCCL_TUNER_PLUGIN=/opt/amazon/efa/lib/libnccl-ofi-tuner.so

# elif [ "$h20_count" -eq 8 ]; then
#     export NCCL_IB_GID_INDEX=3
#     export NCCL_IB_SL=3
#     export NCCL_CHECK_DISABLE=1
#     export NCCL_P2P_DISABLE=0
#     export NCCL_IB_DISABLE=0
#     export NCCL_LL_THRESHOLD=16384
#     export NCCL_IB_CUDA_SUPPORT=1
#     export NCCL_SOCKET_IFNAME=bond1
#     export UCX_NET_DEVICES=bond1
#     export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
#     export NCCL_COLLNET_ENABLE=0
#     export SHARP_COLL_ENABLE_SAT=0
#     export NCCL_NET_GDR_LEVEL=2
#     export NCCL_IB_QPS_PER_CONNECTION=4
#     export NCCL_IB_TC=160
#     export NCCL_PXN_DISABLE=1

#     # export NCCL_LL_THRESHOLD=0
#     # export NCCL_P2P_DISABLE=1
#     # export NCCL_IB_DISABLE=1

# else
#     export NCCL_SOCKET_IFNAME=eth1
#     export NCCL_IB_GID_INDEX=3
#     export NCCL_IB_SL=3
#     export NCCL_CHECKS_DISABLE=1
#     export NCCL_IB_DISABLE=0
#     export NCCL_IBEXT_DISABLE=0
#     export NCCL_P2P_DISABLE=0
#     export NCCL_LL_THRESHOLD=16384
#     export NCCL_IB_CUDA_SUPPORT=1
#     # export NCCL_DEBUG=INFO
#     nccl_ib_hca=$(bash tools/scripts/show_gids | grep $(hostname -I) | grep v2 | awk '{print $1 ":" $2}' )
#     echo "nccl_ib_hca is ${nccl_ib_hca}"
#     export NCCL_IB_HCA="=$nccl_ib_hca"
# fi

# 配置 accelerate
# if [[ -z ${NODE_IP_LIST} ]]; then
#     # taiji环境里有这个变量，只有H200机器环境里没有
#     # 获取参数
#     num=${WORLD_SIZE}
#     HOST_NAME=${JOB_NAME}

#     echo "$num $HOST_NAME"
#     # 指定输出文件名
#     hostfile="/workspace/hostfile"

#     # 清空文件（如果文件已存在）
#     > "$hostfile"

#     # 循环写入字符串到文件
#     for ((i=0; i<1; i++)); do
#         echo "${HOST_NAME}-master-${i} slots=8" >> "$hostfile"
#     done

#     for ((i=0; i<num-1; i++)); do
#         echo "${HOST_NAME}-worker-${i} slots=8" >> "$hostfile"
#     done

#     echo "Successfully written $num lines to $hostfile."

# else
#     echo $NODE_IP_LIST > /dockerdata/env.txt
#     sed "s/:/ slots=/g" /dockerdata/env.txt | sed "s/,/\n/g" > "/dockerdata/hostfile"
#     hostfile="/dockerdata/hostfile"
# fi

# process_num=$(awk -F'slots=' '{sum += $2} END {print sum}' $hostfile)
# if [ "$1" == "DEBUG" ]
# then
#     process_num=1
# fi

# echo "$process_num total process num"

. /root/clashctl/scripts/cmd/clashctl.sh
export WANDB_API_KEY=3c417f941b483432f09ba32cebabecae043cf11f   # ATTETNION:需要改成个人的key
wandb login
# if [ "$h200_count" -eq 8 ]; then
#     export WANDB_API_KEY=f5b896f87730635708b9b81c5d7728ae1a8b4903   # ATTETNION:需要改成个人的key
#     wandb login

#     if [ "$1" == "DEBUG" ]
#     then
#         RANK=0
#         WORLD_SIZE=1
#         MASTER_ADDR=127.0.0.1
#         MASTER_PORT=12345
#         process_num=8
#     fi
    
#     echo "RANK ${RANK} MADDR ${MASTER_ADDR} WORLDSIZE ${WORLD_SIZE} PROCESSNUM ${process_num} MASTERPORT ${MASTER_PORT}" 

#     export accelerate_config_yaml="/workspace/accelerate_multinode.yaml"
#     sed "s#CONFIG_HOSTFILE#${hostfile}#" "accelerate_multinode.yaml" \
#         | sed "s/CONFIG_HOST_INDEX/${RANK}/" \
#         | sed "s/CONFIG_CHIEF_IP/${MASTER_ADDR}/" \
#         | sed "s/CONFIG_HOST_NUM/${WORLD_SIZE}/" \
#         | sed "s/CONFIG_PROCESS_NUM/${process_num}/" \
#         | sed "s/CONFIG_GRADIENT_ACCUMULATION_STEPS/${GRADIENT_ACCUMULATION_STEPS}/" \
#         | sed "s/CONFIG_PROCESS_PORT/${MASTER_PORT}/" \
#         | sed "s/CONFIG_OFFLOAD_DEVICE/${OFFLOAD_DEVICE}/" \
#         | sed "s/CONFIG_ZERO_STAGE/${ZERO_STAGE}/" \
#         | sed "s/CONFIG_MIXED_PRECISION/${MIXED_PRECISION}/" \
#         > ${accelerate_config_yaml}
# else
#     export WANDB_API_KEY=f5b896f87730635708b9b81c5d7728ae1a8b4903   # ATTETNION:需要改成个人的key
#     wandb login

    
#     export accelerate_config_yaml="/dockerdata/accelerate_multinode.yaml"
#     sed "s#CONFIG_HOSTFILE#${hostfile}#" "accelerate_multinode.yaml" \
#         | sed "s/CONFIG_HOST_INDEX/${INDEX}/" \
#         | sed "s/CONFIG_CHIEF_IP/${CHIEF_IP}/" \
#         | sed "s/CONFIG_HOST_NUM/${HOST_NUM}/" \
#         | sed "s/CONFIG_PROCESS_NUM/${process_num}/" \
#         | sed "s/CONFIG_GRADIENT_ACCUMULATION_STEPS/${GRADIENT_ACCUMULATION_STEPS}/" \
#         | sed "s/CONFIG_OFFLOAD_DEVICE/${OFFLOAD_DEVICE}/" \
#         | sed "s/CONFIG_ZERO_STAGE/${ZERO_STAGE}/" \
#         | sed "s/CONFIG_MIXED_PRECISION/${MIXED_PRECISION}/" \
#         > ${accelerate_config_yaml}
# fi

# 启动训练任务
# accelerate launch --config_file=${accelerate_config_yaml} --mixed_precision=$MIXED_PRECISION src/train.py \
#     --model_config ${MODEL_CONFIG} \
#     --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
#     --mixed_precision $MIXED_PRECISION \
#     --adam_epsilon 1e-15 \
#     $EXTRA_ARGS

# accelerate launch \
#     --config_file=${accelerate_config_yaml} \
#     --mixed_precision=bf16 \
#     train_window.py \
#     --model_id 551 \
#     --data_path /vinowan-cfs/ruixu/data/MyTrainData1 \
#     --output_dir $OUTPUT_DIR \
#     --batch_size 9 \
#     --learning_rate 1e-4 \
#     --num_epochs 20000 \
#     --window_size 9000 \
#     --context_length 90000 \
#     --warmup_steps 1000 \
#     --model_path /workspace/pytorch_model.bin \
#     --seed 42
export MASTER_PORT=29500

# export OUTPUT_DIR=checkpoints/1226_gpt_init_debug$(date +%Y-%m-%d_%H)

# torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=1 \
#     --master_addr=127.0.0.1 \
#     --master_port=29501 \
#     train_window_torchrun.py \
#     --model_id 551 \
#     --data_path /vinowan-cfs/ruixu/data/MyTrainData1 \
#     --output_dir $OUTPUT_DIR \
#     --batch_size 9 \
#     --learning_rate 1e-4 \
#     --num_epochs 20000 \
#     --window_size 9000 \
#     --context_length 90000 \
#     --warmup_steps 1000 \
#     --model_path /workspace/checkpoint_batch_9_epoch_7_date_2025-07-09_06-40-08.bin \


# --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM

python third_party/HGDEEMOS/pretrain.py --config_path src/configs/gpt/gpt_0105_michel_a800.yaml --resume False
# torchrun \
#     --nnodes=$MLP_WORKER_NUM \
#     --node_rank=$MLP_ROLE_INDEX \
#     --nproc_per_node=$MLP_WORKER_GPU \
#     --master_addr=$MLP_WORKER_0_HOST \
#     --master_port=$MLP_WORKER_0_PORT \
#     pretrain.py \
#     --train_data_dir /data4/ruixu/LLMDATA/slim \
#     --val_data_dir /data4/ruixu/LLMDATA/slim \
#     --resume /deemos-research-area-d/meshgen/code/HG_DEEMOS/out/HY1024_tsz128x16k_100B_ScaleUp20k_unlockCondition_Diff_LLaMA_551M/Samba-DEEMOS-12-23-06/iter-045000-ckpt.pth
    # --warm_start_ckpt /deemos-research-area-d/meshgen/code/Samba/out/tsz128x16k_100B_Samba_2.2B_22L/Samba-DEEMOS-11-14-06/iter-132000-ckpt.pth
    

# export MASTER_ADDR=localhost
# export MASTER_PORT=29500

# torchrun \
# --nnodes=2 \
# --node_rank=$MLP_ROLE_INDEX \
# --nproc_per_node=$MLP_WORKER_GPU \
# --rdzv_id=samba-421M \
# --rdzv_backend=c10d  \
# --rdzv_endpoint=${MLP_WORKER_0_HOST}:${MLP_WORKER_0_PORT} \
# pretrain.py \
# --train_data_dir /data4/ruixu/LLMDATA/slim \
# --val_data_dir /data4/ruixu/LLMDATA/slim 
