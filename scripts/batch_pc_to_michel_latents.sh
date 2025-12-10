#!/bin/bash

# 配置参数
GPU_LIST=(0 1 2 3 4 5)  # 可用的GPU列表
CONFIG_PATH="third_party/Michelangelo/configs/aligned_shape_latents/shapevae-256.yaml"
CHECKPOINT_PATH="third_party/Michelangelo/checkpoints/aligned_shape_latents/shapevae-256.ckpt"
INPUT_BASE="../data/logan_jsons_pc/abc"
OUTPUT_BASE="../data/logan_jsons_michel_latent/abc"
LOG_DIR="./logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 用于存储后台进程信息的数组
declare -a PIDS=()      # 进程PID
declare -a GPU_IDS=()   # 对应的GPU ID

# 清理函数：杀死所有后台进程
cleanup() {
    echo ""
    echo "正在清理所有后台进程..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "已终止进程: $pid"
        fi
    done
    # 等待所有进程真正结束
    wait 2>/dev/null
    echo "所有进程已清理完毕"
    exit 1
}

# 注册信号处理：捕获 Ctrl+C (SIGINT) 和 SIGTERM
trap cleanup SIGINT SIGTERM

# 获取所有需要处理的目录（遍历 x/xxxx 格式的目录）
echo "正在扫描目录..."
DIRS=($(find "$INPUT_BASE" -mindepth 1 -maxdepth 1 -type d | sort))
TOTAL=${#DIRS[@]}
echo "找到 $TOTAL 个目录需要处理"
echo "可用GPU数量: ${#GPU_LIST[@]} (${GPU_LIST[*]})"
echo "最大并发数: ${#GPU_LIST[@]} (每个GPU一个任务)"
echo "检查点路径: $CHECKPOINT_PATH"
echo "----------------------------------------"

# 计数器
PROCESSED=0
GPU_INDEX=0  # 当前使用的GPU索引

# 遍历所有目录
for input_dir in "${DIRS[@]}"; do
    # 获取相对路径 (x/xxxx 格式)
    rel_path=${input_dir#$INPUT_BASE/}
    
    # 构造输出路径
    output_dir="${OUTPUT_BASE}/${rel_path}"
    
    # 构造日志文件名（使用下划线替换斜杠）
    log_name=$(echo "$rel_path" | tr '/' '_')
    log_file="${LOG_DIR}/batch_pc_to_michel_latents_${log_name}.log"
    
    # 等待直到有空闲的GPU（进程数少于GPU数量）
    while [ ${#PIDS[@]} -ge ${#GPU_LIST[@]} ]; do
        # 检查并移除已完成的进程
        for i in "${!PIDS[@]}"; do
            pid=${PIDS[$i]}
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "进程 $pid (GPU ${GPU_IDS[$i]}) 已完成"
                unset 'PIDS[$i]'
                unset 'GPU_IDS[$i]'
            fi
        done
        # 重建数组索引
        PIDS=("${PIDS[@]}")
        GPU_IDS=("${GPU_IDS[@]}")
        
        # 如果还是满的，等待一下
        if [ ${#PIDS[@]} -ge ${#GPU_LIST[@]} ]; then
            sleep 0.5
        fi
    done
    
    # 选择当前使用的GPU
    current_gpu=${GPU_LIST[$GPU_INDEX]}
    GPU_INDEX=$(( (GPU_INDEX + 1) % ${#GPU_LIST[@]} ))
    
    # 启动新进程
    ((PROCESSED++))
    echo "[$PROCESSED/$TOTAL] 正在处理: $rel_path"
    echo "  GPU: $current_gpu"
    echo "  输入: $input_dir"
    echo "  输出: $output_dir"
    echo "  日志: $log_file"
    
    # 在后台运行命令
    python third_party/Michelangelo/inference_simple.py \
        --config_path $CONFIG_PATH \
        --ckpt_path $CHECKPOINT_PATH \
        --input_path $input_dir \
        --output_dir $output_dir \
        --device cuda:$current_gpu \
        --save_latents \
        > "$log_file" 2>&1 &
    
    # 保存进程PID和对应的GPU ID
    PIDS+=($!)
    GPU_IDS+=($current_gpu)
    echo "  进程PID: $!"
    echo ""
done

# 等待所有剩余的后台进程完成
echo "----------------------------------------"
echo "所有任务已提交，等待完成..."
echo "当前运行中的进程数: ${#PIDS[@]}"

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    gpu=${GPU_IDS[$i]}
    if kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null
        echo "进程 $pid (GPU $gpu) 已完成"
    fi
done

echo "----------------------------------------"
echo "所有任务处理完成！"
echo "总共处理了 $TOTAL 个目录"
echo "日志保存在: $LOG_DIR"
