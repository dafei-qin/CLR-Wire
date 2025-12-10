#!/bin/bash

# 配置参数
MAX_PARALLEL=64  # 最大并发数，可以根据需要修改
MAX_POINTS=4096
INPUT_BASE="../data/logan_jsons/abc"
OUTPUT_BASE="../data/logan_jsons_pc/abc"
LOG_DIR="./logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 用于存储后台进程PID的数组
declare -a PIDS=()

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

# 获取所有需要处理的目录（只遍历一级子目录）
echo "正在扫描目录..."
DIRS=($(find "$INPUT_BASE" -mindepth 2 -maxdepth 2 -type d | sort))
TOTAL=${#DIRS[@]}
echo "找到 $TOTAL 个目录需要处理"
echo "最大并发数: $MAX_PARALLEL"
echo "----------------------------------------"

# 计数器
PROCESSED=0

# 遍历所有目录
for input_dir in "${DIRS[@]}"; do
    # 获取相对路径 (x/xxxx 格式)
    rel_path=${input_dir#$INPUT_BASE/}
    
    # 构造输出路径
    output_dir="${OUTPUT_BASE}/${rel_path}"
    
    # 构造日志文件名（使用下划线替换斜杠）
    log_name=$(echo "$rel_path" | tr '/' '_')
    log_file="${LOG_DIR}/surface_to_samples_${log_name}.log"
    
    # 等待直到运行的进程数少于最大并发数
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
        # 检查并移除已完成的进程
        for i in "${!PIDS[@]}"; do
            pid=${PIDS[$i]}
            if ! kill -0 "$pid" 2>/dev/null; then
                unset 'PIDS[$i]'
            fi
        done
        # 重建数组索引
        PIDS=("${PIDS[@]}")
        
        # 如果还是满的，等待一下
        if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
            sleep 0.5
        fi
    done
    
    # 启动新进程
    ((PROCESSED++))
    echo "[$PROCESSED/$TOTAL] 正在处理: $rel_path"
    echo "  输入: $input_dir"
    echo "  输出: $output_dir"
    echo "  日志: $log_file"
    
    # 在后台运行命令
    python src/process_data/surface_to_samples.py \
        "$input_dir" \
        "$output_dir" \
        --max_points $MAX_POINTS \
        --compute_normal \
        --skip-type bspline_surface \
        --skip-num 15 \
        --skip-num-max 32 \
        > "$log_file" 2>&1 &
    
    # 保存进程PID
    PIDS+=($!)
    echo "  进程PID: $!"
    echo ""
done

# 等待所有剩余的后台进程完成
echo "----------------------------------------"
echo "所有任务已提交，等待完成..."
echo "当前运行中的进程数: ${#PIDS[@]}"

for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null
        echo "进程 $pid 已完成"
    fi
done

echo "----------------------------------------"
echo "所有任务处理完成！"
echo "总共处理了 $TOTAL 个目录"