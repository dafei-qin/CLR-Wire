#!/bin/bash

# 配置参数
MAX_PARALLEL=64  # 最大并发数，可以根据需要修改
NUM_SAMPLES=4096
FPS=True
NUM_FPS=20480
INPUT_BASE="/home/qindafei/CAD/data/abc_step_full/1"
OUTPUT_BASE="/home/qindafei/CAD/data/abc_step_pc/1"
LOG_DIR="./logs/batch_step_to_pc/1"
SCRIPT_PATH="src/tools/sample_step_to_pc_debug.py"

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

# 获取所有需要处理的 STEP 文件（递归查找）
echo "正在扫描 STEP 文件..."
STEP_FILES=($(find "$INPUT_BASE" -type f -name "*.step" | sort))
TOTAL=${#STEP_FILES[@]}
echo "找到 $TOTAL 个 STEP 文件需要处理"
echo "最大并发数: $MAX_PARALLEL"
echo "Num samples: $NUM_SAMPLES"
echo "FPS: $FPS, Num FPS: $NUM_FPS"
echo "----------------------------------------"

# 计数器
PROCESSED=0
SUCCESS=0
FAILED=0

# 遍历所有 STEP 文件
for step_file in "${STEP_FILES[@]}"; do
    # 获取相对路径
    rel_path=${step_file#$INPUT_BASE/}
    
    # 构造输出路径（保持目录结构）
    # 注意：每个 STEP 文件可能生成多个 .npz 文件（每个 unique solid 一个，格式为 xxx_000.npz, xxx_001.npz 等）
    output_base_file="${OUTPUT_BASE}/${rel_path%.step}.npz"
    output_dir=$(dirname "$output_base_file")
    output_base_name=$(basename "$output_base_file" .npz)
    
    # 检查是否已经处理过（如果输出目录存在，则认为已处理）
    if [ -d "$output_dir" ]; then
        ((PROCESSED++))
        echo "[$PROCESSED/$TOTAL] [SKIP] $rel_path (已存在)"
        ((SUCCESS++))
        continue
    fi
    
    # 构造日志文件名（使用下划线替换斜杠）
    log_name=$(echo "$rel_path" | tr '/' '_' | sed 's/\.step$//')
    log_file="${LOG_DIR}/step_to_pc_${log_name}.log"
    
    # 等待直到运行的进程数少于最大并发数
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
        # 检查并移除已完成的进程
        for i in "${!PIDS[@]}"; do
            pid=${PIDS[$i]}
            if ! kill -0 "$pid" 2>/dev/null; then
                unset 'PIDS[$i]'
                # 检查进程是否成功
                wait "$pid" 2>/dev/null
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    ((SUCCESS++))
                else
                    ((FAILED++))
                fi
            fi
        done
        # 重建数组索引
        PIDS=("${PIDS[@]}")
        
        # 如果还是满的，等待一下
        if [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; then
            sleep 0.5
        fi
    done
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 启动新进程
    ((PROCESSED++))
    echo "[$PROCESSED/$TOTAL] 正在处理: $rel_path"
    echo "  输入: $step_file"
    echo "  输出: $output_dir"
    echo "  日志: $log_file"
    
    # 在后台运行命令
    python3 "$SCRIPT_PATH" \
        --single-file \
        "$step_file" \
        --output_dir "$output_dir" \
        --num_samples $NUM_SAMPLES \
        --fps $FPS \
        --num_fps $NUM_FPS \
        --no-debug \
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
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            ((SUCCESS++))
            echo "进程 $pid 已完成 (成功)"
        else
            ((FAILED++))
            echo "进程 $pid 已完成 (失败，退出码: $exit_code)"
        fi
    fi
done

echo "----------------------------------------"
echo "所有任务处理完成！"
echo "总共处理了 $TOTAL 个文件"
echo "成功: $SUCCESS"
echo "失败: $FAILED"
echo "跳过: $((TOTAL - PROCESSED))"
if [ $FAILED -gt 0 ]; then
    echo "请检查日志目录 $LOG_DIR 了解失败详情"
fi

