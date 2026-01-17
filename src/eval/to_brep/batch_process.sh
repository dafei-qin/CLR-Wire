#!/bin/bash
################################################################################
# 批量多线程处理脚本 - 将 JSON+PLY 转换为 STEP 并渲染
# 
# 用法:
#   bash batch_process.sh --input_dir DIR --output_dir DIR [--num_threads N]
#
# 支持 xvfb-run 为每个线程提供独立的虚拟显示环境
################################################################################

# set -e  # 遇到错误立即退出

# ==================== 默认参数 ====================
INPUT_DIR=""
OUTPUT_DIR=""
NUM_THREADS=4
USE_XVFB=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/example.py"
LOG_DIR=""
XVFB_DISPLAY_START=99  # 虚拟显示器起始编号

# ==================== 参数解析 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_threads|-j)
            NUM_THREADS="$2"
            shift 2
            ;;
        --no-xvfb)
            USE_XVFB=false
            shift
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --input_dir DIR --output_dir DIR [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input_dir DIR       输入目录（包含PLY和JSON文件）"
            echo "  --output_dir DIR      输出目录"
            echo "  --num_threads N       并发线程数（默认: 4）"
            echo "  --no-xvfb             不使用 xvfb-run（需要真实显示环境）"
            echo "  --log_dir DIR         日志目录（默认: output_dir/logs）"
            echo "  -h, --help            显示帮助信息"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# ==================== 参数验证 ====================
if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: --input_dir is required"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output_dir is required"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置日志目录
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="${OUTPUT_DIR}/logs"
fi
mkdir -p "$LOG_DIR"

# ==================== 检查依赖 ====================
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# 检查是否有 xvfb-run
if [[ "$USE_XVFB" == true ]]; then
    if ! command -v xvfb-run &> /dev/null; then
        echo "Error: xvfb-run not found. Install it with: sudo apt-get install xvfb"
        echo "Or use --no-xvfb to disable virtual display"
        exit 1
    fi
    echo "✓ xvfb-run detected"
fi

# ==================== 查找匹配的文件 ====================
echo ""
echo "=================================================="
echo "Scanning for PLY+JSON pairs..."
echo "=================================================="
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Threads:          $NUM_THREADS"
echo "Use xvfb-run:     $USE_XVFB"
echo ""

# 临时文件存储任务列表
TASK_FILE=$(mktemp)
trap "rm -f $TASK_FILE" EXIT

# 扫描并匹配文件
python3 << EOF > "$TASK_FILE"
import os
import sys

input_dir = "$INPUT_DIR"
all_files = os.listdir(input_dir)
ply_files = [f for f in all_files if f.endswith('.ply')]
json_files = [f for f in all_files if f.endswith('.json')]

print(f"Found {len(ply_files)} PLY files and {len(json_files)} JSON files", file=sys.stderr)

matches = []
for ply_file in ply_files:
    parts = ply_file.split('_')
    if len(parts) >= 4:
        s_string = "_".join(parts[:3])
        
        # 查找对应的 pred JSON
        target_prefix = s_string + "_pred"
        found_pred = None
        for json_f in json_files:
            if json_f.startswith(target_prefix):
                found_pred = json_f
                break
        
        if found_pred:
            # 输出格式: ply_name|json_name
            print(f"{ply_file}|{found_pred}")
            matches.append((ply_file, found_pred))

print(f"Matched {len(matches)} PLY+JSON pairs", file=sys.stderr)
EOF

NUM_TASKS=$(wc -l < "$TASK_FILE")

if [[ $NUM_TASKS -eq 0 ]]; then
    echo "Error: No matching PLY+JSON pairs found"
    exit 1
fi

echo "✓ Found $NUM_TASKS task(s) to process"
echo ""

# ==================== 清理函数 ====================
declare -a PIDS=()

cleanup() {
    echo ""
    echo "正在清理所有后台进程..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "已终止进程: $pid"
        fi
    done
    wait 2>/dev/null
    echo "所有进程已清理完毕"
    exit 1
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

# ==================== 执行处理 ====================
echo "=================================================="
echo "Starting batch processing..."
echo "=================================================="
START_TIME=$(date +%s)

# 使用纯 bash 并发 + xvfb-run
echo "Using bash concurrent processing with $NUM_THREADS threads"
if [[ "$USE_XVFB" == true ]]; then
    echo "Each thread will use xvfb-run with independent display"
fi
echo ""
echo "Loading tasks into array..."

# 读取任务到数组
mapfile -t TASK_LINES < "$TASK_FILE"
echo "Loaded ${#TASK_LINES[@]} tasks"
echo ""

task_id=0

# 遍历所有任务
for task_line in "${TASK_LINES[@]}"; do
    [[ -z "$task_line" ]] && continue
    
    # 解析任务
    IFS='|' read -r ply_name json_name <<< "$task_line"
    [[ -z "$ply_name" ]] && continue
    
    ((task_id++))
    
    # 等待直到有空闲槽位
    while [ ${#PIDS[@]} -ge $NUM_THREADS ]; do
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
        if [ ${#PIDS[@]} -ge $NUM_THREADS ]; then
            sleep 0.5
        fi
    done
    
    # 启动新任务
    log_file="${LOG_DIR}/${ply_name}.log"
    step_file="${OUTPUT_DIR}/${ply_name}.step"
    
    # 检查是否已存在
    if [[ -f "$step_file" ]]; then
        echo "[${task_id}/${NUM_TASKS}] ⊙ Skip: $ply_name (already exists)"
        continue
    fi
    
    echo "[${task_id}/${NUM_TASKS}] ▶ Start: $ply_name"
    
    # 在后台执行
    (
        start=$(date +%s)
        
        # 根据是否使用 xvfb 选择执行方式
        if [[ "$USE_XVFB" == true ]]; then
            # 使用 xvfb-run
            xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" \
                python "$PYTHON_SCRIPT" \
                --input_path "$INPUT_DIR" \
                --output_path "$OUTPUT_DIR" \
                --num_samples 999999 \
                --cont "${ply_name%.*}" >> "$log_file" 2>&1
            ret=$?
        else
            # 直接执行
            python "$PYTHON_SCRIPT" \
                --input_path "$INPUT_DIR" \
                --output_path "$OUTPUT_DIR" \
                --num_samples 999999 \
                --cont "${ply_name%.*}" >> "$log_file" 2>&1
            ret=$?
        fi
        
        if [ $ret -eq 0 ]; then
            end=$(date +%s)
            duration=$((end - start))
            
            # 检查输出文件
            if [[ -f "$step_file" ]]; then
                jpg_file="${OUTPUT_DIR}/${ply_name}.jpg"
                if [[ -f "$jpg_file" ]]; then
                    echo "[${task_id}/${NUM_TASKS}] ✓ Success: $ply_name (${duration}s, STEP+JPG)"
                else
                    echo "[${task_id}/${NUM_TASKS}] ✓ Success: $ply_name (${duration}s, STEP only)"
                fi
            else
                echo "[${task_id}/${NUM_TASKS}] ✗ Failed: $ply_name - No output file"
            fi
        else
            end=$(date +%s)
            duration=$((end - start))
            echo "[${task_id}/${NUM_TASKS}] ✗ Failed: $ply_name (${duration}s, see $log_file)"
        fi
    ) &
    
    # 保存进程PID
    PIDS+=($!)
    
    # 每处理10个任务输出一次进度
    if [[ $((task_id % 10)) -eq 0 ]]; then
        echo "Progress: $task_id/$NUM_TASKS tasks submitted, ${#PIDS[@]} running"
    fi
    
done

echo ""
echo "All $task_id tasks submitted"
echo "Total PIDs: ${#PIDS[@]}"

# 等待所有剩余的后台进程完成
echo ""
echo "=================================================="
echo "所有任务已提交，等待完成..."
echo "当前运行中的进程数: ${#PIDS[@]}"
echo "=================================================="

# 实时进度监控
LAST_STEP_COUNT=0
LAST_FAILED_COUNT=0

while true; do
    # 检查是否还有进程在运行
    RUNNING=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((RUNNING++))
        fi
    done
    
    # 如果没有进程在运行，退出监控循环
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    
    # 统计当前完成数
    CURRENT_STEP_COUNT=$(find "$OUTPUT_DIR" -name "*.step" 2>/dev/null | wc -l)
    CURRENT_FAILED_COUNT=$(find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | wc -l)
    CURRENT_FAILED_COUNT=$((CURRENT_FAILED_COUNT - CURRENT_STEP_COUNT))
    
    # 如果有新的进展，显示
    if [ $CURRENT_STEP_COUNT -ne $LAST_STEP_COUNT ] || [ $CURRENT_FAILED_COUNT -ne $LAST_FAILED_COUNT ]; then
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TIME))
        TOTAL_DONE=$((CURRENT_STEP_COUNT + CURRENT_FAILED_COUNT))
        
        if [ $ELAPSED -gt 0 ] && [ $TOTAL_DONE -gt 0 ]; then
            RATE=$((TOTAL_DONE * 60 / ELAPSED))  # tasks per minute
            REMAINING=$((NUM_TASKS - TOTAL_DONE))
            if [ $RATE -gt 0 ]; then
                ETA_SEC=$((REMAINING * 60 / RATE))
                ETA_MIN=$((ETA_SEC / 60))
                printf "\r[Progress] Completed: %d/%d | Success: %d | Failed: %d | Running: %d | Rate: %d/min | ETA: %dmin" \
                    $TOTAL_DONE $NUM_TASKS $CURRENT_STEP_COUNT $CURRENT_FAILED_COUNT $RUNNING $RATE $ETA_MIN
            else
                printf "\r[Progress] Completed: %d/%d | Success: %d | Failed: %d | Running: %d" \
                    $TOTAL_DONE $NUM_TASKS $CURRENT_STEP_COUNT $CURRENT_FAILED_COUNT $RUNNING
            fi
        fi
        
        LAST_STEP_COUNT=$CURRENT_STEP_COUNT
        LAST_FAILED_COUNT=$CURRENT_FAILED_COUNT
    fi
    
    sleep 2
done

echo ""  # 换行

# 等待所有进程确保完成
for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# ==================== 统计结果 ====================
echo ""
echo "=================================================="
echo "Processing Complete!"
echo "=================================================="
echo "Total tasks:    $NUM_TASKS"
echo "Duration:       ${DURATION}s"
echo "Output dir:     $OUTPUT_DIR"
echo "Logs dir:       $LOG_DIR"

# 统计成功/失败
STEP_COUNT=$(find "$OUTPUT_DIR" -name "*.step" | wc -l)
JPG_COUNT=$(find "$OUTPUT_DIR" -name "*.jpg" | wc -l)

echo "STEP files:     $STEP_COUNT"
echo "JPG files:      $JPG_COUNT"
echo "=================================================="

