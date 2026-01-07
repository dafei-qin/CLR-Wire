#!/bin/bash

# 配置参数
BASE_DIR="/data/ssd/CAD/data/abc_step_pc"
SCRIPT_PATH="third_party/HGDEEMOS/cache_data_sht.py"
LOG_DIR="${BASE_DIR}/logs/create_caches"
SUFFIX="max_2802_"  # 从命令行参数获取suffix，默认为空
MAX_TOKENS=2802

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

# 检查基础目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 基础目录不存在: $BASE_DIR"
    exit 1
fi

# 准备处理0-9的子文件夹
echo "正在扫描子文件夹..."
SUBDIRS=()
for i in {0..9}; do
    subdir="${BASE_DIR}/${i}"
    if [ -d "$subdir" ]; then
        SUBDIRS+=("$i")
    else
        echo "警告: 子文件夹不存在，跳过: $subdir"
    fi
done

TOTAL=${#SUBDIRS[@]}
if [ $TOTAL -eq 0 ]; then
    echo "错误: 没有找到任何有效的子文件夹 (0-9)"
    exit 1
fi

echo "找到 $TOTAL 个子文件夹需要处理"
if [ -n "$SUFFIX" ]; then
    echo "输出文件后缀: $SUFFIX"
else
    echo "输出文件无后缀"
fi
echo "----------------------------------------"

# 计数器
SUCCESS=0
FAILED=0
SKIPPED=0

# 遍历所有子文件夹
for subdir_num in "${SUBDIRS[@]}"; do
    cache_file="${BASE_DIR}/${subdir_num}"
    
    # 构造输出文件名
    if [ -n "$SUFFIX" ]; then
        save_file="${BASE_DIR}/cache_${subdir_num}_${SUFFIX}.pkl"
    else
        save_file="${BASE_DIR}/cache_${subdir_num}.pkl"
    fi
    
    # 检查是否已经处理过（如果输出文件存在，则跳过）
    if [ -f "$save_file" ]; then
        ((SKIPPED++))
        echo "[SKIP] 子文件夹 $subdir_num (输出文件已存在: $save_file)"
        continue
    fi
    
    # 构造日志文件名
    log_file="${LOG_DIR}/cache_${subdir_num}${SUFFIX:+_${SUFFIX}}.log"
    
    # 启动新进程
    echo "正在处理子文件夹: $subdir_num"
    echo "  输入目录: $cache_file"
    echo "  输出文件: $save_file"
    echo "  日志文件: $log_file"
    
    # 在后台运行命令
    python3 "$SCRIPT_PATH" \
        --cache_file "$cache_file" \
        --save_file "$save_file" \
        --max_tokens "$MAX_TOKENS" \
        > "$log_file" 2>&1 &
    
    # 保存进程PID
    pid=$!
    PIDS+=($pid)
    echo "  进程PID: $pid"
    echo ""
done

# 等待所有后台进程完成
echo "----------------------------------------"
echo "所有任务已提交，等待完成..."
echo "当前运行中的进程数: ${#PIDS[@]}"
echo ""

# 监控所有进程
while [ ${#PIDS[@]} -gt 0 ]; do
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
                echo "进程 $pid 已完成 (成功)"
            else
                ((FAILED++))
                echo "进程 $pid 已完成 (失败，退出码: $exit_code)"
            fi
        fi
    done
    # 重建数组索引
    PIDS=("${PIDS[@]}")
    
    # 如果还有进程在运行，等待一下
    if [ ${#PIDS[@]} -gt 0 ]; then
        sleep 1
    fi
done

echo "----------------------------------------"
echo "所有任务处理完成！"
echo "总共处理了 $TOTAL 个子文件夹"
echo "成功: $SUCCESS"
echo "失败: $FAILED"
echo "跳过: $SKIPPED"
if [ $FAILED -gt 0 ]; then
    echo "请检查日志目录 $LOG_DIR 了解失败详情"
fi

