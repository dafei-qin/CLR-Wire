#!/bin/bash
# setup_ramdisk.sh
# Usage: ./setup_ramdisk.sh [SRC_PATH] [RAMDISK_SIZE_GB] [RAMDISK_PATH]
# Example: ./setup_ramdisk.sh /data/train 120 /mnt/ramdisk

set -euo pipefail  # Exit on error, undefined var, pipe fail

# ===== 配置参数 =====
SRC_PATH="${1:-/data/train}"        # 源数据路径（目录或 .tar.gz 文件）
RAMDISK_SIZE_GB="${2:-120}"         # tmpfs 大小（GB），建议 = 数据大小 × 1.2
RAMDISK_PATH="${3:-/mnt/ramdisk}"   # 内存盘挂载点

# ===== 工具函数 =====
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2; }
error() { log "ERROR: $*" >&2; exit 1; }
warn() { log "WARN: $*" >&2; }

# 检查是否为 root（挂载需要）
[[ $EUID -eq 0 ]] || error "Please run with sudo (mount requires root)."

# 检查系统内存是否充足
TOTAL_MEM_GB=$(free -g | awk 'NR==2 {print $2}')
FREE_MEM_GB=$(free -g | awk 'NR==2 {print $7}')
REQ_MEM_GB=$((RAMDISK_SIZE_GB + 20))  # 预留 20GB 给系统

if (( FREE_MEM_GB < REQ_MEM_GB )); then
    warn "Free memory (${FREE_MEM_GB}G) < required (${REQ_MEM_GB}G). Proceed? [y/N]"
    read -r choice
    [[ "$choice" =~ ^[Yy]$ ]] || exit 1
fi

# ===== 步骤 1：创建/清理挂载点 =====
log "Preparing mount point: $RAMDISK_PATH"
mkdir -p "$RAMDISK_PATH"

# 若已挂载 tmpfs，先卸载（避免覆盖）
if mountpoint -q "$RAMDISK_PATH"; then
    if grep -q "tmpfs" "/proc/mounts" | grep -q "$RAMDISK_PATH"; then
        log "Unmounting existing tmpfs at $RAMDISK_PATH"
        umount "$RAMDISK_PATH"
    else
        error "$RAMDISK_PATH is mounted but not tmpfs — aborting for safety."
    fi
fi

# ===== 步骤 2：挂载 tmpfs =====
log "Mounting tmpfs (${RAMDISK_SIZE_GB}G) at $RAMDISK_PATH"
mount -t tmpfs -o "size=${RAMDISK_SIZE_GB}G,mode=700" tmpfs "$RAMDISK_PATH"
chmod 700 "$RAMDISK_PATH"

# 验证挂载 & 大小
ACTUAL_SIZE=$(df -BG "$RAMDISK_PATH" | awk 'NR==2 {print $2}' | tr -d 'G')
if (( ACTUAL_SIZE < RAMDISK_SIZE_GB - 5 )); then
    error "tmpfs size mismatch: expected ${RAMDISK_SIZE_GB}G, got ${ACTUAL_SIZE}G"
fi
log "✅ tmpfs mounted: $(df -h "$RAMDISK_PATH" | tail -1)"

# ===== 步骤 3：高效拷贝数据 =====
log "Copying data from: $SRC_PATH → $RAMDISK_PATH"

# 自动识别源类型：目录 or .tar.gz
if [[ -d "$SRC_PATH" ]]; then
    log "Source is a directory → using streaming tar (fastest for 100M files)"
    # 使用 tar 流式传输，避免海量 open/close
    time (
        cd "$SRC_PATH" && \
        tar cf - . 2>/dev/null | \
        (cd "$RAMDISK_PATH" && tar xf - 2>/dev/null)
    )
elif [[ "$SRC_PATH" =~ \.tar\.gz$ ]] && [[ -f "$SRC_PATH" ]]; then
    log "Source is a .tar.gz → streaming decompress & extract"
    time (
        zcat "$SRC_PATH" | \
        (cd "$RAMDISK_PATH" && tar xf - 2>/dev/null)
    )
else
    error "Unsupported source: must be a directory or .tar.gz file"
fi

# ===== 步骤 4：校验数据量 =====
RAMDISK_SIZE_ACTUAL=$(du -sb "$RAMDISK_PATH" | awk '{print $1}')
SRC_SIZE=$(du -sb "$SRC_PATH" | awk '{print $1}')
RATIO=$(awk "BEGIN {printf \"%.2f\", $RAMDISK_SIZE_ACTUAL/$SRC_SIZE}")

log "Source size:    $(numfmt --to=iec-i --suffix=B $SRC_SIZE)"
log "Ramdisk size:   $(numfmt --to=iec-i --suffix=B $RAMDISK_SIZE_ACTUAL)"
log "Copy ratio:     $RATIO (should be ~1.0)"

if (( $(echo "$RATIO < 0.99" | bc -l) )); then
    warn "Size mismatch detected — possible incomplete copy!"
fi

# ===== 步骤 5：生成路径索引（可选但强烈推荐）=====
INDEX_PATH="$RAMDISK_PATH/../paths.pkl"
log "Generating path index → $INDEX_PATH (for fast Dataset init)"

# 使用 Python 安全生成（避免 shell glob 爆炸）
python3 -c "
import os, pickle, sys
root = sys.argv[1]
paths = []
for dirpath, _, files in os.walk(root):
    for f in files:
        # 根据你的数据格式调整后缀（例如 '.bin', '.jpg', '.pt'）
        if f.endswith(('.bin', '.jpg', '.png', '.pt', '.npy')):
            paths.append(os.path.join(dirpath, f))
print(f'Found {len(paths):,} files')
with open(sys.argv[2], 'wb') as f:
    pickle.dump(paths, f)
" "$RAMDISK_PATH" "$INDEX_PATH"

# ===== 完成 =====
log "✅ SUCCESS: Data ready at $RAMDISK_PATH"
log "   Use in PyTorch: Dataset(root='$RAMDISK_PATH', index_file='$INDEX_PATH')"
log "--- DDP Tip: Pass --data_root '$RAMDISK_PATH' to all ranks ---"