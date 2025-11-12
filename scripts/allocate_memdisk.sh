#!/bin/bash
# manage_ramdisk.sh
# 功能：创建/重置指定大小的 tmpfs 内存盘（仅管理 mount，不处理数据）
# 用法：sudo ./manage_ramdisk.sh [SIZE_GB] [MOUNT_POINT]
# 示例：sudo ./manage_ramdisk.sh 120 /mnt/ramdisk

set -euo pipefail

SIZE_GB="${1:-120}"
MOUNT_POINT="${2:-/mnt/ramdisk}"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2; }
error() { log "ERROR: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || error "Please run with sudo."

# 检查内存是否够用（预留 10GB 系统开销）
free_mem_gb=$(free -g | awk 'NR==2 {print $7}')
req_mem_gb=$((SIZE_GB + 10))
if (( free_mem_gb < req_mem_gb )); then
    log "WARN: Free memory (${free_mem_gb}G) < required (${req_mem_gb}G). Continue? [y/N]"
    read -r choice || exit 1
    [[ "$choice" =~ ^[Yy]$ ]] || exit 0
fi

log "Managing ramdisk at: $MOUNT_POINT (size=${SIZE_GB}G)"

# 创建挂载点（若不存在）
mkdir -p "$MOUNT_POINT"
chmod 700 "$MOUNT_POINT"

# 检查是否已挂载
if mountpoint -q "$MOUNT_POINT"; then
    # 确保是 tmpfs（防止误操作真实磁盘）
    if ! findmnt -n -o FSTYPE "$MOUNT_POINT" | grep -q "^tmpfs$"; then
        error "$MOUNT_POINT is mounted but NOT tmpfs — aborting for safety."
    fi

    log "→ Existing tmpfs detected. Cleaning & resizing..."
    # 清空内容（比 rm -rf 更快，尤其对海量小文件）
    mount --bind "$MOUNT_POINT" "$MOUNT_POINT"
    mount -o remount,ro "$MOUNT_POINT"  # 先只读防写入
    mount -o remount,rw "$MOUNT_POINT"  # 重挂读写（清空 inode cache 效果类似格式化）
    # 实际清空：删除所有内容（tmpfs 删除极快）
    find "$MOUNT_POINT" -mindepth 1 -delete 2>/dev/null || true
    umount "$MOUNT_POINT"
else
    log "→ No existing mount. Creating fresh tmpfs..."
fi

# 重新挂载指定大小
log "Mounting tmpfs (${SIZE_GB}G) at $MOUNT_POINT"
mount -t tmpfs -o "size=${SIZE_GB}G,mode=700" tmpfs "$MOUNT_POINT"
chmod 700 "$MOUNT_POINT"

# 验证
actual_size=$(df -BG "$MOUNT_POINT" 2>/dev/null | awk 'NR==2 {print $2}' | tr -d 'G')
if (( actual_size < SIZE_GB - 2 )); then
    error "Mount size mismatch: expected ${SIZE_GB}G, got ${actual_size}G"
fi

log "✅ SUCCESS: tmpfs ready at $MOUNT_POINT"
df -h "$MOUNT_POINT" | tail -1
log "--- Now you can:"
log "    1. cp your_data.zip $MOUNT_POINT/"
log "    2. cd $MOUNT_POINT && unzip your_data.zip"
log "    3. Use in Dataset(root='$MOUNT_POINT')"