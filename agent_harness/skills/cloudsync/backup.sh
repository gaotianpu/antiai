#!/usr/bin/env bash
# 云备份：本地 → 云端
# 用法: bash backup.sh [项目目录]
set -euo pipefail

CLOUD_ROOT="/mnt/d/华为云盘"
PROJECT_DIR="${1:-$PWD}"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
TAR_NAME="${PROJECT_NAME}.tar.gz"
TMP_TAR="/tmp/$TAR_NAME"
CLOUD_TAR="$CLOUD_ROOT/$TAR_NAME"

# 1. 打包（排除隐藏文件）
echo "[INFO] 打包项目 $PROJECT_NAME ..."
tar czf "$TMP_TAR" \
  --exclude='.*' \
  -C "$(dirname "$PROJECT_DIR")" "$PROJECT_NAME" \
  --totals || { echo "[ERR] 打包失败"; exit 1; }
echo "[INFO] 打包完成：$(du -h "$TMP_TAR" | cut -f1)"

# 2. 检查云端目录
if [ ! -d "$CLOUD_ROOT" ]; then
  echo "[ERR] 云端目录不存在：$CLOUD_ROOT"
  echo "[ERR] 请确认 Windows 端已挂载（WSL 下 /mnt/d/ 对应 D: 盘）"
  exit 1
fi

# 3. 删除云端旧备份（确保全新替换）
if [ -f "$CLOUD_TAR" ]; then
  echo "[INFO] 删除云端旧备份：$CLOUD_TAR"
  rm -f "$CLOUD_TAR"
  # 等待云同步处理删除，避免与新上传冲突
  sleep 5
fi

# 4. 上传
echo "[INFO] 上传到云端 ..."
rsync --progress "$TMP_TAR" "$CLOUD_TAR"
RC=$?
echo ""
if [ $RC -eq 0 ]; then
  echo "[INFO] 云端备份完成：$CLOUD_TAR"
else
  echo "[ERR] 上传失败，rsync 退出码: $RC"
  rm -f "$TMP_TAR"
  exit $RC
fi

# 5. 清理
rm -f "$TMP_TAR"
echo "[INFO] 本地临时文件已清理"
echo "[INFO] 备份完成！如需恢复，请输入：云恢复"
