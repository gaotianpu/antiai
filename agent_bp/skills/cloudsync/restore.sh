#!/usr/bin/env bash
# 云恢复：云端 → 本地（临时目录解压 + rsync --delete 对齐）
# 用法: bash restore.sh [项目目录]
set -euo pipefail

CLOUD_ROOT="/mnt/d/华为云盘"
PROJECT_DIR="${1:-$PWD}"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
TAR_NAME="${PROJECT_NAME}.tar.gz"
TMP_TAR="/tmp/$TAR_NAME"
TMP_RESTORE="/tmp/${PROJECT_NAME}-restore"
CLOUD_TAR="$CLOUD_ROOT/$TAR_NAME"

# 1. 检查云端是否存在
if [ ! -f "$CLOUD_TAR" ]; then
  echo "[ERR] 云端备份文件不存在：$CLOUD_TAR"
  echo "[ERR] 请先执行云备份"
  exit 1
fi
echo "[INFO] 云端备份大小：$(du -h "$CLOUD_TAR" | cut -f1)"

# 2. 下载
echo "[INFO] 下载云端备份 ..."
rsync --progress "$CLOUD_TAR" "$TMP_TAR"
RC=$?
if [ $RC -ne 0 ]; then
  echo "[ERR] 下载失败，rsync 退出码: $RC"
  exit $RC
fi
echo "[INFO] 下载完成"

# 3. 解压到临时目录
rm -rf "$TMP_RESTORE"
mkdir -p "$TMP_RESTORE"
echo "[INFO] 解压到临时目录 $TMP_RESTORE ..."
tar xzf "$TMP_TAR" -C "$TMP_RESTORE" --totals || {
  echo "[ERR] 解压失败"
  rm -f "$TMP_TAR"
  rm -rf "$TMP_RESTORE"
  exit 1
}
echo "[INFO] 解压完成"

# 4. rsync --delete 对齐（排除隐藏文件，保护 .git）
echo "[INFO] rsync --delete 同步到 $PROJECT_DIR ..."
rsync -a --delete --exclude='.*' "$TMP_RESTORE/$PROJECT_NAME/" "$PROJECT_DIR/"
RC=$?
if [ $RC -ne 0 ]; then
  echo "[ERR] 同步失败，rsync 退出码: $RC"
  rm -f "$TMP_TAR"
  rm -rf "$TMP_RESTORE"
  exit $RC
fi
echo "[INFO] 对齐完成：本地已完全匹配云端备份"

# 5. 清理
rm -f "$TMP_TAR"
rm -rf "$TMP_RESTORE"
echo "[INFO] 临时文件已清理"
echo "[INFO] 恢复完成！本地已完全对齐到云端状态。"
