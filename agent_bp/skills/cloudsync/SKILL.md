---
name: cloudsync
description: 项目打包同步/恢复到云端。脚本见 backup.sh / restore.sh。适用于'云备份'、'云恢复'等指令。
metadata:
  variables:
    - CLOUD_ROOT: 云端同步根目录（默认 /mnt/d/华为云盘）
    - PROJECT_DIR: 当前项目目录（默认 $PWD）
    - TAR_NAME: 备份文件名，格式 <项目名>.tar.gz
    - TMP_TAR: 临时路径 /tmp/<TAR_NAME>
---

# CloudSync — 项目云同步

## 触发信号

| 用户用语 | 操作 |
|:---|:---|
| "云备份" / "备份到云端" | 本地 → 云端（打包 + rsync 上传） |
| "云恢复" / "从云端恢复" | 云端 → 本地（临时目录解压 + rsync --delete 对齐） |

---

## 流程：云备份（本地 → 云端）

执行 `backup.sh`，两步完成：

```bash
bash .agents/skills/cloudsync/backup.sh   # 在当前项目目录执行
```

## 流程：云恢复（云端 → 本地）

使用**临时目录解压 → rsync --delete 对齐**方案，安全地将本地对齐到云端状态：

```bash
bash .agents/skills/cloudsync/restore.sh   # 在当前项目目录执行
```

---

## 注意

- 打包排除所有隐藏文件/目录（`--exclude='.*'`），包括 `.git`
- 操作前最好先 `git commit` 确保工作区干净
- 云端目标路径：`/mnt/d/华为云盘/<项目名>.tar.gz`
- `ls -la /mnt/d/` 会触发 Windows 系统目录（`System Volume Information`、`WindowsApps`）的权限拒绝错误，**直接检查 `$CLOUD_ROOT` 即可**，勿扫描父目录
- 恢复用 `rsync -a --delete --exclude='.*'`，保护 `.git` 等隐藏文件
- 若解压失败 → 本地不受影响
