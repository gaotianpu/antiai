# 对 AI Agent 而言的危险命令清单

## 🔴 极度危险（禁止使用）

### 1. 毁灭性删除
| 命令 | 风险 |
|------|------|
| `rm -rf /` / `rm -rf /*` | 删除整个文件系统 |
| `rm -rf .` / `rm -rf *` | 删除当前目录所有内容（在错误目录下灾难性）|
| `rm -rf --no-preserve-root /` | 绕过保护删除根目录 |
| `shred /dev/sda` | 安全擦除整个磁盘 |

### 2. 直接操作块设备
| 命令 | 风险 |
|------|------|
| `dd if=/dev/zero of=/dev/sda` | 覆写整个磁盘 |
| `dd if=/dev/urandom of=/dev/sda` | 随机数据覆写磁盘 |
| `mkfs.ext4 /dev/sda` / `mkfs.*` | 格式化分区 |
| `fdisk` / `parted` 写入操作 | 修改分区表 |
| `> /dev/sda` | 直接写入块设备 |

### 3. 权限/所有权灾难
| 命令 | 风险 |
|------|------|
| `chmod -R 777 /` | 所有文件可执行，安全灾难 |
| `chown -R user:user /` | 改变系统文件所有权 |
| `chmod 000 /etc/shadow` | 锁定关键系统文件 |

### 4. 系统管理破坏
| 命令 | 风险 |
|------|------|
| `shutdown` / `reboot` / `halt` / `poweroff` | 关闭/重启系统 |
| `init 0` / `init 6` | 切换运行等级导致关机 |
| `kill -9 1` | 杀掉 init/systemd，系统崩溃 |
| `kill -9 -1` | 杀掉所有可杀进程 |
| `iptables -F` | 清空防火墙规则，暴露系统 |

---

## 🟡 高风险（需仔细审查）

### 5. Git 破坏性操作
| 命令 | 风险 |
|------|------|
| `git push --force` / `git push -f` | 强制推送，覆盖远程历史 |
| `git reset --hard HEAD~N` | 丢弃提交历史 |
| `git clean -fd` | 删除所有未跟踪文件 |
| `git branch -D <branch>` | 删除分支（无法恢复） |
| `git rebase --abort` / `git merge --abort` | 中断操作可能导致状态混乱 |

### 6. Docker 破坏性操作
| 命令 | 风险 |
|------|------|
| `docker rm -f $(docker ps -aq)` | 删除所有运行中的容器 |
| `docker rmi -f $(docker images -q)` | 删除所有镜像 |
| `docker system prune -af` | 清理所有未使用资源（含数据卷） |
| `docker run --privileged` | 赋予容器宿主机 root 权限 |

### 7. 数据库危险操作
| 命令 | 风险 |
|------|------|
| `DROP TABLE` / `DROP DATABASE` | 删除表/数据库 |
| `DELETE FROM table` (无 WHERE) | 删除全表数据 |
| `UPDATE table SET ...` (无 WHERE) | 更新所有行 |
| `TRUNCATE TABLE` | 清空表（无法回滚） |
| `ALTER TABLE ... DROP COLUMN` | 删除列 |

### 8. 包管理器破坏
| 命令 | 风险 |
|------|------|
| `apt remove --purge` 系统关键包 | 移除核心组件 |
| `pip uninstall` 关键包（如 pip 自身）| 破坏 Python 环境 |
| `npm uninstall` 核心依赖 | 破坏 Node 项目 |

---

## 🟠 中等风险（需确认上下文）

### 9. 网络变更
| 命令 | 风险 |
|------|------|
| `ifconfig down` / `ip link set down` | 关闭网络接口 |
| `route del default` | 删除默认路由 |
| `ufw disable` / `systemctl stop firewalld` | 关闭防火墙 |

### 10. 文件覆盖/重定向
| 命令 | 风险 |
|------|------|
| `>` 重定向到已有文件 | 覆盖文件内容 |
| `mv` 到已有文件 | 覆盖目标文件 |
| `cp -f` 覆盖已有文件 | 静默覆盖 |
| `ln -sf` | 覆盖符号链接目标 |

### 11. Fork Bomb / DoS
```bash
:(){ :|:& };:              # bash fork bomb
while true; do echo; done  # 无限循环
cat /dev/zero > /tmp/big   # 填满磁盘
```

---

## ✅ 安全原则总结

1. **涉及 `rm -rf` 的一律要三思**，特别是路径以 `/` 开头或包含 `..` 的情况
2. **强制推送 (`--force`) 需要人类确认**，除非明确知晓后果
3. **数据库 DDL/DML 无 WHERE 子句** → 必须加 WHERE 或使用事务包裹
4. **操作块设备 (`/dev/sd*`, `/dev/nvme*`)** → 几乎永远不应该由 Agent 执行
5. **修改系统配置（网络、防火墙、服务）** → 先确认当前状态和预期影响
6. **批量操作容器/包** → 先 `--dry-run` 或列出受影响对象

**建议的防护措施：**
- 在 Agent 的 Tool 实现层加入**黑名单/白名单**过滤
- 关键操作前要求**人类确认（Human-in-the-loop）**
- 危险命令自动添加 `--dry-run` / `--interactive` 标志
- 使用 `trash-cli` 替代 `rm`（可恢复删除）
