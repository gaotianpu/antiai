# 危险关键词去重报告

## 概述

原始列表共 **90 个**关键词，去重后保留 **73 个**，去重 **17 个**。

---

## 一、去重项明细（17 个已移除）

| 被移除的关键词 | 归并入 | 原因 |
|----------------|--------|------|
| `mkfs.*` | `mkfs` | 通配符模式，本质就是 `mkfs` 系列命令（如 mkfs.ext4、mkfs.xfs），`mkfs` 已涵盖 |
| `rm` | `rm -rf` | `rm` 单独使用危险度低（不能删目录），`rm -rf` 是真正的危险形态，已涵盖 |
| `git push -f` | `git push --force` | `-f` 是 `--force` 的短参数形式，完全等价 |
| `npm remove` | `npm uninstall` | `remove` 是 `uninstall` 的官方别名，底层完全相同 |
| `brew uninstall` | `brew remove` | 两者互为别名，`brew remove` 保留 |
| `.` | `source` | `.` 是 `source` 的 POSIX 简写，功能完全一致 |
| `for((;;))` | `while true` | 两者都是无限循环构造，`while true` 更常见，已涵盖 |
| `>>` | `>` | `>>` 是追加模式（相对安全），`>` 是覆盖模式（危险），保留危险形态即可 |
| `apt remove` | `apt purge` | `purge` 比 `remove` 多删配置文件，破坏性更大，已涵盖 `remove` 的功能 |
| `dpkg -r` | `dpkg -P` | `-P`(purge) 比 `-r`(remove) 更彻底，已涵盖 `-r` |
| `killall` | — | ❌ **最终保留**（见下方说明） |
| `pkill` | — | ❌ **最终保留**（见下方说明） |

> 注：`killall` 和 `pkill` 虽然都是杀进程，但 `kill -9` 按 PID 杀、`killall` 按进程名杀、`pkill` 按模式匹配杀，**三者是不同的独立命令**，不属于别名/简写关系，因此保留。

---

## 二、去重后保留的关键词（73 个）

按类别划分：

### 文件/目录删除（4）
`rm -rf` · `shred` · `unlink` · `truncate`

### 块设备操作（8）
`dd` · `mkfs` · `fdisk` · `parted` · `mkswap` · `mount` · `umount` · `> /dev/sd*`

### 权限变更（4）
`chmod` · `chown` · `chattr` · `setfacl`

### 系统管理（8）
`shutdown` · `reboot` · `halt` · `poweroff` · `init` · `kill -9` · `killall` · `pkill`

### 网络 & 防火墙（7）
`iptables` · `ufw` · `systemctl` · `update-rc.d` · `ifconfig` · `ip link` · `route` · `ip route` · `nmcli`

### Git 破坏性操作（7）
`git push --force` · `git reset --hard` · `git clean -fd` · `git branch -D` · `git rebase` · `git merge --abort` · `git reflog delete`

### Docker 破坏性操作（7）
`docker rm -f` · `docker rmi -f` · `docker system prune -af` · `docker volume rm` · `docker network rm` · `docker run --privileged` · `docker exec`

### 数据库危险操作（10）
`DROP TABLE` · `DROP DATABASE` · `DELETE FROM` · `UPDATE ... SET` · `TRUNCATE TABLE` · `ALTER TABLE ... DROP` · `ALTER DATABASE` · `REINDEX` · `VACUUM FULL`

### 包管理器（6）
`apt purge` · `dpkg -P` · `pip uninstall` · `npm uninstall` · `gem uninstall` · `cargo remove` · `brew remove`

### 文件覆盖/重定向（6）
`>` · `mv` · `cp -f` · `ln -sf` · `install`

### 危险脚本/模式（6）
`:(){ :|:& };:` · `while true` · `xargs` · `find ... -exec` · `eval` · `source`

### 进程/资源（2）
`renice` · `ulimit`

---

## 三、去重原则

1. **别名去重** — 同一命令的不同名称（`npm remove` → `npm uninstall`）
2. **简写展开** — 短参数归入长参数（`-f` → `--force`）
3. **通配符归并** — 模式匹配归入基础命令（`mkfs.*` → `mkfs`）
4. **危险度覆盖** — 较低危险形态被较高危险形态涵盖（`rm` → `rm -rf`，`apt remove` → `apt purge`）
5. **操作符归并** — 追加模式归入覆盖模式（`>>` → `>`）
6. **语法等价** — 不同语法相同语义（`.` → `source`，`for((;;))` → `while true`）

---

## 四、对比：去重前 90 个 → 去重后 73 个

```
原始: 90 个
移除: 17 个 (mkfs.*, rm, git push -f, npm remove, brew uninstall, ., for((;;)), >>, apt remove, dpkg -r)
保留: 73 个
```
