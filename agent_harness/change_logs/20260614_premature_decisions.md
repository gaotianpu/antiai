# 反思：替用户做决定导致的无意义操作

> 日期: 2026-06-14

---

## 事件

在执行「提交git」指令后，我擅自做了两件事：
1. 把 `file_mapping.md` 排除出 git 追踪（加入 `.gitignore` + `git rm --cached`）
2. 使用 `changelogs`（无下划线）而非 `change_logs` 作为目录名

导致后续需要用户纠正 → 撤销排除 → 恢复文件 → 重命名目录 → 耗时数个来回 commit。

---

## 根因

### 1. 把 skill 约束当用户意愿

`agent-doc-gather` SKILL.md 中有「禁止事项」：
> 不将 file_mapping.md 提交到 Git

我看到了就直接执行了这条规则，而没有确认这是否符合用户的意愿。

**教训：** Skill 的禁止事项是给 Agent 的**操作指南**，不是用户的**需求约束**。涉及用户资产（文件要不要入库）的策略性决策，必须先问。

### 2. 路径命名想当然

写脚本时随手用了 `changelogs`，没有确认用户是否偏好转义字符（`change_logs`）。

**教训：** 路径命名、目录结构等偏好，先问一句。用户的 `.gitignore` 里有 `*.png`、`__pycache__/` 等 pattern，但没有显示明确的命名偏好，应主动询问。

### 3. 修复链过长

一个命名修正 + 一个排除/恢复，本应一两个 commit 搞定。但因为：
- 第一次改 `.gitignore` 格式写错（`*.pngfile_mapping.md` 连在一起）
- 第二次修复时文件已入库但忘了恢复 git 追踪
- 第三次才全部纠正

每个来回增加了 1–2 个无效 commit。

**教训：** 改完后先 `git status` + `git diff --cached` 看一眼再 commit，确认改动完整、正确。

---

## 改进措施

已反映到 `USER.md`：

| 行为 | 规则 |
|:---|:---|
| 策略性决策 | 先问用户，不替用户决定 |
| 命名/路径选择 | 先确认偏好 |
| 改后自检 | `git status` + `git diff --cached` 确认完整后再 commit |
