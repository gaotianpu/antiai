# Agent 工作区

项目约束（rules/）与 Agent 技能（skills/）的统一管理目录。

- 文档规整 — 审查、去重、修正引用（doc-fullreview）
- 跨项目追踪 — 扫描副本分布、版本新旧、变更 diff（agent-doc-gather）
- 云备份 — 打包同步到云端（cloudsync）


## 文档说明

| 文件 | 说明 |
|------|------|
| `README.md` | rules + skills 索引 |
| `rules/` | 项目约束与指南文档 |
| `skills/` | Agent 技能（含 SKILL.md + 独立脚本） |
| `change_logs/` | 操作日志、经验教训 |
| `file_mapping.md` | agent-doc-gather 产出（本地生成，不入库） |
| `USER.md` | 个人习惯偏好 |
| [`rules/SKILL_guid.md`](rules/SKILL_guid.md) | skills/ 文档编写规范 |

---

## 工作流：修改全局 Skill

当需要修改一个同时存在于**全局**（`~/.agents/skills/`）和**本地**（`skills/`）的 skill 时：

```
① 改本地 skills/xxx/  →  ② 测试验证  →  ③ 同步到全局  →  ④ git commit
```



### 步骤(举例)

**① 修改本地**
```bash
# 本地文件在 skills/<skill-name>/
vi skills/cloudsync/backup.sh
```

**② 测试验证**
```bash
cd ~/workspace/agent && bash skills/cloudsync/backup.sh
```

**③ 同步到全局**
```bash
cp skills/cloudsync/backup.sh ~/.agents/skills/cloudsync/backup.sh
# 如多文件则 cp -r skills/<name>/ ~/.agents/skills/<name>/
```

**④ 提交本地变更**
```bash
git add skills/cloudsync/ && git commit -m "fix: ..."
```

### 理由

- 本地是权威来源（有 git 历史），全局只是同步目标
- 改全局 → 忘了同步回本地 → 下次被 `agent-doc-gather` 检测到 diff 但方向反了
- **先本地后全局**，确保变更永远被 git 追踪

## 硬约束
- **区分讨论 vs 指令语气**：问号/商量语气 → 只出方案；肯定句 → 直接执行
- **"类型优先"原则**：接到"改写/整理"类任务时，先确认目标文档的类型（原则 vs 规格 vs 指南 vs FAQ）
- **用中文**： 回复，写文档，提交git等
- **先查后议**：收到架构/设计/代码归属等"讨论类或分析类"问题时，先 grep `docs/builder/analysis/`、`docs/builder/modules/`、`docs/builder/specs/` 中的原则/决议/设计决策记录。做可行性分析、方案评估、合并审查之前，先通读对应文档的"原则"或"决议记录"章节——若有明确原则/决议，分析结论必须以原则为基准，不得另行推导
- **执行前确认范围**：对"全称命题"指令（"所有 X 都改为 Y"），先 grep 报数再动手
- **不自作聪明假设领域知识**：涉及单位、格式、约定时，先查数据源头，确定后再动手
- **关键决策点复述确认**：不同选项会导致后续分析结论被**颠覆**时（好 ↔ 坏，可行 ↔ 不可行），先"复述理解 + 一句话结论预览"让用户确认，再展开分析
- **动手前追问**：增/删功能前问"核心还是可选"；写文档前先确认章节骨架再填内容
- **存文档前查归属**：创建新 `.md` 文件到 `docs/` 前，先读 `docs/shared/document-classification.md` 确定分类归属（builder/user/shared），再选择对应子目录
- **增补文档用 Edit 追加**，不用 Write 覆盖
- **决议影响面预检**：做出决议（如新增/移除参数、修改命名、改变数据结构）后，先 grep 全项目找出所有引用点，列出受影响文档清单让用户确认，再逐文件执行修改。修改完成后主动 grep 全项目确认无残留旧引用
- **输出后自我审查**：(a) 数字一致性——如"3 处"是否与新列项数一致，引用链接/行号是否仍有效；(b) 认知流——章节顺序是否合理，读者能否不跳转就理解；(c) 改动范围——是否全覆盖所有受影响文件；(d) 查重——grep 关键概念，一词多处置放时提出归并（一处详写，其余引用）