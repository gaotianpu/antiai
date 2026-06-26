# 团队 + AI Agent 协作指南

多成员、多 AI Agent 协同开发同一仓库的完整约定与操作手册。

---

## 1. 核心原则

**`AGENTS.md` 是最高契约**。所有 Agent 启动时读取 AGENTS.md 作为行为准则，因此：

- AGENTS.md 必须精炼、无歧义、覆盖所有硬约束
- 任何新规则或约定变更，必须同步更新 AGENTS.md（或通过 AGENTS.md 指向本文档）
- 禁止 Agent 绕过 AGENTS.md 自行发挥

**项目采用人+AI多角色分工**，详见 `docs/rules/roles.md`。Agent 启动时先确定角色，只加载对应指令文件。

---

## 2. 环境与工具

### 2.1 Python 虚拟环境

每位开发者本地创建统一环境：

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2.2 Docker 标准化 (推荐)

跨平台开发的一致性保障：

```yaml
# docker-compose.yml
services:
  dev:
    build:
      dockerfile: Dockerfile.dev
    volumes:
      - .:/workspace
    environment:
      - PYTHONPATH=/workspace
    command: sleep infinity
```

### 2.3 pre-commit 钩子

统一代码质量门槛：

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: ruff check .
        language: system
      - id: ruff-format
        name: ruff format --check
        entry: ruff format --check .
        language: system
```

安装：`pre-commit install`

### 2.4 提交前验证命令 (必须通过)

```bash
ruff check . && ruff format --check .
mypy trade_os.py apps/ business/ orchestration/ common/
python -m pytest test/
```

---

## 3. 代码一致性

### 3.1 宪法必须强制执行

所有 Agent 编写的代码必须符合 `docs/rules/unix_pythonic_guid.md`（架构宪法）。审查清单见 **[§4.3.2 Agent 审查工作流](#432-agent-审查工作流)** step 2。

### 3.2 Lint & Typecheck

每个 Agent 完成任务后必须运行 lint/typecheck。命令见 §2.4。

### 3.3 风格统一

- 遵循项目已有代码风格（读周围文件，模仿模式）
- 使用已有库和工具，不引入新依赖（除非明确需要）
- 命名遵循项目现有约定
- 禁止添加不必要的注释

---

## 4. Git 管理

### 4.1 分支策略

分支命名格式：`<用户名>/<类型>/<描述>`

```
main                                   # 稳定分支，禁止直接提交
├── zhangsan/feat/delta_neutral        # 张三 — 新功能
├── zhangsan/fix/exposure_bug          # 张三 — 修复
├── lisi/docs/update_index             # 李四 — 文档
├── lisi/refactor/order_manager        # 李四 — 重构
└── wangwu/feat/high_freq              # 王五 — 新功能
```

**类型**: `feat` / `fix` / `refactor` / `docs` / `chore` / `test`

**规则**:
- **禁止直接提交到 main**
- 每个成员在自己的分支上开发，每个任务一个分支
- 分支名使用蛇形小写（snake_case）
- 提交到 main 前必须通过 PR 评审
- 合并后删除远程特性分支

### 4.2 Commit Message

统一使用英文前缀格式：

```
<type>: <简短描述>

<可选详细说明>
```

**Type 定义**:

| 前缀 | 含义 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `refactor` | 重构（无功能变更） |
| `docs` | 文档变更 |
| `chore` | 杂项（配置、脚本、依赖） |
| `test` | 测试相关 |

**禁止事项**:
- 禁止 `git push --force` 到 main 或共享分支
- 禁止提交 secrets（`.env`, `credentials.json` 等）
- commit message 使用中文

### 4.3 PR 流程

#### 4.3.1 作者侧：发起 PR

1. 功能完成后发起 PR
2. PR 描述中写明：做了什么、为什么做、测试情况
3. 至少一人评审通过后合并
4. 合并后删除特性分支

#### 4.3.2 Agent 审查工作流

```
发现待审查分支/PR
    │
    ▼
┌─────────────────────────────┐
│ 1. 信息收集                   │
│   - git fetch origin         │
│   - git log <base>..<head>   │
│   - git diff <base>...<head> │
│   - 识别改动范围和影响模块    │
└─────────┬───────────────────┘
          ▼
┌─────────────────────────────┐
│ 2. 宪法审查（必须，一票否决）  │
│  对照 unix_pythonic_guid.md: │
│  速查清单(12条) → 禁止行为   │
│  → 设计规则 → 代码组织       │
│  详见 §4.3.3 宪法审查嵌入流程│
│  任一违规 → Request Changes  │
└─────────┬───────────────────┘
          ▼
┌─────────────────────────────┐
│ 3. 代码质量审查               │
│   - [ ] 风格模仿现有代码      │
│   - [ ] 未引入不必要的新依赖  │
│   - [ ] 命名遵循项目约定      │
│   - [ ] 无冗余/误导性注释     │
│   - [ ] lint/typecheck 通过   │
└─────────┬───────────────────┘
          ▼
┌─────────────────────────────┐
│ 4. 结构正确性                 │
│   - [ ] 模块放在正确目录下    │
│   - [ ] 未滥用继承 (组合优先)  │
│   - [ ] 依赖方向正确          │
│   - [ ] 错误处理显式且有信息量 │
│   - [ ] 无隐式行为/副作用     │
└─────────┬───────────────────┘
          ▼
┌─────────────────────────────┐
│ 5. 测试完整性                 │
│   - [ ] 新功能有对应测试      │
│   - [ ] 纯函数有单元测试      │
│   - [ ] 现有测试未退化        │
└─────────┬───────────────────┘
          ▼
┌──────────────────────────────────┐
│ 6. 决策                           │
│   ┌─────────────┬────────────────┐│
│   │ 全部通过     │ Approve        ││
│   │             │ → Squash Merge ││
│   ├─────────────┼────────────────┤│
│   │ 小问题       │ Request Changes ││
│   ├─────────────┼────────────────┤│
│   │ 严重违宪     │ Reject          ││
│   ├─────────────┼────────────────┤│
│   │ 业务逻辑     │ 标记 need-human ││
│   │ 不确定       │ → 只审结构合规  ││
│   └─────────────┴────────────────┘│
└─────────┬──────────────────────────┘
          ▼
  记录决策到 PR comment 及 docs/log/log.YYYY-MM-DD.md
```

#### 4.3.3 决策矩阵

| 结果 | 条件 | 动作 |
|------|------|------|
| **Approve** | 全部 checklist 通过，宪法无违规，结构正确 | Squash & Merge 入 main |
| **Request Changes** | 命名/风格/小范围逻辑问题，但总体方向对 | PR comment 指明修改点 |
| **Reject** | 严重违宪（上帝类/全局状态/模块放错目录/引入不必要依赖） | 标注具体违规条款 + 修正建议 |
| **Need Human** | 涉及业务逻辑正确性（策略盈亏/参数合理性/新依赖安全性） | Agent 只审结构合规，标记需人类 review |

#### 4.3.4 Agent 审查能力边界

Agent **能做**：
- 宪法合规性检查（无上帝类、无全局状态、无单例等）
- 模块放置验证（对照识别矩阵和决策树）
- 依赖方向和注入方式检查
- 代码风格一致性（模仿邻近文件）
- lint/typecheck 结果验证
- 文件/函数长度检查

Agent **不能做**（必须由人类判断）：
- 交易策略有效性 — 策略盈亏逻辑需要领域专家
- 业务参数合理性 — 如保证金阈值、仓位比例、风控线
- 新增第三方依赖的安全性审计
- 跨模块业务语义正确性

> Agent 审查时应在 PR comment 中明确区分"结构合规结论（可自动判断）"和"需人类确认的部分"。

#### 4.3.4 审查输出格式

```markdown
## 宪法审查结果

### 自动判定（结构合规）
- [x] 无上帝类/全局状态/单例 — 通过
- [x] 依赖显式注入 — 通过
- [ ] 存在 1 处 except Exception: pass — 第8条违规 (path/to/file.py:42)

### 需人类确认（Agent 无法判断）
- 策略参数阈值是否合理？
- 新增依赖安全性审计
```

### 4.4 Merge 策略

| 场景 | 策略 | 命令 |
|------|------|------|
| 特性分支 → main | **Squash & Merge** | GitHub "Squash and merge" |
| 长期分支同步 main | **Rebase** | `git rebase main` 后 `git push --force-with-lease` |

**禁止**：
- 禁止对已推送的共享分支做 rebase
- 禁止 Merge Commit 合入 main（统一用 Squash）
- 禁止 `git push --force` 到 main

### 4.5 Release Tag

语义化版本：`v<MAJOR>.<MINOR>.<PATCH>`

| 数字 | 何时递增 |
|------|---------|
| MAJOR | 不兼容的 API 变更 |
| MINOR | 向下兼容的新功能 |
| PATCH | 向下兼容的 Bug 修复 |

```bash
git checkout main && git pull
git tag -a v0.1.0 -m "v0.1.0: 首个功能集"
git push origin v0.1.0
```

- 只有 main 分支能打 tag
- tag 必须带注释（`-a`），写版本变更摘要
- 禁止删除或移动已推送的 tag

### 4.6 Hotfix 流程

**触发条件**：生产环境 Bug、资金安全风险、数据损坏

```bash
git checkout main
git checkout -b zhangsan/hotfix/<描述>
git commit -m "fix: <问题描述>"
# 发起 PR → main（标注 HOTFIX），加速评审
# Squash & Merge → main
git tag -a v0.1.2 -m "hotfix: <修复摘要>"
git push origin v0.1.2
```

- hotfix 分支必须从 main 切出，commit 类型用 `fix`
- 合并后必须立即打 tag 并部署
- 通知相关成员 rebase 进行中的分支

### 4.7 文件锁

项目中已有文件锁机制（FRA alignment 和 fix_exposure 共用）。**Agent 不应绕过或移除现有锁机制**。

---

## 5. 需求管理

### 5.1 五阶段生命周期

```
需求捕获  →  需求讨论  →  需求对齐  →  规格固化  →  实现跟进
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
backlog.md  discussions/ reviews/  specs/     <!-- TODO.md → 已合并至 backlog.md -->
 (池子)     (对话记录)  (评审)    (规约)     (状态)
```

### 5.2 需求条目 (`backlog.md`) 格式

```markdown
# TODO

## 进行中
| 编号 | 需求 | 负责人 | AI Agent | 开始日期 | 预期完成 |
|------|------|--------|----------|----------|----------|
| REQ-001 | Delta中性策略 | zhangsan | OpenCode | 05-12 | 05-15 |

## 待开始
| 编号 | 需求 | 优先级 | 前置依赖 |
|------|------|--------|----------|
| REQ-002 | 风控模块实现 | P0 | 无 |

## 已完成 (本周)
| 编号 | 需求 | 完成日期 | 分支 | PR |
|------|------|----------|------|----|
| REQ-000 | 示例 | 05-11 | zhangsan/feat/demo | #42 |
```

每条任务标注责任人/Agent 和状态。开始任务前标注"进行中"，完成后在 `backlog.md` 标记并追加记录到 `docs/log/`。

### 5.3 需求讨论记录

人+AI 的需求讨论沉淀到 `docs/discussions/{主题}.md`，记录：背景、讨论过程（含人/AI发言）、结论、行动项、关联文档。讨论模板见 `docs/discussions/` 目录。

### 5.4 需求固化为规格

讨论结束后，产出规格文档到 `docs/specs/`，包含：目标、范围、功能描述、接口定义、技术约束、验收标准。规格文档是开发的基准。

### 5.5 规则

- **新功能**: 必须经过「讨论→对齐→固化」三步才能开始编码
- **Bug修复/重构**: 允许走简化流程（直接分支+PR）
- **需求变更**: 必须更新对应规格文档
- **人类最终决策**: AI 提供分析和建议，人类做决定

---

## 6. 知识管理

### 6.1 知识分级体系

| 层级 | 载体 | 面向 | 描述 |
|------|------|------|------|
| **L1 入口** | `AGENTS.md` | 所有 AI Agent | 项目全景 + 角色分工 |
| **L2 宪法** | `docs/rules/unix_pythonic_guid.md` | 全体 | 架构宪法 |
| **L3 规约** | `docs/specs/` | 策略开发者 | 一页式规约 + 策略设计 |
| **L4 决策** | `docs/decisions/` (ADR) | 全体 | 每个重大设计决策的记录 |
| **L5 日志** | `docs/log/` | 团队 | 每日任务摘要 |
| **L6 评审** | `docs/reviews/` | 团队 | 设计评审和讨论 |

### 6.2 ADR (架构决策记录)

在 `docs/decisions/` 下按编号记录重大设计决策，回答四个基本问题：背景（为什么做）、决策（选了什么）、后果（正面/负面影响）、备选方案（其他选项及否决原因）。

### 6.3 术语表

`docs/glossary.md` 统一项目术语，消除跨人、跨AI的理解偏差。随代码新增概念同步更新。

### 6.4 文档元数据

所有 `docs/` 下的 `.md` 文件头部必须包含：

```markdown
<!--
创建时间: YYYY-MM-DD
最近修改: YYYY-MM-DD
修改记录:
  YYYY-MM-DD: 描述修改内容
-->
```

### 6.5 新人/新AI 阅读路径

```
docs/specs/overview.md              — 一页式规约总览
  → docs/architecture.md            — 架构模块边界
    → docs/rules/unix_pythonic_guid.md  — 架构宪法
      → docs/rules/team_agent_guide.md  — 本文档
```

---

## 7. 文档维护

### 7.1 索引统一入口

`docs/README.md` 是唯一的文档索引。任何增删 .md 文件后，Agent 应使用 `updatedocs` skill 更新索引。

### 7.2 工作日志

`docs/log/log.YYYY-MM-DD.md` 记录每天的任务摘要。**每个 Agent 完成任务后追加一条记录**，格式：

```markdown
## HH:MM — 简要标题

**任务**: <描述做什么>
**关键变更**:
- <变更1>
- <变更2>
```

### 7.3 异步站会信息源

无需同步会议，从仓库即可获取当日进展：

| 信息 | 来源 |
|------|------|
| 谁做了什么 | `git log --since="1 day ago" --oneline` |
| 任务进度 | `backlog.md` 状态字段 |
| 完成摘要 | `docs/log/log.YYYY-MM-DD.md` |
| 讨论中的需求 | `docs/discussions/` |
| 设计决策 | `docs/decisions/` |

### 7.4 评审记录

涉及设计/架构讨论时，记录到 `docs/reviews/`，确保决策过程可追溯。

---

## 8. 沟通与冲突避免

### 8.1 Agent 间异步沟通

| 机制 | 内容 | 读取者 |
|------|------|--------|
| `git log` | commit message (英文前缀) | 全体 |
| `docs/log/` | 每日任务摘要 | 全体 |
| `backlog.md` | 待办 & 搁置清单 | 全体 |
| `docs/discussions/` | 需求讨论过程 | 需要理解需求的开发者 |
| `docs/specs/` | 已固化的规格文档 | 实现者 |
| `docs/decisions/` (ADR) | 设计决策及理由 | 全体 |
| `docs/reviews/` | 评审记录 | 需要理解背景的开发者 |
| PR description | 功能级变更说明 | Reviewer |

### 8.2 人类间同步

- 重大架构调整需 `docs/reviews/` 记录评审过程
- 新成员加入按 §6.5 阅读路径上手
- 定期 review AGENTS.md 的有效性，视情况更新

### 8.3 热点文件与并行协调

以下文件多 Agent 频繁修改，冲突概率高：

| 文件 | 原因 | 应对 |
|------|------|------|
| `apps/__init__.py` | 注册新 App | 修改前拉取最新 |
| `docs/README.md` | 文档索引更新 | 使用 `updatedocs` skill 统一维护 |
| `AGENTS.md` | Agent 指令更新 | 专人负责修改，或在共识后统一改 |
| `backlog.md` | 待办 & 搁置清单 | 标注责任人，改前检查他人状态 |
| `config/` | 配置文件 | 修改前确认不影响他人 |

冲突处理原则：
- 各 Agent 尽量操作不同子目录/模块
- 多人改同一模块时，使用 `git pull --rebase`
- 出现冲突时理解双方意图后手工合并，不要盲目覆盖

---

## 9. 常见问题

### Q: 两个 Agent 同时改了同一个文件怎么办？
A: 后提交的 Agent 会遇到 push 被拒。执行 `git pull --rebase`，手工解决冲突后重试。不要盲目覆盖对方修改。

### Q: Agent 引入了不符合宪法的代码怎么办？
A: PR 评审阶段应捕获。评审人对照宪法 checklist 检查。如果已合并，开 `fix/` 分支修复。

### Q: 如何确保不同 Agent 模型间的行为一致？
A: AGENTS.md 是唯一约束。不同模型可能理解有差异，但 AGENTS.md 越精确，差异越小。必要时可增加示例代码片段。

### Q: 新成员如何快速上手？
A: 从 `docs/README.md` 按 §6.5 顺序阅读核心文档，然后运行 `python trade_os.py` 确认环境可用。

---

## 10. AGENTS.md 检查清单

团队项目中，AGENTS.md 必须包含以下条目：

- [ ] 架构速览（目录结构、核心模式）
- [ ] 角色分工（引用 `roles.md`）
- [ ] 代码风格硬约束（宪法）
- [ ] Lint/typecheck 命令
- [ ] 常用命令（启动、测试、调试）
- [ ] 任务完成后的 log 记录规则
- [ ] Git 分支/提交约定（指向本文档）
- [ ] 明确禁止项（secrets、force push 等）
- [ ] 入口文档索引
