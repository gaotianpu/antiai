# Agent 资源目录索引

> 收集各项目的 agent skills 和项目约束文档。

---

## 目录结构

```
agent/
├── README.md          ← 本文件
├── rules/             ← 18 个约束/指南文档（平铺，含合并附录）
└── skills/            ← 21 个 agent skill
```

---

## rules/ — 项目约束与指南

| 文件 | 大小 | 说明 |
|:---|---:|:---|
| `AGENTS_guid.md` | 74 行 | 编写 AGENTS.md 最佳实践 |
| `SKILL_guid.md` | 220 行 | Skill 编写最佳实践指南 |
| `unix_pythonic_guid.md` | 482 行 | Unix + Pythonic 架构宪法（最高指导原则） |
| `unit_test_guid.md` | 607 行 | Python 单元测试最佳实践 |
| `async_guid.md` | 907 行 | Python Async/Await 生产级最佳实践 |
| `logging_guid.md` | 241 行 | 日志原则 |
| `message_guid.md` | 216 行 | 消息通知原则 |
| `daemon_guid.md` | 414 行 | Python 常驻进程设计开发原则 |
| `app_doc_lifecycle.md` | 302 行 | APP 文档生命周期：职责划分与模板骨架 |
| `team_agent_guide.md` | 523 行 | 团队 + AI Agent 协作指南 |
| `ai_human_guid.md` | 195 行 | 人机交互理解偏差案例与规避建议 |
| `memory_guid.md` | 160 行 | Python Async 内存优化最佳实践 |
| `requirements_writing_principles.md` | 639 行 | 需求编写原则 |
| `DANGEROUS_COMMANDS.md` | 116 行 | 对 AI Agent 而言的危险命令清单 |
| `DANGEROUS_KEYWORDS.md` | 78 行 | 危险命令关键词，安全预检用 |
| `DANGEROUS_KEYWORDS_REPORT.md` | 89 行 | 危险关键词去重报告 |
| `hypothesis.md` | 65 行 | 假设评估准则文档 |

---

## skills/ — Agent Skill 清单

### 公共可用
| Skill | 描述 | 触发关键词 |
|:---|---|:---|
| `doc-fullreview` | 对文档进行全文规整审查 | 文档规整、全文规整、文档审查、消除重复 |
| `workflow-capture` | 将人机交互过程整理为结构化工作流文档 | 整理工作流、复盘工作流、提取工作流 |
| `parking-lot` | 待办搁置清单管理 | 搁置、先不用了、先保存下来 |
| `cloudsync` | 项目打包同步/恢复到云端（含 backup.sh / restore.sh） | 云备份、云恢复 |

### 需求/设计/测试/验收 文档撰写
| Skill | 描述 | 触发关键词 |
|:---|---|:---|
| `require-refine` | 将模糊需求细化为可交付规格文档 | 需求分析、细化需求、需求文档、写需求 |
| `doc-design` | 编写/审查/更新设计文档 | 写设计文档、设计文档审查、细化设计 |
| `modulerefine` | 细化/分析/审查某个模块 | 分析X模块、细化X设计、审查X功能 |
| `test-module` | 为模块编写/补充/运行测试 | 写单元测试、补测试、做验收测试 |
| `acceptance` | 为模块撰写验收文档并执行验收 | 验收、验收文档、自验收、验收检查 |
| `strategy-perf-audit` | 审查策略代码中的 pandas 性能隐患 | 审查X策略性能、pandas 性能审查 |

### 知识库
| Skill | 描述 | 触发关键词 |
|:---|---|:---|
| `github-ingest` | GitHub 项目分析入库，生成项目说明文档 | — |
| `pdf-ingest` | PDF 下载→转换→格式清理，产出 Markdown | pdf 摄入、pdf转md、论文格式化 |
| `wiki-ingest` | 将 raw/ 文档摄入 wiki 知识库 | 论文入库、概念补全、知识抽取 |
| `wiki-lint` | wiki 质量检查（孤岛扫描、死链清理、冲突检测等） | — |
| `wiki-query` | 基于 wiki 内容生成带引用的回答并沉淀 synthesis 页 | 查询、对比分析、知识检索 |

### 商业资源配置优化
| Skill | 描述 | 触发关键词 |
|:---|---|:---|
| `news-idea-generator` | 基于热力学类比从新闻生成商业创意 | 新闻灵感、新闻商业创意、根据新闻创业 |
| `news-to-business-plan` | 串联4个热力学技能，从新闻输出商业计划书 | 新闻生成商业计划、新闻创业计划 |
| `business-launch-advisor` | 根据资金量级提供分级精算数据需求与冷启动行动路线 | 启动建议、精算数据、冷启动、我有X万怎么做 |
| `business-pitch-generator` | 基于热力学类比构建电梯游说 | 电梯游说、pitch、融资演讲稿、商业计划 |
| `business-viability-judge` | 基于热力学类比评估商业 idea 可行性 | 判断商业模式、这个想法靠谱吗、可行性判断 |


---

## 文件统计

| 类别 | 文件数 | 总行数 |
|:---|---:|---:|
| rules/ | 18 | ~5,300 |
| skills/ | 21 SKILL.md + 2 脚本 | ~3,300 |
| **总计** | **41** | **~8,600** |
