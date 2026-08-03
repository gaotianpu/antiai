---
type: log
---
## [Ingest] RL 重要论文摄入 ×7（2022-2025）

- 新建 Source ×7: [[bai_2022_constitutional_ai]]（宪法 AI）、[[reed_2022_gato]]（通用智能体）、[[rafailov_2023_dpo]]（DPO）、[[hafner_2023_dreamerv3]]（世界模型）、[[shao_2024_deepseekmath]]（GRPO 提出）、[[openai_2024_o1]]（推理模型）、[[kimi_2025_k2]]（多轮 RL）
- 新建 Concept ×6: [[constitutional_ai]], [[generalist_agent]], [[direct_preference_optimization]], [[world_model]], [[model_based_rl]], [[test_time_compute]]
- 更新 Concept: [[grpo]] 追加 GRPO 提出论文来源
- 更新 Synthesis: [[rl_evolution]] — ④阶段补 CAI/DPO，⑤阶段补 DeepSeekMath/o1/K2，新增旁支二（通用智能体与世界模型），转折点表更新
- 更新 Index: sources/index.md（2022/2023/2024/2025 段共 7 条）、concepts/index.md（6 条）
- 验证: 链接闭环无死链、索引完整性全通过

## [Query] RL 演进路线综述

- 用户查询: 强化学习演进路线是否有综述页 → wiki 无现成页，按 wiki-query 沉淀
- 新建 Synthesis: [[rl_evolution]] — RL 演进五阶段（DQN→TRPO/DDPG/A3C→PPO→RLHF→GRPO）+ 模仿学习旁支
- 更新 Index: synthesis/index.md、wiki/index.md 快速导航

## [Skill] skill 与规范完善（基于本轮 lint 实战发现）

- **wiki-lint/SKILL.md** ×4：①检查2 死链扫描改用 Python（GNU grep BRE 的 `[^\\]\\[]` 字符类静默失效，实测 grep 返回 0 而实际 228 个链接），并处理 `\\|` 转义/锚点/aliases；②死链二分类指引（笔误 vs 缺失概念页）；③检查4 修复非法 grep 语法（`! --include` / `! --path` → `--exclude` / `--exclude-dir`）；④检查5 表格转义正则改 -E 消除贪婪误报；⑤检查6 补充方向性豁免说明
- **wiki-ingest/SKILL.md** ×2：新增 §2.3.1 链接闭环验证（入库必做，缺失概念当场补建或登记 todo）；§2.3 明确正文链接即反向引用 + 中心页方向性豁免
- **schema/concept.md**：补充紧凑格式（定义+要点+来源）作为实践默认格式
- **schema/tags.md**：补录 6 个实践已广泛使用 tag（computer-vision/RL/autonomous-driving/imitation-learning/transformer/lane-detection）

## [Merge] 概念页收尾修复

- 合并同义重复: [[transformer]] → [[transformer_architecture]]（定义与来源并入，删除 transformer.md，21 处 related_nodes + 2 处链接更新）
- 别名冲突清零 ×8: ppo（PPO-RLHF 专指 RLHF 应用）、principle_driven_alignment（去 SELF-ALIGN）、goodhart_law（去重）、feature_pyramid（去多尺度特征）、in_context_learning（去少样本学习）、normalization（去层归一化）、gpt（去生成式预训练）
- 验证: 别名冲突 0、死链 0、索引完整（concepts 204/204）

## [Ingest] 缺失概念页补齐 ×145

- 新建 Concept ×145（wiki/concepts/ 60 → 205），覆盖：语言模型（BERT/GPT/MLM/分词）、Transformer 组件、序列模型、生成模型、自监督与对比学习、CNN 高效架构、目标检测/分割、强化学习/对齐、多模态等
- 更新 Index: concepts/index.md 追加 145 条，新增 6 个分类（语言模型/序列模型/生成模型/自监督与对比学习/多模态）
- 删除 wiki/todo.md（145 条待办全部完成）
- 验证: 新页面无死链、表格转义正常、索引完整性全通过

## [Lint] wiki-lint 全量质量检查

### 修复

- **表格转义** ×3: `wiki/synthesis/deepseek_papers.md` 中 `[[mixture_of_experts|MoE]]`、`[[reinforcement_learning|RL]]`、`[[chain_of_thought|CoT]]` → `\|` 转义
- **raw 索引补录** ×7: raw/index.md 新增 2024/2025 分组（6 篇 DeepSeek 论文 arXiv 原文），2014 分组 `[[cnn/googlenet.md]]` → `[[cnn/inception_v1.md]]`（文件实际存在名）
- **死链修复** ×2: `wiki/sources/szegedy_2014_googlenet.md` 阅读笔记链接 `googlenet.md` → `inception_v1.md`；删除空占位文件 `raw/cnn/googlenet.md`（0 字节）
- **别名冲突清理** ×9: 领域页/上位页 aliases 移除越界子概念名——information_theory（熵/KL散度/互信息）、optimization_fundamentals（凸优化）、calculus（链式法则）、linear_algebra（矩阵运算）、optimizer（梯度下降/SGD）、attention_mechanism（self-attention/自注意力）
- **related_nodes 双向补全** ×4: mixture_of_experts（+bi_2024_deepseek_llm, deepseek_papers）、conditional_memory（+deepseek_papers）、multi_head_latent_attention（+wei_2025_deepseek_ocr）
- **脚本改进**: `scripts/check_index_completeness.py` 排除 10 个非论文文件（7 个子目录 README + NewBeer/openai/deepseek_index 导航笔记）

### 生成

- **wiki/todo.md**: 145 种缺失概念页待办（Source 页引用但 concepts/ 未创建，156 次引用）

### 检查通过项

- 孤岛扫描: 0 孤岛
- wiki 内部死链: 0（除上述已记录的概念缺失待办）
- raw/index.md 死链: 0
- 索引完整性: wiki 3 目录 + raw 全部通过
- TODO 标记: 无（除新建 todo.md）
