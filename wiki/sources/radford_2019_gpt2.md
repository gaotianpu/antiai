---
id: radford_2019_gpt2
type: source
tags: [NLP, machine-learning, empirical-study]
aliases: [GPT-2, Language Models are Unsupervised Multitask Learners]
related_nodes: [attention_mechanism]
last_verified: 2026-06-06
---

# Language Models are Unsupervised Multitask Learners

- **元数据**: Tech Report | Alec Radford et al. (OpenAI) | 2019 | 相关: [[attention_mechanism]]
- **概述**: 扩展 GPT-1 至 15 亿参数，证明语言模型在无监督下可零样本执行多种下游任务。
- **关键要点**: 1. 48 层 Transformer，15 亿参数 2. WebText 数据集（45GB 文本） 3. 零样本迁移能力
- **方法/发现**: 在零样本设置下，在 7/8 个任务上取得当时最佳
- **局限/意义**: 证实规模扩展带来能力涌现；因安全顾虑延迟发布（分阶段发布策略）

## 引用
- **原始论文**: [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [阅读笔记](../../raw/nlp/gpt_2.md)
- **相关概念**: [[gpt]] | [[gpt_3]]
