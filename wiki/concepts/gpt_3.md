---
id: gpt_3
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [GPT-3, 1750亿参数模型]
related_nodes: [brown_2020_gpt3, gpt, few_shot_learning, in_context_learning]
last_verified: 2026-08-03
---

# GPT-3

## 定义
GPT 系列第三代：1750 亿参数的巨型语言模型，通过上下文学习（In-Context Learning）在少量示例或无示例条件下完成任务，无需梯度更新。

## 关键要点
- **上下文学习**：把任务示例写进提示，模型"现学现用"，替代微调
- **规模跃迁**：从 15 亿到 1750 亿参数，验证了大规模训练的涌现能力
- **few-shot 能力**：少量示例即可接近甚至超越微调模型的性能

## 来源
- [[brown_2020_gpt3]] — GPT-3 原始论文
