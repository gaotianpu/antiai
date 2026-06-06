---
id: in_context_learning
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [上下文学习, ICL, few-shot learning, 少样本学习]
related_nodes: [brown_2020_gpt3]
last_verified: 2026-06-06
---

# In-Context Learning

## 定义
上下文学习（In-Context Learning, ICL）是指大语言模型在推理时，仅通过在输入中提供若干示例，即可执行未曾训练过的新任务，无需梯度更新。

## 三种模式
- **Zero-shot**: 仅提供任务描述，无示例
- **One-shot**: 提供一个示例
- **Few-shot**: 提供少量示例（通常 3-64 个）

## 来源
- [[brown_2020_gpt3]] — 系统研究并命名此现象
