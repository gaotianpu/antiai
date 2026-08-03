---
id: zero_shot_learning
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [零样本学习, zero-shot]
related_nodes: [radford_2019_gpt2, few_shot_learning, gpt_2]
last_verified: 2026-08-03
---

# Zero-Shot Learning

## 定义
不提供任何任务示例，仅凭任务描述/指令让模型直接执行新任务的能力，无需针对任务的任何训练数据。

## 关键要点
- **GPT-2 的发现**：预训练语言模型天然具备零样本任务执行能力
- **实现方式**：任务以自然语言描述，模型按条件生成答案
- **与提示工程**：零样本提示（指令式）是成本最低的适配方式

## 来源
- [[radford_2019_gpt2]] — 零样本能力验证
