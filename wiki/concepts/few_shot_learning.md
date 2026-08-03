---
id: few_shot_learning
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [少样本学习, few-shot]
related_nodes: [brown_2020_gpt3, in_context_learning, gpt_3]
last_verified: 2026-08-03
---

# Few-Shot Learning

## 定义
仅凭少量标注示例（通常 1-10 个）完成新任务的学习范式。在 LLM 语境下指将示例写入提示（In-Context Learning），无需梯度更新。

## 关键要点
- **零样本 → 少样本 → 微调**：按示例数量与参数更新程度划分的谱系
- **GPT-3 的发现**：175B 规模下少样本提示即可接近微调性能
- **应用**：提示工程的基础，任务适配的轻量方案

## 来源
- [[brown_2020_gpt3]] — GPT-3 少样本能力
