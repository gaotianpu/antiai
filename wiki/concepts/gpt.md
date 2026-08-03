---
id: gpt
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [GPT]
related_nodes: [radford_2018_gpt, generative_pretraining, transformer_architecture]
last_verified: 2026-08-03
---

# GPT

## 定义
Generative Pre-trained Transformer：基于 Transformer 解码器的单向自回归语言模型，开创"生成式预训练 + 微调/提示"范式。

## 关键要点
- **单向性**：因果注意力，只看到左侧上下文，天然适配生成
- **演进**：GPT-1（预训练-微调范式）→ GPT-2（零样本）→ GPT-3（上下文学习）→ GPT-4（多模态）
- **影响**：LLM 时代的基础架构，与 BERT 形成"生成 vs 理解"两条技术路线

## 来源
- [[radford_2018_gpt]] — 提出生成式预训练
- [[radford_2019_gpt2]] — 规模扩展与零样本能力
- [[brown_2020_gpt3]] — 175B 参数与上下文学习
