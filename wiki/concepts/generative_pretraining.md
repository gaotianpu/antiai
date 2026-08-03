---
id: generative_pretraining
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [生成式预训练, 预训练-微调范式]
related_nodes: [radford_2018_gpt, gpt, transfer_learning]
last_verified: 2026-08-03
---

# Generative Pretraining

## 定义
在无标注文本上以自回归语言建模目标预训练 Transformer 解码器，再在下游任务上微调的范式，验证了预训练-微调在 NLP 的通用性。

## 关键要点
- **两阶段**：无监督预训练（学语言）→ 有监督微调（学任务）
- **与 BERT 对比**：GPT 用解码器单向建模，生成任务友好
- **范式演进**：被"预训练 + 提示/指令"进一步取代

## 来源
- [[radford_2018_gpt]] — 提出生成式预训练方法
