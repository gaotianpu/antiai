---
id: permutation_language_modeling
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [排列语言建模, PLM]
related_nodes: [xlnet_2019, masked_language_modeling, bert]
last_verified: 2026-08-03
---

# Permutation Language Modeling

## 定义
XLNet 提出的预训练目标：对序列做随机排列，以自回归方式预测排列中的每个位置，同时获得双向上下文（无 [MASK] 标记）。

## 关键要点
- **双向 + 自回归**：融合 BERT 的双向与 GPT 的生成能力
- **无掩码偏差**：训练与微调输入分布一致（无 [MASK]）
- **实现**：借助注意力掩码实现排列，不真正打乱输入

## 来源
- [[xlnet_2019]] — XLNet：广义自回归预训练
