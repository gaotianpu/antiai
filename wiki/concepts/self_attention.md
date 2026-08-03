---
id: self_attention
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [自注意力, 自注意力机制]
related_nodes: [vaswani_2017_transformer, attention_mechanism, multi_head_attention, attention_variants]
last_verified: 2026-08-03
---

# Self-Attention

## 定义
同一序列内部的位置相互计算注意力权重，每个位置聚合全序列信息，捕获任意距离依赖的机制。

## 关键要点
- **任意距离依赖**：O(1) 步可达任意位置（对比 RNN 的线性步数）
- **复杂度**：O(n²) 时间和内存，是长序列处理的主要瓶颈
- **变体**：因果掩码（生成）、稀疏/滑动窗口（长序列）等

## 来源
- [[vaswani_2017_transformer]] — 自注意力核心机制
