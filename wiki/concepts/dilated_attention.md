---
id: dilated_attention
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [扩张注意力, 空洞注意力]
related_nodes: [longnet_2023, self_attention, attention_variants]
last_verified: 2026-08-03
---

# Dilated Attention

## 定义
LongNet 提出的注意力变体：将 token 按扩张率分成多个稀疏子序列，每个头只关注一个子序列，以线性复杂度捕获超长距离依赖。

## 关键要点
- **线性复杂度**：注意力范围从 O(n²) 降为 O(n)
- **多尺度**：多头使用不同扩张率，兼顾局部与全局
- **应用**：将 Transformer 扩展到 10 亿 token 序列

## 来源
- [[longnet_2023]] — 提出扩张注意力
