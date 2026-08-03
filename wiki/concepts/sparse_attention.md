---
id: sparse_attention
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [稀疏注意力]
related_nodes: [fu_2025_moba, deepseek_2025_v32, dilated_attention, attention_variants]
last_verified: 2026-08-03
---

# Sparse Attention

## 定义
只计算部分查询-键对的注意力模式：固定模式（滑动窗口/扩张/局部全局）与可学习路由（MoBA 块路由、DSA 索引器选择）两类，将长上下文计算量从平方降至近线性。

## 关键要点
- **固定模式**：LongNet 扩张注意力、Swin 窗口注意力
- **可学习路由**：MoBA 键块路由（MoE 思想）、DSA Lightning Indexer 选择
- **质量保障**：稀疏化需配合全局锚点 token 与局部窗口

## 来源
- [[fu_2025_moba]] — MoBA 块路由注意力
- [[deepseek_2025_v32]] — DSA 索引器稀疏注意力
- [[longnet_2023]] — 扩张注意力
