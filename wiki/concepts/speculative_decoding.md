---
id: speculative_decoding
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [投机解码, 推测解码]
related_nodes: [leviathan_2023_speculative, transformer_architecture]
last_verified: 2026-08-03
---

# Speculative Decoding

## 定义
用草稿模型快速生成候选 token 序列、目标大模型并行验证的加速解码范式：接受合理前缀、拒绝采样保证分布无偏，解码加速 2-3 倍。

## 关键要点
- **草稿-验证**：小模型猜、大模型判，把串行生成变并行
- **无偏性**：拒绝采样保证输出分布与目标模型一致
- **变体**：Medusa（多头草稿）、EAGLE（特征级草稿）等

## 来源
- [[leviathan_2023_speculative]] — 投机解码原始论文
