---
id: rotary_position_embedding
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [旋转位置编码, RoPE]
related_nodes: [roformer_2021, positional_encoding, relative_position_encoding]
last_verified: 2026-08-03
---

# Rotary Position Embedding (RoPE)

## 定义
将位置信息以旋转矩阵的形式融入查询/键向量，使注意力分数自然依赖相对位置，兼具绝对位置实现与相对位置语义。

## 关键要点
- **旋转操作**：按位置角度旋转向量，内积自动携带相对位置信息
- **长度外推**：支持扩展到训练长度之外的序列
- **主流地位**：LLaMA、DeepSeek、Qwen 等现代 LLM 的标准位置编码

## 来源
- [[roformer_2021]] — 提出 RoPE
