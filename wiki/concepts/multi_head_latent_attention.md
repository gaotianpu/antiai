---
id: multi_head_latent_attention
type: concept
tags: [empirical-study, NLP]
aliases: ["MLA", "Multi-head Latent Attention", "多头潜在注意力"]
related_nodes: [attention_mechanism, attention_variants, deepseek_2024_v2, deepseek_2024_v3]
---

# Multi-Head Latent Attention (MLA)

## 概述
MLA 是 DeepSeek 提出的高效注意力变体，通过将 Key-Value (KV) cache 压缩为潜在向量，显著降低推理时的显存占用和带宽需求。

## 核心机制
- 标准 MHA 需缓存完整的 Key 和 Value 矩阵
- MLA 将 KV 压缩到一个低维潜在向量中，推理时只需缓存该向量
- Key-Value Joint Compression 将 KV 压缩到单个潜空间

## 量化效果
- DeepSeek-V2：KV cache 压缩 93.3%，吞吐量提升 5.76×（相对 DeepSeek 67B）
- DeepSeek-V3：继承 MLA 架构，实现高效推理

## 来源
- [[deepseek_2024_v2]]
- [[deepseek_2024_v3]]
