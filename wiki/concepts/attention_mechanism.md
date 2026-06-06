---
id: attention_mechanism
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [注意力机制, self-attention, 自注意力]
related_nodes: [vaswani_2017_transformer, bahdanau_2014_attention, devlin_2018_bert]
last_verified: 2026-06-06
---

# Attention Mechanism

## 定义
注意力机制允许模型在生成输出时动态聚焦于输入序列的不同部分，通过加权聚合所有位置的表示来捕获长程依赖，而不受距离限制。

## 关键变体

| 变体 | 特点 | 代表工作 |
|:---|:---|:---|
| Bahdanau Attention | 加性注意力，用于 Seq2Seq 对齐 | [[bahdanau_2014_attention]] |
| Dot-Product Attention | 缩放点积注意力，并行友好 | Luong 2015 |
| Self-Attention | 同一序列内注意力 | [[vaswani_2017_transformer]] |
| Multi-Head Attention | 并行多头、捕捉不同子空间 | [[vaswani_2017_transformer]] |
| Cross-Attention | 编码器-解码器间的注意力 | [[vaswani_2017_transformer]] |
| Causal (Masked) Attention | 单向注意力，禁止看到未来 | GPT 系列 |

## 来源
- [[bahdanau_2014_attention]] — 首次将注意力用于 NMT
- [[vaswani_2017_transformer]] — 纯注意力架构，奠定现代基础
