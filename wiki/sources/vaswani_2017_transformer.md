---
id: vaswani_2017_transformer
type: source
tags: [NLP, machine-learning, theoretical]
aliases: [Transformer, Attention Is All You Need, 1706.03762]
related_nodes: [attention_mechanism]
arxiv_id: 1706.03762
authors: Ashish Vaswani et al.
authors_institution: Google
last_verified: 2026-06-06
---

# Attention Is All You Need

- **元数据**: Conference | 2017 | **作者**: Ashish Vaswani et al. | **机构**: Google | 相关: [[attention_mechanism]]
- **概述**: 提出完全基于注意力机制的 Transformer 架构，摒弃了循环和卷积，成为后续 LLM 的基础范式。
- **关键要点**: 1. 纯注意力架构（编码器-解码器） 2. 多头注意力 + 位置编码 3. 并行化训练，显著减少训练时间
- **方法/发现**: WMT 2014 英德翻译 28.4 BLEU（+2 BLEU 超此前最佳），训练 3.5 天/8 GPU
- **局限/意义**: O(n²) 复杂度限制长序列；奠定了现代 LLM 的架构基础

## 引用
- **原始论文**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | [阅读笔记](../../raw/nlp/transformer.md)
- **相关概念**: [[attention_mechanism]] | [[self_attention]] | [[positional_encoding]]
