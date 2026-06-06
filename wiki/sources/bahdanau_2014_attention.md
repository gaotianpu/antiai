---
id: bahdanau_2014_attention
type: source
tags: [NLP, machine-learning, theoretical]
aliases: [Bahdanau Attention, 注意力机制, Neural Machine Translation by Jointly Learning to Align and Translate, 1409.0473]
related_nodes: [attention_mechanism]
arxiv_id: 1409.0473
authors: Dzmitry Bahdanau et al.
last_verified: 2026-06-06
---

# Neural Machine Translation by Jointly Learning to Align and Translate

- **元数据**: Conference | 2014 | **作者**: Dzmitry Bahdanau et al. | 相关: [[attention_mechanism]]
- **概述**: 提出在神经机器翻译中联合学习对齐和翻译的注意力机制（Bahdanau Attention），打破固定编码向量的瓶颈。
- **关键要点**: 1. 编码器双向 RNN 2. 解码器每次生成时关注源句不同位置 3. 对齐权重可解释
- **方法/发现**: 在英法翻译任务上超越固定编码向量的传统 Seq2Seq 模型
- **局限/意义**: 奠定了注意力机制在 Seq2Seq 中的标准用法，后续被 Transformer 发扬光大

## 引用
- **原始论文**: [arXiv:1409.0473](https://arxiv.org/abs/1409.0473) | [阅读笔记](../../raw/nlp/Attention.md)
- **相关概念**: [[attention_mechanism]] | [[transformer]]
