---
id: raffel_2019_t5
type: source
tags: [NLP, machine-learning, empirical-study]
aliases: [T5, Text-to-Text Transfer Transformer, 统一文本到文本框架]
related_nodes: [attention_mechanism]
last_verified: 2026-06-06
---

# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

- **元数据**: Conference | 2019 | **作者**: Colin Raffel et al. | **机构**: Google | 相关: [[attention_mechanism]]
- **概述**: 提出 T5 模型，将所有 NLP 任务统一为 Text-to-Text 格式，系统研究了预训练方法的影响。
- **新颖概念**: [[text_to_text_framework]], [[encoder_decoder_architecture]]
- **关键要点**: 1. 统一文本到文本框架（输入/输出均为文本） 2. Encoder-Decoder Transformer 3. C4 数据集系统消融实验
- **方法/发现**: 在 GLUE、SuperGLUE、SQuAD 等基准上取得当时最佳
- **局限/意义**: Text-to-Text 范式简洁统一；计算成本高

## 引用
- **原始论文**: [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) | [阅读笔记](../../raw/nlp/T5.md)
