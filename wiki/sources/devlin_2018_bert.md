---
id: devlin_2018_bert
type: source
tags: [NLP, machine-learning, empirical-study]
aliases: [BERT, 深度双向Transformers语言理解预训练, 1810.04805]
related_nodes: [attention_mechanism, google]
arxiv_id: 1810.04805
authors: Jacob Devlin et al.
authors_institution: Google
last_verified: 2026-06-06
---

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- **元数据**: Conference | 2018 | **作者**: Jacob Devlin et al. | **机构**: Google | 相关: [[attention_mechanism]]
- **概述**: 提出深度双向 Transformer 预训练模型 BERT，通过 MLM + NSP 预训练目标从无标注文本学习双向表示，微调后统治 11 项 NLP 任务。
- **新颖概念**: [[masked_language_modeling]], [[bert]]
- **关键要点**: 1. 双向编码器预训练（MLM 掩码语言模型） 2. NSP 下一句预测 3. 微调仅需一个额外输出层
- **方法/发现**: GLUE 80.5%（+7.7%），SQuAD v1.1 F1 93.2，SQuAD v2.0 F1 83.1
- **局限/意义**: 开创预训练-微调范式；单向（GPT）vs 双向（BERT）成为 NLP 两大路线

## 引用
- **原始论文**: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | [阅读笔记](../../raw/nlp/bert.md)
- **相关实体**: [[google]]
