---
id: radford_2018_gpt
type: source
tags: [NLP, machine-learning, empirical-study]
aliases: [GPT-1, Generative Pre-Training, 通过生成式预训练提高语言理解]
related_nodes: [attention_mechanism]
last_verified: 2026-06-06
---

# Improving Language Understanding by Generative Pre-Training

- **元数据**: Tech Report | Alec Radford et al. (OpenAI) | 2018 | 相关: [[attention_mechanism]]
- **概述**: 提出生成式预训练方法（GPT），在无标注文本上预训练 Transformer 解码器后微调，验证了预训练-微调范式在 NLP 任务的通用性。
- **关键要点**: 1. 单向（从左到右）Transformer 解码器 2. 无监督预训练 + 有监督微调 3. 12 层 Transformer，1.17 亿参数
- **方法/发现**: 在多个 NLP 基准（自然语言推理、问答、分类等）上取得当时最佳
- **局限/意义**: 单向性限制了表示能力（后被 BERT 超越）；但开创了 GPT 系列路线

## 引用
- **原始论文**: [PDF](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | [阅读笔记](../../raw/nlp/gpt.md)
- **相关概念**: [[gpt_2]] | [[gpt_3]]
