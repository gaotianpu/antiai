---
id: text_to_text_framework
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [文本到文本框架, Text-to-Text]
related_nodes: [raffel_2019_t5, encoder_decoder_architecture]
last_verified: 2026-08-03
---

# Text-to-Text Framework

## 定义
T5 提出的统一范式：将所有 NLP 任务（翻译、摘要、分类、问答）都形式化为"输入文本 → 输出文本"，用同一个编码器-解码器模型训练。

## 关键要点
- **统一接口**：任务通过前缀指令区分（如 "summarize: ..."），无需任务专用头
- **系统研究**：同一框架下可比预训练目标、数据规模、模型大小的影响
- **影响**：奠定了"通用模型 + 指令前缀"的范式基础

## 来源
- [[raffel_2019_t5]] — T5：统一的文本到文本框架
