---
id: alibi_position_encoding
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [ALiBi, 注意力线性偏置]
related_nodes: [alibi_2021, positional_encoding, relative_position_encoding]
last_verified: 2026-08-03
---

# ALiBi

## 定义
Attention with Linear Biases：不修改词嵌入，而是按相对距离向注意力分数添加线性偏置，距离越远的 token 得分越低，实现长度外推。

## 关键要点
- **零参数**：不增加任何可学习参数，实现简单
- **长度外推**：线性偏置随距离衰减，训练短序列可推理长序列
- **应用**：BLOOM 等模型采用此方案

## 来源
- [[alibi_2021]] — 提出 ALiBi 位置编码
