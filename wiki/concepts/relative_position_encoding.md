---
id: relative_position_encoding
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [相对位置编码, RPE]
related_nodes: [dai_2019_transformer_xl, positional_encoding, rotary_position_embedding]
last_verified: 2026-08-03
---

# Relative Position Encoding

## 定义
以两位置之间的相对距离（而非绝对位置）编码位置信息，使注意力对序列平移不变，并支持长度外推。

## 关键要点
- **优势**：训练时未见过的长度也能合理外推
- **实现**：将相对距离偏置加入注意力分数计算
- **演进**：RoPE、ALiBi 等均为相对位置思想的现代实现

## 来源
- [[dai_2019_transformer_xl]] — 提出相对位置编码
