---
id: length_extrapolation
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [长度外推, 上下文扩展]
related_nodes: [peng_2023_yarn, alibi_position_encoding, rotary_position_embedding, relative_position_encoding]
last_verified: 2026-08-03
---

# Length Extrapolation

## 定义
让模型处理超出训练序列长度的输入的能力/技术：位置编码设计（ALiBi/相对位置）与后处理插值（YaRN 的 NTK 插值）两条路线，配合少量微调可扩展上下文数倍。

## 关键要点
- **设计路线**：相对位置编码天然外推（ALiBi、RoPE 的周期延拓）
- **插值路线**：缩放旋转角（NTK-aware）适配更长的位置范围
- **工程极限**：Gemini 1.5 等以百万 token 验证"工程化外推"可行

## 来源
- [[peng_2023_yarn]] — YaRN 高效外推
- [[gemini_2024_15]] — 百万上下文工程化
