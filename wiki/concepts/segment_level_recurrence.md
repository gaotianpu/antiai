---
id: segment_level_recurrence
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [片段级循环, 段级递归]
related_nodes: [dai_2019_transformer_xl, transformer_architecture]
last_verified: 2026-08-03
---

# Segment-Level Recurrence

## 定义
Transformer-XL 提出的机制：处理长文本时缓存上一片段的隐藏状态，与当前片段拼接参与注意力，使信息跨片段流动。

## 关键要点
- **突破上下文上限**：固定长度窗口 → 理论上可捕获任意长依赖
- **状态复用**：计算当前片段时重复使用前一片段表示，不重复计算
- **配合相对位置编码**：解决缓存状态下绝对位置错乱问题

## 来源
- [[dai_2019_transformer_xl]] — Transformer-XL 核心创新
