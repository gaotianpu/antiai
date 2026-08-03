---
id: consistency_model
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [一致性模型, Consistency Model]
related_nodes: [song_2023_consistency, diffusion_model]
last_verified: 2026-08-03
---

# Consistency Model

## 定义
Song 等提出的生成模型：学习将概率流 ODE 轨迹上的任意噪声点直接映射到数据分布，实现单步生成，无需对抗训练。

## 关键要点
- **一致性函数**：同一轨迹上任意点映射到同一终点
- **单步生成**：对比扩散的多步迭代采样
- **蒸馏/独立训练**：可从扩散模型蒸馏，也可单独训练

## 来源
- [[song_2023_consistency]] — Consistency Models
