---
id: structural_reparameterization
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [结构重参数化, 重参数化]
related_nodes: [ding_2021_repvgg, ding_2021_repmlp]
last_verified: 2026-08-03
---

# Structural Reparameterization

## 定义
训练与推理使用不同但数学等价的结构：训练时用多分支（高表达力），推理时将分支合并为单路径（高速度），典型如 RepVGG 的 3×3 重参数化。

## 关键要点
- **训练-推理解耦**：多分支训练效果好，单路径推理速度快
- **合并原理**：卷积/BN/恒等分支线性叠加可等价合并
- **泛化**：RepMLP、DBB（ Diverse Branch Block）等变体

## 来源
- [[ding_2021_repvgg]] — RepVGG
- [[ding_2021_repmlp]] — RepMLP 卷积重参数化为全连接
