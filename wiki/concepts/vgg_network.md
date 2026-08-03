---
id: vgg_network
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [VGG, VGG网络]
related_nodes: [simonyan_2014_vgg, convolutional_neural_network]
last_verified: 2026-08-03
---

# VGG Network

## 定义
系统研究深度对精度影响的 CNN 架构：统一使用 3×3 小卷积堆叠（16-19 层），验证"更深更宽"的收益，是深度 CNN 的经典基准。

## 关键要点
- **设计哲学**：两个 3×3 ≈ 一个 5×5 感受野，但参数更少、非线性更强
- **结构规整**：卷积层+池化层规则堆叠，易于移植
- **影响**：作为特征提取主干被检测/分割广泛复用

## 来源
- [[simonyan_2014_vgg]] — VGG 原始论文
