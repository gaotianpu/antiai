---
id: fully_convolutional_network
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [全卷积网络, FCN]
related_nodes: [long_2014_fcn, semantic_segmentation, convolutional_neural_network]
last_verified: 2026-08-03
---

# Fully Convolutional Network (FCN)

## 定义
将分类网络的全连接层替换为卷积层，实现端到端、像素到像素的语义分割，接受任意尺寸输入并产生对应尺寸输出。

## 关键要点
- **卷积化**：全连接 → 1×1 卷积 + 上采样（反卷积）
- **任意尺寸**：去除固定输入约束
- **里程碑**：深度学习语义分割的开山之作，后续分割架构的基准

## 来源
- [[long_2014_fcn]] — FCN 原始论文
