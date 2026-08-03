---
id: inception_module
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [Inception模块, GoogLeNet模块]
related_nodes: [szegedy_2014_googlenet, factorized_convolution, convolutional_neural_network]
last_verified: 2026-08-03
---

# Inception Module

## 定义
GoogLeNet 提出的多分支卷积模块：在同一层并行使用 1×1/3×3/5×5 卷积与池化，拼接输出，在增加深度宽度的同时控制计算量。

## 关键要点
- **多尺度感知**：不同分支捕获不同尺度特征
- **1×1 瓶颈**：先降维再卷积，大幅压缩计算
- **演进**：v2/v3 用因式分解卷积替代大卷积核

## 来源
- [[szegedy_2014_googlenet]] — Inception/GoogLeNet
