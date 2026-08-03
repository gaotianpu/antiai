---
id: dropout
type: concept
tags: [machine-learning, theoretical]
aliases: [随机失活, 丢弃法]
related_nodes: [krizhevsky_2012_alexnet, regularization, overfitting]
last_verified: 2026-08-03
---

# Dropout

## 定义
训练时按概率随机"关闭"神经元（输出置零），迫使网络学习冗余表示，是缓解过拟合最经典的正则化手段。

## 关键要点
- **集成视角**：每次前向对应不同子网络，等价于指数级子网络的集成
- **推理缩放**：测试时保留全部神经元，权重乘 1-p 补偿
- **演进**：DropBlock、Stochastic Depth 等结构化变体

## 来源
- [[krizhevsky_2012_alexnet]] — AlexNet 中 Dropout 的关键作用
