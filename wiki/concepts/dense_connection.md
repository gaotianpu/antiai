---
id: dense_connection
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [密集连接, Dense 连接]
related_nodes: [huang_2016_densenet, residual_connection, convolutional_neural_network]
last_verified: 2026-08-03
---

# Dense Connection

## 定义
DenseNet 提出的连接模式：每层与之前所有层的输出拼接作为输入，实现特征复用与梯度直达，缓解深层网络退化与梯度消失。

## 关键要点
- **特征复用**：每层只学习少量新特征（增长率），参数效率高
- **梯度流动**：任意层可直达损失，训练深层网络更稳定
- **代价**：特征图拼接导致内存占用较高

## 来源
- [[huang_2016_densenet]] — DenseNet
