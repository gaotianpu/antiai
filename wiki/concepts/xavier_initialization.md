---
id: xavier_initialization
type: concept
tags: [machine-learning, theoretical]
aliases: [Xavier初始化, Glorot初始化, 均匀缩放初始化]
related_nodes: [glorot_2010_xavier, gradient_descent, activation_function]
last_verified: 2026-08-03
---

# Xavier Initialization

## 定义
Glorot/Xavier 提出的权重初始化方案：按输入输出维度控制初始方差，使信号在前向与反向传播中方差保持恒定，缓解深层网络训练困难。

## 关键要点
- **方差守恒**：$Var(W) = 2/(n_{in} + n_{out})$
- **前提假设**：激活函数近似线性（Sigmoid/Tanh 适用）
- **演进**：ReLU 时代由 He 初始化（Kaiming）取代

## 来源
- [[glorot_2010_xavier]] — Xavier/Glorot 初始化
