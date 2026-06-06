---
id: gradient_descent
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["梯度下降", "SGD", "gradient descent", "随机梯度下降"]
related_nodes: [optimization_fundamentals, optimizer, derivative_and_gradient]
---

# 梯度下降 (Gradient Descent)

## 定义

通过沿损失函数梯度的负方向迭代更新参数来最小化目标函数的一阶优化算法。

## 更新规则

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

其中 $\eta$ 为学习率（步长），$\nabla_\theta L$ 为损失函数对参数的梯度。

## 变体

| 变体 | 每次更新使用的样本数 | 特点 |
|:---|:---:|:---|
| **Batch GD** | 全部样本 | 准确但计算量大，不适合大规模数据 |
| **Mini-batch SGD** | 一个子集（batch） | 最常用；兼顾效率与随机性带来的正则化效果 |
| **Stochastic GD** | 1 个样本 | 高方差、高波动性，很少在现代 DL 中使用 |

## 收敛与分析

- **凸目标**：梯度下降可实现线性收敛率（$\mathcal{O}(1/k)$ 或更好）
- **非凸目标**：保证收敛到一阶驻点（$\|\nabla f\| \to 0$），但无法保证全局最优
- **学习率**：太大导致发散，太小收敛过慢；理想策略通常配合 warmup + decay
- **局部极小值 vs 鞍点**：高维非凸优化中鞍点比局部极小值更普遍，但随机噪声和动量有助于逃离鞍点

## 与深度学习的关系

- **核心地位**：梯度下降（及其变体）是**所有神经网络训练的基础算法**
- **动量加速**：在历史梯度方向上累积动量，越过局部极小并加速收敛：
  $$v_{t+1} = \beta v_t + \nabla L(\theta_t),\quad \theta_{t+1} = \theta_t - \eta v_{t+1}$$
- **自适应方法**：Adam 等优化器为每个参数独立调整学习率，成为现代 DL 的事实标准
- **反向传播**：多层网络的梯度通过链式法则从输出层逐层回传，构成反向传播算法的核心

## 相关概念
- [[optimization_fundamentals]] — 优化基础总览
- [[optimizer]] — 优化器分类，SGD 到 AdamW 的演进
- [[derivative_and_gradient]] — 导数和梯度的数学定义
