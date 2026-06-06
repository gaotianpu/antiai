---
id: optimization_fundamentals
type: concept
tags: [theoretical, robust-optimization, machine-learning]
aliases: ["优化基础", "凸优化", "numerical optimization", "优化理论"]
related_nodes: [math_for_deep_learning, calculus, optimizer, taylor_expansion]
---

# 优化基础

优化是深度学习的引擎——训练神经网络本质上是求解一个高维非凸优化问题。

## 核心概念

- **凸优化 vs 非凸优化**：
  - 凸：目标函数和约束集均为凸，任何局部最小值即全局最小值
  - 非凸：存在多个局部极小值和鞍点，DL 的训练目标几乎都是非凸的
- **局部极小值 vs 全局极小值**：非凸问题中，梯度下降通常收敛到局部极小值，但在高维空间中局部极小值与全局极小值的差距往往不大
- **鞍点 (Saddle Point)**：梯度为零但 Hessian 矩阵不定（兼具正负特征值），高维非凸优化的主要困难来源
- **梯度下降法**：$w_{t+1} = w_t - \eta_t \nabla L(w_t)$，一阶迭代优化算法
- **约束优化与 Lagrange 乘子法**：$\min f(x) \text{ s.t. } g(x)=0$，引入乘子 $\lambda$ 将约束加入目标函数 $\mathcal{L}(x,\lambda) = f(x) + \lambda g(x)$
- **收敛率 (Convergence Rate)**：描述算法随迭代次数逼近最优解的速度。凸问题可达线性/二次收敛率，非凸问题通常只有次线性

## 关键直觉

高维非凸优化的"意外之喜"：在高维参数空间中，局部极小值比鞍点少得多（鞍点的 Hessian 有正有负，坏曲率方向随机存在）。这使得简单的 SGD 在深度网络中出人意料地有效。

## 为什么 DL 需要它

| 应用 | 优化基础 |
|:---|:---|
| 神经网络训练 | 非凸目标上的随机梯度下降，动量和 Adam 加速 |
| TRPO/PPO | 约束策略优化中引入 KL 散度约束，使用 Lagrange 乘子或代理目标 |
| 学习率与收敛 | warmup + cosine decay 等策略影响收敛质量 |
| 损失平面几何 | 平坦极小值的泛化优势 (Sharp vs Flat Minima) |
