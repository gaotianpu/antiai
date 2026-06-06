---
id: calculus
type: concept
tags: [theoretical, machine-learning]
aliases: ["微积分", "微分", "链式法则", "calculus for DL"]
related_nodes: [math_for_deep_learning, linear_algebra, optimization_fundamentals, derivative_and_gradient, chain_rule, taylor_expansion]
---

# 微积分

微积分为深度学习提供了参数更新的核心机制——梯度计算与优化路径。

## 核心概念

- **导数 (Derivative)**：$f'(x) = \lim_{h\to 0} \frac{f(x+h)-f(x)}{h}$，描述函数在某点的瞬时变化率
- **偏导数 (Partial Derivative)**：多元函数对单个变量的导数，$\frac{\partial f}{\partial x_i}$
- **链式法则 (Chain Rule)**：$\frac{\partial f(g(x))}{\partial x} = \frac{\partial f}{\partial g}\frac{\partial g}{\partial x}$，反向传播的数学本质
- **梯度 (Gradient)**：$\nabla f = \left(\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right)^\top$，函数上升最快的方向
- **Jacobian 矩阵**：向量值函数的一阶偏导矩阵，$J_{ij} = \frac{\partial f_i}{\partial x_j}$
- **Hessian 矩阵**：标量函数的二阶偏导矩阵，$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$，用于判断极值性质和二阶优化
- **Taylor 展开**：$f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2}f''(a)(x-a)^2 + \dots$，函数局部近似的核心工具

## 为什么 DL 需要它

| 应用 | 微积分基础 |
|:---|:---|
| 反向传播 | 链式法则的递归应用，逐层计算损失对各参数的梯度 |
| 梯度下降 | $w_{t+1} = w_t - \eta\nabla L(w_t)$，沿负梯度方向更新参数 |
| 学习率调度 | 理解步长 $\eta$ 对收敛行为的影响 |
| 二阶优化 (K-FAC, Newton) | 利用 Hessian 矩阵加速收敛 |
| 激活函数导数 | sigmoid/tanh/ReLU 的导数直接影响梯度流动 |
