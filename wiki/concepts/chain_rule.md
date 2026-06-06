---
id: chain_rule
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["链式法则", "chain rule", "反向传播"]
related_nodes: [calculus, derivative_and_gradient]
---

# 链式法则

链式法则是复合函数求导的规则，也是反向传播算法的数学本质。

## 单变量链式法则

若 \(y = f(u)\) 且 \(u = g(x)\)，则：

\[
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = f'(g(x)) \cdot g'(x)
\]

## 多变量链式法则

对于 \(z = f(x, y)\)，其中 \(x = g(t), y = h(t)\)：

\[
\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt}
\]

推广到向量形式：\(\frac{\partial z}{\partial t_i} = \sum_j \frac{\partial z}{\partial u_j} \frac{\partial u_j}{\partial t_i}\)，即 Jacobian 矩阵乘法。

## 计算图 (Computational Graph)

计算图将复合函数分解为基本运算节点，前向计算值、反向传播梯度：

- 每个节点代表一个运算（加、乘、激活函数等）
- 边代表数据依赖
- 反向传播沿计算图反向应用链式法则

## DL 中的关键角色

- **反向传播 = 链式法则 + 计算图**：从损失函数出发，逐层递归计算梯度
- **自动微分**：框架（PyTorch/TensorFlow）自动构建计算图并应用链式法则
- **梯度流**：链式法则决定梯度如何从输出层流回输入层
  - 连续乘入小于 1 的导数 → 梯度消失
  - 连续乘入大于 1 的值 → 梯度爆炸
- **残差连接**：通过恒等捷径改变计算图结构，使梯度能够直接回传

## 相关概念
- [[calculus]] — 链式法则是微积分的基本工具
- [[derivative_and_gradient]] — 导数与梯度是链式法则的应用产物
