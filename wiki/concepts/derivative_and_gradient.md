---
id: derivative_and_gradient
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["导数", "梯度", "偏导数", "derivative", "gradient"]
related_nodes: [calculus, optimizer, chain_rule]
---

# 导数与梯度

导数和梯度是深度学习反向传播与参数更新的数学基础。

## 导数 (Derivative)

导数是函数在某一点处的瞬时变化率。一元函数 \(f(x)\) 在 \(x\) 处的导数定义为：

\[
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
\]

几何意义：函数曲线在该点切线的斜率。

## 偏导数 (Partial Derivative)

多元函数 \(f(x_1, x_2, \dots, x_n)\) 对单个变量 \(x_i\) 的偏导数：

\[
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \dots, x_i+h, \dots, x_n) - f(x_1, \dots, x_n)}{h}
\]

保持其他变量不变，衡量该方向上函数的变化率。

## 梯度 (Gradient)

梯度是偏导数构成的向量：

\[
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)^\top
\]

- **方向**：函数值上升最快的方向
- **大小**：该方向上的变化率
- **负梯度方向**：函数值下降最快的方向——梯度下降的基础

## 方向导数 (Directional Derivative)

函数在单位向量 \(\mathbf{u}\) 方向上的变化率：

\[
D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}
\]

梯度方向取得最大方向导数。

## 数值梯度 vs 解析梯度

| 类型 | 方法 | 精度 | 计算代价 |
|:---|:---|:---:|:---:|
| 数值梯度 | 有限差分近似 | \(O(h)\) 或 \(O(h^2)\) | 需 \(O(n)\) 个前向传播 |
| 解析梯度 | 链式法则求导 | 精确 | 需 \(O(1)\) 个前反向传播 |

DL 训练使用解析梯度（反向传播），数值梯度仅用于梯度检查（gradient check）。

## DL 中的关键角色

- **梯度下降**：\(w_{t+1} = w_t - \eta \nabla L(w_t)\)，负梯度方向更新参数
- **反向传播**：通过链式法则递归计算各层参数的梯度
- **梯度消失/爆炸**：深层网络中梯度幅值指数衰减或增长，源于激活函数导数与权重矩阵的连乘
- **梯度截断**：将梯度范数限制在阈值内，应对梯度爆炸
- **梯度累积**：小批量间累加梯度，实现有效大批量训练

## 相关概念
- [[calculus]] — 微积分是导数与梯度的基础学科
- [[chain_rule]] — 链式法则是复合函数求导的核心工具
- [[optimizer]] — 优化器利用梯度进行参数更新
