---
id: taylor_expansion
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["泰勒展开", "泰勒级数", "Taylor series", "Taylor approximation"]
related_nodes: [calculus, derivative_and_gradient, optimization_fundamentals]
---

# 泰勒展开

泰勒展开用多项式逼近光滑函数，是优化算法的理论基础之一。

## 定义

函数 \(f(x)\) 在点 \(a\) 处的泰勒展开：

\[
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x-a)^n
\]

## 常见阶数逼近

### 一阶逼近（线性逼近）

\[
f(x) \approx f(a) + f'(a)(x-a)
\]

**梯度下降的数学本质**：在当前位置做一阶泰勒展开，沿负梯度方向最小化线性逼近。

### 二阶逼近（二次逼近）

\[
f(x) \approx f(a) + f'(a)(x-a) + \frac{1}{2} f''(a)(x-a)^2
\]

多元形式：

\[
f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top (\mathbf{x} - \mathbf{a}) + \frac{1}{2} (\mathbf{x} - \mathbf{a})^\top H(\mathbf{a}) (\mathbf{x} - \mathbf{a})
\]

### 余项 (Remainder)

泰勒展开的误差由余项刻画。Lagrange 余项形式：

\[
R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} (x-a)^{n+1}, \quad \xi \in [a, x]
\]

## DL 中的关键角色

- **梯度下降 = 一阶泰勒逼近**：仅利用梯度信息，计算简单但收敛较慢
- **牛顿法 = 二阶泰勒逼近**：利用 Hessian 矩阵，收敛更快但计算 Hessian 代价高 (\(O(n^2)\))
- **TRPO**：在 trust region 内对目标做二阶泰勒逼近，约束新旧策略的 KL 散度
- **曲率逼近**：Hessian 的特征值反映损失曲面的曲率，影响优化难度
- **K-FAC**：用 Fisher 信息矩阵近似 Hessian，实现二阶优化的高效计算

## 相关概念
- [[calculus]] — 泰勒展开基于微积分
- [[derivative_and_gradient]] — 一阶导数用于线性逼近
- [[optimization_fundamentals]] — 优化算法的收敛性分析依赖泰勒展开
