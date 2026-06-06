---
id: lagrange_multiplier
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["拉格朗日乘数", "Lagrange multiplier", "KKT", "约束优化"]
related_nodes: [optimization_fundamentals, chain_of_thought, policy_gradient]
---

# 拉格朗日乘数法 (Lagrange Multiplier)

## 定义

将约束优化问题转化为无约束优化问题的经典方法。通过引入乘子变量将约束条件吸收进目标函数。

## 等号约束：Lagrange 乘子

对于等号约束问题：
$$\min f(x) \quad \text{s.t.} \quad g(x) = 0$$

定义 Lagrange 函数：
$$\mathcal{L}(x, \lambda) = f(x) + \lambda g(x)$$

最优解的**必要条件**：
$$\nabla_x \mathcal{L} = 0 \quad \text{且} \quad \frac{\partial \mathcal{L}}{\partial \lambda} = 0$$

即在满足约束 $g(x)=0$ 的前提下，$f(x)$ 的梯度与约束的梯度方向平行。

## 不等号约束：KKT 条件

对于一般约束问题（等号 + 不等号）：
$$\min f(x) \quad \text{s.t.} \quad g_i(x) = 0,\; h_j(x) \leq 0$$

KKT 必要条件：
1. **驻点条件**：$\nabla f(x^*) + \sum \lambda_i \nabla g_i(x^*) + \sum \mu_j \nabla h_j(x^*) = 0$
2. **原始可行**：$g_i(x^*) = 0,\; h_j(x^*) \leq 0$
3. **对偶可行**：$\mu_j \geq 0$
4. **互补松弛**：$\mu_j h_j(x^*) = 0$ — 要么约束激活（$h_j=0$），要么乘子为零

## 原始-对偶 (Primal-Dual)

- **原始问题**：直接在原始变量 $x$ 上优化
- **对偶问题**：最大化 Lagrange 函数关于乘子的下界
- **强对偶性**：在凸问题中，原始与对偶最优值相等（Slater 条件）
- **弱对偶性**：非凸问题中，对偶最优值 ≤ 原始最优值，提供下界

## 与深度学习的关系

- **约束强化学习**：TRPO/PPO 使用 KL 散度约束策略更新，通过 Lagrange 乘子或代理目标实现"信赖域"优化
- **正则化的 Lagrange 视角**：L2 正则化 $\min L(\theta) + \frac{\lambda}{2}\|\theta\|^2$ 可看作约束 $\|\theta\|^2 \leq C$ 的 Lagrange 形式，$\lambda$ 即为乘子
- **SVM 的对偶形式**：支持向量机的最大间隔分类器通过 KKT 条件推导出对偶表示，使核技巧成为可能
- **最大间隔方法**：从感知机到 SVM 的演进中，Lagrange 乘子法提供了统一的理论框架

## 相关概念
- [[optimization_fundamentals]] — 优化基础，含约束优化的概述
- [[chain_of_thought]] — 直觉上，KL 约束策略更新可视为一种"约束思维链"
- [[policy_gradient]] — TRPO/PPO 中 Lagrange 乘子的具体应用
