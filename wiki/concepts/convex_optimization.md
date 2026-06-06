---
id: convex_optimization
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["凸优化", "convex", "凸函数"]
related_nodes: [optimization_fundamentals, optimizer]
---

# 凸优化 (Convex Optimization)

## 定义

凸优化是一类特殊的优化问题，其中目标函数和可行域都是凸的（convex），具有"局部最优即全局最优"的关键性质。

## 核心概念

### 凸集 (Convex Set)
集合 $C$ 中任意两点的连线上的所有点仍在 $C$ 中：
$$\forall x,y \in C,\; \forall \lambda \in [0,1],\; \lambda x + (1-\lambda)y \in C$$

### 凸函数 (Convex Function)
函数 $f$ 满足对任意 $x,y$ 和 $\lambda \in [0,1]$：
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

几何意义：函数图像下方区域（epigraph）是凸集；弦（chord）总在函数图像之上。

### Jensen 不等式
凸函数的推广形式——凸函数在期望处的值不超过期望的函数值：
$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

这是信息论（KL 散度非负性）、概率论、统计学习理论中无数不等式的根源。

### 凸 vs 非凸

| 性质 | 凸 | 非凸 |
|:---|:---|:---|
| 局部极小值 | 即全局最小值 | 可能有很多局部极小值 |
| 鞍点 | 不存在 | 高维空间的主要困难 |
| 收敛保证 | 可达全局最优 | 仅收敛到驻点 |
| 例子 | 线性回归、SVM（原始形式）、逻辑回归 | 所有带隐层的神经网络 |

## 与深度学习的关系

- **神经网络训练是高度非凸的**：非线性激活函数和深层堆叠使损失平面充满局部极小值和鞍点
- **子问题仍可凸**：如最后一层线性分类头、某些正则化项（L2 weight decay）
- **凸松弛技术**：某些非凸约束被松弛为凸约束以简化求解（如某些对抗训练方法）
- **理论分析工具**：凸优化提供的收敛率分析为理解非凸优化提供了基准

## 相关概念
- [[optimization_fundamentals]] — 优化基础总览，包含凸与非凸的对比
- [[optimizer]] — 具体优化算法，多数最初在凸假设下设计
