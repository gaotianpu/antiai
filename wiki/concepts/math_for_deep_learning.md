---
id: math_for_deep_learning
type: concept
tags: [theoretical, survey, machine-learning]
aliases: ["深度学习数学", "数学基础", "math for ML", "foundational math"]
related_nodes: [linear_algebra, calculus, probability_statistics, information_theory, optimization_fundamentals]
---

# 深度学习数学基础

深度学习本质上是基于数学框架的表示学习与优化问题。本文作为导航页，简要介绍支撑深度学习的五大数学子领域及其核心作用。

## 子领域概览

| 领域 | 核心作用 | 典型应用 |
|:---|:---|:---|
| [[linear_algebra]] | 张量运算、线性变换、低秩分解 | 权重矩阵、Embedding、注意力 Q·K^T、LoRA |
| [[calculus]] | 梯度计算、优化理论 | 反向传播、梯度下降、二阶优化 |
| [[probability_statistics]] | 不确定性建模、统计推断 | 损失函数 (MLE)、贝叶斯推断、Dropout |
| [[information_theory]] | 信息度量、分布差异 | 交叉熵损失、KL 散度、互信息 |
| [[optimization_fundamentals]] | 参数搜索、约束满足 | 非凸训练、SGD 家族、TRPO 约束优化 |

这五个领域并非孤立——线性代数提供运算框架，微积分驱动梯度更新，概率统计解释不确定性，信息论度量分布差异，优化理论指导参数搜索。理解它们之间的联系是深入 DL 的基础。
