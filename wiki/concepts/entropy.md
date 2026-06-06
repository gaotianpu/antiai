---
id: entropy
type: concept
tags: ["machine-learning", "theoretical", "information-theory"]
aliases: ["熵", "信息熵", "entropy"]
related_nodes: [information_theory, loss_function, kl_divergence, mutual_information]
---

# Entropy（熵）

## 定义
$$H(X) = -\sum_{x} p(x) \log p(x) = \mathbb{E}_{x\sim p}[-\log p(x)]$$

熵衡量分布 $p$ 的平均不确定性或信息量。单位：使用 $\log_2$ 为比特（bits），使用 $\ln$ 为纳特（nats）。

## 关键性质

- **非负性**：$H(X) \ge 0$，当且仅当 $p$ 是退化分布（确定事件）时取零
- **最大熵**：固定取值空间下，均匀分布的熵最大
- **联合熵**：$H(X,Y) = -\sum_{x,y} p(x,y) \log p(x,y)$
- **条件熵**：$H(Y|X) = -\sum_{x,y} p(x,y) \log p(y|x) = H(X,Y) - H(X)$
- **链式法则**：$H(X_1,\dots,X_n) = \sum_{i=1}^n H(X_i|X_{i-1},\dots,X_1)$
- **Perplexity**：$\text{PPL} = 2^{H(p)} = e^{H(p)}$，语言模型评价的核心指标

## Deep Learning 中的连接

| 应用 | 关系 |
|:---|:---|
| 交叉熵损失 | $H(p,q) = H(p) + D_{KL}(p\|q)$，固定真实分布 $p$ 时等价于 KL 最小化 |
| 语言模型评估 | Perplexity = $2^{\text{CE}}$，衡量模型对下一个 token 的预测能力 |
| 最大熵原则 | 在约束下选择熵最大的分布，用于强化学习中的最大熵策略（SAC） |
| 决策树 | 信息增益 = $H(Y) - H(Y|X)$ 用于特征分裂选择 |
| 主动学习 | 选择熵最大的样本进行标注（不确定性采样） |

## 相关概念
- [[information_theory]] — 信息论总览
- [[loss_function]] — 交叉熵作为损失函数
