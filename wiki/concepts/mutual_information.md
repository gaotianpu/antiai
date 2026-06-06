---
id: mutual_information
type: concept
tags: ["machine-learning", "theoretical", "information-theory"]
aliases: ["互信息", "MI", "mutual information"]
related_nodes: [information_theory, entropy, kl_divergence]
---

# Mutual Information（互信息）

## 定义
$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} = D_{KL}(p(x,y) \| p(x)p(y))$$

互信息衡量两个随机变量之间的依赖程度，即知道 $Y$ 能减少 $X$ 的不确定性的量。

## 等价形式
- $I(X;Y) = H(X) - H(X|Y)$
- $I(X;Y) = H(Y) - H(Y|X)$
- $I(X;Y) = H(X) + H(Y) - H(X,Y)$
- $I(X;Y) = H(X,Y) - H(X|Y) - H(Y|X)$

## 关键性质

- **非负性**：$I(X;Y) \ge 0$，当且仅当 $X$ 与 $Y$ 独立时为零
- **对称性**：$I(X;Y) = I(Y;X)$
- **与相关系数的关系**：对于高斯分布，$I(X;Y) = -\frac{1}{2} \log(1 - \rho^2)$，其中 $\rho$ 是相关系数
- **数据压缩不等式（DPI）**：若 $X \to Y \to Z$ 构成马尔可夫链，则 $I(X;Y) \ge I(X;Z)$
- **信息瓶颈**：$I(X;Z) - \beta I(Z;Y)$，在压缩输入 $X$ 的同时最大化与目标 $Y$ 的互信息

## Deep Learning 中的连接

| 应用 | MI 角色 |
|:---|:---|
| InfoNCE 损失 | 对比学习的核心目标：最大化正样本对的互信息下界（CPC, SimCLR, CLIP） |
| 表示学习理论 | 好的表示应最大化与下游任务标签的互信息，同时最小化与输入的互信息（信息瓶颈） |
| 特征选择 | 最大化特征与标签的互信息，最小化特征间的冗余 |
| 生成模型评估 | 评估生成多样性（如 FID 间接与 MI 相关） |
| 互信息神经估计（MINE） | 使用神经网络变分估计高维互信息 |

## 相关概念
- [[information_theory]] — 信息论总览
- [[entropy]] — 互信息的熵形式定义
- [[kl_divergence]] — 互信息作为 KL 散度的特例
