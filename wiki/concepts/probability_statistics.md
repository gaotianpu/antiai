---
id: probability_statistics
type: concept
tags: [theoretical, statistical-learning, machine-learning]
aliases: ["概率论", "统计学", "概率统计", "probability for DL"]
related_nodes: [math_for_deep_learning, information_theory]
---

# 概率与统计

概率论为深度学习提供了不确定性建模的语言，统计学则支撑了从数据中推断参数的理论框架。

## 核心概念

- **随机变量**：分为离散型（伯努利、类别分布）和连续型（高斯分布），是建模数据生成过程的基本单元
- **常见分布**：
  - **高斯分布** $\mathcal{N}(\mu, \sigma^2)$：中心极限定理的产物，广泛用于初始化、噪声建模
  - **伯努利分布** $\text{Bern}(p)$：二值结果建模，如 Dropout mask
  - **类别分布** $\text{Cat}(K, \boldsymbol{p})$：多分类输出的自然形式
- **条件概率**：$P(A|B) = \frac{P(A\cap B)}{P(B)}$，构成贝叶斯推断和概率图模型的基础
- **贝叶斯规则**：$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$，先验 + 证据 → 后验的更新框架
- **期望与方差**：$E[X]$ 描述中心趋势，$\text{Var}(X) = E[(X-\mu)^2]$ 描述离散程度
- **MLE (最大似然估计)**：$\theta^* = \arg\max_\theta \log P(D|\theta)$，寻找最可能产生观测数据的参数
- **MAP (最大后验估计)**：$\theta^* = \arg\max_\theta \log P(D|\theta) + \log P(\theta)$，MLE + 先验正则化

## 为什么 DL 需要它

| 应用 | 概率/统计基础 |
|:---|:---|
| 交叉熵损失 | $- \sum y \log \hat{y}$ 等价于分类任务的负对数似然 (MLE) |
| Dropout | 视为 Bernoulli 分布的乘法噪声，是贝叶斯近似的变分推断 |
| 贝叶斯神经网络 | 对权重分布而非点估计进行推断，提供不确定性量化 |
| 生成模型 | 极大似然训练 (VAE, Flow-based) 或对抗训练 (GAN) |
| 初始化策略 | Xavier/Glorot 初始化基于方差保持原则 |
