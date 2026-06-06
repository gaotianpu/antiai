---
id: bayesian_inference
type: concept
tags: ["machine-learning", "theoretical", "statistical-learning"]
aliases: ["贝叶斯", "Bayes", "后验", "先验", "MLE"]
related_nodes: [probability_statistics, regularization]
---

# Bayesian Inference（贝叶斯推断）

## 贝叶斯定理
$$P(A|B) = \frac{P(B|A) \, P(A)}{P(B)}$$

后验概率 $\propto$ 似然 $\times$ 先验概率。

## 核心概念

- **先验 $P(\theta)$**：在观察数据之前对参数的信念
- **似然 $P(D|\theta)$**：给定参数下数据出现的概率
- **后验 $P(\theta|D)$**：观察数据后更新过的参数信念
- **证据 $P(D)$**：数据的边际概率，作为归一化常数

## MLE vs MAP

| 方法 | 目标 | 形式 | DL 等价物 |
|:---|:---|:---|:---|
| MLE | $\arg\max_\theta \log P(D\|\theta)$ | 仅似然 | 交叉熵损失（分类） |
| MAP | $\arg\max_\theta \log P(D\|\theta) + \log P(\theta)$ | 似然 + 先验 | MLE + 权重衰减（L2） |

### Deep Learning 中的连接

- **MLE = 交叉熵最小化**：分类任务中最小化负对数似然等价于最小化交叉熵
- **MAP = 加入 L2 正则的 MLE**：高斯先验 $P(\theta) = \mathcal{N}(0, \lambda^{-1})$ 下的 MAP 等价于 L2 正则化（Weight Decay）
- **贝叶斯神经网络**：对权重求后验分布而非点估计，提供不确定性量化
- **Dropout 作为贝叶斯近似**：Gal & Ghahramani (2016) 证明 Dropout 等价于变分推断
- **深度集成**：多个模型预测的均值可视为后验分布的蒙特卡洛近似

## 相关概念
- [[probability_statistics]] — 概率与统计总览
- [[regularization]] — MAP / L2 正则化的联系
