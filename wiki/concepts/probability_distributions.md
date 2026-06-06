---
id: probability_distributions
type: concept
tags: ["machine-learning", "theoretical", "statistical-learning"]
aliases: ["概率分布", "分布", "Gaussian", "伯努利", "distributions"]
related_nodes: [probability_statistics, loss_function]
---

# Probability Distributions（概率分布）

## 概述
概率分布描述随机变量取值的概率规律，是深度学习中建模数据、初始化参数、定义损失函数的数学基础。

## 核心分布类型

### Bernoulli（伯努利分布）
- 二值随机变量 $X \in \{0,1\}$，参数 $p = P(X=1)$
- DL 连接：二分类输出的 BCE 损失等价于最大化 Bernoulli 似然；Dropout mask 采样自 Bernoulli

### Categorical（类别分布）
- 多类别随机变量 $X \in \{1,\dots,K\}$，参数 $\boldsymbol{p} = (p_1,\dots,p_K)$
- DL 连接：分类网络的 Softmax 输出层直接输出 Categorical 分布参数；交叉熵损失等价于负对数似然

### Gaussian / Normal（高斯分布）
- $\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$
- DL 连接：VAE 假设隐空间服从各向同性高斯先验 $\mathcal{N}(0, I)$；权重初始化（Xavier, He）基于高斯采样；扩散模型的前向加噪过程

### Uniform（均匀分布）
- 连续均匀 $\mathcal{U}(a,b)$ 或离散均匀
- DL 连接：权重初始化常用 $\mathcal{U}(-c, c)$；Dropout 的保留概率可视为均匀采样的伯努利参数

### Exponential Family（指数族）
- 统一形式 $p(x|\theta) = h(x)\exp(\eta(\theta)^T T(x) - A(\theta))$
- 包含 Bernoulli、Gaussian、Categorical、Poisson 等常见分布
- DL 连接：指数族与广义线性模型（GLM）的理论基础；输出层激活函数的选择（linear = Gaussian, softmax = Categorical, sigmoid = Bernoulli）

## Deep Learning 中的分布角色

| 角色 | 典型分布 | 示例 |
|:---|:---|:---|
| 输出层 = 分布参数 | Categorical / Bernoulli | 分类任务 Softmax + CE |
| 隐空间先验 | Gaussian | VAE 的 KL 正则项 |
| 权重初始化 | Uniform / Gaussian | Xavier, He 初始化 |
| 随机正则化 | Bernoulli | Dropout |
| 噪声注入 | Gaussian | 数据增强、Diffusion 前向加噪 |

## 相关概念
- [[probability_statistics]] — 概率统计总览
- [[loss_function]] — 损失函数与似然的关系
