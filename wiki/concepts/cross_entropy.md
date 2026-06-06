---
id: cross_entropy
type: concept
tags: ["machine-learning", "theoretical", "information-theory"]
aliases: ["交叉熵", "cross-entropy loss", "CrossEntropy", "交叉熵损失"]
related_nodes: ["entropy", "kl_divergence", "loss_function", "information_theory"]
---

# Cross Entropy（交叉熵）

## 定义

给定真实分布 $p$ 和预测分布 $q$，交叉熵定义为：

$$H(p,q) = -\sum_{x} p(x) \log q(x)$$

连续形式为 $H(p,q) = -\int p(x) \log q(x) \, dx$.

## 与熵和 KL 散度的关系

交叉熵可分解为熵与 KL 散度之和：

$$H(p,q) = H(p) + D_{KL}(p\|q)$$

证明：
$$
\begin{aligned}
H(p,q) &= -\sum p(x) \log q(x) \\
       &= -\sum p(x) \log p(x) + \sum p(x) \log \frac{p(x)}{q(x)} \\
       &= H(p) + D_{KL}(p\|q)
\end{aligned}
$$

当真实分布 $p$ 固定时，最小化 $H(p,q)$ 等价于最小化 $D_{KL}(p\|q)$.

## 分类中的交叉熵损失

对于 $C$ 类分类任务，模型输出 $\hat{y} \in \mathbb{R}^C$ 经 Softmax 归一化后得到概率分布 $\hat{p} = \text{softmax}(\hat{y})$，真实标签为 one-hot 分布 $y$：

$$\mathcal{L}_{\text{CE}} = -\sum_{i=1}^C y_i \log \hat{p}_i$$

由于 $y$ 是 one-hot，等价于 $\mathcal{L}_{\text{CE}} = -\log \hat{p}_{\text{true}}$，即最小化真实类别的负对数概率。

## BCE vs Categorical CE

| 变体 | 适用场景 | 公式 |
|:---|:---|:---|
| **Binary CE (BCE)** | 二分类 / 多标签分类 | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| **Categorical CE** | 多分类（单标签） | $-\sum_i y_i \log \hat{y}_i$，配合 Softmax |

BCE 使用 Sigmoid 输出，每个类别独立判定；Categorical CE 使用 Softmax 输出，类别间互斥。

## 为什么分类用交叉熵而非 MSE

| 损失 | 梯度形式（配合 Softmax） | 饱和问题 |
|:---|:---|:---|
| **CE** | $\nabla_{\hat{y}} = \hat{p} - y$ | 线性梯度，无饱和 |
| **MSE** | $\nabla_{\hat{y}} = (\hat{p} - y) \cdot \hat{p} \cdot (1-\hat{p})$ | 当 $\hat{p}$ 接近 0 或 1 时梯度极小，收敛慢 |

MSE 在 Softmax 输出端会产生梯度饱和（当预测极度错误或极度正确时梯度趋近 0），而 CE 的梯度始终正比于预测误差 $(\hat{p} - y)$，学习信号稳定。

## 与极大似然估计（MLE）的关系

交叉熵最小化与 MLE 等价：

$$
\begin{aligned}
\theta^*_{\text{MLE}} &= \arg\max_\theta \sum_{n=1}^N \log p_\theta(y_n | x_n) \\
&= \arg\min_\theta -\frac{1}{N} \sum_{n=1}^N \log p_\theta(y_n | x_n) \\
&= \arg\min_\theta H(\hat{p}_{\text{data}}, p_\theta)
\end{aligned}
$$

其中 $\hat{p}_{\text{data}}$ 是经验分布。训练神经网络分类器本质上是在对条件分布 $p_\theta(y|x)$ 做 MLE。

## 相关概念
- [[entropy]] — 熵：交叉熵的构成基础
- [[kl_divergence]] — KL 散度：交叉熵与熵之差
- [[loss_function]] — 损失函数总览
- [[information_theory]] — 信息论总览
