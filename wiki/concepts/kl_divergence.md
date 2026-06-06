---
id: kl_divergence
type: concept
tags: ["machine-learning", "theoretical", "information-theory"]
aliases: ["KL散度", "相对熵", "Kullback-Leibler", "relative entropy"]
related_nodes: [information_theory, entropy, loss_function, reinforcement_learning_from_human_feedback, mutual_information, cross_entropy]
---

# KL Divergence（KL 散度）

## 定义
$$D_{KL}(P \| Q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x\sim p}\left[\log \frac{p(x)}{q(x)}\right]$$

KL 散度衡量用分布 $Q$ 近似分布 $P$ 时的信息损失，是两个概率分布之间差异的非对称度量。

## 关键性质

- **非负性**：$D_{KL}(P\|Q) \ge 0$，当且仅当 $P=Q$ 时为零（Gibbs 不等式）
- **非对称性**：$D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$（一般情况），因此 KL 散度不是真正的距离度量
- **与交叉熵的关系**：$H(P,Q) = H(P) + D_{KL}(P\|Q)$
- **前向 KL vs 反向 KL**：
  - $D_{KL}(P\|Q)$：均值搜索（mode covering），$Q$ 弥散覆盖 $P$ 的支撑集
  - $D_{KL}(Q\|P)$：众数搜索（mode seeking），$Q$ 聚焦拟合 $P$ 的一个众数

## Deep Learning 中的连接

| 应用 | KL 角色 |
|:---|:---|
| 交叉熵损失 | 分类中最小化 $H(P,Q)$ 等价于最小化 $D_{KL}(P\|Q)$（$P$ 固定） |
| VAE | ELBO = EL - $D_{KL}(q(z\|x)\|p(z))$，KL 正则化隐空间 |
| 知识蒸馏 | 蒸馏损失 = $\alpha \cdot \text{CE} + (1-\alpha) \cdot T^2 D_{KL}(\text{softmax}(z_s/T)\|\text{softmax}(z_t/T))$ |
| PPO / RLHF | PPO 使用 KL 惩罚约束策略更新幅度；RLHF 中 KL 正则防止模型偏离 SFT 基础 |
| 变分推断 | 最小化 $D_{KL}(q(\theta)\|p(\theta\|D))$ 将后验近似问题转化为优化问题 |

## 相关概念
- [[information_theory]] — 信息论总览
- [[entropy]] — 熵与 KL 散度的关系
- [[loss_function]] — KL 散度作为损失项
- [[reinforcement_learning_from_human_feedback]] — RLHF 中的 KL 正则
