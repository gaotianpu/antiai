---
id: information_theory
type: concept
tags: [theoretical, information-theory, machine-learning]
aliases: ["信息论", "熵", "KL散度", "互信息", "info theory for DL"]
related_nodes: [math_for_deep_learning, probability_statistics, loss_function]
---

# 信息论

信息论为深度学习提供了量化"信息"和"不确定性的数学语言，是许多损失函数和训练目标的根源。

## 核心概念

- **熵 (Entropy)**：$H(p) = - \sum p(x) \log p(x)$，衡量分布 $p$ 的平均信息量/不确定性
- **交叉熵 (Cross-Entropy)**：$H(p,q) = - \sum p(x) \log q(x)$，衡量用分布 $q$ 编码来自 $p$ 的数据所需的平均比特数
- **KL 散度 (KL Divergence)**：$D_{KL}(p\|q) = \sum p(x) \log \frac{p(x)}{q(x)} = H(p,q) - H(p)$，两个分布间距离的非对称度量
- **互信息 (Mutual Information)**：$I(X;Y) = D_{KL}(p(x,y)\|p(x)p(y))$，衡量两个随机变量间的依赖程度
- **困惑度 (Perplexity)**：$\text{PPL} = 2^{H(p)}$，语言模型评价中熵的指数变换

## 关键关系

交叉熵 = 熵 + KL 散度：$H(p,q) = H(p) + D_{KL}(p\|q)$。当 $p$ 固定时，最小化交叉熵等价于最小化 KL 散度。

## 为什么 DL 需要它

| 应用 | 信息论基础 |
|:---|:---|
| 交叉熵损失 | 分类任务的标准损失函数，等价于最小化预测分布与真实分布间的 KL 散度 |
| VAE | ELBO 优化包含 $D_{KL}(q(z|x)\|p(z))$ 正则项 |
| 知识蒸馏 | 教师输出的软标签分布与学生输出的 KL 散度作为蒸馏损失 |
| 互信息最大化 (InfoNCE) | 对比学习的理论基础，最大化正样本对的互信息 |
| 语言模型评估 | Perplexity 衡量模型对测试集的建模能力 |
