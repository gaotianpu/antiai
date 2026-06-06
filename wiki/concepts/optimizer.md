---
id: optimizer
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["优化器", "梯度下降", "SGD", "Adam"]
related_nodes: ["loss_function", "derivative_and_gradient"]
last_verified: 2026-06-06
---

# Optimizer（优化器）

## 定义
优化器是神经网络训练中更新权重以最小化损失函数的算法，决定了收敛速度、泛化性能和训练稳定性。

## 常见优化器

| 优化器 | 核心思想 | 特点 | 出处 |
|:---|:---|:---|:---|
| **SGD** | θ = θ - η∇L | 基础方法，需精细调学习率 | Robbins & Monro, 1951 |
| **Momentum** | 累加历史梯度方向加速 | 加速收敛，越过局部极小 | Polyak, 1964 |
| **AdaGrad** | 自适应学习率（稀疏梯度放大） | 适合稀疏特征，学习率单调衰减 | Duchi et al., 2011 |
| **RMSProp** | 梯度平方指数移动平均 | 解决 AdaGrad 学习率消失 | Hinton, 2012 |
| **Adam** | Momentum + RMSProp 结合 | 自适应学习率 + 动量，最广泛使用 | Kingma & Ba, 2014 |
| **AdamW** | Adam + 权重衰减解耦 | 改进正则化，Transformer 标配 | Loshchilov & Hutter, 2017 |
| **Nadam** | Adam + Nesterov 动量 | 加速收敛变体 | Dozat, 2016 |

## 演进趋势

```
SGD ─→ Momentum ─→ Nesterov
  │
  └→ AdaGrad ─→ RMSProp ─→ Adam ─→ AdamW
                 │
                 └→ Nadam
```

- **SGD** → 简单可靠，但收敛慢
- **Momentum** → 加速 + 越过局部极小
- **AdaGrad** → 自适应学习率，但学习率过早消失
- **RMSProp** → 解决学习率消失问题
- **Adam** → 自适应 + 动量，成为默认选择
- **AdamW** → 正确解耦权重衰减，LLM 训练首选

## 关键发现
- Adam 在 Transformer/BERT 上优于 SGD，但泛化有时不及 SGD
- AdamW 在 LLM 训练中成为事实标准（BERT、GPT、LLaMA 均使用）
- 学习率预热（Warmup）是现代 LLM 训练的必要技巧

## 相关概念
- [[transfer_learning]] — 微调中优化器选择策略
- [[batch_normalization]] — 与优化器协同的归一化方法
