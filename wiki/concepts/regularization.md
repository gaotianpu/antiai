---
id: regularization
type: concept
tags: [machine-learning, theoretical]
aliases: [正则化, weight decay, dropout]
related_nodes: [batch_normalization, optimizer]
---

# Regularization

## 概述

Regularization（正则化）是一类用于降低模型泛化误差、防止过拟合的技术集合。其核心思想是在训练过程中施加约束或噪声，使模型不能完美记忆训练数据，从而提升对未见数据的适应能力。

## 详细阐述

### 主要方法

#### 参数范数正则化

- **L1 正则化（Lasso）**：在损失中加入权重的 L1 范数，诱导稀疏解
- **L2 正则化（Weight Decay）**：在损失中加入权重的 L2 范数，等价于 [[optimizer]] 更新时的权重衰减

#### 结构化正则化

| 方法 | 机制 | 适用场景 |
|:---|:---|:---|
| Dropout | 训练时随机丢弃神经元，强制冗余表示 | 全连接层 |
| DropPath | 随机丢弃整个路径（如残差分支） | 深度网络 / ViT |
| Stochastic Depth | 随机层丢弃 | 深层 CNN |
| Label Smoothing | 软化 one-hot 标签，防止置信度过高 | 分类任务 |
| [[batch_normalization]] | 通过噪声注入（mini-batch 统计量）带来微弱正则化 | 通用 |
| Early Stopping | 验证集指标不再提升时终止训练 | 通用 |

#### 数据层面正则化

- [[data_augmentation]] — 通过变换扩展训练分布（参见独立页面）
- Mixup / CutMix — 样本间线性插值

### 正则化与优化器

Weight Decay 与 AdamW 的结合是现代训练的标准实践。[[optimizer]] 页详细讨论了 AdamW 将权重衰减与学习率解耦的设计。

### 不同架构中的正则化实践

- **CNN**：Dropout 在卷积层效果有限；[[batch_normalization]] 更有效
- **Transformer**：Dropout + DropPath + Label Smoothing + Stochastic Depth 并用
- **大语言模型**：Weight Decay 常仅作用于非 bias / non-LayerNorm 参数

## 相关概念网络

- [[batch_normalization]] — 归一化带来的正则化效应
- [[optimizer]] — 优化器中权重衰减的实现
- [[data_augmentation]] — 数据层面的正则化
- [[normalization]] — 不同归一化方法的正则化效果

## 引用资料

1. [[ioffe_2015_batchnorm]] — BN 的正则化效应
2. [[layer_norm_2016]] — LayerNorm 的稳定训练作用

## 更新记录

- **2026-06-06**: 初始创建
