---
id: batch_normalization
type: concept
tags: [machine-learning, parameter-optimization, theoretical]
aliases: [BatchNorm, BN, 批归一化]
related_nodes: [ioffe_2015_batchnorm, resnet_2015]
last_verified: 2026-06-06
---

# Batch Normalization

## 定义
Batch Normalization（批归一化）是一种通过标准化每层输入的均值和方差来加速深度网络训练的技术。对每个 mini-batch 计算均值与方差，施加归一化后再做可学习的仿射变换（scale & shift），有效缓解内部协变量偏移（Internal Covariate Shift）。

## 核心优势
- **加速收敛**：允许更高学习率，减少对初始化的依赖
- **正则化效果**：因 mini-batch 引入的噪声带来轻微正则化
- **梯度平滑**：缓解梯度消失/爆炸，利于深层网络训练

## 关键变体
- **LayerNorm**（[[layer_norm_2016]]）— 沿特征维度归一化，适用于 RNN/Transformer
- **GroupNorm**（[[group_norm_2018]]）— 分组归一化，小 batch 场景更稳定

## 来源
- [[ioffe_2015_batchnorm]] — 首次提出 Batch Normalization
- [[resnet_2015]] — 深度依赖 BN 的代表架构
