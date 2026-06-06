---
id: normalization
type: concept
tags: [machine-learning, theoretical]
aliases: [归一化, normalization layer, 层归一化]
related_nodes: [batch_normalization, transformer_architecture]
---

# Normalization

## 概述

Normalization（归一化/规范化）是一类对神经网络中间层输出进行重中心化（re-centering）和重缩放（re-scaling）的技术，用于稳定训练、加速收敛、提升泛化能力。不同归一化方法在归一化的轴上有所区别。

## 详细阐述

### 定义与内涵

给定形状为 `(N, C, H, W)` 的激活张量（N=batch, C=channel, H=height, W=width），不同归一化方法计算均值和方差的维度集合不同。

### 主要方法对比

| 方法 | 归一化轴 | 适用场景 | 核心特征 | 出处 |
|:---|:---|:---|:---|:---|
| BatchNorm | N×H×W | CNN、固定 batch size | 依赖 batch 统计量，训练/推理行为不同 | [[ioffe_2015_batchnorm]] |
| LayerNorm | C×H×W | Transformer、RNN | 对 batch 不敏感，适合变长序列 | [[layer_norm_2016]] |
| InstanceNorm | H×W | 图像风格迁移 | 每个样本独立归一化 | — |
| GroupNorm | 按 channel 分组 | 小 batch 视觉任务 | CHW 分成 G 组后归一化 | [[group_norm_2018]] |
| RMSNorm | C×H^ | Transformer | LayerNorm 简化版，仅做缩放 | — |

### 详细讨论

#### Batch Normalization（[[batch_normalization]]）

对每个 channel 在 batch 和空间维度上归一化。优点：加速训练、允许更大学习率、提供一定正则化。缺点：batch size 小时统计不准确。参见 [[batch_normalization]] 独立页面。

#### Layer Normalization

对每个样本在特征维度上归一化，不受 batch size 影响。Transformer 架构的标准组件，位于每个子层之前（Pre-LN）或之后（Post-LN）。

#### Group Normalization

将 channel 分为 G 组，在组内做归一化。当 batch size=1 或非常小时（视频、3D 医学图像），BN 失效，GN 是有效替代。

#### RMSNorm

去掉 LayerNorm 中的均值中心化步骤，仅做均方根缩放。训练更稳定、计算更高效，被 LLaMA 等大语言模型采用。

### 应用场景

| 方法 | 典型架构 |
|:---|:---|
| BatchNorm | ResNet, VGG, ConvNeXt V2 |
| LayerNorm | Transformer, BERT, GPT, ViT |
| GroupNorm | Mask R-CNN, 小 batch 视觉 |
| RMSNorm | LLaMA, Mistral |

## 相关概念网络

- [[batch_normalization]] — 单独页面详述 BatchNorm
- [[regularization]] — BN 的正则化效应
- [[residual_connection]] — Pre-LN 与残差连接配合使用

## 引用资料

1. [[ioffe_2015_batchnorm]] — Batch Normalization
2. [[layer_norm_2016]] — Layer Normalization
3. [[group_norm_2018]] — Group Normalization

## 更新记录

- **2026-06-06**: 初始创建
