---
id: data_augmentation
type: concept
tags: [machine-learning, empirical-study, parameter-optimization]
aliases: [数据增强, augmentation]
related_nodes: [chen_2020_simclr, he_2019_moco, flip, mae, deit]
last_verified: 2026-06-06
---

# Data Augmentation

## 定义
Data Augmentation（数据增强）通过对原始数据施加保持标签语义的变换（旋转、翻转、裁剪、色彩抖动、噪声注入等）来扩展训练数据集，在不收集新数据的前提下提升模型的泛化能力和鲁棒性。

## 增强方法
### 基本变换
- 几何变换：旋转、翻转、裁剪、缩放
- 色彩变换：亮度/对比度/饱和度调整
- 噪声注入：高斯噪声、椒盐噪声

### 高级增强
- **混合增强**：Mixup、CutMix — 样本间线性插值
- **自动增强**：AutoAugment、RandAugment — 搜索最优增强策略
- **数据驱动增强**：SimCLR 对比学习中作为正对构建的关键组件

## 来源
- [[chen_2020_simclr]] — 增强作为对比学习核心
- [[flip]] — Frequency-domain augmentation
- [[mae]] — 随机掩码作为自监督增强
- [[deit]] — 数据增强驱动的 Transformer 训练
