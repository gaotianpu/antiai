---
id: data_augmentation
type: concept
tags: [machine-learning, empirical-study, parameter-optimization]
aliases: [数据增强, augmentation]
related_nodes: [self_supervised_learning, chen_2020_simclr, he_2019_moco, flip, mae, deit]
last_verified: 2026-06-06
---

# Data Augmentation

## 概述

Data Augmentation（数据增强）通过对原始数据施加保持标签语义的变换（旋转、翻转、裁剪、色彩抖动、噪声注入、混合等）来扩展训练数据集，在不收集新数据的前提下提升模型的泛化能力和鲁棒性。它是 [[self_supervised_learning]] 和对比学习的核心组件。

## 详细阐述

### 定义与内涵

数据增强通过**语义保持变换**（label-preserving transformations）生成逼真的新样本，使模型对输入的不变性（invariance）或等价性（equivariance）更强。在 CV 中具体表现为几何/色彩变换，在 NLP 中表现为回译/词替换/掩码，在语音中表现为 SpecAugment 等频域扰动。

### 图像增强方法

#### 基本几何变换
- **空间变换**：随机旋转（±15°~±30°）、水平翻转、随机裁剪（RandomResizedCrop）、仿射变换
- **色彩变换**：亮度/对比度/饱和度/色调的随机抖动（ColorJitter）
- **噪声注入**：高斯噪声、椒盐噪声、模糊

#### 高级增强
| 方法 | 机制 | 出处 |
|:---|:---|:---|
| Cutout | 随机遮挡正方形区域 | — |
| Mixup | 样本对线性插值（$x = \lambda x_i + (1-\lambda)x_j$），标签同时插值 | — |
| CutMix | 剪切一幅图的一块贴到另一幅上，标签按面积比例混合 | — |
| RandAugment | 从增强池中随机采样 N 个操作，幅度参数 M 统一控制 | — |
| AugMix | 多条增强链混合 + Jensen-Shannon 一致性损失 | — |
| Frequency Augmentation | 频域扰动，如 [[flip]] | [[flip]] |

### 增强在自监督学习中的作用

在 [[self_supervised_learning]] 中，数据增强是构建正对（positive pairs）的唯一手段。SimCLR 系统性研究了增强组合的影响，发现 RandomResizedCrop + ColorJitter 组合最有效 [[chen_2020_simclr]]。MoCo 系列继承类似增强管线 [[he_2019_moco]]、[[chen_2020_mocov2]]。

### 自动数据增强

手工设计增强策略需要大量调参。自动增强方法将策略选择转化为搜索/优化问题：AutoAugment（强化学习搜索）、RandAugment（简化搜索空间）、Fast AutoAugment（基于密度估计）。

### 各领域的增强实践

| 领域 | 常用增强 |
|:---|:---|
| ImageNet 分类 | RandomResizedCrop + Flip + ColorJitter |
| 检测/分割 | 几何增强 + Mixup/CutMix + Mosaic |
| NLP | 回译（back-translation）、随机词掩码、EDA |
| 语音 | SpecAugment（频带掩码 + 时间掩码） |
| 时间序列 | 缩放/扭曲/漂移/置换 |

## 相关概念网络

- [[self_supervised_learning]] — SSL 依赖增强构建正对
- [[regularization]] — 增强的正则化效应
- [[generative_model]] — 增强与生成模型的交叉

## 引用资料

1. [[chen_2020_simclr]] — SimCLR 中增强组合的系统研究
2. [[he_2019_moco]] — MoCo 对比学习增强管线
3. [[chen_2020_mocov2]] — MoCo v2 增强改进
4. [[flip]] — 频域增强
5. [[mae]] — 掩码增强
6. [[deit]] — 数据增强驱动的 Transformer 训练

## 更新记录

- **2026-06-06**: 初始创建（扩展重写）
