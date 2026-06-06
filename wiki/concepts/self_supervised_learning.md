---
id: self_supervised_learning
type: concept
tags: [machine-learning, empirical-study, theoretical]
aliases: [自监督学习, SSL, 无监督表示学习]
related_nodes: [attention_mechanism, data_augmentation, generative_model]
---

# Self-Supervised Learning

## 概述

Self-Supervised Learning（自监督学习，SSL）是一种从无标签数据中构造监督信号的表征学习范式。通过设计预文本任务（pretext task），模型在不依赖人工标注的条件下学习通用特征表示。

## 详细阐述

### 定义与内涵

SSL 的核心思想是利用数据自身的结构生成伪标签（pseudo-labels），将无监督问题转化为有监督问题。与 [[transfer_learning]] 紧密关联——SSL 预训练 + 下游微调是目前 NLP 和 CV 的主流范式。

### 三大范式

#### 对比学习（Contrastive Learning）

拉近正对、推开负对，学习判别性表示。

| 方法 | 核心设计 | 出处 |
|:---|:---|:---|
| SimCLR | 大 batch + NT-Xent 损失 + 强增强 | [[chen_2020_simclr]] |
| MoCo | 动量编码器 + 队列维护负样本 | [[he_2019_moco]] |
| MoCo v2 | 融合 SimCLR 改进（MLP 投影层等） | [[chen_2020_mocov2]] |
| MoCo v3 | 扩展到 ViT 架构 | [[moco_v3]] |

#### 掩码建模（Masked Modeling）

- **MAE**（[[ho_2020_ddpm]] 同作者）：对图像 patch 随机掩码 + 解码器重建
- **BERT**：掩码语言模型（MLM）
- **iGPT**：自回归像素预测 [[igpt]]

#### 聚类方法（Clustering）

SwAV（Swapping Assignments between Views）等，通过在线聚类分配与对比学习结合。

### 关键设计要素

- **数据增强**：正对构建的质量直接影响 SSL 性能，参见 [[data_augmentation]]
- **负样本策略**：是否需要负样本（对比 vs 非对比方法）
- **架构设计**：预测头（projection head / prediction head）的影响

### 应用场景

- 视觉基础模型预训练（[[vitdet]]、[[simplevit]]）
- NLP 大型语言模型预训练
- 多模态表示学习（[[blip]]、[[kosmos_1]]）

## 相关概念网络

- [[data_augmentation]] — SSL 中增强是正对构建的关键
- [[attention_mechanism]] — 自注意力是 SSL 模型的常用骨干
- [[generative_model]] — 掩码建模类 SSL 与生成模型交叉
- [[transfer_learning]] — SSL 预训练的目标是迁移到下游任务

## 引用资料

1. [[chen_2020_simclr]] — SimCLR 对比学习框架
2. [[he_2019_moco]] — MoCo 动量对比
3. [[chen_2020_mocov2]] — MoCo v2 改进
4. [[moco_v3]] — MoCo v3 扩展到 ViT

## 更新记录

- **2026-06-06**: 初始创建
