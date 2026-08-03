---
id: label_smoothing
type: concept
tags: [machine-learning, theoretical]
aliases: [标签平滑, 软标签]
related_nodes: [szegedy_2015_inception_v3, loss_function, regularization]
last_verified: 2026-08-03
---

# Label Smoothing

## 定义
将硬标签（one-hot）替换为软标签（真实类 1-ε，其余类均分 ε），防止模型对训练标签过度自信，提升泛化能力。

## 关键要点
- **机理**：软化交叉熵目标，抑制 logits 无限增大（过度自信）
- **效果**：通常提升 0.2-0.5% 精度，对噪声标签更鲁棒
- **应用**：分类、蒸馏、检测（标签分配软化）广泛使用

## 来源
- [[szegedy_2015_inception_v3]] — Inception-v3 引入标签平滑
