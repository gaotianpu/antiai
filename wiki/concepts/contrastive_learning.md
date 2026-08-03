---
id: contrastive_learning
type: concept
tags: [machine-learning, empirical-study]
aliases: [对比学习]
related_nodes: [chen_2020_simclr, momentum_contrast, self_supervised_learning]
last_verified: 2026-08-03
---

# Contrastive Learning

## 定义
自监督学习范式：将同一样本的不同增强视图视为正对拉近，不同样本视为负对推远，学习对扰动鲁棒的判别表示。

## 关键要点
- **核心目标**：正对相似度 ↑，负对相似度 ↓（InfoNCE 类损失）
- **关键设计**：数据增强组合、MLP 投影头、大批量（SimCLR）
- **演化**：负样本依赖 → MoCo 队列 → SimSiam 无需负样本

## 来源
- [[chen_2020_simclr]] — SimCLR
