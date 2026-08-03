---
id: siamese_network
type: concept
tags: [machine-learning, empirical-study]
aliases: [孪生网络, 双胞胎网络]
related_nodes: [simsiam_2020, contrastive_learning, dual_encoder]
last_verified: 2026-08-03
---

# Siamese Network

## 定义
两个（或共享权重的）相同网络分别编码两个输入，再比较输出相似度的架构，广泛用于对比学习、人脸验证、检索。

## 关键要点
- **权重共享**：两分支通常共享参数（或互相蒸馏）
- **SimSiam 变体**：两分支互为目标（不对称停止梯度），无需负样本
- **应用**：度量学习（验证/检索）与自监督表示学习

## 来源
- [[simsiam_2020]] — SimSiam 孪生自监督
