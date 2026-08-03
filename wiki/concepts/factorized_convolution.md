---
id: factorized_convolution
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [因式分解卷积, 卷积分解]
related_nodes: [szegedy_2015_inception_v3, inception_module]
last_verified: 2026-08-03
---

# Factorized Convolution

## 定义
将大卷积核分解为小卷积组合（如 5×5 → 两个 3×3，3×3 → 1×3 + 3×1），在保持感受野的同时显著减少参数与计算量。

## 关键要点
- **等价感受野**：分解后感受野不变，非线性增加
- **节省参数**：5×5（25）→ 3×3+3×3（18）参数
- **应用**：Inception-v3 的系统设计原则之一

## 来源
- [[szegedy_2015_inception_v3]] — Inception-v3 设计原则
