---
id: focal_loss
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [焦点损失, 聚焦损失]
related_nodes: [focal_loss_2017, loss_function, single_stage_detector]
last_verified: 2026-08-03
---

# Focal Loss

## 定义
针对单阶段检测器正负样本极端不平衡设计的损失：在交叉熵上按样本难度加权，$(1-p_t)^\gamma$ 降低易分负样本的损失贡献。

## 关键要点
- **两类问题**：负样本远多于正样本 + 大量易分样本淹没难例
- **公式**：$FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)$
- **效果**：RetinaNet 精度超越两阶段检测器，开创单阶段新时代

## 来源
- [[focal_loss_2017]] — Focal Loss 原始论文
