---
id: cross_stage_partial_connection
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [跨阶段局部连接, CSP]
related_nodes: [wang_2019_cspnet, dense_connection]
last_verified: 2026-08-03
---

# Cross Stage Partial Connection (CSP)

## 定义
CSPNet 提出的连接策略：将特征图沿通道分为两部分，一部分经密集块变换，另一部分直接拼接（shortcut），减少重复梯度计算。

## 关键要点
- **梯度裁剪**：截断密集连接的重复梯度，降低计算冗余
- **收益**：同等精度下减少 20%+ 计算量，内存更省
- **应用**：CSPDarknet（YOLOv4 骨干）、CSP 变体广泛使用

## 来源
- [[wang_2019_cspnet]] — CSPNet
