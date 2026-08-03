---
id: receptive_field_block
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [感受野块, RFB]
related_nodes: [liu_2017_rfb, atrous_convolution]
last_verified: 2026-08-03
---

# Receptive Field Block (RFB)

## 定义
受人类视觉感受野结构（大小与离心率相关）启发的模块：用多个不同膨胀率的空洞卷积模拟不同大小与偏心率感受野，增强轻量特征判别力。

## 关键要点
- **多分支**：不同膨胀率分支 + 1×1 卷积组合
- **仿生设计**：模拟视网膜感受野的多尺度非均匀结构
- **应用**：RFB Net 检测器、SSD 变体

## 来源
- [[liu_2017_rfb]] — RFB Net
