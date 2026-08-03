---
id: ghost_module
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [Ghost模块, 幽灵模块]
related_nodes: [han_2019_ghostnet, cheap_operation]
last_verified: 2026-08-03
---

# Ghost Module

## 定义
GhostNet 提出的高效模块：先用少量卷积生成核心特征图，再通过廉价线性变换（depthwise 卷积）生成"幽灵"冗余特征图，两者拼接。

## 关键要点
- **冗余利用**：特征图存在大量冗余，不必全部重算
- **压缩比**：同等输出下计算量约为标准卷积的 1/2
- **即插即用**：可替换任意 CNN 中的卷积层

## 来源
- [[han_2019_ghostnet]] — GhostNet
