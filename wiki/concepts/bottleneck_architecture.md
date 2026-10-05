---
id: bottleneck_architecture
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [瓶颈结构, Bottleneck]
related_nodes: [resnet_2015, mobilebert_2020_mobilebert, inverted_residual, knowledge_distillation]
last_verified: 2026-10-05
---

# Bottleneck Architecture

## 定义
"降维 → 变换 → 升维"的窄腰结构：先用 1×1 卷积压缩通道，核心计算在低维空间完成，再恢复维度，显著降低参数量与计算量。

## 关键要点
- **1×1 瓶颈**：由 [[resnet_2015]] 首次引入并推广，Inception 与 ResNet Bottleneck 的共同设计
- **MobileBERT 应用**：将瓶颈引入 Transformer 层压缩嵌入与 FFN
- **代价**：信息在瓶颈处可能丢失，需配合蒸馏等补偿

## 来源
- [[resnet_2015]] — 以「1×1 降维 / 3×3 变换 / 1×1 升维」的瓶颈块把 ResNet 推至 152 层，复杂度仍低于 VGG-19
- [[mobilebert_2020_mobilebert]] — 瓶颈结构压缩 BERT
