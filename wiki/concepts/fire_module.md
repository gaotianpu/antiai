---
id: fire_module
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [Fire模块, 挤压模块]
related_nodes: [iandola_2016_squeezenet, model_pruning]
last_verified: 2026-08-03
---

# Fire Module

## 定义
SqueezeNet 的核心构建块：Squeeze 层（1×1 卷积降维）+ Expand 层（1×1 与 3×3 混合卷积），以极小参数实现 AlexNet 级精度。

## 关键要点
- **设计策略**：用 1×1 卷积替代部分 3×3，减少 3×3 输入通道
- **成果**：参数量为 AlexNet 的 1/50，精度相当
- **意义**：早期模型压缩架构的经典案例

## 来源
- [[iandola_2016_squeezenet]] — SqueezeNet
