---
id: tiny_ml
type: concept
tags: [machine-learning, empirical-study]
aliases: [微型机器学习, TinyML, 端侧智能]
related_nodes: [lin_2020_mcunet, neural_architecture_search]
last_verified: 2026-08-03
---

# TinyML

## 定义
在毫瓦级微控制器（MCU）等极端资源受限设备上运行机器学习模型的领域，需系统-模型协同设计突破内存/算力瓶颈。

## 关键要点
- **约束**：KB 级内存、几十 MHz 主频、电池供电
- **MCUNet 方案**：TinyNAS 搜索匹配硬件限制的架构 + TinyEngine 优化推理
- **突破**：首次在商用 MCU 上实现 >70% ImageNet 精度

## 来源
- [[lin_2020_mcunet]] — MCUNet
