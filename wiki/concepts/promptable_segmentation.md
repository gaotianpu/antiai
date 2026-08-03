---
id: promptable_segmentation
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [可提示分割, 提示分割]
related_nodes: [segment_anything, foundation_model, instance_segmentation]
last_verified: 2026-08-03
---

# Promptable Segmentation

## 定义
SAM 提出的交互式分割范式：模型接受点、框、掩码、文本等任意提示输出对应分割结果，将分割从"任务专用"变为"通用可提示"。

## 关键要点
- **提示类型**：前景/背景点、包围框、粗糙掩码均可作为输入
- **歧义处理**：同一提示可输出多个有效掩码
- **泛化**：零样本迁移到未见过的数据分布与任务

## 来源
- [[segment_anything]] — SAM：分割基础模型
