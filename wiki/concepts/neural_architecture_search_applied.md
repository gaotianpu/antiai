---
id: neural_architecture_search_applied
type: concept
tags: [machine-learning, empirical-study]
aliases: [NAS应用, 自动搜索实践]
related_nodes: [howard_2019_mobilenet_v3, neural_architecture_search]
last_verified: 2026-08-03
---

# NAS in Practice

## 定义
将 NAS 应用于实际模型设计的工程范式：如 MobileNetV3 用 MnasNet（NAS）+ NetAdapt 两阶段搜索优化移动端架构。

## 关键要点
- **硬件感知**：搜索目标直接是延迟/能耗等硬件指标
- **两阶段**：平台感知 NAS 搜全局结构 + NetAdapt 逐层精调
- **成果**：MobileNetV3 同时优化精度与延迟，成为移动端标配

## 来源
- [[howard_2019_mobilenet_v3]] — MobileNetV3 搜索实践
