---
id: foundation_model
type: concept
tags: [machine-learning, theoretical]
aliases: [基础模型, 基石模型]
related_nodes: [segment_anything, transfer_learning, promptable_segmentation]
last_verified: 2026-08-03
---

# Foundation Model

## 定义
在大规模数据上预训练、可通过提示/微调适配广泛下游任务的通用模型，如 GPT、CLIP、SAM——"规模 + 通用性 + 适配性"三位一体。

## 关键要点
- **规模前提**：数据与参数规模是能力涌现的基础
- **通用适配**：提示（prompting）、微调、少样本均可迁移
- **SAM 案例**：分割基础模型，提示驱动的通用分割能力

## 来源
- [[segment_anything]] — SAM 作为分割基础模型
