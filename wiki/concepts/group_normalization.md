---
id: group_normalization
type: concept
tags: [computer-vision, machine-learning, theoretical]
aliases: [组归一化, GroupNorm, GN]
related_nodes: [group_norm_2018, normalization, batch_normalization]
last_verified: 2026-08-03
---

# Group Normalization

## 定义
将通道分成若干组，组内做归一化的方法，不依赖 batch 维度，在 batch size 很小（检测/分割/视频）时比 BatchNorm 更稳定。

## 关键要点
- **折中设计**：LN（全通道一组）与 BN（batch 归一化）之间
- **适用场景**：显存受限、batch 小（如 Mask R-CNN 检测头）
- **效果**：小 batch 下精度显著优于 BN，大 batch 略逊

## 来源
- [[group_norm_2018]] — Group Normalization
