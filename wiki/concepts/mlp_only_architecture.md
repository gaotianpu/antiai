---
id: mlp_only_architecture
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [纯MLP架构, 无注意力架构]
related_nodes: [tolstikhin_2021_mlpmixer, touvron_2021_resmlp]
last_verified: 2026-08-03
---

# MLP-Only Architecture

## 定义
完全由 MLP（全连接层）构建的视觉架构，用"通道混合 + 空间混合"两类 MLP 替代卷积与注意力，如 MLP-Mixer、ResMLP。

## 关键要点
- **两类混合**：token-mixing MLP（跨空间位置）+ channel-mixing MLP（跨通道）
- **意义**：证明注意力和卷积并非精度必需，简单 MLP 亦可 SOTA 级
- **局限**：空间混合依赖输入分辨率，大图显存开销高

## 来源
- [[tolstikhin_2021_mlpmixer]] — MLP-Mixer
- [[touvron_2021_resmlp]] — ResMLP
