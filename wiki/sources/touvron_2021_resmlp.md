---
id: touvron_2021_resmlp
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["ResMLP", "2105.03404"]
related_nodes: []
arxiv_id: 2105.03404
authors: Hugo Touvron et al.
authors_institution: Facebook AI Research
last_verified: 2026-06-06
---

# ResMLP: Feedforward networks for image classification with data-efficient training

- **元数据**: arXiv:2105.03404 | NeurIPS 2021 | **作者**: Hugo Touvron et al. | **机构**: Facebook AI Research
- **概述**: 完全基于 MLP 的图像分类架构，无需自注意力机制
- **关键要点**: 1. 纯 MLP 残差网络（线性层 + 两层前馈网络交替） 2. 依赖强数据增广和蒸馏策略 3. 支持自监督训练
- **方法/发现**: 在 ImageNet 上取得与 CNN/Transformer 竞争的精度-复杂度权衡
- **局限/意义**: 对数据增广和蒸馏依赖强；但证明了 MLP 在视觉任务中的潜力，推动 MLP-Mixer 等后续工作

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/2105.03404) | [阅读笔记](../../raw/cnn/ResMLP.md)
