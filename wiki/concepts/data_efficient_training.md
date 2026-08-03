---
id: data_efficient_training
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [数据高效训练, 小数据训练]
related_nodes: [deit, knowledge_distillation, vision_transformer]
last_verified: 2026-08-03
---

# Data-Efficient Training

## 定义
在有限训练数据下获得强模型的技术集合：DeiT 证明通过强数据增强 + 蒸馏 token + 正则化，纯 Transformer 在 ImageNet-1k 即可训练成功。

## 关键要点
- **问题背景**：ViT 依赖 JFT-300M 级大数据，小数据下不如 CNN
- **DeiT 配方**：增强（Rand-Augment）+ EMA + 蒸馏 token + 重复增强
- **意义**：降低 Transformer 的数据门槛，推动 CV 高效训练

## 来源
- [[deit]] — DeiT 数据高效训练
