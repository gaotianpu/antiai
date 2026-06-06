---
id: krizhevsky_2012_alexnet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["AlexNet", "ImageNet Classification with Deep Convolutional Neural Networks"]
related_nodes: []
authors: Alex Krizhevsky et al.
authors_institution: University of Toronto
last_verified: 2026-06-06
---

# ImageNet Classification with Deep Convolutional Neural Networks

- **元数据**: NeurIPS | 2012 | **作者**: Alex Krizhevsky et al. | **机构**: University of Toronto
- **概述**: 首个在 ImageNet 上大幅超越传统方法的深度 CNN，引爆深度学习革命
- **新颖概念**: [[convolutional_neural_network]], [[dropout]], [[data_augmentation]]
- **关键要点**: 1. 5 卷积层 + 3 全连接层的 60M 参数网络 2. 使用 ReLU 非饱和激活函数加速训练 3. 引入 Dropout 正则化与 GPU 并行训练
- **方法/发现**: 在 ImageNet LSVRC-2010 上取得 top-1 37.5%/top-5 17.0% 错误率，ILSVRC-2012 夺冠
- **局限/意义**: 参数量大、计算成本高；但证明了深度 CNN 在大规模图像分类中的可行性，开启深度学习时代

## 引用
- **原始论文**: [NeurIPS](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | [阅读笔记](../../raw/cnn/alexnet.md)
