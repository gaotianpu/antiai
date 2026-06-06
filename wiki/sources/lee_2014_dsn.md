---
id: lee_2014_dsn
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["Deeply-Supervised Nets", "DSN", "1409.5185"]
related_nodes: []
arxiv_id: 1409.5185
authors: Chen-Yu Lee et al.
authors_institution: UC San Diego
last_verified: 2026-06-06
---

# Deeply-Supervised Nets

- **元数据**: arXiv | 2014 | **作者**: Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, Zhuowen Tu | **机构**: UC San Diego, Microsoft Research
- **概述**: 提出深度监督网络（DSN），通过在隐藏层引入辅助监督信号（companion objective）来解决深层网络的梯度消失问题，使学习过程更加直接透明。
- **关键要点**: 1. 在中间隐藏层添加辅助分类器，与输出层主目标联合优化 2. 提高早期层特征的判别力和鲁棒性 3. 在 MNIST、CIFAR-10/100、SVHN 上取得当时最佳结果
- **方法/发现**: 扩展 SGD 方法分析辅助损失对梯度流动的影响，证明 DSN 缓解了梯度消失/爆炸问题。
- **局限/意义**: DSN 思想被后续 GoogLeNet（Inception v1）的辅助分类器和 Deep Supervision 方法继承。需注意辅助损失的权重调节以避免干扰主目标。

## 引用
- **原始论文**: [arXiv:1409.5185](https://arxiv.org/abs/1409.5185) | [阅读笔记](../../raw/cnn/Deeply-Supervised.md)
