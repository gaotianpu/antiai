---
id: hu_2017_senet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["SENet", "Squeeze-and-Excitation", "1709.01507"]
related_nodes: ["resnet_2015", "xie_2016_resnext", "huang_2016_densenet"]
arxiv_id: 1709.01507
authors: Jie Hu et al.
authors_institution: Momenta
last_verified: 2026-06-06
---

# Squeeze-and-Excitation Networks

- **元数据**: CVPR 2018 | arXiv | 2017 | **作者**: Jie Hu, Li Shen, Gang Sun | **机构**: Momenta
- **概述**: 提出 SE（Squeeze-and-Excitation）块，通过显式建模通道间依赖关系实现自适应通道特征重校准，以极小的计算开销显著提升 CNN 性能，获 ILSVRC 2017 分类冠军。
- **关键要点**: 1. Squeeze 操作通过全局平均池化将空间信息压缩为通道描述符 2. Excitation 操作使用两个全连接层生成通道权重，实现非线性通道门控 3. 可无缝嵌入 ResNet、ResNeXt、Inception 等架构，仅增加少量参数
- **方法/发现**: SE-ResNet-152 在 ImageNet 上将 top-5 错误率降至 2.251%，相对提升约 25%。
- **局限/意义**: SE 是通道注意力机制的里程碑，启发了后续 CBAM、ECA、GE 等工作。但 SE 的全连接层在极窄网络中可能成为瓶颈。

## 引用
- **原始论文**: [arXiv:1709.01507](https://arxiv.org/abs/1709.01507) | [阅读笔记](../../raw/cnn/senet.md)
