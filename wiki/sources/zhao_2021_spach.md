---
id: zhao_2021_spach
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["SPACH", "2108.13002"]
related_nodes: ["tolstikhin_2021_mlpmixer", "liu_2022_convnext"]
arxiv_id: 2108.13002
authors: Yucheng Zhao et al.
authors_institution: Microsoft Research Asia
last_verified: 2026-06-06
---

# A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP

- **元数据**: arXiv | 2021 | **作者**: Yucheng Zhao, Guangting Wang, Chuanxin Tang, Chong Luo, Wenjun Zeng, Zheng-Jun Zha | **机构**: Microsoft Research Asia, USTC
- **概述**: 提出 SPACH 统一框架对 CNN、Transformer、MLP 三种结构进行公平对比实验，发现混合卷积与 Transformer 的模型可达到与精心设计的 SOTA 模型相当的性能。
- **新颖概念**: —
- **关键要点**: 1. 设计 SPACH 框架，分别使用独立模块处理空间和通道信息 2. 中等规模下三种结构性能相近，但网络扩缩时行为各异 3. 混合使用卷积和 Transformer 模块的 Hybrid-MS-S+ 以 63M 参数达到 83.9% top-1 精度
- **方法/发现**: 在统一框架下系统比较不同结构，提出面向实践的混合模型设计原则。
- **局限/意义**: SPACH 为结构选择提供了实证基础，验证了混合架构在视觉任务中的潜力。实验局限于 ImageNet 分类，未覆盖检测/分割。

## 引用
- **原始论文**: [arXiv:2108.13002](https://arxiv.org/abs/2108.13002) | [阅读笔记](../../raw/cnn/spach.md)
