---
id: howard_2019_mobilenet_v3
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["MobileNet v3", "MobileNetV3", "Searching for MobileNetV3", "1905.02244"]
related_nodes: []
arxiv_id: 1905.02244
authors: Andrew Howard et al.
authors_institution: Google
last_verified: 2026-06-06
---

# Searching for MobileNetV3

- **元数据**: arXiv:1905.02244 | ICCV 2019 | **作者**: Andrew Howard et al. | **机构**: Google
- **概述**: 结合 NAS（MnasNet）和 NetAdapt 自动搜索优化 MobileNet 架构
- **关键要点**: 1. 结合 MnasNet 平台感知 NAS 和 NetAdapt 网络适应算法 2. 引入 h-swish 激活函数和 SE 模块 3. 提出 Lite R-ASPP 分割解码器
- **方法/发现**: 自动搜索与人工设计协同，在移动端取得精度-速度最佳权衡
- **局限/意义**: 搜索空间限制了架构多样性；但开创了 NAS + 人工设计混合范式

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1905.02244) | [阅读笔记](../../raw/cnn/MobileNet_v3.md)
