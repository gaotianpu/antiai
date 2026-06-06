---
id: zhan_2018_cdp
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["CDP", "1809.01407"]
related_nodes: []
arxiv_id: 1809.01407
authors: Xiaohang Zhan et al.
authors_institution: The Chinese University of Hong Kong
last_verified: 2026-06-06
---

# Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition

- **元数据**: arXiv | 2018 | **作者**: Xiaohang Zhan, Ziwei Liu, Dahua Lin, Chen Change Loy | **机构**: The Chinese University of Hong Kong, SenseTime Group
- **概述**: 提出 CDP（共识驱动传播），通过 committee 和 mediator 模块从海量未标注人脸数据中稳健选择正样本对，仅用 9% 标签即可接近全监督性能。
- **关键要点**: 1. Committee 由多种 CNN 架构构成，提供多视图信息 2. Mediator（MLP 分类器）聚合 committee 意见，以高精度选择正样本对 3. 伪标签传播后在多任务框架中联合训练
- **方法/发现**: Query-by-Committee 启发 + Mediator 二分类器 + k-NN 图构建 + 伪标签传播
- **局限/意义**: MegaFace 上 9% 标签达 78.18%（全监督 78.52%），开创了海量未标注人脸数据半监督学习的实用范式。

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/1809.01407) | [阅读笔记](../../raw/cnn/cdp.md)
