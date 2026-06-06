---
id: zhang_2023_adalora
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["AdaLoRA", "Adaptive Budget Allocation", "2303.10512"]
related_nodes: ["hu_2021_lora"]
arxiv_id: 2303.10512
authors: Qingru Zhang et al.
authors_institution: Microsoft
last_verified: 2026-06-06
---

# AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning

- **元数据**: arXiv | 2023 | **作者**: Qingru Zhang et al. | **机构**: Microsoft | 相关: [[hu_2021_lora]]
- **概述**: 在 LoRA 基础上按重要性动态分配参数预算，通过 SVD 参数化 + 剪枝不重要奇异值实现自适应。
- **关键要点**: 1. SVD 参数化增量更新 2. 按重要性分数动态分配预算 3. 剪枝不重要奇异值
- **方法/发现**: 替代 LoRA 的固定秩，根据权重重要性自适应分配秩
- **局限/意义**: 低预算场景显著优于 LoRA 基线

## 引用
- **原始论文**: [arXiv:2303.10512](https://arxiv.org/abs/2303.10512) | [阅读笔记](../../raw/AdaLoRA.md)
- **相关概念**: [[hu_2021_lora]]
