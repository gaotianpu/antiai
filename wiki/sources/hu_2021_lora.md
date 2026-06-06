---
id: hu_2021_lora
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["LoRA", "Low-Rank Adaptation", "低秩适配", "2106.09685"]
related_nodes: [dettmers_2023_qlora, zhang_2023_adalora]
arxiv_id: 2106.09685
authors: Edward Hu et al.
authors_institution: Microsoft
last_verified: 2026-06-06
---

# LoRA: Low-Rank Adaptation of Large Language Models

- **元数据**: arXiv | 2021 | **作者**: Edward Hu et al. | **机构**: Microsoft
- **概述**: 提出低秩适配方法，冻结预训练权重，插入可训练的低秩矩阵，大幅降低微调参数量和显存需求。
- **关键要点**: 1. 冻结原权重，插入低秩分解矩阵 2. 推理时零额外延迟 3. 可与量化等正交技术叠加
- **方法/发现**: 预训练权重变化量 = A×B（低秩分解），秩 r ≪ d
- **局限/意义**: 成为参数高效微调事实标准，衍生 QLoRA、AdaLoRA 等大量工作

## 引用
- **原始论文**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | [阅读笔记](../../raw/LoRA.md)
