---
id: wang_2022_selfinstruct
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["Self-Instruct", "自我指导", "2212.10560"]
related_nodes: []
arxiv_id: 2212.10560
authors: Yizhong Wang et al.
authors_institution: University of Washington
last_verified: 2026-06-06
---

# Self-Instruct: Aligning Language Model with Self Generated Instructions

- **元数据**: arXiv | 2022 | **作者**: Yizhong Wang et al. | **机构**: University of Washington
- **概述**: 提出 Self-Instruct 方法，让 LLM 自我生成指令数据指导自身微调，大幅降低人工标注成本。
- **新颖概念**: [[self_instruct]], [[instruction_tuning]]
- **关键要点**: 1. 种子指令集 + LLM 自生成扩展 2. 指令多样性过滤 3. 自生成数据微调效果匹敌人工标注
- **方法/发现**: 175 个种子指令 → LLM 生成更多指令 → 过滤 → SFT 微调
- **局限/意义**: 成为 Alpaca/Vicuna 等指令微调数据生成的底层方法

## 引用
- **原始论文**: [arXiv:2212.10560](https://arxiv.org/abs/2212.10560) | [阅读笔记](../../raw/self-Instruct.md)
