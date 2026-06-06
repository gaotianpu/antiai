---
id: sun_2023_dromedary
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["Dromedary", "Self-Align", "原则驱动自对齐", "2305.03047"]
related_nodes: ["touvron_2023_llama", "reinforcement_learning_from_human_feedback"]
arxiv_id: 2305.03047
authors: Zhiqing Sun et al.
authors_institution: IBM Research
last_verified: 2026-06-06
---

# Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision

- **元数据**: arXiv | 2023 | **作者**: Zhiqing Sun et al. | **机构**: IBM Research | 相关: [[touvron_2023_llama]], [[reinforcement_learning_from_human_feedback]]
- **概述**: 提出 SELF-ALIGN 方法，通过 16 条原则 + 上下文学习实现 LLM 自对齐，仅需 <300 行人工标注。
- **新颖概念**: [[self_align]], [[principle_driven_alignment]]
- **关键要点**: 1. 四阶段：合成提示生成 → 原则驱动上下文学习 → SFT 微调 → 改进 2. 仅需 200 种子提示 + 16 原则 + 5 演示 3. 在 LLaMA-65B 上训练 Dromedary，超越 Text-Davinci-003
- **方法/发现**: 基于原则驱动的推理替代 RLHF，大幅降低人工标注依赖
- **局限/意义**: 对齐成本大幅降低，为 Constitutional AI 实践提供了替代方案

## 引用
- **原始论文**: [arXiv:2305.03047](https://arxiv.org/abs/2305.03047) | [阅读笔记](../../raw/Dromedary.md)
- **相关概念**: [[reinforcement_learning_from_human_feedback]] | [[touvron_2023_llama]]
