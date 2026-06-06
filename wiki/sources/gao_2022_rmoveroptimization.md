---
id: gao_2022_rmoveroptimization
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["Reward Model Overoptimization", "Scaling Laws for RM", "2210.10760"]
related_nodes: ["ouyang_2022_instructgpt"]
arxiv_id: 2210.10760
authors: Leo Gao et al.
authors_institution: OpenAI
last_verified: 2026-06-06
---

# Scaling Laws for Reward Model Overoptimization

- **元数据**: arXiv | 2022 | **作者**: Leo Gao et al. | **机构**: OpenAI | 相关: [[ouyang_2022_instructgpt]]
- **概述**: 研究 RLHF 中奖励模型过优化现象（Goodhart's Law），提出 scaling law 预测最佳优化强度。
- **新颖概念**: [[reward_overoptimization]], [[goodhart_law]]
- **关键要点**: 1. 过度优化导致 Reward Hacking 2. Goodhart's Law 3. 最优 KL 预算存在 scaling law
- **方法/发现**: 奖励得分 ≠ 真实能力，存在"过度优化"临界点
- **局限/意义**: 指导 RLHF 中 KL 正则化强度的选择

## 引用
- **原始论文**: [arXiv:2210.10760](https://arxiv.org/abs/2210.10760) | [阅读笔记](../../raw/RM_Overoptimization.md)
- **相关概念**: [[ouyang_2022_instructgpt]]
