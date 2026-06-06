---
id: ouyang_2022_instructgpt
type: source
tags: [NLP, machine-learning, empirical-study]
aliases: [InstructGPT, 根据人类反馈的指令学习, 2203.02155]
related_nodes: [reinforcement_learning_from_human_feedback, openai, gao_2022_rmoveroptimization]
arxiv_id: 2203.02155
authors: Long Ouyang et al.
authors_institution: OpenAI
last_verified: 2026-06-06
---

# Training language models to follow instructions with human feedback

- **元数据**: Conference | 2022 | **作者**: Long Ouyang et al. | **机构**: OpenAI | 相关: [[reinforcement_learning_from_human_feedback]]
- **概述**: 使用人类反馈强化学习（RLHF）微调 GPT-3，使模型更好地遵循用户指令，降低有害和不实输出。
- **关键要点**: 1. 监督微调（SFT）→ 奖励模型（RM）→ PPO 强化学习 2. 1.3B InstructGPT 优于 175B GPT-3 3. 对齐研究的基础范式
- **方法/发现**: InstructGPT 在有用性和真实性上显著优于 GPT-3，编造减少
- **局限/意义**: 依赖高质量人类标注；奖励模型有局限性

## 引用
- **原始论文**: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) | [阅读笔记](../../raw/nlp/gpt_InstructGPT.md)
- **相关概念**: [[reinforcement_learning_from_human_feedback]] | [[ppo]]
- **相关实体**: [[openai]]
