---
id: bai_2022_constitutional_ai
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["Constitutional AI", "宪法AI", "Harmlessness from AI Feedback", "2212.08073"]
related_nodes: [constitutional_ai, reinforcement_learning_from_human_feedback]
authors: Yuntao Bai et al.
authors_institution: Anthropic
arxiv_id: 2212.08073
last_verified: 2026-08-03
---

# Constitutional AI: Harmlessness from AI Feedback

- **元数据**: arXiv 2212.08073 | 2022 | **作者**: Yuntao Bai et al. | **机构**: Anthropic
- **概述**: 提出宪法 AI（CAI），用一组宪法原则（Constitution）替代人类偏好标注：模型自我批评与修订生成对齐数据，再经 RL 训练实现无害性对齐。
- **新颖概念**: [[constitutional_ai]]
- **关键要点**: 1. 原则集驱动自我修订（RLAIF 数据生成） 2. 两阶段：监督式自我批评 + RL 偏好优化 3. 无需人类标注偏好即可对齐
- **方法/发现**: 模型按原则批评并修订自身输出 → 生成偏好对 → RLHF/RL 训练，有害性显著低于纯 RLHF 基线
- **局限/意义**: 原则仍由人写（"元原则"成本）；证明 AI 反馈（RLAIF）可替代人类反馈，是 Dromedary/SELF-ALIGN 等后续工作的思想源头

## 引用
- **原始论文**: [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- **相关概念**: [[constitutional_ai]], [[reinforcement_learning_from_human_feedback]]
