---
id: constitutional_ai
type: concept
tags: [machine-learning, theoretical, RL]
aliases: [宪法AI, 宪法人工智能, CAI]
related_nodes: [bai_2022_constitutional_ai, principle_driven_alignment, reinforcement_learning_from_human_feedback]
last_verified: 2026-08-03
---

# Constitutional AI (CAI)

## 定义
Anthropic 提出的对齐范式：以一组人工撰写的"宪法原则"指导模型自我批评与修订，生成 AI 反馈数据（RLAIF），替代人类偏好标注进行 RL 对齐。

## 关键要点
- **两阶段**：监督阶段（模型按原则修订自身输出）→ RL 阶段（AI 反馈偏好对训练）
- **RLAIF**：反馈来源从人类转移到 AI，标注成本大幅降低
- **思想源头**：启发了 SELF-ALIGN（[[principle_driven_alignment]]）、Dromedary 等无 RLHF 对齐路线

## 来源
- [[bai_2022_constitutional_ai]] — 提出宪法 AI
