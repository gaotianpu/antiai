---
id: reed_2022_gato
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["Gato", "A Generalist Agent", "2205.06175"]
related_nodes: [generalist_agent, foundation_model]
authors: Scott Reed et al.
authors_institution: DeepMind
arxiv_id: 2205.06175
last_verified: 2026-08-03
---

# A Generalist Agent (Gato)

- **元数据**: arXiv 2205.06175 | 2022 | **作者**: Scott Reed et al. | **机构**: DeepMind
- **概述**: 提出 Gato，一个 11.8B 参数的通用智能体：将文本、图像、机器人动作统一 token 化，单模型在 604 个任务上达到专家水平 50%+，涵盖游戏、对话、机器人操控。
- **新颖概念**: [[generalist_agent]]
- **关键要点**: 1. 多模态统一 token 序列（与 LLM 同构） 2. 单一 Transformer 权重处理 604 任务 3. 规模与数据多样性驱动通用性
- **方法/发现**: 离线 RL + 监督学习混合训练（行为克隆式），证明"通用智能体"可从多任务数据直接涌现
- **局限/意义**: 各任务精度未超越专用模型；但验证了"一个模型 + 多模态数据"的通用路线，是机器人基础模型（RT-1/RT-2 等）的先声

## 引用
- **原始论文**: [arXiv:2205.06175](https://arxiv.org/abs/2205.06175)
- **相关概念**: [[generalist_agent]], [[foundation_model]]
