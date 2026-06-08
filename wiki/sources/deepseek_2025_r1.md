---
id: deepseek_2025_r1
type: source
tags: [empirical-study, NLP, reinforcement_learning]
aliases: ["DeepSeek-R1", "DeepSeek-R1 Technical Report", "arXiv 2501.12948"]
related_nodes: [deepseek_ai, chain_of_thought, reinforcement_learning, grpo]
arxiv_id: 2501.12948
authors: DeepSeek-AI
authors_institution: DeepSeek
---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- **元数据**: arXiv | 2025 | **作者**: DeepSeek-AI | **机构**: DeepSeek | 相关: [[deepseek_ai]], [[reinforcement_learning]]
- **概述**: 纯 RL 激发 LLM 推理能力，涌现自我反思、验证等高级推理模式，被 Nature 封面报道。
- **新颖概念**: [[grpo]], DeepSeek-R1-Zero
- **关键要点**: 1. 无需人类标注推理轨迹，纯 RL 训练 2. 涌现自验证/反思/策略调整 3. 在数学、代码、STEM 超越监督学习基线
- **方法/发现**: 基于 DeepSeek-V3-Base，使用 GRPO 框架，多阶段训练管线（RL+冷启动 SFT）。
- **局限/意义**: 推理能力局限于可验证任务；非推理任务泛化需进一步研究。
