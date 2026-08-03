---
id: openai_2024_o1
type: source
tags: ["machine-learning", "empirical-study", "RL"]
aliases: ["OpenAI o1", "o1 System Card", "慢思维推理模型"]
related_nodes: [test_time_compute, chain_of_thought, grpo]
authors: OpenAI
authors_institution: OpenAI
last_verified: 2026-08-03
---

# OpenAI o1 System Card

- **元数据**: OpenAI 技术报告 | 2024-09 | **作者**: OpenAI | **机构**: OpenAI
- **概述**: 提出 o1 系列推理模型：以大规模强化学习训练模型"思考"（内部思维链），推理时间可扩展（test-time compute scaling），在数学（AIME 94.8%）、代码（Codeforces 89th 百分位）等基准大幅超越 GPT-4o。
- **新颖概念**: [[test_time_compute]], [[chain_of_thought]]
- **关键要点**: 1. RL 训练模型生成并优化内部推理链（慢思维） 2. 推理时计算量越大、精度越高（scaling law） 3. 思维链长度与质量随 RL 训练涌现
- **方法/发现**: 训练数据包含高质量推理轨迹，RL 目标奖励正确最终答案；部署时控制推理预算（低/中/高）
- **局限/意义**: 思维链内容保密（安全考虑）；开启"推理模型"时代，o3/DeepSeek-R1 等沿此路线演进，验证 RL 可激发推理能力

## 引用
- **原始论文**: [o1 System Card](https://openai.com/index/openai-o1-system-card/)
- **相关概念**: [[test_time_compute]], [[chain_of_thought]]
