---
id: test_time_compute
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [推理时计算, 测试时扩展, test-time scaling]
related_nodes: [openai_2024_o1, chain_of_thought, gpt_3]
last_verified: 2026-08-03
---

# Test-Time Compute

## 定义
推理阶段按需扩展计算量的范式：o1 等推理模型在生成答案前先"思考"（内部思维链），分配更多推理预算可获得更高精度，形成推理时的 scaling law。

## 关键要点
- **训练-推理协同**：RL 训练模型学会利用推理时间；部署时控制思考预算（低/中/高）
- **与预训练 scaling 互补**：预训练扩展模型能力，test-time compute 扩展单次任务表现
- **应用**：o1/o3、DeepSeek-R1、Kimi K2 Thinking 等推理模型的共同特征

## 来源
- [[openai_2024_o1]] — o1 推理时间扩展
- [[kimi_2025_k2]] — K2 Thinking 长思维链
