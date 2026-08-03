---
id: web_enhanced_qa
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [联网问答, 搜索增强问答]
related_nodes: [webgpt_2021, reward_modeling]
last_verified: 2026-08-03
---

# Web-Enhanced QA

## 定义
让语言模型调用浏览器/搜索引擎获取实时信息后回答问题的范式，结合人类反馈（RLHF）训练模型学会"何时搜索、如何利用结果"。

## 关键要点
- **超越参数知识**：访问训练数据之外的最新信息，可引用来源
- **训练目标**：同时优化答案质量与事实准确性（含引证）
- **演进**：现代 RAG（检索增强生成）的直接前身

## 来源
- [[webgpt_2021]] — WebGPT：浏览器搜索 + 人类反馈
