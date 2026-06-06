---
id: brown_2020_gpt3
type: source
tags: [NLP, machine-learning, empirical-study]
aliases: [GPT-3, Language Models are Few-Shot Learners, 2005.14165]
related_nodes: [in_context_learning, openai]
arxiv_id: 2005.14165
last_verified: 2026-06-06
---

# Language Models are Few-Shot Learners

- **元数据**: Conference | Tom Brown et al. (OpenAI) | 2020 | 相关: [[attention_mechanism]]
- **概述**: 将 GPT 扩展到 1750 亿参数，通过上下文学习（In-Context Learning）在少量示例下完成各种任务，无需梯度微调。
- **关键要点**: 1. 175B 参数 2. 上下文学习（Few-shot / One-shot / Zero-shot） 3. 涌现出推理、翻译、代码生成等能力
- **方法/发现**: 在多个 NLP 基准上，few-shot 设置匹配或超越微调模型
- **局限/意义**: 计算成本极高；推理速度慢；可能包含偏见与有害内容

## 引用
- **原始论文**: [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) | [阅读笔记](../../raw/nlp/gpt_3.md)
- **相关概念**: [[gpt]] | [[gpt_2]] | [[in_context_learning]]
- **相关实体**: [[openai]]
