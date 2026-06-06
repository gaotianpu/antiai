---
id: dettmers_2023_qlora
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["QLoRA", "Efficient Finetuning of Quantized LLMs", "2305.14314"]
related_nodes: ["hu_2021_lora", "touvron_2023_llama"]
arxiv_id: 2305.14314
authors: Tim Dettmers et al.
authors_institution: University of Washington
last_verified: 2026-06-06
---

# QLoRA: Efficient Finetuning of Quantized LLMs

- **元数据**: arXiv | 2023 | **作者**: Tim Dettmers et al. | **机构**: University of Washington | 相关: [[hu_2021_lora]], [[touvron_2023_llama]]
- **概述**: 在 LoRA 基础上引入 4-bit NormalFloat + 双重量化 + 分页优化器，65B 模型单 48GB GPU 微调。
- **新颖概念**: [[low_rank_adaptation]], [[model_quantization]]
- **关键要点**: 1. NF4 数据类型 2. 双重量化 3. 分页优化器防 OOM 4. Guanaco 达 ChatGPT 99.3% 性能
- **方法/发现**: 4-bit 量化预训练模型 + LoRA 适配器反向传播，千余模型消融实验
- **局限/意义**: 使大模型微调平民化，单 GPU 即可微调 65B 模型

## 引用
- **原始论文**: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | [阅读笔记](../../raw/QLoRA.md)
- **相关概念**: [[hu_2021_lora]], [[touvron_2023_llama]]
