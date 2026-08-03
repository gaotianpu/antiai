---
id: peng_2023_yarn
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["YaRN", "RoPE外推", "2309.00071"]
related_nodes: [rotary_position_embedding, length_extrapolation]
authors: Bowen Peng et al.
authors_institution: Independent
arxiv_id: 2309.00071
last_verified: 2026-08-03
---

# YaRN: Efficient Context Window Extension of Large Language Models

- **元数据**: arXiv 2309.00071 | 2023 | **作者**: Bowen Peng et al. | **机构**: Independent
- **概述**: 提出 YaRN（Yet another RoPE extensioN）：结合 NTK 感知插值与注意力温度修正，仅需微调 1/10 数据即可将 RoPE 模型上下文扩展数倍（如 4K→128K）。
- **新颖概念**: [[length_extrapolation]]
- **关键要点**: 1. NTK-aware 插值：按频率段缩放 RoPE 旋转角 2. 注意力温度（λ）修正长距离分数衰减 3. 少量微调即可高质量外推
- **方法/发现**: LLaMA-65B 上 4K→128K 外推，困惑度与长文本任务保持；成为开源社区上下文扩展的标配
- **局限/意义**: 位置编码工程化里程碑；长度外推从"能否"走向"高效可行"，与 ALiBi/位置插值构成完整技术谱系

## 引用
- **原始论文**: [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- **相关概念**: [[rotary_position_embedding]], [[length_extrapolation]]
