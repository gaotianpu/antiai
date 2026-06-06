# deepseek 重磅论文

DeepSeek在AI领域的技术贡献主要通过以下途径发布：发表在预印本平台arXiv上的正式论文、GitHub上发布即开源的技术报告（例如多模态模型），以及传统学术期刊和会议论文。

### 🧠 核心模型系列报告
*   **DeepSeek-V3**：671B总参数、37B激活参数的MoE语言模型，核心创新包括无辅助损失的负载均衡策略和多Token预测训练目标。**引用**：DeepSeek-AI et al. (2024). DeepSeek-V3 Technical Report. *arXiv:2412.19437*.
*   **DeepSeek-R1**：通过强化学习激发推理能力，被《Nature》封面报道。**引用**：DeepSeek-AI et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.
*   **DeepSeek-V3.2**：通过稀疏注意力DSA和强化学习提升效率与推理能力，性能比肩GPT-5。**引用**：DeepSeek-AI. (2025). DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.
*   **DeepSeek LLM (67B)**：67B参数，详细分享了Scaling Laws、超参数选择、数据质量影响等细节。**引用**：DeepSeek-AI et al. (2024). DeepSeek LLM: Scaling Open-Source Language Models with Longtermism. *arXiv:2401.02954*.
*   **DeepSeek-V2**：一个强大、经济且高效的混合专家语言模型。**引用**：DeepSeek-AI et al. (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv:2405.04434*.
*   **DeepSeek-Prover-V2**：通过RL推进形式化数学推理。**引用**：DeepSeek-AI. (2025). DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via RL.
*   **DeepSeek-V3.1**：支持Think/非Think模式的混合推理模型和Agent。

### 🧬 前沿架构创新
*   **Engram ("条件记忆")**：一种全新的神经网络模块，旨在将模型的“记忆”与“计算”能力解耦。
*   **mHC ("流形约束超连接")**：一种新型网络架构，通过为信息加装“智能调节阀”，提升大规模训练稳定性。

### 📡 专项领域模型
*   **DeepSeek-OCR (视觉压缩)**：将长文本内容通过光学二维映射进行压缩，以提高处理效率。**引用**：DeepSeek-AI. (2025). DeepSeek-OCR: Contexts Optical Compression. *arXiv:2510.18234*.
*   **Thinking with Visual Primitives (多模态推理)**：提出全新的多模态推理范式，通过将坐标等视觉原语融入思维链，解决空间参照任务。
*   **DeepSeekMath-V2**：具备自验证能力的数学推理模型，能解答竞赛级问题。
*   **DeepSeek-Coder**：专注代码智能的大模型系列。**引用**：DeepSeek-AI. (2024). DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence. *arXiv:2401.14196*.
*   **DeepSeek-VL**：多模态系列模型。

### ⚙️ 底层系统与效率
*   **FlashMLA (双向管道并行)**：为DeepSeek-V3/R1设计的高效并行算法，用于压缩流水线气泡、共享梯度传输。
*   **Native Sparse Attention (NSA) (原生稀疏注意力)**：专为重新定义长上下文建模的效率与性能而设计。
*   **辅助文档**
    *   **DeepSeek-V3 技术报告深度解析**：一份深度解析，涵盖架构创新与工程实践。
    *   **DeepSeek系列论文技术要点PPT**：一份60页的PPT，系统整理了系列论文的技术要点。
    *   **DeepSeek 进化史**：一篇基于21篇论文梳理的技术蜕变文章。

