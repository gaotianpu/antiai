# 概念索引

## 模型架构
- [[attention_mechanism]] — 注意力机制：动态聚焦输入关键部分
- [[in_context_learning]] — 上下文学习：无需梯度更新的少样本学习
- [[reinforcement_learning_from_human_feedback]] — RLHF：人类反馈强化学习的对齐技术

## 模型设计
- [[conditional_memory]] — 条件记忆：通过稀疏查表实现静态知识检索
- [[mixture_of_experts]] — MoE：稀疏激活的条件计算范式
- [[sparsity_allocation]] — 稀疏分配：MoE 与记忆之间的最优容量配比

## 神经网络基础
- [[batch_normalization]] — 批归一化：标准化层输入以加速训练
- [[convolutional_neural_network]] — 卷积神经网络：权值共享的视觉特征提取
- [[residual_connection]] — 残差连接：恒等捷径缓解深层网络退化
- [[generative_model]] — 生成模型：学习数据分布并采样新样本
- [[diffusion_model]] — 扩散模型：逐步去噪的概率生成范式

## 训练与优化
- [[transfer_learning]] — 迁移学习：预训练+微调的知识迁移范式
- [[data_augmentation]] — 数据增强：变换扩展训练分布
- [[knowledge_distillation]] — 知识蒸馏：教师指导学生模型的压缩范式
- [[low_rank_adaptation]] — LoRA：低秩分解的参数高效微调

## 模型压缩
- [[model_pruning]] — 模型剪枝：移除冗余权重/通道参数量
- [[model_quantization]] — 模型量化：低位宽表示降低存储与推理成本

## 强化学习
- [[reinforcement_learning]] — 强化学习：交互试错学习决策策略
- [[policy_gradient]] — 策略梯度：直接优化参数化策略的方法

## 计算机视觉
- [[object_detection]] — 目标检测：定位并识别图像中物体
- [[anchor_box]] — 锚框：预定义参考框用于检测回归
