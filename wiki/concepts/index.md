# 概念索引

## 数学基础
- [[math_for_deep_learning]] — 深度学习数学基础：五大子领域总览
- [[linear_algebra]] — 线性代数：向量/矩阵/特征分解/SVD
- [[matrix_operations]] — 矩阵运算：乘法/转置/范数/秩及 DL 应用
- [[eigendecomposition]] — 特征分解：特征值方程/对角化/谱定理及 DL 应用
- [[singular_value_decomposition]] — 奇异值分解：SVD 分解/低秩近似/Eckart-Young 定理及 DL 应用
- [[calculus]] — 微积分：导数/链式法则/梯度/二阶优化
- [[chain_rule]] — 链式法则：复合函数求导与反向传播的数学本质
- [[derivative_and_gradient]] — 导数与梯度：变化率/偏导/梯度向量
- [[probability_statistics]] — 概率与统计：分布/MLE/MAP/贝叶斯推断
- [[probability_distributions]] — 概率分布：伯努利/类别/高斯/均匀分布及 DL 角色
- [[bayesian_inference]] — 贝叶斯推断：MLE vs MAP / 后验估计与正则化
- [[information_theory]] — 信息论：熵/交叉熵/KL散度/互信息
- [[entropy]] — 熵：不确定性度量 / 交叉熵 / 困惑度
- [[cross_entropy]] — 交叉熵：分类损失的核心 / CE = H(p) + D_KL / MLE 等价性
- [[kl_divergence]] — KL 散度：分布差异度量 / VAE / 蒸馏 / RLHF
- [[mutual_information]] — 互信息：依赖度量 / InfoNCE / 表示学习
- [[optimization_fundamentals]] — 优化基础：凸与非凸/鞍点/约束优化
- [[taylor_expansion]] — 泰勒展开：多项式逼近与一阶/二阶优化理论基础
- [[convex_optimization]] — 凸优化：凸集/凸函数/Jensen 不等式/全局最优保证
- [[gradient_descent]] — 梯度下降：批量/小批量/随机/动量/自适应方法
- [[lagrange_multiplier]] — 拉格朗日乘数法：约束优化/KKT 条件/对偶形式

## 模型架构
- [[attention_mechanism]] — 注意力机制：动态聚焦输入关键部分
- [[attention_variants]] — 注意力变体：多头/因果/高效注意力的分类
- [[multi_head_latent_attention]] — MLA：将 KV cache 压缩为潜在向量的高效注意力
- [[in_context_learning]] — 上下文学习：无需梯度更新的少样本学习
- [[positional_encoding]] — 位置编码：为注意力引入顺序感知
- [[reinforcement_learning_from_human_feedback]] — RLHF：人类反馈强化学习的对齐技术
- [[tokenization]] — 分词：NLP 文本切分策略
- [[transformer_architecture]] — Transformer 架构：完全基于自注意力的序列模型
- [[encoder_decoder_architecture]] — 编解码器架构：编码器-解码器结构与变体
- [[feed_forward_network]] — 前馈网络：Transformer 中的 Position-wise FFN

## 模型设计
- [[conditional_memory]] — 条件记忆：通过稀疏查表实现静态知识检索
- [[mixture_of_experts]] — MoE：稀疏激活的条件计算范式
- [[sparsity_allocation]] — 稀疏分配：MoE 与记忆之间的最优容量配比

## 神经网络基础
- [[activation_function]] — 激活函数：引入非线性的关键组件
- [[batch_normalization]] — 批归一化：标准化层输入以加速训练
- [[convolutional_neural_network]] — 卷积神经网络：权值共享的视觉特征提取
- [[normalization]] — 归一化：层归一化/批归一化等稳定训练技术
- [[residual_connection]] — 残差连接：恒等捷径缓解深层网络退化
- [[generative_model]] — 生成模型：学习数据分布并采样新样本
- [[diffusion_model]] — 扩散模型：逐步去噪的概率生成范式

## 训练与优化
- [[loss_function]] — 损失函数：分类/回归/检测/对比学习的损失设计
- [[optimizer]] — 优化器：从 SGD 到 AdamW 的参数更新算法
- [[transfer_learning]] — 迁移学习：预训练+微调的知识迁移范式
- [[data_augmentation]] — 数据增强：变换扩展训练分布
- [[knowledge_distillation]] — 知识蒸馏：教师指导学生模型的压缩范式
- [[learning_rate_schedule]] — 学习率策略：从 warmup 到 cosine decay 的调度
- [[low_rank_adaptation]] — LoRA：低秩分解的参数高效微调
- [[regularization]] — 正则化：防止过拟合的技术集合
- [[self_supervised_learning]] — 自监督学习：从无标签数据构造监督信号
- [[multi_token_prediction]] — MTP：同时预测多个未来 token 的训练目标

## 模型压缩
- [[model_pruning]] — 模型剪枝：移除冗余权重/通道参数量
- [[model_quantization]] — 模型量化：低位宽表示降低存储与推理成本

## 强化学习
- [[reinforcement_learning]] — 强化学习：交互试错学习决策策略
- [[policy_gradient]] — 策略梯度：直接优化参数化策略的方法
- [[grpo]] — GRPO：无 critic 的组相对策略优化

## 计算机视觉
- [[feature_pyramid]] — 特征金字塔：多尺度特征融合的检测标配
- [[object_detection]] — 目标检测：定位并识别图像中物体
- [[anchor_box]] — 锚框：预定义参考框用于检测回归

## 推理与提示
- [[chain_of_thought]] — 思维链：中间推理步骤提升复杂推理

## 强化学习
- [[actor_critic]] — Actor-Critic：策略价值双网络
- [[asynchronous_learning]] — 异步学习：A3C 并行框架
- [[dagger_algorithm]] — DAgger：数据集聚合
- [[deep_q_network]] — DQN：深度 Q 网络
- [[deterministic_policy_gradient]] — DPG：确定性策略梯度
- [[end_to_end_driving]] — 端到端驾驶：像素到转向
- [[experience_replay]] — 经验回放：去相关与样本复用
- [[goodhart_law]] — 古德哈特定律：指标目标脱节
- [[imitation_learning]] — 模仿学习：专家示范学习
- [[imitative_models]] — 模仿模型：分布+规划融合
- [[ppo]] — PPO 应用：RLHF 对齐优化器
- [[proximal_policy_optimization]] — PPO：裁剪代理目标
- [[reward_modeling]] — 奖励建模：人类偏好学习
- [[reward_overoptimization]] — 奖励过优化：Goodhart 实证
- [[trust_region_method]] — 信任区域方法：TRPO KL 约束

## 训练与优化
- [[adaptive_budget_allocation]] — 自适应预算分配：AdaLoRA 动态参数
- [[label_smoothing]] — 标签平滑：软标签正则化
- [[rationale_distillation]] — 理由蒸馏：推理链作为训练信号

## 模型架构
- [[alibi_position_encoding]] — ALiBi：线性偏置注意力，长度外推
- [[deepnorm]] — DeepNorm：千层 Transformer 稳定训练
- [[dilated_attention]] — 扩张注意力：线性复杂度超长程建模
- [[flash_attention]] — FlashAttention：IO 感知的高效精确注意力
- [[io_aware_attention]] — IO 感知注意力：以内存访问为核心的算法设计
- [[layer_normalization]] — 层归一化：Transformer 标准组件
- [[mlp_only_architecture]] — 纯 MLP 架构：MLP-Mixer / ResMLP
- [[multi_head_attention]] — 多头注意力：并行子空间捕捉不同模式
- [[parallel_training_sequential_inference]] — 并行训练顺序推理：高效序列架构目标
- [[relative_position_encoding]] — 相对位置编码：平移不变位置建模
- [[retention_mechanism]] — 保留机制：RetNet 线性递推注意力
- [[rotary_position_embedding]] — RoPE：旋转位置编码，现代 LLM 标配
- [[segment_level_recurrence]] — 片段级循环：跨片段信息流动
- [[self_attention]] — 自注意力：序列内部动态聚焦
- [[shifted_window_attention]] — 移动窗口注意力：Swin 分层注意力
- [[transformer_architecture]] — Transformer：纯注意力序列架构，LLM 基础
- [[vision_transformer]] — ViT：图像 patch 序列化的视觉 Transformer

## 计算机视觉
- [[anchor_free_detector]] — 无锚框检测：YOLOX 范式
- [[bag_of_freebies]] — 免费赠品集：零推理代价技巧
- [[decoupled_head]] — 解耦头：分类回归分离
- [[focal_loss]] — Focal Loss：难例加权
- [[fully_convolutional_network]] — FCN：全卷积分割
- [[implicit_knowledge]] — 隐式知识：YOLOR 统一表征
- [[instance_segmentation]] — 实例分割：检测+掩码
- [[multimodal_fusion]] — 多模态融合：相机+LiDAR
- [[path_aggregation_network]] — PANet：双向特征路径
- [[promptable_segmentation]] — 可提示分割：SAM 交互范式
- [[region_proposal]] — 区域候选：两阶段检测第一步
- [[region_proposal_network]] — RPN：可学习候选网络
- [[semantic_segmentation]] — 语义分割：像素级分类
- [[single_stage_detector]] — 单阶段检测：YOLO 范式
- [[trainable_bag_of_freebies]] — 可训练技巧：YOLOv7
- [[transfuser]] — TransFuser：注意力跨模态融合
- [[visual_inertial_odometry]] — VIO：视觉惯性里程计

## 神经网络基础
- [[atrous_convolution]] — 空洞卷积：感受野扩张
- [[atrous_spatial_pyramid_pooling]] — ASPP：多尺度空洞池化
- [[bottleneck_architecture]] — 瓶颈结构：降维-变换-升维
- [[channel_attention]] — 通道注意力：SE 重校准
- [[channel_shuffle]] — 通道重排：组间信息流动
- [[cheap_operation]] — 廉价操作：低成本线性变换
- [[compound_scaling]] — 复合缩放：EfficientNet 三维协同
- [[cross_stage_partial_connection]] — CSP 连接：跨阶段梯度裁剪
- [[data_efficient_training]] — 数据高效训练：DeiT 配方
- [[deeply_supervised_net]] — 深度监督：辅助监督信号
- [[dense_connection]] — 密集连接：DenseNet 特征复用
- [[depthwise_separable_convolution]] — 深度可分离卷积：高效算子
- [[design_space_design]] — 设计空间设计：RegNet 方法论
- [[dropout]] — 随机失活：经典正则化
- [[efficient_network_design_principles]] — 高效网络设计准则：MAC 导向
- [[factorized_convolution]] — 因式分解卷积：大核分解
- [[fire_module]] — Fire 模块：SqueezeNet 构建块
- [[gelu_activation]] — GELU：高斯误差线性单元
- [[ghost_module]] — Ghost 模块：冗余特征生成
- [[group_convolution]] — 分组卷积：基数维度
- [[group_normalization]] — 组归一化：小 batch 稳定训练
- [[inception_module]] — Inception 模块：多分支卷积
- [[inverted_residual]] — 倒残差：MobileNetV2 扩展-卷积-压缩
- [[linear_bottleneck]] — 线性瓶颈：低维免激活
- [[multi_scale_representation]] — 多尺度表示：Res2Net 细粒度
- [[multi_scale_training]] — 多尺度训练：动态输入分辨率
- [[neural_architecture_search]] — NAS：自动架构搜索
- [[neural_architecture_search_applied]] — NAS 实践：MobileNetV3 搜索
- [[non_local_operation]] — 非局部操作：长距离依赖
- [[pyramid_pooling_module]] — 金字塔池化模块：PSPNet 上下文聚合
- [[receptive_field_block]] — 感受野块：仿生多尺度模块
- [[regnet]] — RegNet：规律缩放网络家族
- [[sparse_connection]] — 稀疏连接：SparseNet O(L) 复杂度
- [[spatial_pyramid_pooling]] — 空间金字塔池化：任意尺寸输入
- [[structural_reparameterization]] — 结构重参数化：训练推理解耦
- [[tiny_ml]] — TinyML：MCU 端侧智能
- [[vgg_network]] — VGG：3×3 卷积堆叠经典
- [[xavier_initialization]] — Xavier 初始化：方差守恒

## 语言模型
- [[bert]] — BERT：双向编码器表示
- [[byte_pair_encoding]] — BPE：字节对编码子词算法
- [[cbow]] — CBOW：上下文预测中心词
- [[dual_encoder]] — 双编码器：检索匹配架构
- [[few_shot_learning]] — 少样本学习：示例提示即适配
- [[generative_pretraining]] — 生成式预训练：预训练-微调范式
- [[gpt]] — GPT：生成式预训练 Transformer
- [[gpt_2]] — GPT-2：零样本能力验证
- [[gpt_3]] — GPT-3：175B 参数与上下文学习
- [[instruction_tuning]] — 指令微调：遵循人类指令
- [[knowledge_enhanced_pretraining]] — 知识增强预训练：ERNIE 实体知识注入
- [[masked_language_modeling]] — MLM：掩码语言建模
- [[permutation_language_modeling]] — 排列语言建模：XLNet 双向自回归
- [[replaced_token_detection]] — 替换词检测：ELECTRA 预训练目标
- [[self_instruct]] — 自我指令：模型自举指令数据
- [[skip_gram]] — Skip-gram：中心词预测上下文
- [[span_boundary_objective]] — 跨度边界目标：SBO 辅助任务
- [[span_masking]] — 跨度掩码：连续片段掩码
- [[subword_tokenization]] — 子词分词：BPE/WordPiece/Unigram
- [[text_to_text_framework]] — 文本到文本框架：T5 统一任务范式
- [[weakly_supervised_speech_recognition]] — 弱监督语音识别：Whisper 大规模弱标签
- [[web_enhanced_qa]] — 联网问答：搜索增强回答
- [[whole_word_masking]] — 全词掩码：WWM 粒度变体
- [[word_embedding]] — 词向量：稠密分布式表示
- [[zero_shot_learning]] — 零样本学习：无需示例的任务执行

## 生成模型
- [[cascaded_diffusion]] — 级联扩散：逐级超分生成
- [[consistency_model]] — 一致性模型：单步生成
- [[diffusion_transformer]] — DiT：Transformer 扩散骨干
- [[guided_diffusion]] — 引导扩散：条件注入控制生成
- [[hierarchical_image_generation]] — 层次化图像生成：DALL·E 2 两阶段
- [[latent_diffusion]] — 潜空间扩散：Stable Diffusion 基础
- [[masked_image_modeling]] — 掩码图像建模：MAE/BEiT/iBOT
- [[masked_video_modeling]] — 掩码视频建模：MAE-ST
- [[online_tokenizer]] — 在线 tokenizer：iBOT 动态目标
- [[pixel_level_pretraining]] — 像素级预训练：iGPT
- [[vector_quantized_tokenizer]] — VQ tokenizer：离散视觉词表

## 自监督与对比学习
- [[contrastive_language_image_pretraining]] — CLIP：对比图文预训练
- [[contrastive_learning]] — 对比学习：正负对拉近推远
- [[momentum_contrast]] — MoCo：动量对比字典查找
- [[siamese_network]] — 孪生网络：共享权重双分支
- [[stop_gradient]] — 停止梯度：防坍塌关键机制

## 模型设计
- [[engram]] — Engram：条件记忆 O(1) 查表

## 多模态
- [[foundation_model]] — 基础模型：通用预训练范式
- [[multimodal_chain_of_thought]] — 多模态思维链：图文联合推理
- [[multimodal_language_model]] — 多模态语言模型：Kosmos-1 范式
- [[multimodal_mixture_of_encoders_decoders]] — 多模态编解码混合：BLIP MoE-D

## 序列模型
- [[gated_recurrent_unit]] — GRU：简化门控循环单元
- [[gating_mechanism]] — 门控机制：信息流控制
- [[long_short_term_memory]] — LSTM：门控长程记忆
- [[recurrent_neural_network]] — RNN：循环网络与时序建模

## 模型压缩
- [[huffman_coding]] — 霍夫曼编码：无损压缩权重存储
- [[sparse_routing]] — 稀疏路由：MoE 条件计算

## 推理与提示
- [[principle_driven_alignment]] — 原则驱动对齐：SELF-ALIGN 范式
- [[self_align]] — SELF-ALIGN：无 RLHF 自对齐
- [[zero_shot_chain_of_thought]] — 零样本思维链：触发短语
