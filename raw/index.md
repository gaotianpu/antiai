# raw/ 索引

按年份分组（正序），年份不详的留在末尾。

## 1990

### NLP
| [[nlp/rnn.md]] | 提出简单循环神经网络（Elman RNN）用于处理时序结构（`NLP, machine-learning`）|
## 1997

### NLP
| [[nlp/lstm.md]] | 提出 LSTM，通过门控机制解决长序列梯度消失问题（`NLP, machine-learning`）|

## 1998

### CNN
| [[cnn/LeNet.md]] | 提出 LeNet-5 卷积神经网络，开创 CNN 在手写文档识别中的应用（`computer-vision, machine-learning, empirical-study`）|

## 2010

### 自动驾驶
| [[Autonomous_Robot/DAgger.md]] | 分析 DAgger 数据聚合技术在自动驾驶中的泛化问题并提出改进方案（`autonomous-driving, imitation-learning, data-aggregation`）|

### 其他
| [[Xavier_init.md]] | 分析深层网络训练困难的原因，提出 Xavier/Glorot 初始化，在前向和反向传播中维持激活方差恒定。（`machine-learning, theoretical`）|

## 2012

### CNN
| [[cnn/alexnet.md]] | 首个在 ImageNet 上大幅超越传统方法的深度 CNN，引爆深度学习革命（`computer-vision, machine-learning, empirical-study`）|

### 其他
| [[OCR.md]] | OCR |

## 2013

### NLP
| [[nlp/word2vec.md]] | 提出 Word2Vec（CBOW + Skip-gram），高效学习词向量（`NLP, machine-learning`）|

### CNN
| [[cnn/R-CNN.md]] | 提出 R-CNN，将 CNN 与区域候选结合，在 VOC 2012 上 mAP 提升超 30%。（`computer-vision, machine-learning, empirical-study`）|

### RL
| [[RL/DQN.md]] | 提出 DQN，将深度神经网络与 Q-learning 结合，在 49 个 Atari 游戏中达到人类水平。（`machine-learning, empirical-study, RL`）|

## 2014

### NLP
| [[nlp/Attention.md]] | 提出在神经机器翻译中联合学习对齐和翻译的注意力机制（Bahdanau Attention），打破固定编码向量的瓶颈。（`NLP, machine-learning, theoretical`）|
| [[nlp/gru.md]] | 提出 GRU，简化的门控循环单元，性能与 LSTM 相当但计算更高效（`NLP, machine-learning`）|

### CNN
| [[cnn/Deeply-Supervised.md]] | 提出深度监督网络（DSN），通过在隐藏层引入辅助监督信号（companion objective）来解决深层网络的梯度消失问题，使学习过程更加直接透明。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/FCN.md]] | 提出全卷积网络（FCN），将分类网络中的全连接层替换为卷积层，实现端到端、像素到像素的语义分割，接受任意尺寸输入并产生对应尺寸输出。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/Pooling.md]] | 通过系统实验推翻"池化层是 CNN 变形稳定性必要条件"的假设，发现池化既非必要也非充分，且网络在训练中会主动抵消池化的过度稳定性偏置。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/googlenet.md]] | 提出 Inception 模块，在增加深度和宽度的同时控制计算复杂度（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/sppnet.md]] | 空间金字塔池化，让 CNN 支持任意尺寸输入（`computer-vision, machine-learning`）|
| [[cnn/vgg.md]] | 系统研究网络深度对精度的影响，提出 16-19 层 VGG 架构（`computer-vision, machine-learning, empirical-study`）|

## 2015

### NLP
| [[nlp/BPE.md]] | 提出 BPE（字节对编码）子词分词方法，解决稀有词翻译问题（`NLP, machine-learning`）|

### CNN
| [[cnn/BatchNorm.md]] | 提出 Batch Normalization (BN)，通过将每层输入归一化到统一分布来解决内部协变量偏移问题，使训练可以使用更高学习率并减少对 Dropout 的依赖。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/Faster_R-CNN.md]] | 提出区域候选网络（RPN），将候选框生成融入检测网络实现端到端训练，近乎实时检测。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/inception_v3.md]] | 系统提出 Inception 架构的设计原则与因式分解卷积技巧（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/resnet.md]] | 提出残差学习（ResNet），通过跳跃连接解决深层网络退化问题，可训练 152 层（`computer-vision, machine-learning`）|
| [[cnn/yolo_v1.md]] | 提出 YOLO 单阶段目标检测范式，将检测视为回归问题，实现实时端到端检测。（`computer-vision, machine-learning, empirical-study`）|

### RL
| [[RL/DDPG.md]] | 将 DQN 思路扩展到连续动作空间，结合 Actor-Critic 和 DQN 的经验回放 + Target Network。（`machine-learning, empirical-study, RL`）|
| [[RL/TRPO.md]] | 提出信任区域策略优化，通过 KL 散度约束保证策略更新的单调改进。（`machine-learning, theoretical, RL`）|

### 效率/压缩
| [[Deep_Compression.md]] | 提出三阶段模型压缩流程：剪枝 + 量化 + Huffman 编码，在不损失精度前提下压缩 35–49 倍。（`machine-learning, empirical-study`）|
| [[Distilling_ss.md]] | 提出 LLM rationale 作为多任务训练信号的蒸馏方法，770M T5 超越 540B PaLM。（`NLP, machine-learning, empirical-study`）|

## 2016

### NLP
| [[nlp/LayerNorm.md]] | 提出 Layer Normalization 归一化方法，适用于 RNN/Transformer（`NLP, machine-learning`）|

### CNN
| [[cnn/GELUs.md]] | 提出 GELU 激活函数 $x\Phi(x)$（$\Phi$ 为标准高斯 CDF），按输入值而非符号进行加权，在 CV、NLP、语音任务上一致优于 ReLU 和 ELU。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/SqueezeNet.md]] | 在保持 AlexNet 级精度的前提下实现 50 倍参数压缩（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/densenet.md]] | 提出密集连接机制，每层与之前所有层直接连接（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/fpn.md]] | 特征金字塔网络，利用多尺度特征图提升检测精度（`computer-vision, machine-learning`）|
| [[cnn/inception_v4.md]] | 探索 Inception 与残差连接的结合，提出 Inception-v4 和 Inception-ResNet（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/pspnet.md]] | 提出金字塔场景解析网络（PSPNet），通过金字塔池化模块在不同尺度上聚合全局上下文信息，解决场景解析中上下文误匹配问题。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/resnext.md]] | 提出"基数"（cardinality）维度，通过分组卷积提升模型性能（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/sparsenet.md]] | 稀疏化 DenseNet 连接，将连接复杂度从 O(L²) 降至 O(L)（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/xception.md]] | 将 Inception 模块重新解释为常规卷积与深度可分离卷积之间的中间步骤，并提出 Xception 架构——用深度可分离卷积完全替代 Inception 模块。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/yolo_v2.md]] | YOLOv2 在 v1 基础上引入批量归一化、锚框先验、多尺度训练，YOLO9000 联合检测和分类数据集实现 9000 类检测。（`computer-vision, machine-learning, empirical-study`）|

### RL
| [[RL/A2C.md]] | 提出异步多线程训练框架（A3C/A2C），通过并行 Actor 替代经验回放，稳定训练。（`machine-learning, empirical-study, RL`）|
| [[RL/ACER.md]] | 将经验回放引入 Actor-Critic，结合 Retrace 和截断重要性采样，提升样本效率。（`machine-learning, empirical-study, RL`）|

### 自动驾驶
| [[Autonomous_Robot/DAVE-2.md]] | 使用 CNN 从单面前向相机像素直接映射到转向命令的经典端到端学习（`autonomous-driving, end-to-end-learning, cnn`）|

### 效率/压缩
| [[LayerNorm.md]] | LayerNorm |

## 2017

### NLP
| [[nlp/transformer.md]] | 提出完全基于注意力机制的 Transformer 架构，摒弃了循环和卷积，成为后续 LLM 的基础范式。（`NLP, machine-learning, theoretical`）|

### CNN
| [[cnn/senet.md]] | 提出 SE 块，通过通道注意力自适应重校准特征，极小开销提升 CNN 性能（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/DeepLab_v3.md]] | 系统地重新审视空洞卷积（Atrous Convolution）在语义分割中的应用，改进空洞空间金字塔池化（ASPP）模块，集成图像级特征编码全局上下文，在 PASCAL VOC 2012 上达到 SOTA 性能。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/Focal_Loss.md]] | 提出 Focal Loss，解决一阶段检测器中正负样本极端不平衡问题（`computer-vision, machine-learning`）|
| [[cnn/Instance_Segmentation.md]] | 实例分割综述 |
| [[cnn/Mask_R-CNN.md]] | 在 Faster R-CNN 基础上添加分割分支，同时做目标检测和实例分割（`computer-vision, machine-learning`）|
| [[cnn/MobileNet_v1.md]] | 提出深度可分离卷积构建高效移动端 CNN（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/Non-local.md]] | 提出非局部（Non-local）操作作为通用构建块来捕获长距离依赖，在视频分类和图像检测/分割任务上取得显著提升。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/Semantic_Segmentation.md]] | 语义分割综述 |
| [[cnn/ShuffleNet.md]] | 提出逐点分组卷积 + 通道重排操作，实现极低算力下的高效 CNN（`computer-vision, machine-learning, empirical-study`）|

### RL
| [[RL/PPO.md]] | 提出 PPO，通过裁剪的替代目标实现稳定策略更新，兼顾 TRPO 的可靠性和实现简单性。（`machine-learning, empirical-study, RL`）|

### 自动驾驶
| [[Autonomous_Robot/VINS-Mono.md]] | 提出单目视觉惯性系统在线时间校准方法（`visual-inertial-odometry, slam, sensor-calibration`）|

### 其他
| [[meta_learning_survey.md]] | 元学习综述，系统分类：基于度量、基于模型、基于优化的三大范式。（`machine-learning, survey`）|

## 2018

### NLP
| [[nlp/Self-Attention_Sentiment_Analysis.md]] | 探索自注意力在情感分析中的应用（`NLP, machine-learning`）|
| [[nlp/bert.md]] | 提出深度双向 Transformer 预训练模型 BERT，通过 MLM + NSP 预训练目标从无标注文本学习双向表示，微调后统治 11 项 NLP 任务。（`NLP, machine-learning, empirical-study`）|
| [[nlp/gpt.md]] | 提出生成式预训练方法（GPT），在无标注文本上预训练 Transformer 解码器后微调，验证了预训练-微调范式在 NLP 任务的通用性。（`NLP, machine-learning, empirical-study`）|

### CNN
| [[cnn/GroupNorm.md]] | 提出 Group Normalization，在 batch 维度较小时比 Batch Norm 更稳定（`computer-vision, machine-learning`）|
| [[cnn/MobileNet_v2.md]] | 提出倒残差结构（Inverted Residual）和线性瓶颈（Linear Bottleneck）（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/ShuffleNet_v2.md]] | 指出 FLOPs 不能准确衡量实际速度，提出 4 条高效网络设计准则（`computer-vision, machine-learning, practical-guide`）|
| [[cnn/panet.md]] | 提出路径聚合网络（PANet），通过自底向上的路径增强和自适应特征池化来改进基于提议的实例分割框架中的信息流动，获得 COCO 2017 实例分割冠军。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/rfb.md]] | 受人类视觉感受野结构启发，提出感受野块（RFB）模块，通过考虑感受野大小与离心率的关系增强轻量级特征的判别能力，构建 RFB Net 检测器。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/yolo_v3.md]] | YOLOv3 引入 Darknet-53 骨架 + FPN 多尺度预测，在不牺牲速度的前提下大幅提升小目标检测能力。（`computer-vision, machine-learning, empirical-study`）|

### 自动驾驶
| [[Autonomous_Robot/ChauffeurNet.md]] | 通过模仿学习和合成扰动数据训练鲁棒的自动驾驶策略（`autonomous-driving, imitation-learning, behavior-cloning`）|
| [[Autonomous_Robot/H-Net.md]] | 提出基于注意力机制的无监督立体深度估计框架 H-Net（`depth-estimation, autonomous-driving, self-supervised-learning`）|
| [[Autonomous_Robot/Learning_Situational_Driving.md]] | 提出基于混合策略的情境驾驶框架，捕获不同场景下的多种驾驶模式（`autonomous-driving, imitation-learning, mixture-of-experts`）|

### 效率/压缩
| [[Vector_quantized.md]] | Vector Quantized |

### 其他
| [[online_active_learning_survey.md]] | 在线主动学习综述，覆盖数据流场景下的查询策略、预算分配和概念漂移处理。（`machine-learning, survey`）|

## 2019

### NLP
| [[nlp/ALBERT.md]] | ALBERT |
| [[nlp/BART.md]] | BART |
| [[nlp/BERT-wwm.md]] | 全词掩码（Whole Word Masking）中文 BERT 预训练，掩码粒度从子词提升到整词（`NLP, machine-learning, empirical-study`）|
| [[nlp/Marian.md]] | Marian |
| [[nlp/RoBERTa.md]] | 系统优化 BERT 预训练：更大 batch、更多数据、动态掩码、移除 NSP，显著提升（`NLP, machine-learning, empirical-study`）|
| [[nlp/SpanBERT.md]] | 提出 Span 级别的预训练目标（Span Masking + Span Boundary Objective）（`NLP, machine-learning, empirical-study`）|
| [[nlp/StructBERT.md]] | 将语言结构信息（词序、句子结构）融入 BERT 预训练（`NLP, machine-learning, empirical-study`）|
| [[nlp/T5.md]] | 提出 T5 模型，将所有 NLP 任务统一为 Text-to-Text 格式，系统研究了预训练方法的影响。（`NLP, machine-learning, empirical-study`）|
| [[nlp/TinyBERT.md]] | 提出两阶段 Transformer 蒸馏方法，将 BERT 压缩为小模型（`NLP, machine-learning, empirical-study`）|
| [[nlp/Transformer-XL.md]] | 提出片段级循环机制和相对位置编码，突破 Transformer 固定上下文长度限制（`NLP, machine-learning`）|
| [[nlp/XLNet.md]] | 广义自回归预训练，结合自回归和自编码优势，排列语言建模（`NLP, machine-learning`）|
| [[nlp/ernie.md]] | 将知识图谱实体信息融入语言表示预训练（`NLP, machine-learning`）|
| [[nlp/ernie_v2.md]] | 持续预训练框架，多任务增量学习（`NLP, machine-learning`）|
| [[nlp/gpt_2.md]] | 扩展 GPT-1 至 15 亿参数，证明语言模型在无监督下可零样本执行多种下游任务。（`NLP, machine-learning, empirical-study`）|

### CNN
| [[cnn/EfficientNet.md]] | 提出复合缩放方法，统一平衡深度、宽度和分辨率（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/HarDNet.md]] | 提出谐波密集连接网络，降低内存流量以加速推理（`computer-vision, machine-learning, practical-guide`）|
| [[cnn/MoCo.md]] | 提出 MoCo（动量对比），将对比学习视为字典查找，通过队列和动量编码器构建大规模一致字典，缩小无监督与监督表示学习在视觉任务上的差距。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/MoCo_v2.md]] | 在 MoCo 框架中集成 SimCLR 的两项改进（MLP 投影头 + 更强数据增广），无需大批量（仅 256）即可超越 SimCLR，称为 MoCo v2。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/MobileNet_v3.md]] | 结合 NAS（MnasNet）和 NetAdapt 自动搜索优化 MobileNet 架构（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/Res2Net.md]] | 提出 Res2Net 模块，通过在单个残差块内构建层次化残差连接来实现细粒度的多尺度特征表示。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/cspnet.md]] | 提出跨阶段局部连接，减少重复梯度计算提升学习能力（`computer-vision, machine-learning, empirical-study`）|

### RL
| [[RL/hp_RL.md]] | 提出从人类偏好中训练奖励模型，再用 RL 优化策略，实现复杂任务的对齐学习。（`machine-learning, empirical-study, RL`）|

### 自动驾驶
| [[Autonomous_Robot/cheating.md]] | 将驾驶任务分解为"特权智能体"（可访问真值信息）和"纯视觉智能体"两阶段训练（`autonomous-driving, imitation-learning, knowledge-distillation`）|
| [[Autonomous_Robot/limit_Behavior_Cloning.md]] | 系统实验研究行为克隆在自动驾驶中的可扩展性和局限性（`autonomous-driving, imitation-learning, behavior-cloning`）|

## 2020

### NLP
| [[nlp/ELECTRA.md]] | 提出取代 MLM 的 Replaced Token Detection（RTD）预训练任务，鉴别器比生成器更高效（`NLP, machine-learning, empirical-study`）|
| [[nlp/MacBERT.md]] | 针对中文的 BERT 改进，提出 MacBERT 使用纠错式掩码语言模型（`NLP, machine-learning, empirical-study`）|
| [[nlp/MobileBERT.md]] | 为移动端设计的紧凑 BERT，使用瓶颈结构和知识蒸馏（`NLP, machine-learning, empirical-study`）|
| [[nlp/gpt_3.md]] | 将 GPT 扩展到 1750 亿参数，通过上下文学习（In-Context Learning）在少量示例下完成各种任务，无需梯度微调。（`NLP, machine-learning, empirical-study`）|
| [[nlp/summarize_HF.md]] | 使用人类反馈训练总结模型，结合 RLHF 技术（`NLP, machine-learning`）|

### CNN
| [[cnn/CNN_summary.md]] | 对卷积网络核心组件（卷积层、归一化、激活函数、堆叠方式）及目标检测、实例分割、语义分割等应用领域的综合知识总结。（`computer-vision, machine-learning, survey`）|
| [[cnn/Consensus-Driven_Propagation.md]] | Consensus-Driven Propagation |
| [[cnn/GhostNet.md]] | 利用特征图冗余，通过廉价线性变换生成 Ghost 特征图（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/MCUNet.md]] | 提出 MCUNet 系统-模型协同设计框架，联合优化 TinyNAS 神经架构搜索和 TinyEngine 轻量推理引擎，首次在商用 MCU 上实现 >70% ImageNet 精度。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/SimCLR.md]] | 提出 SimCLR，一个简洁的对比学习框架，无需专门架构或记忆库，通过数据增广组合 + MLP 投影头 + 大批量训练实现 SOTA。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/SimSiam.md]] | 提出 SimSiam，无需负样本、无需动量编码器的简单孪生网络自监督学习方法（`computer-vision, machine-learning`）|
| [[cnn/blendmask.md]] | 提出 BlendMask，通过 blender 模块结合自上而下的实例级信息与自下而上的像素级语义信息，在保持单阶段检测框架速度的同时超越 Mask R-CNN 的掩码精度。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/cdp.md]] | 提出 CDP（共识驱动传播），通过 committee 和 mediator 模块从海量未标注人脸数据中稳健选择正样本对，仅用 9% 标签即可接近全监督性能。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/regnet.md]] | 提出从设计空间发现最优网络架构的系统方法论（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/yolo_v4.md]] | 系统性梳理检测器各组件的影响，组合 CSPDarknet-53 + PANet + MISH + CIoU 等技巧实现 SOTA。（`computer-vision, machine-learning, empirical-study`）|

### ViT
| [[vit/DeiT.md]] | 数据高效的图像 Transformer 训练和注意力蒸馏（`computer-vision, machine-learning`）|
| [[vit/ViT.md]] | 将 Transformer 直接应用于图像分类，ViT 成为 CV 基础架构（`computer-vision, machine-learning`）|
| [[vit/iGPT.md]] | 像素级生成式预训练（Image GPT）（`computer-vision, machine-learning`）|

### 生成模型
| [[Generative/DDPM.md]] | 提出去噪扩散概率模型（DDPM），将扩散模型与去噪分数匹配建立联系，实现高质量图像生成。（`machine-learning, theoretical`）|

### 自动驾驶
| [[Autonomous_Robot/CAB.md]] | 提出 CAB 运动预测模型，通过训练过程促进语义上下文信息的利用（`autonomous-driving, motion-forecasting, context-awareness`）|
| [[Autonomous_Robot/E2E-LD.md]] | 提出面向驾驶的车道检测评估指标 E2E-LD 和 PSLD（`lane-detection, autonomous-driving, evaluation-metric`）|
| [[Autonomous_Robot/Imitative_Models.md]] | 提出模仿模型（Imitative Models）结合模仿学习和目标导向规划的优势（`autonomous-driving, imitation-learning, planning`）|
| [[Autonomous_Robot/Label_Efficient.md]] | 系统量化减少语义分割标注成本对行为克隆智能体的影响（`autonomous-driving, imitation-learning, label-efficiency`）|
| [[Autonomous_Robot/RadarPerception.md]] | 利用连续雷达帧的时间关系增强雷达目标识别能力（`radar-perception, autonomous-driving, temporal-modeling`）|

### 视频
| [[video/Review_video_prediction.md]] | 视频预测深度学习技术综述，覆盖 ConvLSTM、3D CNN、GAN 和基于光流的方法。（`computer-vision, machine-learning, survey`）|
| [[video/Unsupervised_Spatiotemporal.md]] | 大规模对比实验研究无监督时空表征学习方法，覆盖多种代理任务和架构。（`computer-vision, machine-learning, empirical-study`）|

### 其他
| [[DeepAL_survery_2009.00236.md]] | 深度主动学习综述，系统回顾 DAL 方法在深度学习各领域的应用。（`machine-learning, survey`）|

## 2021

### NLP
| [[nlp/Alibi.md]] | 提出 ALiBi 位置编码方案，通过线性偏置实现长度外推（`NLP, machine-learning`）|
| [[nlp/Multi-turn_Dialogue_Survey.md]] | 多轮对话理解综述（`NLP, machine-learning`）|
| [[nlp/Prompting_Survey.md]] | 提示方法系统综述，统一分析预训练-提示-预测范式（`NLP, machine-learning`）|
| [[nlp/RoBERTa-wwm.md]] | 在 RoBERTa 基础上使用全词掩码（WWM）的中文预训练模型（`NLP, machine-learning, empirical-study`）|
| [[nlp/RoFormer.md]] | 提出旋转位置编码（RoPE），将相对位置编码融入绝对位置编码（`NLP, machine-learning`）|
| [[nlp/Transformers_Survey.md]] | Transformer 综述论文，系统回顾注意力机制和 Transformer 变体（`NLP, machine-learning`）|
| [[nlp/ernie_v3.md]] | 百亿参数知识增强预训练模型（`NLP, machine-learning`）|
| [[nlp/gpt_WebGPT.md]] | 让语言模型使用浏览器搜索并回答问题，结合人类反馈训练（`NLP, machine-learning, empirical-study`）|

### CNN
| [[cnn/DetCo.md]] | DetCo 在 MoCo v2 基础上引入多级监督和全局-局部对比学习，同时提升图像分类和目标检测的迁移性能。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/ResMLP.md]] | 完全基于 MLP 的图像分类架构，无需自注意力机制（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/repvgg.md]] | 提出 RepVGG，通过结构重参数化技术在训练时使用多分支拓扑、推理时合并为纯 3×3 卷积堆叠的 VGG 风格单路径网络，实现速度与精度的最佳权衡。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/yolor.md]] | 提出统一的隐式/显式知识编码网络，在多任务上共享统一表征。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/yolox.md]] | 将 YOLO 切换为无锚框检测器，引入解耦头 + SimOTA 标签分配，刷新 COCO SOTA。（`computer-vision, machine-learning, empirical-study`）|

### ViT
| [[vit/BEiT.md]] | BERT 风格图像预训练（`computer-vision, machine-learning`）|
| [[vit/MAE.md]] | 掩码自编码器，高效视觉表示学习（`computer-vision, machine-learning`）|
| [[vit/MoCo_v3.md]] | 自监督 ViT 训练的实证研究（`computer-vision, machine-learning`）|
| [[vit/SwinT.md]] | 层次化移动窗口 Transformer（`computer-vision, machine-learning`）|
| [[vit/ViTDet_Benchmarking.md]] | ViT 检测迁移学习的基准（`computer-vision, machine-learning`）|
| [[vit/iBOT.md]] | 在线标记器的图像 BERT 预训练（`computer-vision, machine-learning`）|

### 多模态
| [[Multimodal/CLIP.md]] | 对比语言-图像预训练，零样本迁移到下游视觉任务（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/GLIDE.md]] | 文本引导扩散模型实现逼真图像生成和编辑（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/TrOCR.md]] | 基于 Transformer 的光学字符识别（`computer-vision, machine-learning, NLP`）|

### 生成模型
| [[Generative/LatentDiffusion.md]] | 将扩散模型引入预训练自编码器的潜在空间，大幅降低计算成本，引入交叉注意力实现文本/条件控制，即 Stable Diffusion。（`machine-learning, computer-vision`）|

### 自动驾驶
| [[Autonomous_Robot/LAV.md]] | 从所有观察到的车辆（不仅自车）收集经验训练驾驶策略（`autonomous-driving, imitation-learning, multi-agent`）|
| [[Autonomous_Robot/SAM.md]] | 通过挤压网络和模仿网络进行条件视觉驾驶策略学习（`autonomous-driving, imitation-learning, knowledge-distillation`）|
| [[Autonomous_Robot/TokenFusion.md]] | 提出针对 ViT 的多模态令牌融合方法 TokenFusion（`transformer, multimodal-fusion, computer-vision`）|
| [[Autonomous_Robot/TransFuser.md]] | 提出 TransFuser 使用注意力机制融合图像和 LiDAR 表示（`autonomous-driving, transformer, multimodal-fusion`）|
| [[Autonomous_Robot/UTT.md]] | 提出统一 Transformer 跟踪器 UTT，用同一框架解决 SOT 和 MOT（`object-tracking, transformer, unified-framework`）|
| [[Autonomous_Robot/v2r_rl.md]] | 提出虚拟到现实强化学习框架，通过现实主义翻译网络实现 sim-to-real 迁移（`autonomous-driving, reinforcement-learning, domain-transfer`）|

### 效率/压缩
| [[LoRA.md]] | 提出低秩适配方法，冻结预训练权重，插入可训练的低秩矩阵，大幅降低微调参数量和显存需求。（`NLP, machine-learning, empirical-study`）|
| [[Switch_Transformers.md]] | 提出简化稀疏路由的 Switch Transformer，将 MoE 扩展到 1.6T 参数，训练速度提升 7 倍。（`NLP, machine-learning, empirical-study`）|
| [[model_compression.md]] | 模型压缩与加速综述，覆盖剪枝、量化、低秩分解、紧凑卷积滤波器和知识蒸馏四大类。（`machine-learning, survey`）|

### 其他
| [[diffusion.md]] | Diffusion 综述 |
| [[mlp-mixer.md]] | 提出仅用 MLP 的视觉架构，通过通道混合 + 空间混合 MLP 替代卷积和注意力。（`computer-vision, machine-learning, empirical-study`）|
| [[repmlp.md]] | 将卷积重参数化为全连接层，在训练时使用卷积结构，推理时等效为 MLP。（`computer-vision, machine-learning, empirical-study`）|

## 2022

### NLP
| [[nlp/CoT-Auto.md]] | 自动生成思维链示例的零样本 CoT 方法，无需人工设计提示（`NLP, machine-learning, empirical-study`）|
| [[nlp/CoT-Zero-shot.md]] | 零样本思维链：仅用 Let's think step by step 即可激发推理（`NLP, machine-learning, empirical-study`）|
| [[nlp/CoT.md]] | 提出思维链提示（CoT）方法，通过在提示中加入中间推理步骤大幅提升大模型推理能力（`NLP, machine-learning, empirical-study`）|
| [[nlp/FlashAttention.md]] | 通过 IO 感知的算法设计实现快速且内存高效的精确注意力（`NLP, machine-learning`）|
| [[nlp/LaMDA.md]] | 专为对话应用设计的语言模型，训练 1.56T 词元（`NLP, machine-learning`）|
| [[nlp/MorphTE.md]] | 将形态信息注入张量化嵌入（`NLP, machine-learning`）|
| [[nlp/Self-critiquing.md]] | 训练模型对自己的输出进行批评和改进，辅助人类评估（`NLP, machine-learning, empirical-study`）|
| [[nlp/UCD.md]] | 现实世界有害内容检测的整体方法（`NLP, machine-learning`）|
| [[nlp/XPOS.md]] | 提出 XPOS 位置编码，支持长度外推（`NLP, machine-learning`）|
| [[nlp/cpt-txt.md]] | 通过对比预训练学习文本和代码嵌入（`NLP, machine-learning`）|
| [[nlp/dual_encoder_qa.md]] | 探索双编码器架构在问答任务中的应用（`NLP, machine-learning`）|
| [[nlp/gpt_InstructGPT.md]] | 使用人类反馈强化学习（RLHF）微调 GPT-3，使模型更好地遵循用户指令，降低有害和不实输出。（`NLP, machine-learning, empirical-study`）|
| [[nlp/nlp_generalisation.md]] | NLP 泛化研究综述，分类体系（`NLP, machine-learning`）|

### CNN
| [[cnn/ConvNeXt.md]] | 借鉴 Swin Transformer 设计策略，现代化纯 CNN 架构（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/spach.md]] | 提出 SPACH 统一框架对 CNN、Transformer、MLP 三种结构进行公平对比实验，发现混合卷积与 Transformer 的模型可达到与精心设计的 SOTA 模型相当的性能。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/yolo_v6.md]] | 美团工业级检测框架，集成量化感知训练（PTQ/QAT），COCO 35.9% AP @ 1234 FPS（Nano）。（`computer-vision, machine-learning, empirical-study`）|
| [[cnn/yolo_v7.md]] | 提出可训练的 Bag-of-Freebies（重参数化、动态标签分配、辅助头），在实时检测器上刷新 SOTA。（`computer-vision, machine-learning, empirical-study`）|

### ViT
| [[vit/BEiT_2.md]] | 向量量化视觉标记器的掩码图像建模（`computer-vision, machine-learning`）|
| [[vit/DiTs.md]] | Transformer 作为扩散模型骨干（`computer-vision, machine-learning`）|
| [[vit/ViTDet.md]] | ViT 作为目标检测骨架的探索（`computer-vision, machine-learning`）|
| [[vit/ViT_derivatives.md]] | 视觉 Transformer 变体综述（`computer-vision, machine-learning`）|
| [[vit/simpleViT.md]] | 简化的高效 ViT 实现（`computer-vision, machine-learning`）|
| [[vit/vit_jpeg.md]] | 语言模型的视觉令牌助手（`computer-vision, machine-learning`）|

### 多模态
| [[Multimodal/BEiT_v3.md]] | 图像作为外语的统一多模态预训练（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/BLIP.md]] | 引导式语言-图像预训练，统一理解和生成（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/DeepNet.md]] | 通过 DeepNorm 将 Transformer 扩展到 1000 层（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/FLIP.md]] | 通过掩码策略扩展语言-图像预训练（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/Imagen.md]] | 文本到图像的逼真扩散模型（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/MAGNETO.md]] | 统一 Transformer 基础架构（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/MetaLM.md]] | 语言模型作为通用接口（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/OpenCLIP.md]] | CLIP 的开源复现（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/TorchScale.md]] | Transformer 大规模训练工具库（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/dall-e_v2.md]] | 层次化文本条件图像生成，使用 CLIP 隐空间（`computer-vision, machine-learning, NLP`）|

### 自动驾驶
| [[Autonomous_Robot/AdaRIP.md]] | 提出认知不确定性感知的鲁棒模仿规划（RIP/AdaRIP），检测并适应 OOD 驾驶场景（`autonomous-driving, uncertainty-estimation, distribution-shift`）|
| [[Autonomous_Robot/Choice_data.md]] | 系统研究数据设计选择对端到端驾驶模型训练和验证的影响（`autonomous-driving, end-to-end-learning, data-efficiency`）|
| [[Autonomous_Robot/Curve_LD.md]] | 提出基于参数贝塞尔曲线的车道检测方法，兼顾速度和精度（`lane-detection, autonomous-driving, parametric-curve`）|
| [[Autonomous_Robot/HiMODE.md]] | 提出基于 CNN+Transformer 混合架构的单目全向深度估计模型（`depth-estimation, autonomous-driving, transformer`）|
| [[Autonomous_Robot/MLDA.md]] | 在像素、实例和类别三个互补语义层级处理跨域车道检测（`lane-detection, domain-adaptation, autonomous-driving`）|
| [[Autonomous_Robot/MUTR3D.md]] | 通过 3D 轨迹查询实现端到端多相机 3D 跟踪（`3d-tracking, multi-camera, autonomous-driving`）|
| [[Autonomous_Robot/MoT_survey.md]] | 全面综述深度学习在多目标跟踪中的四个主要步骤（`multi-object-tracking, survey, deep-learning`）|
| [[Autonomous_Robot/ONCE-3DLanes.md]] | 构建真实世界 3D 车道检测数据集 ONCE-3DLanes 和 SALAD 方法（`lane-detection, 3d-detection, autonomous-driving-dataset`）|
| [[Autonomous_Robot/Time3D.md]] | 以端到端方式从单目视频联合训练 3D 检测和 3D 跟踪（`3d-detection, 3d-tracking, monocular`）|
| [[Autonomous_Robot/mmTTransformer.md]] | 提出基于堆叠 Transformer 的多模态运动预测框架 mmTransformer（`motion-prediction, transformer, multimodal`）|

### 视频
| [[video/MAE_st.md]] | 将 MAE 扩展到视频时空学习，提出 MAE-ST，在高掩码率下学习视频表示。（`computer-vision, machine-learning, empirical-study`）|

### 效率/压缩
| [[FlashAttention.md]] | FlashAttention |
| [[Sparse_Expert_review.md]] | 稀疏专家模型综述，覆盖 MoE 架构的设计、训练和推理（`computer-vision, machine-learning`）|
| [[X-MoE.md]] | 分析 MoE 的表示坍塌问题（token 向专家质心聚类），提出低维超球路由缓解坍塌。（`NLP, machine-learning, empirical-study`）|

### 其他
| [[DeepAL_survery_2203.13450.md]] | 对 19 种 DAL 方法进行公平比较实验，构建 DeepAL+ 工具包，评估 batch size / epoch 等影响因素。（`machine-learning, survey, empirical-study`）|
| [[RM_Overoptimization.md]] | 研究 RLHF 中奖励模型过优化现象（Goodhart's Law），提出 scaling law 预测最佳优化强度。（`NLP, machine-learning, empirical-study`）|
| [[self-Instruct.md]] | 提出 Self-Instruct 方法，让 LLM 自我生成指令数据指导自身微调，大幅降低人工标注成本。（`NLP, machine-learning, empirical-study`）|
| [[whisper.md]] | 在 68 万小时多语言弱监督数据上训练语音识别系统，零样本迁移达到有监督 SOTA 水平。（`machine-learning, NLP`）|

## 2023

### NLP
| [[nlp/LongNet.md]] | 提出扩张注意力（Dilated Attention），将 Transformer 扩展到 10 亿 token（`NLP, machine-learning`）|
| [[nlp/RetNet.md]] | 提出保留网络（RetNet），兼具并行训练和高效推理的优势（`NLP, machine-learning`）|
| [[nlp/gpt_4.md]] | 多模态大语言模型，在多种专业和学术基准上达到人类水平（`NLP, machine-learning, empirical-study`）|

### CNN
| [[cnn/ConvNeXt_v2.md]] | 将 ConvNeXt 与掩码自编码器（MAE）协同设计，提升自监督学习性能（`computer-vision, machine-learning, empirical-study`）|

### 多模态
| [[Multimodal/CoT-Multimodal.md]] | 将 CoT 推理扩展到多模态场景，两阶段框架（基本原理生成 → 答案推理），融合文本和图像信息。（`NLP, computer-vision, machine-learning`）|
| [[Multimodal/Kosmos-1.md]] | 多模态大语言模型，融合文本/图像/语音（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/SegGPT.md]] | 上下文中的全部分割统一模型（`computer-vision, machine-learning, NLP`）|
| [[Multimodal/Segment_Anything.md]] | 可提示分割一切的目标分割基础模型（SAM）（`computer-vision, machine-learning, NLP`）|

### 生成模型
| [[Generative/Consistency_models.md]] | 提出一致性模型，通过概率流 ODE 将任意噪声点直接映射到数据分布，实现单步生成，无需对抗训练。（`machine-learning, computer-vision, theoretical`）|

### 效率/压缩
| [[AdaLoRA.md]] | 在 LoRA 基础上按重要性动态分配参数预算，通过 SVD 参数化 + 剪枝不重要奇异值实现自适应。（`NLP, machine-learning, empirical-study`）|
| [[MegaByte.md]] | 提出多尺度 Transformer 架构，通过分块预测实现百万字节级别的序列建模，突破 tokenization 限制。（`NLP, machine-learning, empirical-study`）|
| [[QLoRA.md]] | 在 LoRA 基础上引入 4-bit NormalFloat + 双重量化 + 分页优化器，65B 模型单 48GB GPU 微调。（`NLP, machine-learning, empirical-study`）|

### 其他
| [[Dromedary.md]] | 提出 SELF-ALIGN 方法，通过 16 条原则 + 上下文学习实现 LLM 自对齐，仅需 <300 行人工标注。（`NLP, machine-learning, empirical-study`）|
| [[LLaMA.md]] | 提出 LLaMA 系列基础语言模型（7B–65B），仅使用公开数据训练，LLaMA-13B 以 1/10 参数量超越 GPT-3（175B）。（`NLP, machine-learning, empirical-study`）|

## 2026

### 其他
| [[2601.07372.md]] | 提出条件记忆（Engram），将 N-gram 嵌入改造为 O(1) 查表，与 MoE 互补（`NLP, machine-learning`） |

