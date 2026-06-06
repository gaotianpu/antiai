# raw/ 索引

## 索引性质
| 文件 | 说明 |
|:---|:---|
| [[deepseek_index.md]] | DeepSeek 重磅论文 |
| [[openai.md]] | OpenAI 论文 |
| [[NewBeer.md]] | 大牛/大神 论文 |

## CNN / 视觉基础
| 文件 | 说明 |
|:---|:---|
| [[cnn/LeNet.md]] | LeNet |
| [[cnn/alexnet.md]] | AlexNet |
| [[cnn/vgg.md]] | VGG |
| [[cnn/resnet.md]] | 提出残差学习（ResNet），通过跳跃连接解决深层网络退化问题（`computer-vision, machine-learning`）|
| [[cnn/resnext.md]] | ResNeXt |
| [[cnn/densenet.md]] | DenseNet |
| [[cnn/googlenet.md]] | GoogLeNet / Inception v1 |
| [[cnn/inception_v1.md]] | Inception v1 |
| [[cnn/inception_v3.md]] | Inception v3 |
| [[cnn/inception_v4.md]] | Inception v4 |
| [[cnn/SqueezeNet.md]] | SqueezeNet |
| [[cnn/MobileNet_v1.md]] | MobileNet v1 |
| [[cnn/MobileNet_v2.md]] | MobileNet v2 |
| [[cnn/MobileNet_v3.md]] | MobileNet v3 |
| [[cnn/ShuffleNet.md]] | ShuffleNet |
| [[cnn/ShuffleNet_v2.md]] | ShuffleNet v2 |
| [[cnn/EfficientNet.md]] | EfficientNet |
| [[cnn/regnet.md]] | RegNet |
| [[cnn/HarDNet.md]] | HarDNet |
| [[cnn/GhostNet.md]] | GhostNet |
| [[cnn/cspnet.md]] | CSPNet |
| [[cnn/sparsenet.md]] | SparseNet |
| [[cnn/ConvNeXt.md]] | ConvNeXt |
| [[cnn/ConvNeXt_v2.md]] | ConvNeXt v2 |
| [[cnn/ResMLP.md]] | ResMLP |
| [[cnn/repvgg.md]] | RepVGG |
| [[cnn/Res2Net.md]] | Res2Net |
| [[cnn/xception.md]] | Xception |
| [[cnn/Pooling.md]] | 池化方法 |
| [[cnn/BatchNorm.md]] | Batch Normalization |
| [[cnn/GroupNorm.md]] | 提出 Group Normalization，在 batch 较小时比 Batch Norm 更稳定（`computer-vision, machine-learning`）|
| [[cnn/GELUs.md]] | GELU 激活函数 |

## CNN — 目标检测 / 分割
| 文件 | 说明 |
|:---|:---|
| [[cnn/R-CNN.md]] | R-CNN |
| [[cnn/Faster_R-CNN.md]] | Faster R-CNN |
| [[cnn/Mask_R-CNN.md]] | 在 Faster R-CNN 基础上添加分割分支，同时做检测和实例分割（`computer-vision, machine-learning`）|
| [[cnn/fpn.md]] | 特征金字塔网络，利用多尺度特征图提升检测精度（`computer-vision, machine-learning`）|
| [[cnn/panet.md]] | PANet |
| [[cnn/sppnet.md]] | 空间金字塔池化，让 CNN 支持任意尺寸输入（`computer-vision, machine-learning`）|
| [[cnn/FCN.md]] | FCN |
| [[cnn/pspnet.md]] | PSPNet |
| [[cnn/DeepLab_v3.md]] | DeepLab v3 |
| [[cnn/yolo_v1.md]] | YOLO v1 |
| [[cnn/yolo_v2.md]] | YOLO v2 |
| [[cnn/yolo_v3.md]] | YOLO v3 |
| [[cnn/yolo_v4.md]] | YOLO v4 |
| [[cnn/yolo_v6.md]] | YOLO v6 |
| [[cnn/yolo_v7.md]] | YOLO v7 |
| [[cnn/yolor.md]] | YOLOR |
| [[cnn/yolox.md]] | YOLOX |
| [[cnn/Focal_Loss.md]] | 提出 Focal Loss，解决一阶段检测器正负样本不平衡问题（`computer-vision, machine-learning`）|
| [[cnn/Instance_Segmentation.md]] | 实例分割综述 |
| [[cnn/Semantic_Segmentation.md]] | 语义分割综述 |
| [[cnn/blendmask.md]] | BlendMask |
| [[cnn/rfb.md]] | RFB |
| [[cnn/spach.md]] | SPACH |
| [[cnn/Non-local.md]] | Non-local Networks |
| [[cnn/Deeply-Supervised.md]] | Deeply-Supervised Nets |

## CNN — 自监督 / 对比学习
| 文件 | 说明 |
|:---|:---|
| [[cnn/MoCo.md]] | MoCo |
| [[cnn/MoCo_v2.md]] | MoCo v2 |
| [[cnn/SimCLR.md]] | SimCLR |
| [[cnn/SimSiam.md]] | 无需负样本和动量编码器的简单孪生网络自监督方法（`computer-vision, machine-learning`）|
| [[cnn/DetCo.md]] | DetCo |
| [[cnn/cdp.md]] | CDP |
| [[cnn/Consensus-Driven_Propagation.md]] | Consensus-Driven Propagation |

## CNN — 其他
| 文件 | 说明 |
|:---|:---|
| [[cnn/CNN_summary.md]] | CNN 综述汇总 |
| [[cnn/MCUNet.md]] | MCUNet |

## NLP — 基础架构
| 文件 | 说明 |
|:---|:---|
| [[nlp/rnn.md]] | 提出简单循环神经网络（Elman RNN）用于处理时序结构（`NLP, machine-learning`）|
| [[nlp/lstm.md]] | 提出 LSTM，通过门控机制解决长序列梯度消失问题（`NLP, machine-learning`）|
| [[nlp/gru.md]] | 提出 GRU，简化的门控循环单元，性能与 LSTM 相当（`NLP, machine-learning`）|
| [[nlp/transformer.md]] | 提出完全基于注意力机制的 Transformer 架构，摒弃了循环和卷积，成为后续 LLM 的基础范式（`NLP, machine-learning`）|
| [[nlp/Transformer-XL.md]] | 提出片段级循环机制和相对位置编码，突破 Transformer 固定上下文长度限制（`NLP, machine-learning`）|
| [[nlp/XLNet.md]] | 广义自回归预训练，结合自回归和自编码优势（`NLP, machine-learning`）|
| [[nlp/Attention.md]] | 提出在神经机器翻译中联合学习对齐和翻译的注意力机制（Bahdanau Attention）（`NLP, machine-learning`）|
| [[nlp/Self-Attention_Sentiment_Analysis.md]] | 探索自注意力在情感分析中的应用（`NLP, machine-learning`）|
| [[nlp/Alibi.md]] | 提出 ALiBi 位置编码方案，通过线性偏置实现长度外推（`NLP, machine-learning`）|
| [[nlp/XPOS.md]] | 提出 XPOS 位置编码，支持长度外推（`NLP, machine-learning`）|
| [[nlp/RoFormer.md]] | 提出旋转位置编码（RoPE），将相对位置编码融入绝对位置编码（`NLP, machine-learning`）|
| [[nlp/LongNet.md]] | 提出扩张注意力（Dilated Attention），将 Transformer 扩展到 10 亿 token（`NLP, machine-learning`）|
| [[nlp/RetNet.md]] | 提出保留网络（RetNet），兼具并行训练和高效推理的优势（`NLP, machine-learning`）|
| [[nlp/FlashAttention.md]] | 通过 IO 感知的算法实现快速且内存高效的精确注意力（`NLP, machine-learning`）|

## NLP — 预训练模型
| 文件 | 说明 |
|:---|:---|
| [[nlp/gpt.md]] | 提出生成式预训练方法，验证了预训练-微调范式在 NLP 任务的通用性（`NLP, machine-learning`）|
| [[nlp/gpt_2.md]] | 扩展 GPT-1 至 15 亿参数，证明零样本迁移能力（`NLP, machine-learning`）|
| [[nlp/gpt_3.md]] | 扩展到 1750 亿参数，通过上下文学习完成各种任务（`NLP, machine-learning`）|
| [[nlp/gpt_4.md]] | 多模态大语言模型，在各种专业和学术基准上达到人类水平（`NLP, machine-learning`）|
| [[nlp/bert.md]] | 提出深度双向 Transformer 预训练模型，通过 MLM + NSP 统治 11 项 NLP 任务（`NLP, machine-learning`）|
| [[nlp/BERT-wwm.md]] | 全词掩码中文 BERT 预训练，掩码粒度从子词提升到整词（`NLP, machine-learning`）|
| [[nlp/RoBERTa.md]] | 系统优化 BERT 预训练：更大 batch、动态掩码、移除 NSP（`NLP, machine-learning`）|
| [[nlp/RoBERTa-wwm.md]] | RoBERTa 基础上使用全词掩码的中文预训练模型（`NLP, machine-learning`）|
| [[nlp/ALBERT.md]] | ALBERT |
| [[nlp/ELECTRA.md]] | 提出 Replaced Token Detection 预训练任务，比 MLM 更高效（`NLP, machine-learning`）|
| [[nlp/ernie.md]] | 将知识图谱实体信息融入语言表示预训练（`NLP, machine-learning`）|
| [[nlp/ernie_v2.md]] | 持续预训练框架，多任务增量学习（`NLP, machine-learning`）|
| [[nlp/ernie_v3.md]] | 百亿参数知识增强预训练模型（`NLP, machine-learning`）|
| [[nlp/SpanBERT.md]] | 提出 Span 级别预训练目标（Span Masking + SBO）（`NLP, machine-learning`）|
| [[nlp/StructBERT.md]] | 将语言结构信息（词序、句子结构）融入 BERT 预训练（`NLP, machine-learning`）|
| [[nlp/MacBERT.md]] | 针对中文的 BERT 改进，使用纠错式掩码语言模型（`NLP, machine-learning`）|
| [[nlp/TinyBERT.md]] | 两阶段 Transformer 蒸馏方法，将 BERT 压缩为小模型（`NLP, machine-learning`）|
| [[nlp/MobileBERT.md]] | 为移动端设计的紧凑 BERT，使用瓶颈结构和知识蒸馏（`NLP, machine-learning`）|
| [[nlp/BART.md]] | BART |
| [[nlp/T5.md]] | 将所有 NLP 任务统一为 Text-to-Text 格式，系统研究预训练方法影响（`NLP, machine-learning`）|
| [[nlp/Marian.md]] | Marian |
| [[nlp/LaMDA.md]] | 专为对话应用设计的语言模型，训练 1.56T 词元（`NLP, machine-learning`）|
| [[nlp/cpt-txt.md]] | 通过对比预训练学习文本和代码嵌入（`NLP, machine-learning`）|
| [[nlp/word2vec.md]] | 提出 Word2Vec（CBOW + Skip-gram），高效学习词向量（`NLP, machine-learning`）|
| [[nlp/BPE.md]] | 提出 BPE 子词分词方法，解决稀有词翻译问题（`NLP, machine-learning`）|
| [[nlp/LayerNorm.md]] | 提出 Layer Normalization，适用于 RNN/Transformer（`NLP, machine-learning`）|
| [[nlp/UCD.md]] | 现实世界有害内容检测的整体方法（`NLP, machine-learning`）|
| [[nlp/MorphTE.md]] | 将形态信息注入张量化嵌入（`NLP, machine-learning`）|

## NLP — 对齐与微调
| 文件 | 说明 |
|:---|:---|
| [[nlp/gpt_InstructGPT.md]] | 使用 RLHF 微调 GPT-3，使模型更好地遵循用户指令（`NLP, machine-learning`）|
| [[nlp/gpt_WebGPT.md]] | 让语言模型使用浏览器搜索并回答问题（`NLP, machine-learning`）|
| [[nlp/Self-critiquing.md]] | 训练模型对自己的输出进行批评和改进，辅助人类评估（`NLP, machine-learning`）|

## NLP — Prompt / CoT
| 文件 | 说明 |
|:---|:---|
| [[nlp/CoT.md]] | 提出思维链提示，通过中间推理步骤提升大模型推理能力（`NLP, machine-learning`）|
| [[nlp/CoT-Zero-shot.md]] | 零样本思维链：仅用「Let's think step by step」即可激发推理（`NLP, machine-learning`）|
| [[nlp/CoT-Auto.md]] | 自动生成思维链示例的零样本 CoT 方法（`NLP, machine-learning`）|
| [[nlp/Prompting_Survey.md]] | 提示方法系统综述，统一分析预训练-提示-预测范式（`NLP, machine-learning`）|
| [[nlp/Transformers_Survey.md]] | Transformer 综述，系统回顾注意力机制和 Transformer 变体（`NLP, machine-learning`）|

## NLP — 其他
| 文件 | 说明 |
|:---|:---|
| [[nlp/Multi-turn_Dialogue_Survey.md]] | 多轮对话理解综述（`NLP, machine-learning`）|
| [[nlp/dual_encoder_qa.md]] | 探索双编码器架构在问答任务中的应用（`NLP, machine-learning`）|
| [[nlp/summarize_HF.md]] | 使用人类反馈训练总结模型，结合 RLHF 技术（`NLP, machine-learning`）|
| [[nlp/nlp_generalisation.md]] | NLP 泛化研究综述及分类体系（`NLP, machine-learning`）|

## 视觉 Transformer (ViT)
| 文件 | 说明 |
|:---|:---|
| [[vit/ViT.md]] | 将 Transformer 直接应用于图像分类，ViT 成为 CV 基础架构（`computer-vision, machine-learning`）|
| [[vit/vit_jpeg.md]] | 语言模型的视觉令牌助手（`computer-vision, machine-learning`）|
| [[vit/ViT_derivatives.md]] | 视觉 Transformer 变体综述（`computer-vision, machine-learning`）|
| [[vit/ViTDet.md]] | ViT 作为目标检测骨架的探索（`computer-vision, machine-learning`）|
| [[vit/ViTDet_Benchmarking.md]] | ViT 检测迁移学习的基准（`computer-vision, machine-learning`）|
| [[vit/DeiT.md]] | 数据高效的图像 Transformer 训练和注意力蒸馏（`computer-vision, machine-learning`）|
| [[vit/SwinT.md]] | 层次化移动窗口 Transformer（`computer-vision, machine-learning`）|
| [[vit/simpleViT.md]] | 简化的高效 ViT 实现（`computer-vision, machine-learning`）|
| [[vit/MAE.md]] | 掩码自编码器，高效视觉表示学习（`computer-vision, machine-learning`）|
| [[vit/BEiT.md]] | BERT 风格图像预训练（`computer-vision, machine-learning`）|
| [[vit/BEiT_2.md]] | 向量量化视觉标记器的掩码图像建模（`computer-vision, machine-learning`）|
| [[vit/iGPT.md]] | 像素级生成式预训练（Image GPT）（`computer-vision, machine-learning`）|
| [[vit/iBOT.md]] | 在线标记器的图像 BERT 预训练（`computer-vision, machine-learning`）|
| [[vit/MoCo_v3.md]] | 自监督 ViT 训练的实证研究（`computer-vision, machine-learning`）|
| [[vit/DiTs.md]] | Transformer 作为扩散模型骨干（`computer-vision, machine-learning`）|

## 多模态
| 文件 | 说明 |
|:---|:---|
| [[Multimodal/CLIP.md]] | 对比语言-图像预训练，零样本迁移到下游视觉任务（`computer-vision, NLP`）|
| [[Multimodal/OpenCLIP.md]] | CLIP 的开源复现（`computer-vision, NLP`）|
| [[Multimodal/FLIP.md]] | 通过掩码策略扩展语言-图像预训练（`computer-vision, NLP`）|
| [[Multimodal/BLIP.md]] | 引导式语言-图像预训练，统一理解和生成（`computer-vision, NLP`）|
| [[Multimodal/Segment_Anything.md]] | 可提示分割一切的目标分割基础模型（SAM）（`computer-vision, NLP`）|
| [[Multimodal/SegGPT.md]] | 上下文中的全部分割统一模型（`computer-vision, NLP`）|
| [[Multimodal/Kosmos-1.md]] | 多模态大语言模型，融合文本/图像/语音（`computer-vision, NLP`）|
| [[Multimodal/MetaLM.md]] | 语言模型作为通用接口（`computer-vision, NLP`）|
| [[Multimodal/TrOCR.md]] | 基于 Transformer 的光学字符识别（`computer-vision, NLP`）|
| [[Multimodal/GLIDE.md]] | 文本引导扩散模型实现逼真图像生成和编辑（`computer-vision, NLP`）|
| [[Multimodal/dall-e_v2.md]] | 层次化文本条件图像生成，使用 CLIP 隐空间（`computer-vision, NLP`）|
| [[Multimodal/Imagen.md]] | 文本到图像的逼真扩散模型（`computer-vision, NLP`）|
| [[Multimodal/BEiT_v3.md]] | 图像作为外语的统一多模态预训练（`computer-vision, NLP`）|
| [[Multimodal/CoT-Multimodal.md]] | Multimodal CoT |
| [[Multimodal/DeepNet.md]] | 通过 DeepNorm 将 Transformer 扩展到 1000 层（`computer-vision, NLP`）|
| [[Multimodal/MAGNETO.md]] | 统一 Transformer 基础架构（`computer-vision, NLP`）|
| [[Multimodal/TorchScale.md]] | Transformer 大规模训练工具库（`computer-vision, NLP`）|

## 生成模型
| 文件 | 说明 |
|:---|:---|
| [[Generative/DDPM.md]] | DDPM |
| [[Generative/LatentDiffusion.md]] | Latent Diffusion |
| [[Generative/Consistency_models.md]] | Consistency Models |

## 强化学习
| 文件 | 说明 |
|:---|:---|
| [[RL/DQN.md]] | DQN |
| [[RL/DDPG.md]] | DDPG |
| [[RL/A2C.md]] | A2C |
| [[RL/ACER.md]] | ACER |
| [[RL/PPO.md]] | PPO |
| [[RL/TRPO.md]] | TRPO |
| [[RL/hp_RL.md]] | 超参数 RL |

## 自动驾驶
| 文件 | 说明 |
|:---|:---|
| [[Autonomous_Robot/DAVE-2.md]] | DAVE-2 |
| [[Autonomous_Robot/Imitative_Models.md]] | Imitative Models |
| [[Autonomous_Robot/TransFuser.md]] | TransFuser |
| [[Autonomous_Robot/LAV.md]] | LAV |
| [[Autonomous_Robot/Label_Efficient.md]] | Label Efficient |
| [[Autonomous_Robot/ChauffeurNet.md]] | ChauffeurNet |
| [[Autonomous_Robot/Learning_Situational_Driving.md]] | Learning Situational Driving |
| [[Autonomous_Robot/cheating.md]] | Cheating |
| [[Autonomous_Robot/H-Net.md]] | H-Net |
| [[Autonomous_Robot/E2E-LD.md]] | E2E-LD |
| [[Autonomous_Robot/Curve_LD.md]] | Curve LD |
| [[Autonomous_Robot/DAgger.md]] | DAgger |
| [[Autonomous_Robot/limit_Behavior_Cloning.md]] | Limit Behavior Cloning |
| [[Autonomous_Robot/RadarPerception.md]] | Radar Perception |
| [[Autonomous_Robot/VINS-Mono.md]] | VINS-Mono |
| [[Autonomous_Robot/AdaRIP.md]] | AdaRIP |
| [[Autonomous_Robot/CAB.md]] | CAB |
| [[Autonomous_Robot/Choice_data.md]] | Choice Data |
| [[Autonomous_Robot/HiMODE.md]] | HiMODE |
| [[Autonomous_Robot/MLDA.md]] | MLDA |
| [[Autonomous_Robot/mmTTransformer.md]] | mmTTransformer |
| [[Autonomous_Robot/MUTR3D.md]] | MUTR3D |
| [[Autonomous_Robot/ONCE-3DLanes.md]] | ONCE-3DLanes |
| [[Autonomous_Robot/SAM.md]] | SAM |
| [[Autonomous_Robot/Time3D.md]] | Time3D |
| [[Autonomous_Robot/TokenFusion.md]] | TokenFusion |
| [[Autonomous_Robot/UTT.md]] | UTT |
| [[Autonomous_Robot/v2r_rl.md]] | V2R-RL |
| [[Autonomous_Robot/MoT_survey.md]] | MOT Survey |

## 视频
| 文件 | 说明 |
|:---|:---|
| [[video/MAE_st.md]] | MAE-ST |
| [[video/Review_video_prediction.md]] | Video Prediction 综述 |
| [[video/Unsupervised_Spatiotemporal.md]] | Unsupervised Spatiotemporal |

## 效率 / 压缩
| 文件 | 说明 |
|:---|:---|
| [[LoRA.md]] | LoRA |
| [[QLoRA.md]] | QLoRA |
| [[AdaLoRA.md]] | AdaLoRA |
| [[Deep_Compression.md]] | Deep Compression |
| [[model_compression.md]] | 模型压缩综述 |
| [[Distilling_ss.md]] | 知识蒸馏 |
| [[Sparse_Expert_review.md]] | 稀疏专家模型综述，覆盖 MoE 设计、训练和推理（`computer-vision, machine-learning`）|
| [[Switch_Transformers.md]] | Switch Transformers |
| [[X-MoE.md]] | X-MoE |
| [[MegaByte.md]] | MegaByte |
| [[Vector_quantized.md]] | Vector Quantized |
| [[FlashAttention.md]] | FlashAttention |
| [[LayerNorm.md]] | LayerNorm |

## 其他
| 文件 | 说明 |
|:---|:---|
| [[Dromedary.md]] | Dromedary |
| [[LLaMA.md]] | LLaMA |
| [[whisper.md]] | Whisper |
| [[self-Instruct.md]] | Self-Instruct |
| [[RM_Overoptimization.md]] | RM Overoptimization |
| [[Xavier_init.md]] | Xavier Init |
| [[mlp-mixer.md]] | MLP-Mixer |
| [[repmlp.md]] | RepMLP |
| [[diffusion.md]] | Diffusion 综述 |
| [[OCR.md]] | OCR |
| [[meta_learning_survey.md]] | Meta-Learning 综述 |
| [[online_active_learning_survey.md]] | Online Active Learning 综述 |
| [[DeepAL_survery_2009.00236.md]] | Deep Active Learning (2009.00236) |
| [[DeepAL_survery_2203.13450.md]] | Deep Active Learning (2203.13450) |
| [[2601.07372.md]] | 提出条件记忆（Engram），将 N-gram 嵌入改造为 O(1) 查表，与 MoE 互补（`NLP, machine-learning`）|
