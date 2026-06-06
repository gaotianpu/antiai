# 论文阅读

## 历史&趋势
* 深度学习的层次结构
    1. 基础组件：FFN, 激活函数，残差连接，归一化, 嵌入, 损失函数, 卷积, self-attention等
    2. 模型架构：RNN(LSTM/GRU), CNN, Transformers, MLP-Mixer等
    3. 架构范式：掩码自动编码、解码自回归、对比学习
    4. 通用大模型： 图、文、语音、视频、多模态
    5. 各种下游任务：微调 -> 基于提示
* 模型架构:
    1. SVM,GBDT,MLP 多层感知机(与深度学习有很深的渊源); 依赖大量的特征工程,非端到端
    2. RNN([LSTM](./nlp/lstm.md),[GRU](./nlp/gru.md)) | CNN ([ResNet](./cnn/resnet.md), [CSPnet](./cnn/cspnet.md)); 
    3. [Transformer](./nlp/transformer.md)
    4. MLP, 受Transformer启发，人们重新思考设计MLP，CNN？
* 范式：
    1. 监督学习，依赖大量人工标注数据。各种NLP任务，图像分类、目标检测等;  
    2. 自监督预训练 + 微调
        * 掩码自动编码 masked auto-encoding: 掩码部分输入token，预测掩码部分内容，代表：[BERT](./nlp/bert.md),[MAE](./vit/MAE.md); 
        * 自回归 autoregressive: 预测下一个token或像素，代表：[GPT](./nlp/gpt.md), [iGPT](./vit/iGPT.md); 再试试预测上一个token？
        * 对比学习 contrastive learning： [MoCo](./cnn/MoCo.md), [SimCLR](./cnn/SimCLR.md), [CLIP](./Multimodal/CLIP.md) 等; 
        * 思考：再将对比学习与自回归、掩码模型统一起来？
    3. 多任务预训练 + 提示学习 [GPT-3](./nlp/gpt_3.md)
    4. 基于人工反馈的强化学习 [InstructGPT](./nlp/gpt_InstructGPT.md)
* 通用人工智能 AGI
    1. [多模态(图像、文本、视频、音频)，多任务融合统一](./Multimodal/README.md) 
    2. 稀疏专家模型，减少推理时的算力


##  神经网络基础
10. [主动学习](./Active_Learning.md) 寻找有价值样本做标注
11. [推理部署](./Inference_deploy.md)
12. [提示学习 Prompt Learning](./Prompt_Learning.md) 



## 二. [自然语言处理(NLP)](./nlp/README.md)
1. 主要任务 和 语料库
1. [BPE](./nlp/BPE.md) 文本编码
2. [文本的向量化表示: n-gram, Skip-gram, CBOW](./nlp/word2vec.md)
3. 循环神经网络: [RNN](./nlp/rnn.md), [LSTM](./nlp/lstm.md), [GRU](./nlp/gru.md)
4. [Transformer](./nlp/transformer.md)
5. 双向掩码语言模型: [BERT](./nlp/bert.md),[Erine](./nlp/ernie_v3.md) 等
6. 单向自回归(生成式)语言模型，[GPT-1](./nlp/gpt.md),[2](./nlp/gpt_2.md),[3](./nlp/gpt_3.md)
7. [InstructGPT](./nlp/gpt_InstructGPT.md) 基于强化学习，引入人类反馈机制, chatGPT引爆 


## 三. [计算机视觉(CV)](./cnn/README.md) 
1. [卷积网络](./cnn/README.md)
2. [ViT](./vit/ViT.md), [MAE](./vit/MAE.md) | [iGPT](./vit/iGPT.md) , [ViTDet](./vit/ViTDet.md)
3. 机器视觉任务： 
    * 图像分类: 主干网络的发展历程
    * [目标检测](./cnn/Object_Detection.md) 怎样实现无监督的学习？
    * [语义分割](./cnn/Semantic_Segmentation.md)
    * [实体分割](./cnn/Instance_Segmentation.md)
    * 人体关键点检测
    * 人脸识别
    * 文字识别(OCR)
4. [视频理解](./video/README.md) : 视频分类、目标追踪、轨迹预测、物理常识学习？

## 四、[多模态 Multimoda](./Multimodal/README.md)
1. 文本、图像、语音等统一，未来通用人工智能(AGI)的基础
2. 稀疏专家模型，减少推理时的算力


## 五、[生成模型](./Generative.md)
1. diffusion model 扩散模型
2. GAN 生成对抗模型
3. VAE 变微分自动编码器
4. Flow-based models 流模型 


## 六、[强化学习 Reinforcement](./Reinforcement_learning.md)
chatGPT 将PPO引入到人机对话系统中，改进大模型的期望输出。 


## 七、 3D
### [NeRF](https://zhuanlan.zhihu.com/p/512538748)
* NeRF: Representing Scenes as neural Radiance Fields for View Synthesis
* GIRAFFE
* Neural Fields in Visual Computing and Beyond
* State of the art on neural rendering
* https://dellaert.github.io/NeRF/
* https://github.com/yenchenlin/awesome-NeRF

## 八、[自主驾驶&机器人](./Autonomous_Robot/README.md)
* 双目摄像头测距
* 视觉+激光雷达|毫米波雷达？

## [牛人论文集](./NewBeer.md)
有时间整理一些牛人的论文集，瞅瞅他们的脑回路是咋连接的.

## [术语翻译](../tools/translate.dict.md)

## Paper 阅读心得
1. 当在某个行业深入到一定程度时，前沿创新已经不会很快的以书籍、视频教程的形式出现了，这个时候，毫无疑问要看paper了；
2. 不用纠结英文水平，使用百度等翻译应用将论文翻译成中文，添加个人理解、笔记等多多益善；所有论文放在gitee等代码托管平台上，方便与人交流；感觉比之前将英文论文打印出来，遇到不认识的单词查词的阅读方式要好些。gitee缺点手机上阅读体验不是很好。 另外pdf转文本，再按段落调取翻译api生成最终按段落区分的英文-中文内容，这个过程尽量做到自动化。
3. 书读百变，其义自现。前沿的内容理解起来可能费点劲，连续多读几遍，或者隔几天回头再读几遍，里面道理才会彻底明白；
4. 论文中通常会有Releated Work 一节，主要讲这篇论文参考了之前的那些研究成功，当篇文章某些知识点可能因为之前的论文中已经介绍过了，可能讲解的就会比较粗略，如果一头雾水的，需要往前翻看之前的相关论文。
5. 会有一些survey形式的论文，针对某个领域对过往论文的梳理调研，对于想切入新方向的人，可以先找出这类论文看看。可以把这类文章当成一个该领域知识体系的索引目录，顺藤摸瓜，快速理解该领域的系统发展情况。
6. 论文的代码复现。通常都能在github上找到论文的实现代码。通过阅读代码，可以对细节有更清晰的把我。
