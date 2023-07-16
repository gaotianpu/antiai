# AI论文和代码实现

创新常常在边缘，请给探索以空间。

## * 最近进展
* [chatGPT](./thinking/chatGPT.md)
    * Google 的 《我们没有护城河，OpenAI也没有》, 总结了时下大语言模型的开源进展~
* 视觉领域
    * Meta的 [Segment Anything](./paper/Multimodal/Segment_Anything.md)，分割一切，GUI形式的人机对齐，chat形式之外的另一种GUI对齐形式，应用场景应该也会比chat形式更广泛？
* 图像生成
    * OpenAI的 [Consistency Models](./paper/Consistency_models.md) 一致性模型，比迭代式的扩散模型更快的生成模型；


## 一、整体概览
分层 | 说明 
--- | --- 
5 - 具体任务 | 微调 -> 人机对齐([Chat形式](./paper/nlp/gpt_InstructGPT.md) -> [GUI形式](./paper/Multimodal/Segment_Anything.md)) -> [提示学习](./basic/Prompt_Learning.md) 
4 - 各类模态 | [自然语言](./paper/nlp/README.md)、[图像](./paper/cnn/README.md) 、音频、[视频](./paper/video/README.md)、雷达、传感器等 -> [多模态融合](./paper/Multimodal/README.md)
3 - 训练范式 | 特定任务全监督 -> 自监督预训练(掩码自动编码，生成式自回归、[对比学习](./basic/contrastive_learning.md)) 
2 - 模型架构 | SVM/GBDT/[MLP](./basic/mlp.md) -> [RNN](./paper/nlp/rnn.md)([LSTM](./paper/nlp/lstm.md)/[GRU](./paper/nlp/gru.md)), [CNN](./paper/cnn/README.md) -> [Transformers](./paper/nlp/transformer.md) -> ? ([MLP-mixer](./paper/mlp-mixer.md))
1 - 基础组件 | [全连接](./basic/linear_regression.md)、[激活函数](./basic/Activation.md)、[归一化](./basic/Normalization.md) 、[初始化策略](./paper/Xavier_init.md)、[损失函数](./basic/Loss_metric.md)、[优化算法](./basic/Optimizer.md)、[残差连接](./paper/cnn/resnet.md)、[正则化策略](./basic/regularization.md)、嵌入

### 重要的时间节点
1. 2009年前，受限于数据规模和算力，尽管[RNN](./paper/nlp/rnn.md)/[CNN](./paper/cnn/LeNet.md)等深度学习架构模型已经发明出来，但与SVM/GBDT等非深度学习算法相比，深度神经网络的优势并不明显。
2. 2009年，[ImageNet](https://www.image-net.org/)图像标注数据集提供了深度学习的数据基础，随后，其他各类开源的标注数据集也陆续提供。互联网(特别是移动互联UGC)的快速发展，使得大规模的数据收集标注工作成为可能。
3. 2012年，[Alexnet](./paper/cnn/alexnet.md)采用GPU加速深度学习的训练过程，为算力需求的解决指明了方向(留意英伟达股票这些年的表现)。随着深度学习越来越成功，影响力越来越大。为深度学习的训练和推理阶段设计专门优化的芯片成为很有潜力的产业，Google的TPU，其他公司也提供了各种替代方案。可以预见，未来的算力需求将依然强劲。
4. 2012 ~ 2017，计算机视觉的[CNN](./paper/cnn/README.md)得到长足发展，图像分类、[目标识别](./paper/cnn/Object_Detection.md)、[语义分割](./paper/cnn/Semantic_Segmentation.md)、[实例分割](./paper/cnn/Instance_Segmentation.md)、姿态估计等任务效果不断提高，同时，也促进了深度神经网络基础组件的发展和完善。但视觉领域的发展无不依赖大量的数据标注工作，人工标注数据成本相当高昂。NLP方向架构上没太大进展，探索了[词嵌入的训练](./paper/nlp/word2vec.md)、句子整体表征能力。
5. 2017年，[Transformers](./paper/nlp/transformer.md)架构提出，随后不久[GPT](./paper/nlp/gpt.md),[BERT](./paper/nlp/bert.md)相继提出，使得自然语言领域大规模自监督预训练成为可能。自监督预训练不再需要人工标注数据，而下游任务只需少量数据微调就可以了。随着预训练数据获取成本的大幅降低，训练超大模型成为可能。与此同时，采用对比学习范式实现图像领域的自监督预训练也开展中。
6. 2020年，[视觉Transformer(ViT)](./paper/vit/ViT.md)提出，[MAE](./paper/vit/MAE.md)/[BEit](./paper/vit/BEiT.md)/[ViTDet](./paper/vit/ViTDet.md)等开启ViT时代的探索，而采用[对比学习](./basic/contrastive_learning.md)模式的[CLIP](./paper/Multimodal/CLIP.md)实现了T时代图像-文本多模融合。[各种模态(NLP,CV等)模型架构的统一](./paper/Multimodal/README.md)将为迈向通用人工智能打下坚实的基础。
7. 2022年底，[chatGPT](./paper/thinking_chatGPT.md)出现。超大模型能够涌现出之前不具备的各种能力，而自然语言的各种任务都能够用对话形式表示(一轮不行就多轮)，因此，针对特定领域的微调就变成了针对所有领域的人机意图对齐。同样，视觉领域图像分类、目标识别、语义\实体分割等，也可以统一到一种形式上--[分割anything](./paper/Multimodal/Segment_Anything.md)，也可以实现人机意图对齐，多模领域也有这个趋势。最终，更像人类的通用智能将会出现。


## 二、基础组件
1. [标量、向量、矩阵、张量](./basic/vector.md)
    * 具体事物、现象，文本，图片，音频等的向量化表示

2. [线性回归 Linear regression](./basic/linear_regression.md)
    * 回归问题，预测连续值，线性函数，随机初始化，均方误差损失，梯度下降/上升，优化方法：随机/小批量随机

3. [逻辑回归 logistic regression](./basic/logistic_regression.md)
    * 分类问题，预测离散分类标签(也可输出连续的概率值)，线性+sigmod/softmax，交叉熵损失/负对数似然，分类的评估指标

4. [多层感知机 MLP(Multilayer Perceptron)](./basic/mlp.md)
    * 也叫：前馈神经网络 FFN（feedforward neural network）
    * 线性+非线性的深度堆叠可以模拟各种多项式 [Polynomial Regression As an Alternative to Neural Nets](https://arxiv.org/abs/1806.06850)。
    * 深度网络训练过程中的梯度消失/爆炸问题会导致训练的崩塌、不收敛： [激活函数](./basic/Activation.md)，[归一化策略](./basic/Normalization.md)
    * 数据集划分：训练集、验证集、测试集; 欠拟合/过拟合，偏差/方差，泛化能力是终极;  
    * 过拟合 -> [正则化策略](./basic/regularization.md)：复杂度惩罚, dropout, 早停等
    * 隐藏层的研究： 梯度变化情况; 梯度消失侦测; 可视化等 

5. [激活函数 Activation functions](./basic/Activation.md)
    * 纯线性函数的堆叠仍然还是线性函数，而线性+非线性的深度堆叠可以模拟各种多项式。像欧几里得能够只用直尺(线性)和圆规(非线性)就能推导出200多个几何定律一样的奇妙。

6. [归一化 Normalization](./basic/Normalization.md) 

7. [损失函数和评估指标 Loss & Metric](./basic/Loss_metric.md)

8. [优化器和学习率 Optimizer & learning rate](./basic/Optimizer.md)

9. [正则化 Regularization](./basic/regularization.md) 
    * 防止过拟合，在超大数据集面前，过拟合似乎不是个事儿


## 三、架构演化
1. SVM/GBDT/[MLP](./basic/mlp.md)
2. 用于序列处理的 [RNN](./paper/nlp/rnn.md): [LSTM](./paper/nlp/lstm.md),[GRU](./paper/nlp/gru.md)
3. 用于图像处理的 [CNN](./paper/cnn/README.md): [LeNet](./paper/cnn/LeNet.md),[AlexNet](./paper/cnn/alexnet.md),[VGG](./paper/cnn/vgg.md),[ResNet](./paper/cnn/resnet.md),[CSPnet](./paper/cnn/cspnet.md),[ConvNeXt](./paper/cnn/ConvNeXt.md)
4. 统一架构：[Transformers](./paper/nlp/transformer.md) 
5. 极简架构的尝试：[MLP-mixer](./paper/mlp-mixer.md), 未来是否还有更好的，能够替代Transformers的架构？


## 四、[Transformers](./basic/intro_transformer.md)
1. 自然语言 [Attention Is All You Need](./paper/nlp/transformer.md) 
2. 视觉 ViT [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](./paper/vit/ViT.md)
3. 多模态的改进：[MAGNETO:post/pre/sub-LN](./paper/Multimodal/MAGNETO.md), [BEiT-3,不同模态，不同的FFN](./paper/Multimodal/BEiT_v3.md)
4. 训练时的稳定性: [DeepNet:初始化+梯度裁剪](./paper/Multimodal/DeepNet.md), [深度神经网络初始化 Xavier_init](./paper/Xavier_init.md)
5. 输入长度可外推: [XPos](./paper/nlp/XPOS.md), [ALibi](./paper/nlp/Alibi.md), [RoPE](./paper/nlp/RoFormer.md)
6. [MegaByte](./paper/MegaByte.md)
    * 多模输入更统一：文本、图像、视频，都当成一串连续字节输入；
    * 可以处理超长字节串；
    * 并行生成，多快好省。 并行分块(多token)生成，是不是意味着RLHF就没用了？
    * [Bytes Are All You Need: Transformers Operating Directly On File Bytes](https://arxiv.org/abs/2306.00238)
6. 稀疏专家模型：[调研](./paper/Sparse_Expert_review.md), [X-MoE](./paper/X-MoE.md), [SwitchT](./paper/Switch_Transformers.md)
7. 开源的基础架构：[TorchScale](./paper/Multimodal/TorchScale.md) 


## 五、 预训练范式
| | 掩码自动编码 | 生成式自回归 | 编码-解码 | [对比学习](./paper/contrastive_learning.md) | 掩码+对比
| --- | --- | --- | --- | --- | ---
| 自然语言 | [BERT](./paper/nlp/bert.md) | [GPT-1](./paper/nlp/gpt.md),[2](./paper/nlp/gpt_2.md),[3](./paper/nlp/gpt_3.md) | [BART](./paper/nlp/BART.md), [T5](./paper/nlp/T5.md), [Marian](./paper/nlp/Marian.md) | [cpt-txt/code](./paper/nlp/cpt-txt.md) | ---
| 视觉 | [BEiT_2](./paper/vit/BEiT_2.md) | [iGPT](./paper/vit/iGPT.md) | [MAE](./paper/vit/MAE.md) | [Moco-1](./paper/cnn/MoCo.md),[2](./paper/cnn/MoCo_v2.md),[3](./paper/vit/MoCo_v3.md) [SimCLR](./cnn/SimCLR.md)  | ---
| 多模 | [BEiT_3](./paper/Multimodal/BEiT_v3.md) < [MetaLM](./paper/Multimodal/MetaLM.md) | [Kosmos-1](./paper/Multimodal/Kosmos-1.md)> | --- | [CLIP](./paper/Multimodal/CLIP.md)   | [FLIP = MAE+CLIP](./paper/Multimodal/FLIP.md)

### 说明
* 三种范式的直观理解：
    * 掩码自动编码，完形填空
    * 生成式自回归，预测下一个token
    * 对比学习
* 三种范式与模型架构的发展是正交的，既，可以使用RNN/CNN/Transfomers实现这三种范式的训练, 只不过Transformers的优点太多了，近几年这三种范式都基于Transformers实现了。
* 词嵌入是早期的预训练尝试，取得很好的成果，代表论文[word2vec](./paper/nlp/word2vec.md)，训练方式：
    * n-gram，根据前几个tokens预测下一个token，有点生成式的意思，和后来的GPT思路相似；
    * skip-gram，给定中间的token，预测它周边的tokens； 
    * CBOW则是给定上下文tokens，预测中间的token，有点像后来的BERT，双向掩码预测的思路； 
    模型则选用MLP, 通过预训练产出词向量，有一定的表义能力，下游可以通过简单输入词向量相加去预测文本分类，也可以使用RNN模型预测情感分类这样复杂点的语义；词向量的缺点是，字词脱离句子就会有歧义产生。后来的 [ELMO](https://arxiv.org/abs/1802.05365) 通过LSTM得到一个能够完整表征句子语义的模型。
* MetaLM,Kosmos-1的核心思想，智慧分2个层次，第一个层次是感知理解，由输入产生潜在表示，掩码自动编码适合；第二个层次，逻辑、推理等高阶形式，由潜在表示产生最终的输出，生成式回归模型适合。 
* 对比学习最早基于CNN架构，在视觉领域取得成功，后来OpenAI的CLIP将其引入多模领域；对比学习替代强化学习PPO的思路？


## 六、微调、RLHF、提示学习
1. 微调 fine-tune
    * [Parameter-Efficient Fine-Tuning (PEFT)](https://github.com/huggingface/peft) 各种微调技术集成
    * [LoRA, Low-Rank Adaptation of LLM](https://github.com/microsoft/LoRA)，插件式的微调
    * [AdaLoRA:Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](./paper/AdaLoRA.md)
    * [QLoRA: Efficient Finetuning of Quantized LLMs](./paper/QLoRA.md)
2. RLHF 基于人类反馈的强化学习
    * [InstructGPT](./paper/nlp/gpt_InstructGPT.md) 2022.3.4
    * [Learning to summarize from human feedback](./paper/nlp/summarize_HF.md) 2020.9.2
    * [PPO](./paper/RL/PPO.md) 2017.7.20
    * [SELF-INSTRUCT](./paper/self-Instruct.md)  基于LLM自动生成各种提示任务
    * [Principle-Driven Self-Alignment](./paper/Dromedary.md)
3. [提示学习 Prompt Learning](./basic/Prompt_Learning.md) 
    * [Chain-of-Thought](./paper/nlp/CoT.md)，思维链，让ai将思维过程展示出来
    * Self-Consistency，自洽性，
    * system/user/assistant, 在同一个大模型基础上做到对话时千人千面？面对不同人，呈现不同的性格？ 看OpenAI的prompt教程，可以给对话指定个背景。
4. 图形界面交互对齐
    * [Segment Anything](./paper/Multimodal/Segment_Anything.md)
    * 基于AI的对话形式很了不起，但判断可能还是像PC早期的DOS等命令行形式那样(命令行+管道的形式，自动化脚本任务)，最终会出现基于AI的GUI交互形式，普通人似乎更能接受。 
    * 目标检测、语义分割、目标追踪  


## 七、工程实践
1. 大规模训练技巧，GPU集群的运维、使用； 
    * [ColossalAI:分布式深度学习模型](https://github.com/hpcaitech/ColossalAI) 
2. 模型轻量化：剪枝、量化、蒸馏
    * MLC LLM https://github.com/mlc-ai/mlc-llm
    * [Distilling Step-by-Step](./paper/Distilling_ss.md) 
    * [Quant-Noise:Training with Quantization Noise for Extreme Model Compression](https://arxiv.org/abs/2004.07320)
    * [MiniViT: Compressing Vision Transformers with Weight Multiplexing](https://arxiv.org/abs/2204.07154)
    * 剪枝：是把连接直接干掉，还是权重为0？
    * 量化：训练后PTQ、训练中QAT；线性量化影响不大，激活函数量化影响较大
3. [推理部署](./basic/Inference_deploy.md)
4. hugging face
    * https://huggingface.co/learn/nlp-course/zh-CN/chapter0/1

## 八、数据
1. 丰富性、多样性，知道训练集中的数据类型分布，能够自动发现新鲜的、没见过，需要补充的数据？
2. [主动学习](./basic/Active_Learning.md) 寻找有价值样本做标注
3. 中文语料数据获取。 内容质量、获取成本
    * 中文语料缺乏，例如很多学术论文都以英语形式，简单翻译后提供给中文模型是否可行？
    * WordNet名词层次结构 中文有类似的？
    * 大规模预训练：
        * 图书(电子书)，纸质图书扫描后转文本，过往的报刊杂志
        * 网络小说
        * 文库
        * 维基百科
        * 学术论文、博客？
        * 问答：知道、知乎
        * 影视动漫介绍、评价
        * 商品介绍、评价；本地生活的商户，菜品、服务，评价等。
        * 企查查，公司信息
        * 招聘jd
    * 多模态
        * query + 图片搜索结果
        * 微博，图片+文本
        * 文章中的图片和上下文 
    * 人类意图对齐
        * 搜索query
        * 问答中的问题-答案？
        * 论坛主贴-回帖？
4. [SELF-INSTRUCT](./paper/self-Instruct.md) 自动生成指令
    * [shareGPT](https://sharegpt.com/) 用户分享的chatGPT聊天记录 

## Others
这里主要放些算法模型的知识，数据和算力今后也会更多的关注些~
1. 数据需求，
    * 预训练大模型需要全网规模的数据(文本、图像、图文对), 考验的是网络爬虫能力，大规模数据清洗、去重能力。gtp-2,3介绍了一些全网数据的处理经验。
    * 人工标注数据重心会放在人机意图对齐上，人类标注员通过比较同一问题下，不同版本大模型生成内容的好坏，给出模型整体生成能力打分，通过强化学习策略指导模型微调。
    * 专业领域，则需要大量相关专家撰写高质量回答，这部分成本将会非常高，但对于改进机器生成会有很大的帮助，会有从“还行”到“惊喜”的效果改进。
    * 瞎想想：生物进化是从感知明暗，处理视觉开始的，随着人类出现，进化出语言和文字，从此，人类的语言和文字就和思维能力一同相互制约、发展，当人类发现了归纳与演绎的方法论后，数学定理和科学规律被陆续发现。目前的AI继承了这些，但不完全，只依赖人类现成的用文字表述的知识，或许永远无法超越人类的整体智慧？AI最终要在信息的最原始来源 -- 视觉领域取得突破，例如，通过一段连续的视频学习物理常识、因果关系等。
2. 算力方面
    * NPU(GPU)会针对模型训练阶段和推理阶段提供不同的优化，训练卡要求大批量、大内存、高浮点精度；推理卡则要求高并发，浮点精度往往要求不高(模型部署前，会做蒸馏、量化、压缩等降低浮点精度的操作)。国内NPU厂家大都先在推理卡上发力(多部署在他们的云端，公开的市面上似乎很难买到)，而训练卡目前似乎是英伟达一家独大。期待国产NPU的爆发，把chatGPT这样从零训练模型的费用打到白菜价(需调研下国内算力方面top5玩家-寒武纪、灵汐、壁仞、燧原)。
    * 算力与模型之间的开发框架，国外Pytorch,Tensorflow,国内的paddle，OneFlow等，加速卡与框架的适配工作也很重要。
    * 超大模型服务的云端部署 + 开放API；    
    * 自主机器人(无人车,无人机,机器人等)的单机部署；多模态大模型的突破将会给自主机器人带来突破，马斯克的人形机器人布局的时机很nice啊！这玩意一旦突破，市场的需求量可海了去了。
3. 产业方面
    * 公域AI，大模型本身，群雄逐鹿，天下未定，将会有一波机会,的Top3胜出窗口期大约2~3年；
    * 另一个角度，把AI当成一种像搜索、推荐、直播这样的功能，除top3头部玩家外，电商、本地生活、游戏、视频站点等都会标配AI功能；
    * 私域AI，chatGPT这样的公域AI，存在人机交互的信息泄露的可能，私域AI也有较大的场景；
    * 围绕大模型的二次开发，类似app机制；办公软件、影视制作工具、CAD设计等等，会有大量基于chatGPT的二次开发。最终的发展，会让更多人收益。比如，乡村农人通过直播向全国的客户卖自家的农产品,这在以前信息、物流条件下是难以想象的。

