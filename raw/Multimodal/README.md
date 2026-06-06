# 多模态
理想的多模态应该能够像人类一样，

## 一些思考
* 基于图片，将各种语言的概念对齐？
* 文本天然适合作为提示。而图像比较难，例如，不太容易根据提示做目标检测、语义分割任务？
* MAE的启发，图像的信息高度冗余，减少预训练成本？
* CLIP的对比学习，图像文本分别采用2个encoder，这个需要统一起来，MetaLM,BEiT_v3的思路？
* [Multi-Turn Dialogue 多轮对话](../nlp/Multi-turn_Dialogue_Survey.md), 相关论文还没怎么看 

## 数据获取
1. 纯文本
    * gpt-1: 
        * BooksCorpus, 7k 多本独特的未出版书籍, 很长一段连续的文本，这使得生成模型能够学习以远程信息为条件; 
        * 1B Word Benchmark, 但在句子级别进行了打乱 —— 破坏了远程结构. 
    * gpt-2: 构建尽可能大且多样的数据集,以便在尽可能多的领域和上下文中收集任务的自然语言演示。
        * Common Crawl, 存在严重的数据质量问题
        * 构造webtext，社交媒体平台reddit的链出文章(4500万); 
    * gpt-3: 
        * 基于与一系列高质量参考语料库的相似性过滤了CommonCrawl, 训练了一个质量分类器，
        * 在文档级别、数据集内部和数据集之间执行了模糊重复数据消除，以防止冗余
        * 将已知的高质量参考语料库添加到训练组合中，以增强CommonCrawl并增加其多样性
    * 维基百科; 开源图书; 网络小说; 百度文库，豆瓣/imdb的电影信息，电商的商品资料，企查查的企业信息
    * 搜索引擎：query+有点网页内容
    * 微博: 口语、短语化表达
2. 图像-文本对
    * CLIP: 50万个查询(维基百科英语版本中出现至少100次的单词,一定搜索量维基百科词条名,WordNet同义词集);每个查询中包含多达2万个图文对; 搜索 图文对 ，作为构建过程的一部分，这些对的文本包括一组500000个查询中的一个(1基本查询列表是维基百科英语版本中出现至少100次的所有单词。这一点得到了进一步的补充，其中包含了高度的相互信息以及一定搜索量以上的所有维基百科文章的名称。最后，将添加查询列表中尚未包含的所有WordNet同义词集)。我们通过在每个查询中包含多达20000对图-文来近似地平衡结果
    * KOSMOS-1: 一篇文章中，图片和上下文, 图片的alt属性. The Pile(排除：GitHub、arXiv、Stack Exchange 和 PubMed Central),CC-Stories 和 RealNews 
    * 搜索引擎：query+有点图片
3. 数据质量: gpt-2/3有比较详细的描述
    * 重复问题：包含 8-grams WebText 训练集标记的布隆过滤器;  HashingTF特征逻辑回归质量分类器
    * 丰富度衡量？



## 一、重要阶段回顾
### 1.文本
* 2017.6.12  [Attention Is All You Need](../nlp/transformer.md) self-attetion机制，Transformer架构首次引入
* 2018.6.11  [Improving Language Understanding by Generative Pre-Training](../nlp/gpt.md) GPT,自回归预测下一个token，生成式. 外推？
* 2018.10.11 [BERT Bidirectional Encoder Representations from Transformer ](../nlp/bert.md) BERT,输入部分掩码，预测掩码token。内插？
* 2022.1.24  [Text and Code Embeddings by Contrastive Pre-Training](../nlp/cpt-txt.md)  对比学习 


### 2.图像
* 2020.6.17  [Generative Pretraining from Pixels](../vit/iGPT.md) iGPT,根据文本gpt思路，图像gpt,预测下一个像素值，应改造为像ViT那样，预测下一个图块？
* 2020.10.22 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](../vit/vit.md) ViT,图像分块输入
    * https://github.com/lucidrains/vit-pytorch vit相关的各种论文实现
* 2021.11.11 [Masked autoencoders are scalable vision learners](../vit/MAE.md)  MAE,掩码思路, 图像的冗余特征，75%的掩码率
* 2022.3.30  [Exploring Plain Vision Transformer Backbones for Object Detection](../vit/ViTDet.md)  ViTDet,金字塔结构的适配，小图训练，大图推理的能力
* 2021.4.5   [An Empirical Study of Training Self-Supervised Vision Transformers](../vit/MoCo_v3.md) 对比学习  

### 3.多模态
* 2021.3.26 CLIP [Learning Transferable Visual Models From Natural Language Supervision](./CLIP.md) 对比学习，正负图像-文本对; 
    * 2022.12.1 FLIP(MAE+CLIP)  [Scaling Language-Image Pre-training via Masking](./FLIP.md) 
    * 2023.1.30 [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
    * https://huggingface.co/docs/transformers/main/model_doc/blip-2
    * 和图像+图像增广一起训练呢？
* 2022.8.22 BEiT_v3 [Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks](./BEiT_v3.md) 掩码模型
* 2023.2.27 Kosmos-1 [Language Is Not All You Need: Aligning Perception with Language Models](./Kosmos-1.md) 基于MetaLM
    * 2022.6.13 MetaLM [Language Models are General-Purpose Interfaces](./MetaLM.md) 智能的2个层次：表述现实，描绘理想。 初级采用掩码模型用于理解感知，表述现实; 高级采用自回归模型，用于推理、规划、创造性等描绘理想; 

输入嵌入的统一，向字节对齐


## 二、主要问题和改进
1. 训练稳定性 
    * 2022.3.1 [DeepNet: Scaling Transformers to 1,000 Layers](./DeepNet.md) DeepNorm + 初始化方法
    * 2022.10.12 [MAGNETO: Foundation Transformers](./MAGNETO.md)  文本、视觉等transformer架构统一，重点改造LN;  
    * 2022.11.23 [TorchScale: Transformers at Scale](./TorchScale.md) 
2. 输入长度外推 
    * 2022.12.20 [A Length-Extrapolatable Transformer](../nlp/XPOS.md) XPOS, 相对位置编码 + 推理阶段使用分块因果注意力。
    * 2021.8.27 [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](../nlp/Alibi.md) Alibi,结合xpos的论文，这个alibi更具有创新型
    * ViTDet也有训练和推理阶段图像分辨率不一致的情况，也有一套解决办法，或许也可以统一起来; 
3. 稀疏专家模型 
    * 2022.4.20 [On the Representation Collapse of Sparse Mixture of Experts](../X-MoE.md) 关于专家稀疏混合的表示坍塌
    * 2022.9.4 [A Review of Sparse Expert Models in Deep Learning](../Sparse_Expert_review.md) 综合调研
    * 2021.1.11 [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](../Switch_Transformers.md) 


## 三、[提示工程](../Prompt_Learning.md)


## 四、基于扩散模型的条件生成
* 2022.5.23  Imgen [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](./Imagen.md) 
* 2022.4.13  DALL-e v2 [Hierarchical Text-Conditional Image Generation with CLIP Latents](./dall-e_v2.md) 
* 2021.12.20 GLIDE [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](./GLIDE.md) 

## 五、其他
* [多模态神经元](https://openai.com/blog/multimodal-neurons/) 特征可视化技术，https://distill.pub/2021/multimodal-neurons/


## 六、开源代码实现
1. BPE, Byte Pair Encoding (其他: WordPiece、ULM)
    * https://github.com/openai/tiktoken Rust实现，效率很高
    * https://github.com/openai/gpt-2/blob/master/src/encoder.py
2. GPT
    * https://github.com/karpathy/nanoGPT ， GPT系列的论文实现
    * https://github.com/karpathy/minGPT
3. BERT
    * https://github.com/codertimo/BERT-pytorch
4. ViT,MAE,ViTDet
    * https://github.com/lucidrains/vit-pytorch, ViT,MAE等实现
    * https://github.com/facebookresearch/mae 官方实现
    * ViTDet 
5. 大模型 
    * https://github.com/microsoft/torchscale, 包括XPos,Magneto,X-MoE,DeepNet等几篇论文实现
    * https://github.com/microsoft/unilm 