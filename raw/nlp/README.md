# 自然语言处理(NLP)

## 主要任务
* ... 
* 文本分类 text classification
* 推理 Entailment 蕴含  给出Premise前提，得出Hypothesis假设，自然语言推理(Natural Language Inference 或者 Textual Entailment)：判断两个句子是包含关系(entailment)，矛盾关系(contradiction)，或者中立关系(neutral); 
* 语义相似性 Semantic Similarity ，判断两个句子是否语义上是相关的
* 多选 Multiple Choice，给出上下文，得出答案，问答和常识推理(Question answering and commonsense reasoning)：类似于多选题，输入一个文章，一个问题以及若干个候选答案，输出为每个答案的预测概率; 
* machine translation , 机器翻译 
* document generation , 文档生成
* syntactic parsing , 句法解析 
* sequence labeling, 序列标记 
* 中分分词、词性标注、命名实体识别（Named Entity Recognition, NER）;  关系抽取？ POS tagging, chunking, named entity recognition, semantic role labeling

* [NLP的评估指标](https://zhuanlan.zhihu.com/p/339379980)
    * BLEU (bilingual evaluation understudy) 翻译质量,与人类翻译结果的一致性,缺点是该指标倾向于选择更短的翻译,不能处理没有字界(word boundary，如汉语)的语言。
    * METEOR(Metric for Evaluation of Translation with Explicit ORdering)
    * ROUGE(Recall-Oriented Understudy for Gisting Evaluation)比较自动生成的摘要或翻译与人类生成的参考摘要或翻译之间的相似性。ROUGE与BELU的区别是ROUGE只考虑召回率，即不关心翻译结果是否流畅，只关注翻译是否准确
    * CIDEr(Consensus-based Image Description Evaluation) 图片摘要

1. [nlp主要任务](https://zhuanlan.zhihu.com/p/109122090), 数据集、衡量指标
1. 实体标注(中文分词处理方式类似)
* BMES, Begin表示一个词的词首位值，Middle表示一个词的中间位置，End表示一个词的末尾位置，Single表示一个单独的字词。
* BIO, B代表实体的开头,Inside代表实体的结尾, Outside代表不属于任何类型
* BIOES, B表示开始,I表示内部,O表示非实体,E实体尾部, S表示该词本身就是一个实体
2. 实体关系抽取 Relation Extraction
* Joint Entity and Relation Extraction Based on A Hybrid Neural Network
* End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures
* Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme 标注策略？
3. 知识图谱构建?

## 数据集
1. [中文语料库](https://www.zhihu.com/question/21177095)
2. 外文语料库
* BooksCorpus
* 1B Word Benchmark
 Stanford Sentiment Treebank-2 [54], CoLA [65]
  MSR Paraphrase Corpus [14], Quora Question Pairs [9], STS Benchmark [6]
   RACE [30], Story Cloze [40]
    SNLI [5], MultiNLI [66], Question NLI [64], RTE [4], SciTail [25]
* Storycloze
Winograd
SuperGLUE
* Penn Tree Bank (PTB) 
* LAMBADA, 要求预测需要阅读一段上下文的句子的最后一个词
* HellaSwag, 为故事或指令集选择最佳结局
* StoryCloze 2016 , 为五句长篇故事选择正确的结尾句
Natural Questions [KPR+19]、
WebQuestions [BCFL13] 和 
TriviaQA [JCWZ17]
Winograd Schemas Challenge [LDM12], 确定代词指的是哪个词，当代词在语法上有歧义但在语义上对人类来说是明确的.
PhysicalQA (PIQA) [BZB+19]，询问有关物理世界如何运作的常识性问题，旨在探索对世界的扎根理解
F1 similarity score, BLEU, or exact match
ANLI reading comprehension datasets like RACE or QuAC
基准套件SuperGLUE
* BIG-Bench, 多项选择
3. 数据处理工具 
* 数据去重
* 测量数据污染的工具？
* html提取内容的工具

## 发展阶段
1. 针对特定任务的有监督训练  
2. 自监督预训练(词嵌入->句子表征) + 下游任务有监督微调; 
3. 基于prompt多任务统一预训练模型，将所有下游任务向预训练模型靠拢 
4. 大模型 + 元学习
5. NOW:引入人工反馈强化学习


## 一、文本的向量化表示
1. 词嵌入:n-gram,Skip-gram,CBOW 
    * word2vec: 2013-1-16 [Efficient Estimation of Word Representations in Vector Space](./word2vec.md)
    * GloVe: 2014 [Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)
    * 早期表示：one-hot,Bag-of-words(BoW),tf-idf;
2. 合适的token粒度,在存储空间和表示能力之间做平衡
    * BPE [Neural Machine Translation of Rare Words with Subword Units](./BPE.md)
        * [OpenAI提供的Rust实现,性能很不错](https://github.com/openai/tiktoken)
    * [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/pdf/1909.03341.pdf) 
    * [MorphTE: Injecting Morphology in Tensorized Embeddings](./MorphTE.md) 词嵌入压缩
    

## 二、RNN 循环神经网络
1. RNN 循环神经网络, 1990s [Finding Structure in Time](./rnn.md)
    * LSTM 长短记忆, 1997 [LONG SHORT-TERM MEMORY](./lstm.md)
    * GRU 门控单元, 2014.12.11 [Empirical evaluation of gated recurrent neural networks on sequence modeling](./gru.md)
2. ELMo 2018-2-15 [Deep contextualized word representations.](https://arxiv.org/abs/1802.05365) 使用神经网络生成更大粒度的表示，为BERT等提供了启发
<!--
3. Attention 2014-9-1 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
2016.1.25 [Long short-term memory-networks for machine reading](https://arxiv.org/abs/1601.06733)
2016.6.6 [A decomposable attention model](https://arxiv.org/abs/1606.01933)
2017.2.3 [Structured attention networks](https://arxiv.org/abs/1702.00887)
2017.3.9 [A structured self-attentive sentence embedding](https://arxiv.org/abs/1703.03130)
2017.5.11 [A deep reinforced model for abstractive summarization](https://arxiv.org/abs/1705.04304)
2015.11.25 [Neural GPUs learn algorithms](https://arxiv.org/abs/1511.08228)
2016.10.31 [Neural machine translation in linear time](https://arxiv.org/abs/1610.10099)
2017.5.8 [Convolutional sequence to sequence learning](https://arxiv.org/abs/1705.03122)
-->

## 三、Transformer
1. Transformer 2017-06 [Attention Is All You Need](./transformer.md)
    * 2022.6.12 MAGNETO [Foundation Transformers](./MAGNETO.md), sub-LayerNorm,更有效的缩放模型,模型初始化方法
    * 2021.6.8 [A Survey of Transformers](./Transformers_Survey.md)
    * Adaptive Attention Span, 动态的调整attention的窗口的大小
    * Reformer(Locality-Sensitive Hashing)
    * Universal Transformer
    * Star-Transformer, 
    * BPT: BP-Transformer,
2. [长序列输入](https://zhuanlan.zhihu.com/p/259835450)：
    * sin/cos位置嵌入，可学习的位置迁移; 相对位置嵌入：ROPE,Alibi 
    * 2022.12.20 XPOS [A Length-Extrapolatable Transformer](./XPOS.md) 相对位置编码,可在短序列上进行训练，而在较长序列上进行测试。 借鉴mae，预训练时接受512输入，微调时，可大量输入？  
    * [Simple local attentions remain competitive for long-context](https://arxiv.org/abs/2112.07210)
    * 2020.7 [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
    * 2020 [Longformer: The Long-Document Transformer](#)  局部(滑动窗口式注意力、空洞滑动窗口)注意力和全局注意力结合的机制 稀疏注意力
    * 2020 [Local Self-Attention over Long Text for Efficient Document Retrieval](#) q-doc查询,
    * 2020 [Recurrent Chunking Mechanisms for Long-Text Machine Reading Comprehension](#),q-doc查询：引入强化学习，模型决定想要处理的下一个部分.
    * 2019.1.9 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](./Transformer-XL.md), 各个小的分段之间建立了联系
    * 2019 [Extractive Summarization of Long Documents by Combining Global and Local Context](#) 摘要生成：
    * 2018 [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](#) 摘要生成：
    * 2018.8.9 [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/abs/1808.04444)
3. 性能问题
    * [线性注意力:使用基于内核或低秩的近似来代替普通注意力; 稀疏注意力:利用结构化稀疏性来减少计算量; 循环式设计;](https://blog.csdn.net/weixin_48167662/article/details/125739453)
    * 2021.9 [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/abs/2109.08668) http://pelhans.com/2020/07/09/various_attention/
    * 2020.9 [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)
    * [Transformer 问题及改进](https://github.com/soledad921/NLP-Interview-Notes/blob/main/DeepLearningAlgorithm/transformer/transformer_error.md)
    * Linformer, 一定的条件下能将attention的计算复杂到降低到线性
    * 最大输入长度、token维度、head数量、层数，缩放规律？  
4. 稀疏性专家模型 
    * 2022.9.4 [A Review of Sparse Expert Models in Deep Learning](./Sparse_review.md) https://zhuanlan.zhihu.com/p/571873617 https://zhuanlan.zhihu.com/p/463352552
    * 2021.12.13 [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)
    * 2021.1.11 [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
    * 2017.1.23 MoE [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
5. 其他 
    * 2022.1.24 [Text and Code Embeddings by Contrastive Pre-Training](./Embeddings_Contrastive.md) #对比学习预训练？


## 四. 掩码语言模型
1. BERT 2018-10 [BERT Bidirectional Encoder Representations from Transformer ](./bert.md) , 变种
* SpanBERT 2019-7-24 [Improving Pre-training by Representing and Predicting Spans](./SpanBERT.md)
* RoBERTa 2019-7-26 [A Robustly Optimized BERT Pretraining Approach](./RoBERTa.md)
* StructBERT 2019-8 [Incorporating Language Structures into Pre-training for Deep Language Understanding](./StructBERT.md)
* ALBERT 2019-9-26
* ELECTRA 2020-3-11 [Pre-training Text Encoders as Discriminators Rather Than Generators](./ELECTRA.md)
* 中文
    * MacBERT, 2019-6 [Pre-Training with Whole Word Masking for Chinese BERT](./BERT-wwm.md)
    * MacBERT_v2 2020-4 [Revisiting Pre-Trained Models for Chinese Natural Language Processing](./MacBERT.md)
    * RoBERTa-wwm, 2021-3 [RoBERTa-wwm-ext Fine-Tuning for Chinese Text Classification](./RoBERTa-wwm.md)
2. Ernie 加入知识图谱先验知识
    * v1 2019-05-17 [Enhanced Language Representation with Informative Entities](./ernie.md)
    * v2, 2019-7-29 [A Continual Pre-training Framework for Language Understanding](./ernie_v2.md)
    * v3, 2021-12-23 [Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](./ernie_v3.md) 加入了生成机制
3. 轻量化：
    * TinyBERT 2019-9-23, [Distilling BERT for Natural Language Understanding](./TinyBERT.md)
    * MobileBERT 2020-4-6 [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](./MobileBERT.md)

## 五. 自回归(生成式)语言模型
1. GPT_v1 2018-6-11 [Improving Language Understanding by Generative Pre-Training](./gpt.md)
2. GPT_v2 2019-2-14 [Language Models are Unsupervised Multitask Learners](./gpt_2.md) 
    * [Release Strategies and the Social Impacts of Language Models](https://arxiv.org/abs/1908.09203) 社会影响探讨
    * 2018.6.20 [The Natural Language Decathlon: Multitask Learning as Question Answering](https://arxiv.org/abs/1806.08730)
3. GPT_v3 2020-5-28 [Language Models are Few-Shot Learners](./gpt_3.md)
4. InstructGPT 2022.3.4 [Training language models to follow instructions with human feedback](./InstructGPT.md) 
5. 其他
* XLNet 2019-6-19 [Generalized Autoregressive Pretraining for Language Understanding](./XLNet.md)
* 大模型最新语料实时更新问题; 
* 人工智能生成的语料在互联网比重会越来越多，模型迭代时如何对待这些生成的语料？




