# ERNIE: Enhanced Language Representation with Informative Entities
ERNIE：信息实体增强的语言表示 2019-05-17 原文：https://arxiv.org/abs/1905.07129

## 阅读笔记
* 

## Abstract
Neural language representation models such as BERT pre-trained on large-scale corpora can well capture rich semantic patterns from plain text, and be fine-tuned to consistently improve the performance of various NLP tasks. However, the existing pre-trained language models rarely consider incorporating knowledge graphs (KGs), which can provide rich structured knowledge facts for better language understanding. We argue that informative entities in KGs can enhance language representation with external knowledge. In this paper, we utilize both large-scale textual corpora and KGs to train an enhanced language representation model (ERNIE), which can take full advantage of lexical, syntactic, and knowledge information simultaneously. The experimental results have demonstrated that ERNIE achieves significant improvements on various knowledge-driven tasks, and meanwhile is comparable with the state-of-the-art model BERT on other common NLP tasks. The source code and experiment details of this paper can be obtained from https://github.com/thunlp/ERNIE . 

神经语言表示模型(如在大规模语料库上预训练的BERT)可以很好地从纯文本中捕获丰富的语义模式，进行微调后可以一致地提高各种NLP任务的性能。然而，现有的预训练语言模型很少考虑合并知识图谱(KG)，KG可以提供丰富的结构化知识事实，以更好地理解语言。我们认为KGs中的信息实体可以利用外部知识来增强语言表达。在本文中，我们利用大规模文本语料库和KG来训练增强的语言表示模型(ERNIE)，该模型可以同时充分利用词汇、句法和知识信息。实验结果表明，ERNIE在各种知识驱动任务上取得了显著的改进，同时在其他常见的NLP任务上与最先进的BERT模型相当。本文的源代码和实验细节可从https://github.com/thunlp/ERNIE .

## 1 Introduction
Pre-trained language representation models, including feature-based (Mikolov et al., 2013; Pennington et al., 2014; Peters et al., 2017, 2018) and fine-tuning (Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018; Devlin et al., 2019) approaches, can capture rich language information from text and then benefit many NLP applications. BERT (Devlin et al., 2019), as one of the most recently proposed models, obtains the stateof-the-art results on various NLP applications by simple fine-tuning, including named entity recognition (Sang and De Meulder, 2003), question answering (Rajpurkar et al., 2016; Zellers et al., 2018), natural language inference (Bowman et al., 2015), and text classification (Wang et al., 2018).

预训练的语言表示模型，包括基于特征的(Mikolovet al., 2013; Penningtonet al., 2014; Peterset al., 2017，2018)和微调(Dai和Le，2015; Howard和Ruder，2018; Radfordet al., 2018; Devlinet al., 2019)方法，可以从文本中捕获丰富的语言信息，然后使许多NLP应用受益。BERT(Devlinet al., 2019)作为最新提出的模型之一，通过简单的微调，获得了各种NLP应用的最新结果，包括命名实体识别(Sang和De Meulder，2003)、问答(Rajpurkaret al., 2016; Zellerset al., 2018)、自然语言推理(Bowmanet al., 2015)和文本分类(Wanget al., 2018年)。

Figure 1: An example of incorporating extra knowledge information for language understanding. The solid lines present the existing knowledge facts. The red dotted lines present the facts extracted from the sentence in red. The green dotdash lines present the facts extracted from the sentence in green. 
图1：结合额外知识信息进行语言理解的样本。实线表示现有的知识事实。红色虚线用红色表示从句子中提取的事实，绿色点划线用绿色表示从句子提取的事实。

Although pre-trained language representation models have achieved promising results and worked as a routine component in many NLP tasks, they neglect to incorporate knowledge information for language understanding. As shown in Figure 1, without knowing Blowin’ in the Wind and Chronicles: Volume One are song and book respectively, it is difficult to recognize the two occupations of Bob Dylan, i.e., songwriter and writer, on the entity typing task. Furthermore, it is nearly impossible to extract the fine-grained relations, such as composer and author on the relation classification task. For the existing pre-trained language representation models, these two sentences are syntactically ambiguous, like “UNK wrote UNK in UNK”. Hence, considering rich knowledge information can lead to better language understanding and accordingly benefits various knowledge-driven applications, e.g. entity typing and relation classification.

尽管预训练的语言表示模型已经取得了很好的效果，并且在许多NLP任务中作为常规组件工作，但它们忽略了将知识信息用于语言理解。如图1所示，如果不知道Blowin‘in the Wind和Chronicles:Volume One分别是歌曲和书籍，很难识别Bob Dylan在实体键入任务中的两种职业，即歌曲作者和作家。此外，几乎不可能在关系分类任务中提取细粒度关系，例如作曲家和作者。对于现有的预训练的语言表示模型，这两个句子在句法上是模糊的，就像“UNK用UNK写UNK”。因此，考虑丰富的知识信息可以导致更好的语言理解，从而有利于各种知识驱动的应用，例如实体类型和关系分类。

For incorporating external knowledge into language representation models, there are two main author arXiv:1905.07129v3 [cs.CL] 4 Jun 2019 composer challenges. (1) Structured Knowledge Encoding: regarding to the given text, how to effectively extract and encode its related informative facts in KGs for language representation models is an important problem; (2) Heterogeneous Information Fusion: the pre-training procedure for language representation is quite different from the knowledge representation procedure, leading to two individual vector spaces. How to design a special pre-training objective to fuse lexical, syntactic, and knowledge information is another challenge.

为了将外部知识融入语言表示模型，有两个(主要作者arXiv:1905.07129v3[cs.CL]4 Jun 2019作曲家)挑战。(1) 结构化知识编码：对于给定文本，如何有效地提取和编码其在KGs中的相关信息事实，用于语言表示模型是一个重要问题; (2) 异构信息融合：语言表示的预训练过程与知识表示过程非常不同，导致两个单独的向量空间。如何设计一个特殊的训练前目标来融合词汇、句法和知识信息是另一个挑战。

To overcome the challenges mentioned above, we propose Enhanced Language RepresentatioN with Informative Entities (ERNIE), which pretrains a language representation model on both large-scale textual corpora and KGs: 

为了克服上述挑战，我们提出了增强型信息实体语言表示(ERNIE)，它在大规模文本语料库和KGs中预处理语言表示模型：

(1) For extracting and encoding knowledge information, we firstly recognize named entity mentions in text and then align these mentions to their corresponding entities in KGs. Instead of directly using the graph-based facts in KGs, we encode the graph structure of KGs with knowledge embedding algorithms like TransE (Bordes et al., 2013), and then take the informative entity embeddings as input for ERNIE. Based on the alignments between text and KGs, ERNIE integrates entity representations in the knowledge module into the underlying layers of the semantic module. 

(1) 为了提取和编码知识信息，我们首先识别文本中提及的命名实体，然后将这些提及与KGs中的对应实体对齐。我们使用知识嵌入算法(如TransE(Bordeset al., 2013))对KGs的图结构进行编码，而不是直接使用基于图的事实，然后将信息实体嵌入作为ERNIE的输入。基于文本和KGs之间的对齐，ERNIE将知识模块中的实体表示集成到语义模块的底层。

(2) Similar to BERT, we adopt the masked language model and the next sentence prediction as the pre-training objectives. Besides, for the better fusion of textual and knowledge features, we design a new pre-training objective by randomly masking some of the named entity alignments in the input text and asking the model to select appropriate entities from KGs to complete the alignments. Unlike the existing pre-trained language representation models only utilizing local context to predict tokens, our objectives require models to aggregate both context and knowledge facts for predicting both tokens and entities, and lead to a knowledgeable language representation model.

(2) 与BERT相似，我们采用掩码语言模型和下一句预测作为预训练目标。此外，为了更好地融合文本和知识特征，我们设计了一个新的预训练目标，通过随机掩码输入文本中的一些命名实体对齐，并要求模型从KGs中选择合适的实体来完成对齐。与现有的仅利用局部上下文预测标记的预训练语言表示模型不同，我们的目标要求模型聚合上下文和知识事实，以预测标记和实体，并生成知识化的语言表示模型。

We conduct experiments on two knowledgedriven NLP tasks, i.e., entity typing and relation classification. The experimental results show that ERNIE significantly outperforms the state-of-theart model BERT on these knowledge-driven tasks, by taking full advantage of lexical, syntactic, and knowledge information. We also evaluate ERNIE on other common NLP tasks, and ERNIE still achieves comparable results. 

我们在两个知识驱动的NLP任务上进行了实验，即实体类型和关系分类。实验结果表明，ERNIE通过充分利用词汇、句法和知识信息，在这些知识驱动的任务上显著优于现有的BERT模型。我们还对ERNIE的其他常见NLP任务进行了评估，ERNIE仍然取得了可比的结果。

## 2 Related Work
Many efforts are devoted to pre-training language representation models for capturing language information from text and then utilizing the information for specific NLP tasks. These pre-training approaches can be divided into two classes, i.e., feature-based approaches and finetuning approaches.

许多工作致力于预训练语言表示模型，用于从文本中捕获语言信息。这些预训练方法可分为两类，即基于特征的方法和微调方法。

The early work (Collobert and Weston, 2008; Mikolov et al., 2013; Pennington et al., 2014) focuses on adopting feature-based approaches to transform words into distributed representations. As these pre-trained word representations capture syntactic and semantic information in textual corpora, they are often used as input embeddings and initialization parameters for various NLP models, and offer significant improvements over random initialization parameters (Turian et al., 2010). Since these word-level models often suffer from the word polysemy, Peters et al. (2018) further adopt the sequence-level model (ELMo) to capture complex word features across different linguistic contexts and use ELMo to generate context-aware word embeddings.

早期工作(Collobert和Weston，2008; Mikolovet al., 2013; Penningtonet al., 2014)侧重于采用基于特征的方法将单词转换为分布式表示。由于这些预训练的单词表示在文本语料库中捕获语法和语义信息，因此它们通常被用作各种NLP模型的输入嵌入和初始化参数，并且比随机初始化参数有显著的改进(Turianet al., 2010)。由于这些词级模型经常受到单词多义的影响，Peterset al (2018)进一步采用了序列级模型(ELMo)来捕捉不同语言语境中的复杂单词特征，并使用ELMo生成上下文感知的单词嵌入。

Different from the above-mentioned featurebased language approaches only using the pretrained language representations as input features, Dai and Le (2015) train auto-encoders on unlabeled text, and then use the pre-trained model architecture and parameters as a starting point for other specific NLP models. Inspired by Dai and Le (2015), more pre-trained language representation models for fine-tuning have been proposed. Howard and Ruder (2018) present AWDLSTM (Merity et al., 2018) to build a universal language model (ULMFiT). Radford et al. (2018) propose a generative pre-trained Transformer (Vaswani et al., 2017) (GPT) to learn language representations. Devlin et al. (2019) propose a deep bidirectional model with multiplelayer Transformers (BERT), which achieves the state-of-the-art results for various NLP tasks.

与上述仅使用预训练语言表示作为输入特征的基于特征的语言方法不同，Dai和Le(2015)在未标记文本上训练自动编码器，然后使用预训练模型架构和参数作为其他特定NLP模型的起点。受Dai和Le(2015)的启发，人们提出了更多用于微调的预训练语言表示模型。Howard和Ruder(2018)提出了AWDLSTM(Merityet al., 2018)，以构建通用语言模型(ULMFiT)。Radfordet al (2018)提出了一种生成式预训练Transformer(Vaswaniet al., 2017)(GPT)来学习语言表示。Devlinet al (2019年)提出了一种具有多层Transformer(BERT)的深度双向模型，该模型为各种NLP任务实现了最先进的结果。

Though both feature-based and fine-tuning language representation models have achieved great success, they ignore the incorporation of knowledge information. As demonstrated in recent work, injecting extra knowledge information can significantly enhance original models, such as reading comprehension (Mihaylov and Frank, 2018; Zhong et al., 2018), machine translation (Zaremoodi et al., 2018), natural language inference (Chen et al., 2018), knowledge acquisition (Han et al., 2018a), and dialog systems (Madotto et al., 2018). Hence, we argue that extra knowledge information can effectively benefit existing pre-training models. In fact, some work has attempted to joint representation learning of words and entities for effectively leveraging external KGs and achieved promising results (Wang et al., 2014; Toutanova et al., 2015; Han et al., 2016; Yamada et al., 2016; Cao et al., 2017, 2018). Sun et al. (2019) propose the knowledge masking strategy for masked language model to enhance language representation by knowledge (It is a coincidence that both Sun et al. (2019) and we chose ERNIE as the model names, which follows the interesting naming habits like ELMo and BERT. Sun et al. (2019) released their code on March 16th and submitted their paper to Arxiv on April 19th while we submitted our paper to ACL whose deadline is March 4th) . In this paper, we further utilize both corpora and KGs to train an enhanced language representation model based on BERT. 

尽管基于特征和微调的语言表示模型都取得了巨大的成功，但它们忽略了知识信息的整合。如最近的研究所示，注入额外的知识信息可以显著增强原始模型，例如阅读理解(Mihaylov和Frank，2018; Zhonget al., 2018)、机器翻译(Zaremoodiet al., 2018年)、自然语言推理(Chenet al., 2018年)、知识获取(Hanet al., 2018a)和对话系统(Madottoet al., 2017年)。因此，我们认为额外的知识信息可以有效地使现有的预训练模型受益。事实上，一些工作试图联合学习词汇和实体，以有效利用外部KGs，并取得了可喜的成果(Wanget al., 2014; Toutanovaet al., 2015; Hanet al., 2016; Yamadaet al., 2016;Caoet al., 2017年，2018年)。Sunet al (2019)提出了掩码语言模型的知识掩码策略，以增强知识的语言表示(Sunet al (2018)和我们选择ERNIE作为模型名称，这是一个巧合，它遵循了ELMo和BERT等有趣的命名习惯。Sunet al (2019年)于3月16日发布了他们的代码，并于4月19日向Arxiv提交了他们的论文，而我们将论文提交给ACL，ACL的截止日期为3月4日)。在本文中，我们进一步利用语料库和KG来训练基于BERT的增强语言表示模型。

## 3 Methodology
In this section, we present the overall framework of ERNIE and its detailed implementation, including the model architecture in Section 3.2, the novel pre-training task designed for encoding informative entities and fusing heterogeneous information in Section 3.4, and the details of the fine-tuning procedure in Section 3.5.

在本节中，我们将介绍ERNIE的总体框架及其详细实施，包括第3.2节中的模型架构、第3.4节中为编码信息实体和融合异构信息而设计的新的预训练任务，以及第3.5节中的微调程序细节。

### 3.1 Notations 表示法
We denote a token sequence as ${\{w_1, . . . , w_n\}}$ (In this paper, tokens are at the subword level. ), where n is the length of the token sequence. Meanwhile, we denote the entity sequence aligning to the given tokens as ${\{e_1, . . . , e_m\}}$, where m is the length of the entity sequence. Note that m is not equal to n in most cases, as not every token can be aligned to an entity in KGs. Furthermore, we denote the whole vocabulary containing all tokens as V, and the entity list containing all entities in KGs as E. If a token w ∈ V has a corresponding entity e ∈ E, their alignment is defined as f(w) = e. In this paper, we align an entity to the first token in its named entity phrase, as shown in Figure 2.

我们将token序列表示为${\{w_1, . . . , w_n\}}$(在本文中，token处于子字级别。)，其中n是token序列的长度。同时，我们将与给定标记对齐的实体序列表示为${\{e_1, . . . , e_m\}}$，其中m是实体序列的长度。请注意，在大多数情况下，m不等于n，因为并非每个token都可以与KGs中的实体对齐。此外，我们将包含所有token的整个词汇表表示为V，将包含KG中所有实体的实体列表表示为E ∈ V具有对应的实体 e ∈ E、 它们的对齐定义为 f(w) = e。在本文中，我们将一个实体与其命名实体短语中的第一个标记对齐，如图2所示。

Figure 2: The left part is the architecture of ERNIE. The right part is the aggregator for the mutual integration of the input of tokens and entities. Information fusion layer takes two kinds of input: one is the token embedding, and the other one is the concatenation of the token embedding and entity embedding. After information fusion, it outputs new token embeddings and entity embeddings for the next layer. 
图2：左边是ERNIE的架构。右侧部分是用于token和实体输入的相互集成的聚合器。信息融合层接受两种输入：一种是token嵌入，另一种是标记嵌入和实体嵌入的级联。信息融合后，它为下一层输出新的token嵌入和实体嵌入。

### 3.2 Model Architecture
As shown in Figure 2, the whole model architecture of ERNIE consists of two stacked modules: (1) the underlying textual encoder (T-Encoder) responsible to capture basic lexical and syntactic information from the input tokens, and (2) the upper knowledgeable encoder (K-Encoder) responsible to integrate extra token-oriented knowledge information into textual information from the underlying layer, so that we can represent heterogeneous information of tokens and entities into a united feature space. Besides, we denote the number of T-Encoder layers as N, and the number of K-Encoder layers as M.

如图2所示，ERNIE的整个模型架构由两个堆叠模块组成：(1)底层文本编码器(T-encoder)，负责从输入token捕获基本词汇和语法信息，从而我们可以将token和实体的异构信息表示为一个统一的特征空间。此外，我们将T-Encoder层数表示为N，将K-Encoder层数表示为M。

To be specific, given a token sequence ${\{w_1, . . . , w_n\}}$ and its corresponding entity sequence ${\{e_1, . . . , e_m\}}$, the textual encoder firstly sums the token embedding, segment embedding, positional embedding for each token to compute its input embedding, and then computes lexical and syntactic features ${\{w_1, . . . , w_n\}}$ as follows, 

具体来说，给定一个标记序列${\{w_1, . . . , w_n\}}$及其对应的实体序列${\{e_1, . . . , e_m\}}$，文本编码器首先对每个标记的标记嵌入、段嵌入和位置嵌入求和，以计算其输入嵌入，然后如下计算词法和句法特征

${\{w_1, . . . , w_n\}} = T-Encoder({\{w_1, . . . , w_n\}})$, (1) 

where T-Encoder(·) is a multi-layer bidirectional Transformer encoder. As T-Encoder(·) is identical to its implementation in BERT and BERT is prevalent, we exclude a comprehensive description of this module and refer readers to Devlin et al. (2019) and Vaswani et al. (2017).

其中T-Encoder(·)是多层双向Transformer编码器。由于T-Encoder(·)与BERT中的实现相同，BERT普遍存在，我们不再对该模块做全面描述，请读者参阅Devlinet al (2019)和Vaswaniet al (2017)。

After computing ${\{w_1, . . . , w_n\}}$, ERNIE adopts a knowledgeable encoder K-Encoder to inject the knowledge information into language representation. To be specific, we represent ${\{e_1, . . . , e_m\}}$ with their entity embeddings ${\{e_1, . . . , e_m\}}$, which are pre-trained by the effective knowledge embedding model TransE (Bordes et al., 2013). Then, both {w1, . . . , wn} and ${\{e_1, . . . , e_m\}}$ are fed into K-Encoder for fusing heterogeneous information and computing fi- nal output embeddings, 

在计算${\{w_1, . . . , w_n\}}$之后，ERNIE采用知识型编码器K-encoder将知识信息注入到语言表示中。具体来说，我们用它们的实体嵌入${\{e_1, . . . , e_m\}}$来表示${\{e_1, . . . , e_m\}}$，这些嵌入由有效的知识嵌入模型TransE预训练(Bordeset al., 2013)。然后，将${\{w_1, . . . , w_n\}}$和${\{e_1, . . . , e_m\}}$输入K-Encoder，用于融合异构信息并计算最终输出嵌入，

${\{w^o_1, . . . , w^o_n\}}, {\{e^o_1, . . . , e^o_n\}} = K-Encoder( {\{w_1, . . . , w_n\}}, {\{e_1, . . . , e_m\}}).$ (2)

${\{w^o_1, . . . , w^o_n\}}$ and ${\{e^o_1, . . . , e^o_n\}}$ will be used as features for specific tasks. More details of the knowledgeable encoder K-Encoder will be introduced in Section 3.3.

${\{w^o_1, . . . , w^o_n\}}$和｛eo1，……，eon｝将用作特定任务的功能。第3.3节将介绍知识丰富的编码器K编码器的更多细节。

### 3.3 Knowledgeable Encoder 知识型编码器
As shown in Figure 2, the knowledgeable encoder K-Encoder consists of stacked aggregators, which are designed for encoding both tokens and entities as well as fusing their heterogeneous features. In the i-th aggregator, the input token embeddings {w(i−1) 1 , . . . , w(i−1) n } and entity embeddings {e(i−1) 1 , . . . , e(i−1) m } from the preceding aggregator are fed into two multi-head self-attentions (MH-ATTs) (Vaswani et al., 2017) respectively, 

如图2所示，知识丰富的编码器K-encoder由堆叠的聚合器组成，这些聚合器设计用于对token和实体进行编码，并融合其异构特征。在第i个聚合器中，输入token嵌入{w(i−1) 1，…，w(i−1) n}和实体嵌入{e(i)−1) 1，…，e(i−1) m}分别馈送到两个多头自关注(MH-ATT)(Vaswaniet al., 

{ ˜w(i) 1 , . . . , ˜w(i) n } = MH-ATT({w(i−1) 1 , . . . , w(i−1) n }), 

{e˜(i) 1 , . . . , e˜(i) m } = MH-ATT({e(i−1) 1 , . . . , e(i−1) m }). (3)

Then, the i-th aggregator adopts an information fusion layer for the mutual integration of the token and entity sequence, and computes the output embedding for each token and entity. For a token wj and its aligned entity ek = f(wj ), the information fusion process is as follows, 

然后，第i个聚合器采用信息融合层对token和实体序列进行相互融合，并计算每个token和实体的输出嵌入。对于tokenwj及其对齐实体ek＝f(wj)，

hj = σ( ˜W(i) t ˜wj(i) + ˜W(i) e e˜(i) k + ˜b(i)), 

wj(i) = σ(Wt(i)hj + b(ti)), 

e(i) k = σ(We(i)hj + b(i) e ). (4) 

where hj is the inner hidden state integrating the information of both the token and the entity. σ(·) is the non-linear activation function, which usually is the GELU function (Hendrycks and Gimpel, 2016). For the tokens without corresponding entities, the information fusion layer computes the output embeddings without integration as follows, 

其中hj是整合token和实体两者的信息的内部隐藏状态。σ(·)是非线性激活函数，通常是GELU函数(Hendrycks和Gimpel，2016)。对于没有对应实体的token，信息融合层计算没有集成的输出嵌入，如下所示：，

hj = σ( ˜W(i) t ˜wj(i) + ˜b(i)), 

wj(i) = σ(Wt(i)hj + b(ti)). (5)

For simplicity, the i-th aggregator operation is denoted as follows, 

为了简单起见，第i个聚合器操作表示如下：，

{w(i) 1 , . . . , w(i) n }, {e(i) 1 , . . . , e(i) m } = Aggregator( {w(i−1) 1 , . . . , w(i−1) n }, {e(i−1) 1 , . . . , e(i−1) m }). (6)

The output embeddings of both tokens and entities computed by the top aggregator will be used as the final output embeddings of the knowledgeable encoder K-Encoder.

顶部聚合器计算的token和实体的输出嵌入将用作知识丰富的编码器K-encoder的最终输出嵌入。

### 3.4 Pre-training for Injecting Knowledge
In order to inject knowledge into language representation by informative entities, we propose a new pre-training task for ERNIE, which randomly masks some token-entity alignments and then requires the system to predict all corresponding entities based on aligned tokens. As our task is similar to training a denoising auto-encoder (Vincent et al., 2008), we refer to this procedure as a denoising entity auto-encoder (dEA). Considering that the size of E is quite large for the softmax layer, we thus only require the system to predict entities based on the given entity sequence instead of all entities in KGs. Given the token sequence {w1, . . . , wn} and its corresponding entity sequence ${\{e_1, . . . , e_m\}}$, we define the aligned entity distribution for the token wi as follows, 

为了将知识注入信息实体的语言表示中，我们为ERNIE提出了一个新的预训练任务，该任务随机掩码一些标记-实体对齐，然后要求系统基于对齐的标记预测所有对应的实体。由于我们的任务类似于训练去噪自动编码器(Vincent et al.，2008)，我们将此过程称为去噪实体自动编码器(dEA)。考虑到E的大小对于softmax层来说相当大，因此我们只需要系统根据给定的实体序列而不是KG中的所有实体来预测实体。给定token序列｛w1，…，wn｝及其对应的实体序列｛e1，……，em｝，我们定义tokenwi的对齐实体分布如下：，

p(ej |wi) = exp(linear(wio) · ej ) P mk=1 exp(linear(wio) · ek), (7)

where linear(·) is a linear layer. Eq. 7 will be used to compute the cross-entropy loss function for dEA.

其中线性(·)是线性层。等式7将用于计算dEA的交叉熵损失函数。

Figure 3: Modifying the input sequence for the specific tasks. To align tokens among different types of input, we use dotted rectangles as placeholder. The colorful rectangles present the specific mark tokens. 

图3：修改特定任务的输入序列。为了在不同类型的输入之间对齐标记，我们使用虚线矩形作为占位符。彩色矩形表示特定的标记标记。

Considering that there are some errors in tokenentity alignments, we perform the following operations for dEA: (1) In 5% of the time, for a given token-entity alignment, we replace the entity with another random entity, which aims to train our model to correct the errors that the token is aligned with a wrong entity; (2) In 15% of the time, we mask token-entity alignments, which aims to train our model to correct the errors that the entity alignment system does not extract all existing alignments; (3) In the rest of the time, we keep tokenentity alignments unchanged, which aims to encourage our model to integrate the entity information into token representations for better language understanding.

考虑到token实体对齐中存在一些错误，我们对dEA执行以下操作：(1)在5%的时间内，对于给定的token实体对齐，我们将该实体替换为另一个随机实体，其目的是训练我们的模型以纠正token与错误实体对齐的错误; (2) 在15%的时间内，我们掩码了token-实体对齐，其目的是训练我们的模型以纠正实体对齐系统无法提取所有现有对齐的错误; (3) 在剩下的时间里，我们保持标记对齐不变，这旨在鼓励我们的模型将实体信息集成到标记表示中，以更好地理解语言。

Similar to BERT, ERNIE also adopts the masked language model (MLM) and the next sentence prediction (NSP) as pre-training tasks to enable ERNIE to capture lexical and syntactic information from tokens in text. More details of these pre-training tasks can be found from Devlin et al. (2019). The overall pre-training loss is the sum of the dEA, MLM and NSP loss.

与BERT类似，ERNIE还采用掩码语言模型(MLM)和下一句预测(NSP)作为预训练任务，使ERNIE能够从文本中的标记中捕获词汇和句法信息。有关这些预训练任务的更多详情，请参见Devlinet al (2019年)。总的训练前损失是dEA、MLM和NSP损失的总和。

### 3.5 Fine-tuning for Specific Tasks
As shown in Figure 3, for various common NLP tasks, ERNIE can adopt the fine-tuning procedure similar to BERT. We can take the final output embedding of the first token, which corresponds to the special [CLS] token, as the representation of the input sequence for specific tasks. For some knowledge-driven tasks (e.g., relation classification and entity typing), we design special finetuning procedure:

如图3所示，对于各种常见的NLP任务，ERNIE可以采用类似于BERT的微调程序。我们可以将对应于特殊[CLS]token的第一个token的最终输出嵌入作为特定任务的输入序列的表示。对于一些知识驱动的任务(例如，关系分类和实体类型)，我们设计了特殊的微调程序：

For relation classification, the task requires systems to classify relation labels of given entity pairs based on context. The most straightforward way to fine-tune ERNIE for relation classification is to apply the pooling layer to the final output embeddings of the given entity mentions, and represent the given entity pair with the concatenation of their mention embeddings for classification. In this paper, we design another method, which modifies the input token sequence by adding two mark tokens to highlight entity mentions. These extra mark tokens play a similar role like position embeddings in the conventional relation classification models (Zeng et al., 2015). Then, we also take the [CLS] token embedding for classification. Note that we design different tokens [HD] and [TL] for head entities and tail entities respectively.

对于关系分类，该任务要求系统基于上下文对给定实体对的关系标签进行分类。为关系分类微调ERNIE的最直接方法是将池层应用于给定实体引用的最终输出嵌入，并用它们的引用嵌入的级联表示给定实体对以进行分类。在本文中，我们设计了另一种方法，该方法通过添加两个标记标记来突出实体提及来修改输入标记序列。这些额外标记标记在传统关系分类模型中起着类似于位置嵌入的作用(Zenget al., 2015)。然后，我们还采用[CLS]标记嵌入进行分类。请注意，我们分别为头部实体和尾部实体设计了不同的标记[HD]和[TL]。

The specific fine-tuning procedure for entity typing is a simplified version of relation classification. As previous typing models make full use of both context embeddings and entity mention embeddings (Shimaoka et al., 2016; Yaghoobzadeh and Sch¨utze, 2017; Xin et al., 2018), we argue that the modified input sequence with the mention mark token [ENT] can guide ERNIE to combine both context information and entity mention information attentively. 

实体类型的特定微调过程是关系分类的简化版本。由于先前的类型模型充分利用了上下文嵌入和实体提及嵌入(Shimaokaet al., 2016; Yaghoobzadeh和Sch¨utze，2017; Xinet al., 2018)，我们认为，带有提及标记标记的修改输入序列[ENT]可以引导ERNIE专注地结合上下文信息和实体提及信息。

## 4 Experiments
In this section, we present the details of pretraining ERNIE and the fine-tuning results on five NLP datasets, which contain both knowledgedriven tasks and the common NLP tasks.

在本节中，我们将详细介绍ERNIE的预培训以及五个NLP数据集的微调结果，这些数据集包含知识驱动任务和常见NLP任务。

### 4.1 Pre-training Dataset
The pre-training procedure primarily acts in accordance with the existing literature on pre-training language models. For the large cost of training ERNIE from scratch, we adopt the parameters of BERT released by Google3 to initialize the Transformer blocks for encoding tokens. Since pre-training is a multi-task procedure consisting of NSP, MLM, and dEA, we use English Wikipedia as our pre-training corpus and align text to Wikidata. After converting the corpus into the formatted data for pre-training, the annotated input has nearly 4, 500M subwords and 140M entities, and discards the sentences having less than 3 entities.

预训练程序主要根据现有的关于预训练语言模型的文献进行。对于从头开始训练ERNIE的巨大成本，我们采用Google3发布的BERT参数来初始化Transformer块以编码token。由于预训练是一个由NSP、MLM和dEA组成的多任务程序，我们使用英语维基百科作为预训练语料库，并将文本与维基百科数据对齐。在将语料库转换为用于预训练的格式化数据之后，带注释的输入有近4500M个子单词和140M个实体，并丢弃少于3个实体的句子。

Before pre-training ERNIE, we adopt the knowledge embeddings trained on Wikidata4 by TransE as the input embeddings for entities. To be specific, we sample part of Wikidata which contains 5, 040, 986 entities and 24, 267, 796 fact triples. The entity embeddings are fixed during training and the parameters of the entity encoding modules are all initialized randomly.

在预训练ERNIE之前，我们采用TransE在Wikidata4上训练的知识嵌入作为实体的输入嵌入。具体来说，我们对Wikidata的一部分进行了采样，其中包含5040986个实体和24267796个事实三元组。实体嵌入在训练期间是固定的，并且实体编码模块的参数都是随机初始化的。

### 4.2 Parameter Settings and Training Details
In this work, we denote the hidden dimension of token embeddings and entity embeddings as Hw, He respectively, and the number of self-attention heads as Aw, Ae respectively. In detail, we have the following model size: N = 6, M = 6, Hw = 768, He = 100, Aw = 12, Ae = 4. The total parameters are about 114M.

在这项工作中，我们将标记嵌入和实体嵌入的隐藏维度分别表示为Hw和He，将自注意头部的数量分别表示为Aw和Ae。具体来说，我们有以下模型尺寸：N=6，M=6，Hw=768，He=100，Aw=12，Ae=4。总参数约为114M。

The total amount of parameters of BERTBASE is about 110M, which means the knowledgeable module of ERNIE is much smaller than the language module and has little impact on the run-time performance. And, we only pre-train ERNIE on the annotated corpus for one epoch. To accelerate the training process, we reduce the max sequence length from 512 to 256 as the computation of selfattention is a quadratic function of the length. To keep the number of tokens in a batch as same as BERT, we double the batch size to 512. Except for setting the learning rate as 5e−5 , we largely follow the pre-training hyper-parameters used in BERT. For fine-tuning, most hyper-parameters are the same as pre-training, except batch size, learning rate, and number of training epochs. We find the following ranges of possible values work well on the training datasets with gold annotations, i.e., batch size: 32, learning rate (Adam): 5e−5, 3e−5, 2e−5 , number of epochs ranging from 3 to 10.

BERTBASE的参数总量约为110M，这意味着ERNIE的知识模块比语言模块小得多，对运行时性能影响不大。而且，我们只对ERNIE进行一个时期的注释语料库预训练。为了加快训练过程，我们将最大序列长度从512减少到256，因为自注意的计算是长度的二次函数。为了使批中的token数量与BERT相同，我们将批大小加倍为512−5，我们主要遵循BERT中使用的预训练超参数。对于微调，除了批量大小、学习速率和训练周期数之外，大多数超参数与预训练相同。我们发现以下范围的可能值在具有黄金注释的训练数据集上很好地工作，即批次大小：32，学习率(Adam)：5e−5、3e−5、2e−5，纪元数从3到10。

We also evaluate ERNIE on the distantly supervised dataset, i.e., FIGER (Ling et al., 2015). As the powerful expression ability of deeply stacked Transformer blocks, we found small batch size would lead the model to overfit the training data. Hence, we use a larger batch size and less training epochs to avoid overfitting, and keep the range of learning rate unchanged, i.e., batch size: 2048, number of epochs: 2, 3.

我们还根据远程监控数据集(即FIGER)评估ERNIE(Linget al., 2015年)。由于深度堆叠的Transformer块的强大表达能力，我们发现小批量会导致模型超出训练数据。因此，我们使用较大的批大小和较少的训练时长来避免过度拟合，并保持学习速率的范围不变，即批大小：2048，时长数：2，3。

Table 2: Results of various models on FIGER (%). 
表2:FIGER上各种模型的结果(%)。

As most datasets do not have entity annotations, we use TAGME (Ferragina and Scaiella, 2010) to extract the entity mentions in the sentences and link them to their corresponding entities in KGs.

由于大多数数据集没有实体注释，我们使用TAGME(Ferragina和Scaiella，2010)提取句子中提到的实体，并将它们与KGs中相应的实体联系起来。

### 4.3 Entity Typing
Given an entity mention and its context, entity typing requires systems to label the entity mention with its respective semantic types. To evaluate performance on this task, we fine-tune ERNIE on two well-established datasets FIGER (Ling et al., 2015) and Open Entity (Choi et al., 2018). The training set of FIGER is labeled with distant supervision, and its test set is annotated by human. Open Entity is a completely manually-annotated dataset. The statistics of these two datasets are shown in Table 1. We compare our model with the following baseline models for entity typing:

给定实体提及及其上下文，实体类型化要求系统用其各自的语义类型标记实体提及。为了评估这项任务的性能，我们在两个成熟的数据集FIGER(Linget al., 2015)和Open Entity(Choiet al., 2018)上微调ERNIE。FIGER的训练集标记为远程监控，其测试集由人类注释。OpenEntity是一个完全手动注释的数据集。这两个数据集的统计数据如表1所示。我们将我们的模型与以下实体类型基线模型进行了比较：

NFGEC. NFGEC is a hybrid model proposed by Shimaoka et al. (2016). NFGEC combines the representations of entity mention, context and extra hand-craft features as input, and is the stateof-the-art model on FIGER. As this paper focuses on comparing the general language representation abilities of various neural models, we thus do not use the hand-craft features in this work.

NFGEC公司。NFGEC是Shimaokaet al (2016)提出的混合模型。NFGEC结合了实体提及、上下文和额外手工艺特征的表示作为输入，是FIGER上最先进的模型。由于本文着重于比较各种神经模型的一般语言表示能力，因此我们在本文中不使用手工艺特征。

UFET. For Open Entity, we add a new hybrid model UFET (Choi et al., 2018) for comparison. UFET is proposed with the Open Entity dataset, which uses a Bi-LSTM for context representation instead of two Bi-LSTMs separated by entity mentions in NFGEC.

UFET。对于开放实体，我们添加了一个新的混合模型UFET(Choiet al., 2018)进行比较。UFET是使用开放实体数据集提出的，该数据集使用一个Bi-LSTM作为上下文表示，而不是NFGEC中由实体提及分开的两个Bi-LST。

Besides NFGEC and UFET, we also report the result of fine-tuning BERT with the same input format introduced in Section 3.5 for fair comparison. Following the same evaluation criteria used in the previous work, we compare NFGEC, BERT, ERNIE on FIGER, and adopt strict accuracy, loose macro, loose micro scores for evaluation. We compare NFGEC, BERT, UFET, ERNIE on Open Entity, and adopt precision, recall, microF1 scores for evaluation.

除了NFGEC和UFET之外，我们还报告了使用第3.5节中介绍的相同输入格式微调BERT的结果，以便进行公平比较。按照之前工作中使用的相同评估标准，我们在FIGER上比较NFGEC、BERT、ERNIE，并采用严格的准确性、宽松的宏观评分和宽松的微观评分进行评估。我们在开放实体上比较NFGEC、BERT、UFET、ERNIE，并采用精确度、召回率、microF1分数进行评估。

Table 3: Results of various models on Open Entity (%).
表3：开放实体的各种模型的结果(%)。

Table 4: The statistics of the relation classification datasets FewRel and TACRED. 
表4：关系分类数据集FewRel和TACRED的统计数据。

The results on FIGER are shown in Table 2. From the results, we observe that: (1) BERT achieves comparable results with NFGEC on the macro and micro metrics. However, BERT has lower accuracy than the best NFGEC model. As strict accuracy is the ratio of instances whose predictions are identical to human annotations, it illustrates some wrong labels from distant supervision are learned by BERT due to its powerful fitting ability. (2) Compared with BERT, ERNIE significantly improves the strict accuracy, indicating the external knowledge regularizes ERNIE to avoid fitting the noisy labels and accordingly benefits entity typing.

FIGER的结果如表2所示。从结果中，我们观察到：(1)BERT在宏观和微观指标上与NFGEC取得了可比的结果。然而，BERT的精确度低于最佳NFGEC模型。由于严格的准确度是其预测与人类注释相同的实例的比率，它说明了BERT由于其强大的拟合能力而从远程监控中学习到的一些错误标签。(2) 与BERT相比，ERNIE显著提高了严格的准确性，这表明外部知识规范了ERNIE以避免拟合噪声标签，从而有利于实体键入。

The results on Open Entity are shown in Table 3. From the table, we observe that: (1) BERT and ERNIE achieve much higher recall scores than the previous entity typing models, which means pre-training language models make full use of both the unsupervised pre-training and manuallyannotated training data for better entity typing. (2) Compared to BERT, ERNIE improves the precision by 2% and the recall by 2%, which means the informative entities help ERNIE predict the labels more precisely.

关于开放实体的结果如表3所示。从表中，我们观察到：(1)BERT和ERNIE的召回分数比以前的实体类型模型高得多，这意味着预训练语言模型充分利用了无监督的预训练和人工注释的训练数据，以更好地进行实体类型。(2) 与BERT相比，ERNIE提高了2%的准确率和2%的召回率，这意味着信息实体帮助ERNIE更准确地预测标签。

In summary, ERNIE effectively reduces the noisy label challenge in FIGER, which is a widely-used distantly supervised entity typing dataset, by injecting the information from KGs. Besides, ERNIE also outperforms the baselines on Open Entity which has gold annotations.

总之，ERNIE通过从KGs注入信息，有效地减少了FIGER中的噪音标签挑战，FIGER是一个广泛使用的远程监控实体打字数据集。此外，ERNIE还优于Open Entity的基线，后者有黄金注释。

Table 5: Results of various models on FewRel and TACRED (%).
表5：关于FewRel和TACRED的各种模型的结果(%)。

### 4.4 Relation Classification
Relation classification aims to determine the correct relation between two entities in a given sentence, which is an important knowledge-driven NLP task. To evaluate performance on this task, we fine-tune ERNIE on two well-established datasets FewRel (Han et al., 2018c) and TACRED (Zhang et al., 2017). The statistics of these two datasets are shown in Table 4. As the original experimental setting of FewRel is few-shot learning, we rearrange the FewRel dataset for the common relation classification setting. Specifi- cally, we sample 100 instances from each class for the training set, and sample 200 instances for the development and test respectively. There are 80 classes in FewRel, and there are 42 classes (including a special relation “no relation”) in TACRED. We compare our model with the following baseline models for relation classification:

关系分类的目的是确定给定句子中两个实体之间的正确关系，这是一个重要的知识驱动NLP任务。为了评估这项任务的性能，我们在两个成熟的数据集FewRel(Hanet al., 2018c)和TACRED(Zhanget al., 2017)上微调ERNIE。这两个数据集的统计数据如表4所示。由于FewRel的原始实验设置是少样本学习，因此我们重新排列了FewRel数据集以用于公共关系分类设置。具体来说，我们为训练集从每个类中抽取100个实例，为开发和测试分别抽取200个实例。FewRel中有80个类，TACRED中有42个类(包括一个特殊关系“无关系”)。我们将我们的模型与以下关系分类基线模型进行比较：

CNN. With a convolution layer, a max-pooling layer, and a non-linear activation layer, CNN gets the output sentence embedding, and then feeds it into a relation classifier. To better capture the position of head and tail entities, position embeddings are introduced into CNN (Zeng et al., 2015; Lin et al., 2016; Wu et al., 2017; Han et al., 2018b).

CNN。通过卷积层、最大池层和非线性激活层，CNN获得输出句子嵌入，然后将其输入到关系分类器中。为了更好地捕捉头部和尾部实体的位置，CNN引入了位置嵌入(Zenget al., 2015; Linet al., 2016; Wuet al., 2017; Hanet al., 2018b)。

PA-LSTM. Zhang et al. (2017) propose PALSTM introducing a position-aware attention mechanism over an LSTM network, which evaluates the relative contribution of each word in the sequence for the final sentence representation.

PA-LSTM。Zhanget al (2017)建议PALSTM在LSTM网络上引入位置感知注意机制，该机制评估每个单词在序列中对最终句子表示的相对贡献。

C-GCN. Zhang et al. (2018) adopt the graph convolution operations to model dependency trees for relation classification. To encode the word order and reduce the side effect of errors in dependency parsing, Contextualized GCN (C-GCN) firstly uses Bi-LSTM to generate contextualized representations as input for GCN models.

C-GCN。Zhanget al (2018)采用图卷积运算对依赖树进行建模，以进行关系分类。为了对词序进行编码并减少依赖分析中错误的副作用，上下文化GCN(C-GCN)首先使用Bi-LSTM生成上下文化表示作为GCN模型的输入。

In addition to these three baselines, we also finetune BERT with the same input format introduced in Section 3.5 for fair comparison.

除了这三个基线之外，我们还使用第3.5节中介绍的相同输入格式对BERT进行微调，以便进行公平比较。

Table 6: Results of BERT and ERNIE on different tasks of GLUE (%).
表6：BERT和ERNIE对GLUE不同任务的结果(%)。

As FewRel does not have any null instance where there is not any relation between entities, we adopt macro averaged metrics to present the model performances. Since FewRel is built by checking whether the sentences contain facts in Wikidata, we drop the related facts in KGs before pre-training for fair comparison. From Table 5, we have two observations: (1) As the training data does not have enough instances to train the CNN encoder from scratch, CNN just achieves an F1 score of 69.35%. However, the pre-training models including BERT and ERNIE increase the F1 score by at least 15%. (2) ERNIE achieves an absolute F1 increase of 3.4% over BERT, which means fusing external knowledge is very effective.

由于FewRel没有实体之间没有任何关系的空实例，因此我们采用宏平均度量来表示模型性能。由于FewRel是通过检查维基数据中的句子是否包含事实而建立的，因此我们在预训练将相关事实放在KGs中，以便进行公平比较。从表5中，我们有两个观察：(1)由于训练数据没有足够的实例从头开始训练CNN编码器，CNN只获得了69.35%的F1分数。然而，包括BERT和ERNIE在内的训练前模型将F1得分提高了至少15%。(2) ERNIE的F1绝对值比BERT增加了3.4%，这意味着融合外部知识非常有效。

In TACRED, there are nearly 80% null instances so that we follow the previous work (Zhang et al., 2017) to adopt micro averaged metrics to represent the model performances instead of the macro. The results of CNN, PA-LSTM, and C-GCN come from the paper by Zhang et al. (2018), which are the best results of CNN, RNN, and GCN respectively. From Table 5, we observe that: (1) The C-GCN model outperforms the strong BERT model by an F1 increase of 0.4%, as C-GCN utilizes the dependency trees and the entity mask strategy. The entity mask strategy refers to replacing each subject (and object similarly) entity with a special NER token, which is similar to our proposed pre-training task dEA. (2) ERNIE achieves the best recall and F1 scores, and increases the F1 of BERT by nearly 2.0%, which proves the effectiveness of the knowledgeable module for relation classification.

在TACRED中，有近80%的空实例，因此我们遵循之前的工作(Zhanget al., 2017)，采用微平均度量来表示模型性能，而不是宏观。CNN、PA-LSTM和C-GCN的结果来自张et al (2018)的论文，分别是CNN、RNN和GCN的最佳结果。从表5中，我们观察到：(1)C-GCN模型优于强BERT模型，F1增加0.4%，因为C-GCN使用依赖树和实体掩码策略。实体掩码策略指的是用一个特殊的NERtoken替换每个主题(和类似的对象)实体，这类似于我们建议的预训练任务dEA。(2) ERNIE获得了最佳的召回率和F1分数，并将BERT的F1提高了近2.0%，这证明了知识模块对关系分类的有效性。

In conclusion, we find that the pre-trained language models can provide more information for relation classification than the vanilla encoder CNN and RNN. And ERNIE outperforms BERT on both of the relation classification datasets, especially on the FewRel which has a much smaller training set. It demonstrates extra knowledge helps the model make full use of small training data, which is important for most NLP tasks as large-scale annotated data is unavailable.

总之，我们发现预训练的语言模型比普通编码器CNN和RNN能够为关系分类提供更多的信息。ERNIE在两个关系分类数据集上都优于BERT，特别是在训练集小得多的FewRel上。它证明了额外的知识有助于模型充分利用小的训练数据，这对于大多数NLP任务非常重要，因为大规模的注释数据是不可用的。

Table 7: Ablation study on FewRel (%). 
表7:FewRel的消融研究(%)。

### 4.5 GLUE
The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018) is a collection of diverse natural language understanding tasks (Warstadt et al., 2018; Socher et al., 2013; Dolan and Brockett, 2005; Agirre et al., 2007; Williams et al., 2018; Rajpurkar et al., 2016; Dagan et al., 2006; Levesque et al., 2011), which is the main benchmark used in Devlin et al. (2019). To explore whether our knowledgeable module degenerates the performance on common NLP tasks, we evaluate ERNIE on 8 datasets of GLUE and compare it with BERT.

通用语言理解评估(GLUE)基准(Wanget al., 2018)是多种自然语言理解任务的集合(Warstadtet al., 2018; Socheret al., 2013; Dolan和Brockett，2005; Agirreet al., 2007; Williamset al., 2018;Rajpurkaret al., 2016; Daganet al., 2006; Levesqueet al., 2011)，这是Devlinet al (2019)使用的主要基准。为了探索我们的知识模块是否会降低常见NLP任务的性能，我们在8个GLUE数据集上评估ERNIE，并将其与BERT进行比较。

In Table 6, we report the results of our evaluation submissions and those of BERT from the leaderboard. We notice that ERNIE is consistent with BERTBASE on big datasets like MNLI, QQP, QNLI, and SST-2. The results become more unstable on small datasets, that is, ERNIE is better on CoLA and RTE, but worse on STS-B and MRPC.

在表6中，我们报告了我们提交的评估结果以及排行榜上的BERT结果。我们注意到ERNIE在MNLI、QQP、QNLI和SST-2等大数据集上与BERTBASE一致。结果在小数据集上变得更加不稳定，即,ERNIE对CoLA和RTE更好，但对STS-B和MRPC更差。

In short, ERNIE achieves comparable results with BERTBASE on GLUE. On the one hand, it means GLUE does not require external knowledge for language representation. On the other hand, it illustrates ERNIE does not lose the textual information after heterogeneous information fusion.

简而言之，ERNIE在胶水方面取得了与BERTBASE相当的结果。一方面，这意味着GLUE不需要外部知识来进行语言表示。另一方面，它说明ERNIE在异构信息融合后不会丢失文本信息。

### 4.6 Ablation Study
In this subsection, we explore the effects of the informative entities and the knowledgeable pretraining task (dEA) for ERNIE using FewRel dataset. w/o entities and w/o dEA refer to finetuning ERNIE without entity sequence input and the pre-training task dEA respectively. As shown in Table 7, we have the following observations: (1) Without entity sequence input, dEA still injects knowledge information into language representation during pre-training, which increases the F1 score of BERT by 0.9%. (2) Although the informative entities bring much knowledge informa- tion which intuitively benefits relation classification, ERNIE without dEA takes little advantage of this, leading to the F1 increase of 0.7%. 

在本小节中，我们使用FewRel数据集探讨信息实体和知识预训练任务(dEA)对ERNIE的影响。w/o实体和w/o dEA分别指在没有实体序列输入的情况下微调ERNIE和预训练任务dEA。如表7所示，我们有以下观察：(1)在没有实体序列输入的情况下，dEA仍然在预训练期间将知识信息注入语言表示中，这将BERT的F1分数提高0.9%。(2) 尽管信息实体带来了大量的知识信息，直观地有利于关系分类，但没有dEA的ERNIE几乎没有利用这一点，导致F1增长了0.7%。

## 5 Conclusion
In this paper, we propose ERNIE to incorporate knowledge information into language representation models. Accordingly, we propose the knowledgeable aggregator and the pre-training task dEA for better fusion of heterogeneous information from both text and KGs. The experimental results demonstrate that ERNIE has better abilities of both denoising distantly supervised data and fine-tuning on limited data than BERT. There are three important directions remain for future research: (1) inject knowledge into feature-based pre-training models such as ELMo (Peters et al., 2018); (2) introduce diverse structured knowledge into language representation models such as ConceptNet (Speer and Havasi, 2012) which is different from the world knowledge database Wikidata; (3) annotate more real-world corpora heuristically for building larger pre-training data. These directions may lead to more general and effective language understanding.

在本文中，我们建议ERNIE将知识信息整合到语言表示模型中。因此，我们提出了知识聚合器和预训练任务dEA，以更好地融合来自文本和KGs的异质信息。实验结果表明，ERNIE比BERT具有更好的去噪能力和对有限数据的微调能力。未来的研究仍有三个重要方向：(1)将知识注入基于特征的预训练模型，如ELMo(Peterset al., 2018); (2) 将不同的结构化知识引入语言表示模型，如ConceptNet(Speer和Havasi，2012)，这与世界知识数据库Wikidata不同; (3) 启发式地注释更多真实世界的语料库，以构建更大的训练前数据。这些指导可能会导致更广泛和有效的语言理解。

## Acknowledgement
This work is funded by the Natural Science Foundation of China (NSFC) and the German Research Foundation (DFG) in Project Crossmodal Learning, NSFC 61621136008 / DFG TRR-169, the National Natural Science Foundation of China (NSFC No. 61572273) and China Association for Science and Technology (2016QNRC001).

本研究由中国自然科学基金会(NSFC)和德国研究基金会(DFG)跨模式学习项目资助，NSFC 61621136008/DFG TRR-169，国家自然科学基金(NSFC No.61572273)和中国科学技术协会(2016QNRC001)资助。

## References

* Eneko Agirre, Llu’is M‘arquez, and Richard Wicentowski.2007. Proceedings of the fourth international workshop on semantic evaluations (semeval-2007). In Proceedings of SemEval-2007. 
* Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multi-relational data. In Proceedings of NIPS, pages 2787–2795. 
* Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of EMNLP, pages 632–642. 
* Yixin Cao, Lei Hou, Juanzi Li, Zhiyuan Liu, Chengjiang Li, Xu Chen, and Tiansi Dong. 2018. Joint representation learning of cross-lingual words and entities via attentive distant supervision. In Proceedings of EMNLP, pages 227–237. 
* Yixin Cao, Lifu Huang, Heng Ji, Xu Chen, and Juanzi Li.2017. Bridge text and knowledge by learning multiprototype entity mention embedding. In Proceedings of ACL, pages 1623–1633. 
* Qian Chen, Xiaodan Zhu, Zhen-Hua Ling, Diana Inkpen, and Si Wei. 2018. Neural natural language inference models enhanced with external knowledge. In Proceedings of ACL, pages 2406–2417. 
* Eunsol Choi, Omer Levy, Yejin Choi, and Luke Zettlemoyer.2018. Ultra-fine entity typing. In Proceedings of ACL, pages 87–96. 
* Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of ICML, pages 160–167. 
* Ido Dagan, Oren Glickman, and Bernardo Magnini. 2006. 
* The PASCAL recognising textual entailment challenge. In Proceedings of MLCW, pages 177–190. 
* Andrew M Dai and Quoc V Le. 2015. Semi-supervised sequence learning. In Proceedings of NIPS, pages 3079– 3087. 
* Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT. 
* William B Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In Proceedings of IWP. 
* Paolo Ferragina and Ugo Scaiella. 2010. Tagme: on-the-fly annotation of short text fragments (by wikipedia entities). In Proceedings of CIKM, pages 1625–1628. 
* Xu Han, Zhiyuan Liu, and Maosong Sun. 2016. Joint representation learning of text and knowledge for knowledge graph completion. arXiv preprint arXiv:1611.04125. 
* Xu Han, Zhiyuan Liu, and Maosong Sun. 2018a. Neural knowledge acquisition via mutual attention between knowledge graph and text. In Proceedings of AAAI. 
* Xu Han, Pengfei Yu, Zhiyuan Liu, Maosong Sun, and Peng Li. 2018b. Hierarchical relation extraction with coarse-to- fine grained attention. In Proceedings of EMNLP, pages 2236–2245. 
* Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. 2018c. Fewrel: A largescale supervised few-shot relation classification dataset with state-of-the-art evaluation. In Proceedings of EMNLP, pages 4803–4809. 
* Dan Hendrycks and Kevin Gimpel. 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415. 
* Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In Proceedings of ACL, pages 328–339. 
* Hector J Levesque, Ernest Davis, and Leora Morgenstern.2011. The Winograd schema challenge. In AAAI Spring Symposium: Logical Formalizations of Commonsense Reasoning, volume 46, page 47. 
* Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. 2016. Neural relation extraction with selective attention over instances. In Proceedings of ACL, volume 1, pages 2124–2133. 
* Xiao Ling, Sameer Singh, and Daniel S Weld. 2015. Design challenges for entity linking. TACL, 3:315–328. 
* Andrea Madotto, Chien-Sheng Wu, and Pascale Fung. 2018. 
* Mem2seq: Effectively incorporating knowledge bases into end-to-end task-oriented dialog systems. In Proceedings of ACL, pages 1468–1478. 
* Stephen Merity, Nitish Shirish Keskar, and Richard Socher.2018. Regularizing and optimizing lstm language models. In Proceedings of ICLR. 
* Todor Mihaylov and Anette Frank. 2018. Knowledgeable reader: Enhancing cloze-style reading comprehension with external commonsense knowledge. In Proceedings of ACL, pages 821–832. 
* Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In Proceedings of NIPS, pages 3111–3119. 
* Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. Glove: Global vectors for word representation. In Proceedings of EMNLP, pages 1532–1543. 
* Matthew Peters, Waleed Ammar, Chandra Bhagavatula, and Russell Power. 2017. Semi-supervised sequence tagging with bidirectional language models. In Proceedings of ACL, pages 1756–1765. 
* Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer.2018. Deep contextualized word representations. In Proceedings of NAACL-HLT, pages 2227–2237. 
* Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding by generative pre-training. In Proceedings of Technical report, OpenAI. 
* Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In Proceedings of EMNLP, pages 2383–2392. 
* Erik F Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the conll-2003 shared task: Languageindependent named entity recognition. In Proceedings of NAACL-HLT, pages 142–147. 
* Sonse Shimaoka, Pontus Stenetorp, Kentaro Inui, and Sebastian Riedel. 2016. An attentive neural architecture for fine-grained entity type classification. In Proceedings of AKBC, pages 69–74. 
* Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of EMNLP, pages 1631–1642. 
* Robert Speer and Catherine Havasi. 2012. Representing general relational knowledge in conceptnet 5. In Proceedings of LREC, pages 3679–3686. 
* Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, and Hua Wu. 2019. Ernie: Enhanced representation through knowledge integration. 
* Kristina Toutanova, Danqi Chen, Patrick Pantel, Hoifung Poon, Pallavi Choudhury, and Michael Gamon. 2015. 
* Representing text for joint embedding of text and knowledge bases. In Proceedings of EMNLP, pages 1499–1509. 
* Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: a simple and general method for semisupervised learning. In Proceedings of ACL, pages 384– 394. 
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of NIPS, pages 5998–6008. 
* Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and PierreAntoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In Proceedings of ICML, pages 1096–1103. 
* Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. Glue: A multitask benchmark and analysis platform for natural language understanding. In Proceedings of EMNLP, pages 353– 355. 
* Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen.2014. Knowledge graph and text jointly embedding. In Proceedings of EMNLP, pages 1591–1601. 
* Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman.2018. Neural network acceptability judgments. arXiv preprint 1805.12471. 
* Adina Williams, Nikita Nangia, and Samuel R. Bowman.2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of NAACLHLT, pages 1112–1122. 
* Yi Wu, David Bamman, and Stuart Russell. 2017. Adversarial training for relation extraction. In Proceedings of EMNLP, pages 1778–1783. 
* Ji Xin, Hao Zhu, Xu Han, Zhiyuan Liu, and Maosong Sun.2018. Put it back: Entity typing with language model enhancement. In Proceedings of EMNLPs, pages 993–998. 
* Yadollah Yaghoobzadeh and Hinrich Sch¨utze. 2017. Multilevel representations for fine-grained typing of knowledge base entities. In Proceedings of EACL, pages 578–589. 
* Ikuya Yamada, Hiroyuki Shindo, Hideaki Takeda, and Yoshiyasu Takefuji. 2016. Joint learning of the embedding of words and entities for named entity disambiguation. In Proceedings of CoNLL, pages 250–259. 
* Poorya Zaremoodi, Wray Buntine, and Gholamreza Haffari.2018. Adaptive knowledge sharing in multi-task learning: Improving low-resource neural machine translation. In Proceedings of ACL, pages 656–661. 
* Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. 2018. Swag: A large-scale adversarial dataset for grounded commonsense inference. In Proceedings of EMNLP, pages 93–104. 
* Daojian Zeng, Kang Liu, Yubo Chen, and Jun Zhao. 2015. 
* Distant supervision for relation extraction via piecewise convolutional neural networks. In Proceedings of EMNLP, pages 1753–1762. 
* Yuhao Zhang, Peng Qi, and Christopher D Manning. 2018. 
* Graph convolution over pruned dependency trees improves relation extraction. In Proceedings of EMNLP, pages 2205–2215. 
* Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D Manning. 2017. Position-aware attention and supervised data improve slot filling. In Proceedings of EMNLP, pages 35–45. 
* Wanjun Zhong, Duyu Tang, Nan Duan, Ming Zhou, Jiahai Wang, and Jian Yin. 2018. Improving question answering by commonsense-based pre-training. arXiv preprint arXiv:1809.03568.
