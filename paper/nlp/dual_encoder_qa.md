# Exploring Dual Encoder Architectures for Question Answering
探索用于问题解答的双编码器架构 2022-4-14 原文：https://arxiv.org/abs/2204.07120

# Abstract
Dual encoders have been used for question-answering (QA) and information retrieval (IR) tasks with good results. There are two major types of dual encoders, Siamese Dual Encoders (SDE), with parameters shared across two encoders, and Asymmetric Dual Encoder (ADE), with two distinctly parameterized encoders. In this work, we explore the dual encoder architectures for QA retrieval tasks. By evaluating on MS MARCO and the MultiReQA benchmark, we show that SDE performs significantly better than ADE. We further propose three different improved versions of ADEs. Based on the evaluation of QA retrieval tasks and direct analysis of the embeddings, we demonstrate that sharing parameters in projection layers would enable ADEs to perform competitively with SDEs.

双编码器已用于问答(QA)和信息检索(IR)任务，并取得了良好的效果。有两种主要类型的双编码器，孪生双编码器(SDE)和非对称双编码器(ADE)，这两种编码器具有两个不同的参数化编码器。在这项工作中，我们探索了QA检索任务的双编码器架构。通过对MS MARCO和MultiReQA基准的评估，我们发现SDE的性能明显优于ADE。我们进一步提出了三种不同的ADE改进版本。基于对QA检索任务的评估和对嵌入的直接分析，我们证明在投影层中共享参数将使ADE能够与SDE竞争。

## 1 Introduction
A dual encoder is an architecture consisting of two encoders, each of which encodes an input (such as a piece of text) into an embedding, and where the model is optimized based on similarity metrics in the embedding space. It has shown excellent performance in a wide range of information retrieval and question answering tasks (Gillick et al., 2018; Karpukhin et al., 2020). This approach is also easy to productionize because the embedding index of dual encoders can grow dynamically for newly discovered or updated documents and passages without retraining the encoders (Gillick et al., 2018). In contrast, generative neural networks used for question answering need to be retrained with new data. This advantage makes dual encoders more robust to freshness.

双编码器是一种由两个编码器组成的架构，每个编码器将输入(如一段文本)编码为嵌入，其中模型基于嵌入空间中的相似性度量进行优化。它在广泛的信息检索和问答任务中表现出色(Gillicket al., 2018; Karpukhinet al., 2020)。这种方法也很容易实现产品化，因为双编码器的嵌入索引可以在新发现或更新的文档和段落中动态增长，而无需重训编码器(Gillicket al., 2018)。相反，用于回答问题的生成神经网络需要用新数据重训。这一优势使双编码器对新鲜度更加稳健。

There are different valid designs for dual encoders. As shown in Table 1, the two major types are: Siamese Dual Encoder (SDE) and Asymmetric Dual Encoder (ADE). In a SDE the parameters are shared between the two encoders. In an ADE only some or no parameters are shared (Gillick et al., 2018). In practice, we often require certain asymmetry in the dual encoders, especially in the case where the inputs of the two towers are of different types. Though all of these models have achieved excellent results in different NLP applications, how these parameter-sharing design choices affect the model performance is largely unexplored.

双编码器有不同的有效设计。如表1所示，两种主要类型是：孪生双编码器(SDE)和非对称双编码器(ADE)。在SDE中，参数在两个编码器之间共享。在ADE中，仅共享一些或不共享参数(Gillicket al., 2018)。在实践中，我们经常要求双编码器具有一定的不对称性，特别是在两个塔的输入是不同类型的情况下。尽管所有这些模型在不同的NLP应用中都取得了优异的结果，但这些参数共享设计选择如何影响模型性能在很大程度上尚未探索。

Table 1: Existing off-the-shelf dual encoders. 
表1：现有现成的双编码器。

This paper explores the impact of parameter sharing in different components of dual encoders on question answering tasks, and whether the impact holds for dual encoders with different capacity. We conduct experiments across six well-established datasets and find that SDEs consistently outperforms ADEs on question answering tasks. We further propose to improve ADEs by sharing the projection layer between the two encoders, and show that this simple approach enables ADEs to achieve comparable or even better performance than SDEs. 

本文探讨了双编码器不同组件中的参数共享对问答任务的影响，以及这种影响是否适用于具有不同容量的双编码器。我们在六个成熟的数据集上进行了实验，发现SDE在回答问题的任务上始终优于ADE。我们进一步提出通过在两个编码器之间共享投影层来改进ADE，并表明这种简单的方法使ADE能够实现与SDE相当甚至更好的性能。

## 2 Related work
Dual encoders have been widely studied in entity linking (Gillick et al., 2018), open-domain question answering (Karpukhin et al., 2020), and dense retrieval (Ni et al., 2021a), etc. This architecture consists of two encoders, where each encoder encodes arbitrary inputs that may differ in type or granularity, such as queries, images, answers, passages, or documents.

双编码器已在实体链接(Gillicket al., 2018)、开放域问答(Karpukhinet al., 2020)和密集检索(Niet al., 2021a)等方面进行了广泛研究。该架构由两个编码器组成，其中每个编码器编码类型或粒度可能不同的任意输入，如查询、图像、答案、段落或文档。

Open-domain question answering (ODQA) is a challenging task that searches for evidence across large-scale corpora and provides answers to user queries (Voorhees, 1999; Chen et al., 2017). One of the prevalent paradigms for ODQA is a two-step approach, consisting of a retriever to find relevant evidence and a reader to synthesize answers. Alternative approaches are directly retrieving from large candidate corpus to provide sentence-level (Guo et al., 2021) or phrase-level (Lee et al., 2021b) answers; or directly generating answers or passage indices using an end-to-end generation approach (Tay et al., 2022). Lee et al. (2021a) compared the performance of SDEs and ADEs for phrase-level QA retrieval tasks. However, they only considered the two extreme cases that two towers have the parameters completely shared or distinct. In this work, we address the missing piece of previous work by exploring parameter sharing in different parts of the model. 

开放域问答(ODQA)是一项具有挑战性的任务，它在大规模语料库中搜索证据，并为用户查询提供答案(Voorhees，1999; Chenet al., 2017)。ODQA的一个流行范例是两步式方法，由检索器查找相关证据和读者综合答案组成。替代方法是直接从大型候选语料库中检索，以提供句子级(郭et al., 2021)或短语级(李et al., 2021b)答案; 或者使用端到端生成方法直接生成答案或文章索引(Tayet al., 2022)。Leeet al (2021a)比较了SDE和ADE在短语级QA检索任务中的性能。然而，他们只考虑了两个极端情况，即两个塔的参数完全相同或不同。在这项工作中，我们通过探索模型的不同部分中的参数共享来解决先前工作中缺失的部分。

Figure 1: Architectures of dual encoders. We study whether parameter sharing in different dual encoder components (i.e. token embedder, transformer encoder, and projection layer) can lead to better representation learning. Different color within each figure represents distinctly parameterized components, and grey components are frozen during the fine-tuning. 

图1：双编码器的结构。我们研究不同的双编码器组件(即令牌嵌入器、变换器编码器和投影层)中的参数共享是否可以导致更好的表示学习。每个图形中的不同颜色代表了明显的参数化组件，灰色组件在微调过程中被冻结。

## 3 Method
We follow a standard setup of QA retrieval: given a question q and a corpus of answer candidates A, the goal is to retrieve k relevant answers Ak ∈ A for q. The answer can be either a passage, a sentence, or a phrase.

我们遵循QA检索的标准设置：给定问题q和候选答案a的语料库，目标是检索k个相关答案Ak∈ 答案可以是短文、句子或短语。

We adopt a dual encoder architecture (Gillick et al., 2018; Reimers and Gurevych, 2019) as the model to match query and answers for retrieval. The model has two encoders, where each is a transformer that encodes a question or an answer. Each encoder first produces a fixed-length representation for its input and then applies a projection layer to generate the final embedding. We train the dual encoder model by optimizing the contrastive loss with an in-batch sampled softmax (Henderson et al., 2017):

我们采用双编码器架构(Gillicket al., 2018; Reimers和Gurevych，2019)作为模型，以匹配查询和答案进行检索。该模型有两个编码器，每个编码器都是一个对问题或答案进行编码的转换器。每个编码器首先为其输入生成固定长度的表示，然后应用投影层来生成最终嵌入。我们通过使用批内采样softmax优化对比度损失来训练双编码器模型(Hendersonet al., 2017)：

L = e sim(qi,ai)/τ P j∈B e sim(qi,aj )/τ , (1) 

where qi is a question and a∗ is a candidate answer. ai is ground-truth answer, or a positive sample, for qi . All other answers aj in the same batch B are considered as negative samples during training. τ is the softmax temperature and sim is a similarity function to measure the relevance between the question and the answer. In this work, we use cosine distance as the similarity function.

qi是一个问题和一个∗ 是候选答案。ai是qi的基本真实答案，或正样本。同一批次B中的所有其他答案aj在训练期间被视为负样本。τ是软最大温度，sim是一个相似函数，用于测量问题和答案之间的相关性。在这项工作中，我们使用余弦距离作为相似函数。

We want to explore the architectures of dual encoder with different degrees of parameter sharing. In particular, we aim to evaluate the importance of parameter sharing in dual encoder training. To this end, we study five different variants of dual encoders as shown in Figure 1: 
* Siamese Dual-Encoder (SDE),
* Asymmetric Dual-Encoder (ADE),
* ADE with shared token embedder (ADE-STE),
* ADE with frozen token embedder (ADE-FTE),
* ADE with shared projection layer (ADE-SPL). 

我们想探索具有不同参数共享程度的双编码器的架构。特别是，我们旨在评估参数共享在双编码器训练中的重要性。为此，我们研究了双编码器的五种不同变体，如图1所示：
* 孪生双编码器(SDE)，
* 非对称双编码器(ADE)，
* 具有共享令牌嵌入器(ADE-STE)的ADE，
* 具有冻结令牌嵌入器的ADE(ADE-FTE)，
* 具有共享投影层(ADE-SPL)的ADE。

## 4 Experiments and Analysis
We evaluate the proposed dual encoder architectures on six question-answering retrieval tasks from MS MARCO (Bajaj et al., 2016) and MultiReQA (Guo et al., 2021). In MS MARCO, we consider the relevant passages as answer candidates, while for the five QA datasets in MultiReQA the answer candidates are individual sentences.

我们评估了MS MARCO(Bajajet al., 2016)和MultiReQA(郭et al., 2021)提出的六个问答检索任务的双编码器架构。在MS MARCO中，我们将相关段落视为候选答案，而对于MultiReQA中的五个QA数据集，候选答案是单个句子。 

Table 2: Precision at 1(P@1)(%) and Mean Reciprocal Rank (MRR)(%) on QA retrieval tasks. SDE and ADE stand for Siamese Dual-Encoder and Asymmetric Dual-Encoder, respectively. ADE-STE, -FTE and -SPL are the ADEs with shared token-embedders, frozen token-embedders, and shared projection-layers, respectively. BERT-DE, which stands BERT (Devlin et al., 2019) Dual-Encoder, and USE-QA (Yang et al., 2020) are the baselines reported in MultiReQA (Guo et al., 2021). The most performant models are marked in bold.

表2：1时的精度(P@1)%和平均倒数排名(MRR)(%)。SDE和ADE分别代表孪生双编码器和非对称双编码器。ADE-STE、-FTE和-SPL分别是具有共享令牌嵌入器、冻结令牌嵌入器和共享投影层的ADE。BERT-DE代表BERT(Devlinet al., 2019)双编码器和USE-QA(Yanget al., 2020)是MultiReQA中报告的基线(郭et al., 2021)。性能最好的模型以粗体标记。

Table 3: Scaling effect. Precision at 1 (P@1)(%) and Mean Reciprocal Rank (MRR)(%) on MS MARCO (Bajaj et al., 2016) QA retrieval tasks, with dual encoders initialized from t5.1.1-small, -base, and -large checkpoints. The most performant models are marked in bold.

表3：缩放效果。精度为1(P@1)在MS MARCO(Bajajet al., 2016)QA检索任务中，(%)和平均倒数排名(MRR)(%)，双编码器从t5.1.1-小、基和大检查点初始化。性能最好的模型以粗体标记。

To initialize the parameters of dual encoders, we use pre-trained t5.1.1 encoders (Raffel et al., 2020). Following Ni et al. (2021b), we take the average embeddings of the T5 encoder’s outputs and send to a projection layer to get the final embeddings. The projection layers are randomly initialized, with variance scaling initialization with scale 1.0. For the retrieval, we use mean embeddings from the encoder towers. To make a fair comparison, the same hyper-parameters are applied across all the models for the fine-tuning with Adafactor optimizer (Shazeer and Stern, 2018), using learning rate 10−3 and batch size 512. The models are fine-tuned for 20, 000 steps, with linear decay of learning rate from 10−3 to 0 at the final steps. The fine-tuned models are benchmarked with precision at 1 (P@1) and mean reciprocal rank (MRR) on the QA retrieval tasks, in Table 2.

为了初始化双编码器的参数，我们使用预训练的t5.1.1编码器(Raffelet al., 2020)。根据Niet al (2021b)，我们获取T5编码器输出的平均嵌入，并发送到投影层以获得最终嵌入。投影层被随机初始化，方差缩放初始化为1.0。对于检索，我们使用来自编码器塔的平均嵌入。为了进行公平的比较，使用Adafactor优化器(Shazeer和Stern，2018)对所有模型应用相同的超参数进行微调，使用学习率10−3和批量大小512。模型被微调为20000步，学习率从10−最后一步为3到0。微调模型的基准精度为1(P@1)和QA检索任务的平均倒数排名(MRR)，见表2。

### 4.1 Comparing SDE and ADE
SDE and ADE in Figure 1 (a) and (b) are the two most distinct dual-encoders in terms of parameter sharing. Experiment results show that, on QA retrieval tasks, ADE performs consistently worse than SDE. To explain that, our assumption is that, at inference time, the two distinct encoders in ADE that do not share any parameters, map the questions and the answers into two parameter spaces that are not perfectly aligned. However, for SDE, parameter sharing enforces the embeddings from the two encoders to be in the same space. We verify this assumption in Section 4.3.

图1(a)和(b)中的SDE和ADE是参数共享方面最不同的两种双编码器。实验结果表明，在QA检索任务中，ADE的表现始终比SDE差。为了解释这一点，我们的假设是，在推理时，ADE中不共享任何参数的两个不同编码器将问题和答案映射到两个不完全对齐的参数空间中。然而，对于SDE，参数共享强制来自两个编码器的嵌入在同一空间中。我们在第4.3节中验证了这一假设。

### 4.2 Improving the Asymmetric Dual Encoder
Although the dual encoders with maximal parameter sharing (SDEs) performs significantly better than the ones without parameter sharing (ADEs), we often require certain asymmetry in the dual encoders in practice. Therefore, trying to improve the performance of ADEs, we construct dual-encoders with parameters shared at different levels between the two encoders.

尽管具有最大参数共享(SDE)的双编码器的性能明显优于没有参数共享(ADE)的编码器，但在实践中，我们通常要求双编码器具有一定的不对称性。因此，为了提高ADE的性能，我们构建了两个编码器之间在不同级别共享参数的双编码器。

Shared and Frozen Token Embedders. Token embedders are the lowest layers close to the input text. In ADEs, token embedders are initialized from the same set of pre-trained parameters, but fine-tuned separately. A straightforward way to bring ADEs closer to SDEs is to share the token embedders between the two towers, or to an extreme, to simply freeze the token embedders during training.

共享和冻结令牌嵌入。令牌嵌入器是靠近输入文本的最低层。在ADE中，令牌嵌入器从同一组预训练参数初始化，但单独微调。让ADE更接近SDE的一种简单方法是在两个塔之间共享令牌嵌入器，或者在训练期间简单地冻结令牌嵌入器。

Evaluated on MS MARCO and MultiReQA, the results in Table 2 show that both freezing (ADEFTE) and sharing (ADE-STE) token embedders bring consistent, albeit marginal, improvements for ADEs. However, ADEs with common token embedders still leave a significant gap compared to SDEs on most tasks. These results suggest token embedders might not be the key to close this gap.

在MS MARCO和MultiReQA上进行评估，表2中的结果表明，冻结(ADEFTE)和共享(ADE-STE)令牌嵌入器都为ADE带来了一致的(尽管是微不足道的)改进。然而，与大多数任务中的SDE相比，具有通用令牌嵌入器的ADE仍然存在很大差距。这些结果表明，token嵌入可能不是缩小这一差距的关键。

Figure 2: t-SNE clustering of the embeddings of the NaturalQuestions eval set generated by five dual encoders. The blue and orange points represent the embeddings of the questions and answers, respectively.
图2：由五个双编码器生成的NaturalQuestions评估集嵌入的t-SNE聚类。蓝色和橙色点分别表示问题和答案的嵌入。

Figure 3: Relative performance improvements of different models relative to ADE on QA retrieval tasks. ∆MRR = (MRR − MRRADE)/MRRADE) × 100.
图3：不同模型在QA检索任务中相对于ADE的相对性能改进。∆MRR=(MRR− MRRADE)/mrrrade)×100。

Figure 4: The impact of model size on the performance of different dual encoder architectures, measured by MRR on the eval set of MS MARCO. 
图4：模型大小对不同双编码器架构性能的影响，通过MRR对MS MARCO评估集进行测量。

Shared Projection Layers. Another way of improving retrieval quality of ADEs is to share the projection layers between the two encoders. Table 2 shows that sharing projection layers drastically improves the quality of ADEs. As in Figure 3, ADE-SPL (purple curve) performs on-par and, sometimes, even better than SDE (blue curve). This observation reveals that sharing projection layers is a valid approach to enhance the performance of ADEs. This technique would be vital if asymmetry is required by a modeling task. We further interpret this result in the next section.

共享投影层。提高ADE检索质量的另一种方法是在两个编码器之间共享投影层。表2显示，共享投影层大大提高了ADE的质量。如图3所示，ADE-SPL(紫色曲线)的表现与标准杆相当，有时甚至优于SDE(蓝色曲线)。这一观察结果表明，共享投影层是提高ADE性能的有效方法。如果建模任务需要不对称性，这种技术将非常重要。我们将在下一节中进一步解释这个结果。

### 4.3 Analysis on the Embeddings
The experiments corroborate our assumption that sharing the projection layer enforces the two encoders to produce embeddings in the same parameter space, which improves the retrieval quality.

实验证实了我们的假设，即共享投影层会强制两个编码器在相同的参数空间中生成嵌入，从而提高检索质量。

To further substantiate our assumption, we first generate the question and answer embeddings from the NaturalQuestions eval set, and then use t-SNE (van der Maaten and Hinton, 2008) to project and cluster the embeddings into 2-dimensional space (For efficiently clustering with t-SNE, we randomly sampled questions and answers, 400 each, from the NQ eval set. ). Figure 2 shows that, for ADE, ADE-STE and ADEFTE that have separate projection layers, the question and answer embeddings are projected and clustered into two disjoint groups. In comparison, ADE-SPL that shares the projection layers, the embeddings of questions and answers are not separable by t-SNE, which is similar to the behavior of SDE. This verifies our assumption that the projection layer plays an important role in bringing together the representations of questions and answers, and is the key for retrieval performance.

为了进一步证实我们的假设，我们首先从NaturalQuestions评估集生成问答嵌入，然后使用t-SNE(van der Maaten和Hinton，2008)将嵌入投影并聚类到二维空间中(为了有效地利用t-SNE聚类，我们从NQ评估集随机抽取了400个问题和答案)。图2显示，对于具有独立投影层的ADE、ADE-STE和ADEFTE，问题和答案嵌入被投影并聚类为两个不相交的组。相比之下，共享投影层的ADE-SPL，问题和答案的嵌入不能通过t-SNE分离，这与SDE的行为类似。这验证了我们的假设，即投影层在汇集问题和答案的表示方面起着重要作用，并且是检索性能的关键。

### 4.4 Impact of Model Size.
To assess the impact of model size, we fine-tune and evaluate the dual-encoders initialized from t5.1.1-small (∼ 77M parameters), -base (∼ 250M), and -large (∼ 800M) on the MS MARCO. Table 3 and Figure 4 show that, across different model sizes, sharing projection layers consistently improves the retrieval performance of ADE, and ADE-SPL performs competitively with SDE. This observation further validates our recommendation to share the projection layer in ADEs. 

为了评估模型大小的影响，我们对从t5.1.1开始初始化的双编码器进行了微调和评估(∼ 77M参数)，-基础(∼ 250M)和-大型(∼ 800M)。表3和图4显示，在不同的模型大小中，共享投影层一致地提高了ADE的检索性能，并且ADE-SPL的性能与SDE具有竞争力。这一观察进一步验证了我们在ADE中共享投影层的建议。

## 5 Conclusion and Future Work
Based on the experiments on six QA retrieval tasks with three different model sizes, we conclude that, although SDEs outperforms ADEs, sharing the projection layer between the two encoders enables ADEs to perform competitively with SDEs. By directly probing the embedding space, we demonstrate that the shared projection layers in SDE and ADE-SPL can map the embeddings of the two encoders into coinciding parameter spaces, which is crucial for improving the retrieval quality. Therefore, we recommend to share the projection layers between two encoders of ADEs in practice.

基于对三种不同模型大小的六个QA检索任务的实验，我们得出结论，尽管SDE优于ADE，但在两个编码器之间共享投影层使ADE能够与SDE竞争。通过直接探测嵌入空间，我们证明了SDE和ADE-SPL中的共享投影层可以将两个编码器的嵌入映射到一致的参数空间，这对于提高检索质量至关重要。因此，我们建议在实践中在ADE的两个编码器之间共享投影层。

## References