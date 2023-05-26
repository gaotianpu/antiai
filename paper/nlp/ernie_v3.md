# ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation
ERNIE 3.0 Titan：探索更大规模的知识增强的语言理解和生成训练 2021-12-23 原文：https://arxiv.org/abs/2112.12731

## 阅读笔记
* 中文大型语料数据集的构建
* 知识增强模型?
* 自监督的对抗性损失, 可控的语言建模损失
* 在线蒸馏框架，教师模型将同时教授学生和训练自己

## Abstract
Pre-trained language models have achieved state-of-the-art results in various Natural Language Processing (NLP) tasks. GPT-3 has shown that scaling up pre-trained language models can further exploit their enormous potential. A unified framework named ERNIE 3.0 was recently proposed for pre-training large-scale knowledge enhanced models and trained a model with 10 billion parameters. ERNIE 3.0 outperformed the state-of-the-art models on various NLP tasks. In order to explore the performance of scaling up ERNIE 3.0, we train a hundred-billion-parameter model called ERNIE 3.0 Titan with up to 260 billion parameters on the PaddlePaddle platform. Furthermore, we design a self-supervised adversarial loss and a controllable language modeling loss to make ERNIE 3.0 Titan generate credible and controllable texts. To reduce the computation overhead and carbon emission, we propose an online distillation framework for ERNIE 3.0 Titan, where the teacher model will teach students and train itself simultaneously. ERNIE 3.0 Titan is the largest Chinese dense pre-trained model so far. Empirical results show that the ERNIE 3.0 Titan outperforms the state-of-the-art models on 68 NLP datasets.

预训练的语言模型在各种自然语言处理(NLP)任务中取得了最先进的结果。GPT-3已经表明，扩大预训练语言模型可以进一步挖掘其巨大潜力。最近提出了一个名为ERNIE 3.0的统一框架，用于预训练大规模知识增强模型，并训练了一个具有100亿个参数的模型。ERNIE 3.0在各种NLP任务上表现优于最先进的模型。为了探索扩大ERNIE 3.0的性能，我们在飞桨平台上训练了一个1000亿参数的模型，称为ERNIE 3.0 Titan，参数高达2600亿。此外，我们设计了一个自监督的对抗性损失和一个可控的语言建模损失，以使ERNIE 3.0 Titan生成可信和可控的文本。为了减少计算开销和碳排放，我们提出了ERNIE 3.0 Titan的在线蒸馏框架，教师模型将同时教授学生和训练自己。ERNIE 3.0 Titan是迄今为止最大的中文密集预训练模型。经验结果表明，ERNIE 3.0 Titan在68个NLP数据集上优于最先进的模型。

## 1 Introduction
Pre-trained language models such as ELMo [4], GPT [5], BERT [6], and ERNIE [7] have proven effective for improving the performances of various natural language understanding and generation tasks. Pre-trained language models are generally learned on a large amount of text data in a self-supervised manner and then fine-tuned on downstream tasks or directly deployed through zero-shot learning without task-specific fine-tuning. Such pre-trained language models start to serve as foundation models and bring a new paradigm for natural language processing tasks. This new paradigm changes the focus of NLP research from designing specialized models for different tasks to studying pre-trained language models and using them in various tasks. Recent advances such as GPT-3 [1] have demonstrated the promising trend towards scaling up pre-trained language models with billions of parameters. These studies show surprising potentials by scaling up pre-trained models. 

ELMo [4]、GPT [5]、BERT [6] 和 ERNIE [7] 等预训练语言模型已被证明可有效提高各种自然语言理解和生成任务的性能。 预训练语言模型通常以自监督的方式在大量文本数据上学习，然后在下游任务上进行微调，或者直接通过零样本学习部署，无需针对特定任务进行微调。 这种预训练的语言模型开始充当基础模型，并为自然语言处理任务带来新的范式。 这种新范式将 NLP 研究的重点从为不同任务设计专门模型转变为研究预训练语言模型并将其用于各种任务。 GPT-3 [1] 等最新进展展示了扩展具有数十亿参数的预训练语言模型的大好趋势。 这些研究通过扩大预训练模型显示出惊人的潜力。

Most of existing large-scale models were pre-trained on plain texts without integrating knowledge. ERNIE 3.0 [2] tried to incorporate knowledge such as linguistic knowledge and world knowledge into large-scale pre-trained language models. ERNIE 3.0 pre-trained Transformers on massive unstructured texts and knowledge graphs to learn different levels of knowledge, such as lexical, syntactic, and semantic information. ERNIE 3.0 can handle both natural language understanding tasks and natural language generation tasks through zero-shot learning, few-shot learning, or fine-tuning. Furthermore, it supports introducing various customized tasks at any time. These tasks share the same encoding networks that are pre-trained through multi-task learning. This method makes it possible to encode lexical, syntactic, and semantic information across different tasks.

大多数现有的大规模模型都是在没有集成知识的情况下在纯文本上进行预训练的。 ERNIE 3.0 [2] 试图将语言知识和世界知识等知识融入到大规模预训练语言模型中。 ERNIE 3.0 在海量非结构化文本和知识图谱上预训练 Transformer，学习不同层次的知识，如词汇、句法和语义信息。 ERNIE 3.0 可以通过零样本学习、少样本学习或微调来处理自然语言理解任务和自然语言生成任务。 此外，它还支持随时引入各种自定义任务。 这些任务共享通过多任务学习预训练的相同编码网络。 这种方法使得跨不同任务编码词汇、句法和语义信息成为可能。

<!--人工抽取的那点知识(图谱)，真的有增益？-->

This work explores the performance of knowledge-enhanced pre-trained models with larger-scale parameters based on the ERNIE 3.0 framework. We train a Chinese dense pre-trained language model with 260 billion parameters (named as ERNIE 3.0 Titan) on the PaddlePaddle platform. Although large-scale language models like GPT-3 have shown promising text generation capabilities, it is still challenging for users to control the generation results and obtain generated texts that are factually consistent with the real world. To fill the gap, we propose a highly credible and controllable generative pre-training technique (see Figure. 2), in which a self-supervised adversarial loss and a controllable language modeling loss are optimized during the pre-training phase. In detail, a self-supervised adversarial loss allows the model to learn to distinguish whether a text is the original one or generated by ERNIE 3.0. Besides accelerating the convergence of the model, this loss enables ERNIE 3.0 Titan to re-ranking the credibility of the generated results. Meanwhile, a controllable language modeling loss is applied to enable the model to generate texts with specific attributes. We prompt the original text with a diverse attribute set, including the genre, topic, keywords, sentiment, and length, which can be easily expanded for more user-defined attributes. The users can freely combine these attributes for controllable generations of different types of downstream application scenarios. We conduct several experiments on 68 datasets. The results show that ERNIE 3.0 Titan significantly outperforms previous models on various tasks by a large margin and achieves new state-of-the-art results.

这项工作探索了基于 ERNIE 3.0 框架的具有更大规模参数的知识增强型预训练模型的性能。 我们在 PaddlePaddle 平台上训练了一个具有 2600 亿个参数的中文密集预训练语言模型(命名为 ERNIE 3.0 Titan)。 尽管像 GPT-3 这样的大规模语言模型已经显示出很有前途的文本生成能力，但用户控制生成结果并获得与现实世界事实一致的生成文本仍然具有挑战性。 为了填补这一空白，我们提出了一种高度可信和可控的生成预训练技术(见图 2)，其中在预训练阶段优化了自监督的对抗性损失和可控的语言建模损失。 详细来说，自监督的对抗性损失允许模型学习区分文本是原始文本还是由 ERNIE 3.0 生成的。 除了加速模型的收敛之外，这种损失还使 ERNIE 3.0 Titan 能够重新对生成结果的可信度进行排名。 同时，应用可控的语言建模损失，使模型能够生成具有特定属性的文本。 我们使用多样化的属性集提示原始文本，包括类型、主题、关键字、情感和长度，可以轻松扩展以获取更多用户定义的属性。 用户可以自由组合这些属性，实现可控生成不同类型的下游应用场景。 我们对 68 个数据集进行了多项实验。 结果表明，ERNIE 3.0 Titan 在各项任务上明显优于之前的模型，取得了新的 state-of-the-art 结果。

<!-- 对抗性损失和可控的语言建模损失 -->

Furthermore, we propose a distillation framework to distill the ERNIE 3.0 Titan for easy deployment. Intuitively, we can apply current knowledge distillation methods [8, 9, 10, 11] to ERNIE 3.0 Titan. However, current distillation methods require an additional inference stage on a fully trained teacher to transfer knowledge to the student models, which is not environment-friendly concerning carbon emissions [12]. Another problem of current methods is that only one student model can be produced after the distillation phase is completed, requiring the teacher to infer multiple times to distill multiple students. In addition to the computation resource problems, previous studies [13, 14, 15, 16] reveal that distillation from oversized teachers can lead to unexpected performance degradation problems. [13, 14, 17] indicates that the difficulty comes from the large gap between the teacher’s and student’s parameter numbers, causing significant differences between their representation spaces. To this end, we propose an online distillation framework to efficiently distill the ERNIE 3.0 Titan into multiple small models during the pre-training stage, which results in little additional computation cost as compared to current distillation methods. Our distillation framework contains four key features: i) teaching multiple students at the same time, ii) proposing On-the-Fly Distillation (OFD), where the teacher instructs the students during the training stage for a more environmentally friendly distillation, iii) introducing teacher assistants [13] for better distilling large scale models, iv) introducing Auxiliary Layer Distillation (ALD), a technique to improve distillation performance by stacking an additional student layer in distillation stage and discarding it at the fine-tuning stage. We compare our distilled ERNIE 3.0 Titan with previous compact models on five typical types of downstream tasks. The results demonstrate the effectiveness of our proposed distillation framework, and show that the distilled ERNIE 3.0 Titan achieves SOTA results on all tasks. 

此外，我们提出了一个蒸馏框架来蒸馏 ERNIE 3.0 Titan 以便于部署。 直观地说，我们可以将当前的知识蒸馏方法 [8、9、10、11] 应用于 ERNIE 3.0 Titan。 然而，目前的蒸馏方法需要在训练有素的教师身上进行额外的推理阶段，以将知识传递给学生模型，这在碳排放方面不利于环境 [12]。 当前方法的另一个问题是蒸馏阶段完成后只能产生一个学生模型，需要教师进行多次推理才能蒸馏出多个学生。 除了计算资源问题，之前的研究 [13, 14, 15, 16] 表明，超大规模教师的蒸馏会导致意想不到的性能下降问题。 [13,14,17]表明困难来自教师和学生的参数数量之间的巨大差距，导致他们的表示空间之间存在显著差异。 为此，我们提出了一种在线蒸馏框架，可以在预训练阶段将 ERNIE 3.0 Titan 有效地蒸馏成多个小模型，与目前的蒸馏方法相比，这几乎不会产生额外的计算成本。 我们的蒸馏框架包含四个关键特征：
1. 同时教授多个学生，
2. 提出即时蒸馏(OFD)，教师在训练阶段指导学生进行更环保的蒸馏，
3. 引入教师助理 [13] 以更好地蒸馏大型模型，
4. 引入辅助层蒸馏(ALD)，这是一种通过在蒸馏阶段堆叠额外的学生层并在微调阶段丢弃它来提高蒸馏性能的技术。 
我们将我们的蒸馏 ERNIE 3.0 Titan 与之前的紧凑型模型在五种典型的下游任务类型上进行了比较。 结果证明了我们提出的蒸馏框架的有效性，并表明蒸馏后的 ERNIE 3.0 Titan 在所有任务上都取得了 SOTA 结果。

## 2 Related Work
### 2.1 Large-scale Pre-training
Due to the rapid development of deep learning algorithms and the iterations of high-performance chips, pre-trained language models such as BERT [6], GPT [5], and ERNIE [7] have made significant breakthroughs in many fields of natural language processing, such as natural language understanding, language generation, machine translation, humanmachine dialogue, and question answering. These methods use unified Transformer models for learning universal representations on large unsupervised corpora. This technique taps into the advantages of scale of unsupervised data brings to natural language processing and significantly breaks the high reliance on costly annotated data.

由于深度学习算法的快速发展和高性能芯片的迭代，BERT[6]、GPT[5]、ERNIE[7]等预训练语言模型在自然语言的多个领域取得了重大突破 处理，如自然语言理解、语言生成、机器翻译、人机对话和问答。 这些方法使用统一的 Transformer 模型来学习大型无监督语料库的通用表示。 该技术利用了无监督数据规模给自然语言处理带来的优势，并显著打破了对昂贵的标注数据的高度依赖。

Some recent works [18, 19, 1, 20] had demonstrated that increasing the size of pre-trained models can further exploit the potential value of unsupervised data. For example, the T5 model [19] was proposed to push the performance for both natural language understanding and natural language generation tasks with 11 billion parameters. The T5 model converts all text-based language tasks into a text-to-text format by a unified framework and fully explores the effectiveness of pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors.  

最近的一些工作 [18、19、1、20] 已经证明，增加预训练模型的大小可以进一步利用无监督数据的潜在价值。 例如，T5 模型 [19] 被提出来推动具有 110 亿个参数的自然语言理解和自然语言生成任务的性能。 T5 模型通过统一的框架将所有基于文本的语言任务转换为文本到文本的格式，并充分探索了预训练目标、架构、未标注数据集、迁移方法等因素的有效性。

<!--T5的论文-->

GPT-3 [1], with 175 billion parameters, achieved an amazing performance on a wide range of tasks under the few-shot and zero-shot settings with prompts. Several work have investigated the effects of larger pre-trained models such as Jurassic-1 [21], Gopher [22], Megatron-Turing NLG [23], PanGu-α [24], Yuan 1.0 [25], etc. An alternative route to increasing the size of the model parameters is to sparse the model parameters. These models [20, 26, 27, 28, 29] use mixture-of-experts to select different parameters for each incoming example. As far as we know, sparse models have not achieved better performance than dense models up to now, although they can scale more efficiently with trillion parameters or even more. Therefore, this paper mainly focuses on large-scale distributed training and inference techniques for dense models. Most of the previous models learned only on plain texts confronted the problem of lacking common sense [30]. In addition, most large-scale models are trained in an auto-regressive way, but [6] shows that such models have poorer performance with traditional fine-tuning when adapting to downstream language understanding tasks. In order to solve these problems, a unified framework called ERNIE 3.0 [2] was proposed to train large-scale knowledge-enhanced models on large-scale plain texts and a large-scale knowledge graph by fusing the auto-regressive network and the auto-encoding network.

GPT-3 [1] 拥有 1750 亿个参数，在带提示的少样本和零样本设置下，在广泛的任务上取得了惊人的性能。 几项工作已经研究了大型预训练模型的效果，例如 Jurassic-1 [21]、Gopher [22]、Megatron-Turing NLG [23]、PanGu-α [24]、Yuan 1.0 [25] 等。 增加模型参数大小的替代方法是稀疏模型参数。 这些模型 [20、26、27、28、29] 使用混合专家为每个传入样本选择不同的参数。 据我们所知，到目前为止，稀疏模型并没有取得比密集模型更好的性能，尽管它们可以更有效地扩展万亿甚至更多的参数。 因此，本文主要关注密集模型的大规模分布式训练和推理技术。 大多数以前的模型只在纯文本上学习，都面临着缺乏常识的问题 [30]。 此外，大多数大型模型都是以自回归方式训练的，但[6]表明，当适应下游语言理解任务时，此类模型与传统微调相比性能较差。为了解决这些问题，提出了一个名为ERNIE 3.0[2]的统一框架，通过融合自回归网络和自动编码网络，在大规模纯文本和大规模知识图上训练大规模知识增强模型。

<!--稀疏模型, 使用混合专家为每个传入样本选择不同的参数 -->

The exponential increment of the pre-trained language model’s size has posed a great challenge for efficient training due to memory constraints and unaffordable training time. The size of pre-trained language models exceeds the memory limit of a single modern processor. In addition, it is inevitable to store momentum and other optimizer states in widely used optimization algorithms such as Adam [31]. Therefore there are lots of work focusing on achieving efficient training of large-scale models. The first category is the Pipeline Model Parallelism, splitting different Transformer layers into separate cards. GPipe [32] utilizes a novel batch-splitting pipelining algorithm, resulting in almost linear speedup when partition model across multiple accelerators. TeraPipe [33] proposed a high-performance token-level pipeline parallel algorithm for synchronous model-parallel training of uni-directional language models. PTD-P [34] scheduled with interleaved stages in a fine-grained way to reduce the size of the pipeline bubble further. Another category is the Tensor Model Parallelism [18], in which individual layers of the model are partitioned over multiple devices. They took advantage of the structure of transformer networks to create a simple model parallel implementation by adding a few synchronization primitives. PTD-P [34] also combine pipeline, tensor, and data parallelism across multi-GPU servers to combine their advantages.

由于内存限制和无法承受的训练时间，预训练语言模型大小的指数增长对高效训练提出了巨大挑战。 预训练语言模型的大小超过了单个现代处理器的内存限制。 此外，在 Adam [31] 等广泛使用的优化算法中，不可避免地会存储动量和其他优化器状态。 因此，有很多工作专注于实现大规模模型的高效训练。 第一类是 Pipeline Model Parallelism，将不同的 Transformer 层拆分成单独的卡。 GPipe [32] 利用一种新颖的批处理流水线算法，在跨多个加速器划分模型时产生几乎线性的加速。 TeraPipe [33] 提出了一种高性能的令牌级流水线并行算法，用于单向语言模型的同步模型并行训练。 PTD-P [34] 以细粒度的方式安排交错阶段，以进一步减小管道气泡的大小。 另一类是 Tensor Model Parallelism [18]，其中模型的各个层被划分到多个设备上。 他们利用变压器网络的结构，通过添加一些同步原语来创建一个简单的模型并行实现。 PTD-P [34] 还结合了跨多 GPU 服务器的流水线、张量和数据并行性，以结合它们的优势。

<!--模型训练中的工程能力-->

### 2.2 Credible Text Generation 可信文本生成
The credibility of a text contains multiple aspects such as coherency, clarity, veracity, etc. For non-pretraining methods, previous works explored many approaches to promote several aspects of credible generation: Plan-and-write [35] uses a hierarchical method that first plans a storyline and then generates a story based on it to improve the text coherency; CGRG [36] uses grounding knowledge to generate veritable answers. With the development of pre-training technology, large-scale language models provide a simple yet powerful solution for credible text generation. [37] shows that GPT-2 [38] synthetic text samples can achieve a promising convincing score compared with authentic articles from the New York Times (72% vs. 83% in one cohort judged the articles to be credible). Furthermore, GROVER [39] utilizes an adversarial framework to train a classifier with the generated text for incredible news detection. Inspired by this, ERNIE 3.0 Titan introduces an auxiliary adversarial loss to train a credibility ranker for self-ranking and select the most credible text for output.

文本的可信度包含多个方面，如连贯性、清晰度、准确性等。对于非预训练方法，之前的工作探索了许多方法来促进可信生成的几个方面：Plan-and-write [35] 使用分层方法， 先规划故事情节，再据此生成故事，提高文本的连贯性;  CGRG [36] 使用基础知识来生成真实的答案。 随着预训练技术的发展，大规模语言模型为可信文本生成提供了一种简单而强大的解决方案。 [37] 表明，与纽约时报的真实文章相比，GPT-2 [38] 合成文本样本可以获得令人信服的分数(72% 对 83% 的队列认为文章是可信的)。 此外，GROVER [39] 利用对抗框架使用生成的文本训练分类器，以进行令人难以置信的新闻检测。 受此启发，ERNIE 3.0 Titan 引入了辅助对抗损失来训练可信度排序器进行自排序，并选择最可信的文本进行输出。

### 2.3 Controllable Text Generation 可控文本生成
As large-scale pre-training models have shown growing effectiveness in generating high-quality sentences, controllable text generation is getting more attention [40]. GPT-3 [1] used in-context few-shot learning to prompt text generation for various tasks, while it is still challenging for users in controlling the generation results. CTRL [40] provided an effective way for controllable text generation. It trained a language model conditioned on control codes that govern style, content, and task-specific behavior. However, these control codes, derived from the structure that naturally co-occurs with raw texts, cover constrained controllable attributes. Following works like Arg-CTRL [41] and Tailor [42] extended the control codes of CTRL either by crowdsourcing aspect annotations or deriving from the PropBank formalism to control the semantic-specific generation. Another line of work [43, 44] aims to generate texts of desired attributes through relatively small ‘pluggable’ attribute models focusing on light-weight controllable fine-tuning on pre-trained models. In this paper, we focus on directly providing a highly controllable model pre-trained on ERNIE 3.0 controllable dataset powered by conditional LM loss.

随着大规模预训练模型在生成高质量句子方面的有效性越来越高，可控文本生成越来越受到关注[40]。 GPT-3 [1] 使用上下文中的小样本学习来提示各种任务的文本生成，而用户在控制生成结果方面仍然具有挑战性。 CTRL [40] 为可控文本生成提供了一种有效的方法。 它训练了一个语言模型，该模型以控制风格、内容和任务特定行为的控制代码为条件。 然而，这些控制代码源自与原始文本自然共现的结构，涵盖了受限的可控属性。 Arg-CTRL [41] 和 Tailor [42] 等后续工作通过众包方面注释或从 PropBank 形式派生来控制特定语义生成来扩展 CTRL 的控制代码。 另一项工作 [43、44] 旨在通过相对较小的“可插入”属性模型生成所需属性的文本，重点是对预训练模型进行轻量级可控微调。 在本文中，我们专注于直接提供在由条件 LM 损失提供支持的 ERNIE 3.0 可控数据集上预训练的高度可控模型。

### 2.4 Model Distillation of Language Models 语言模型的模型蒸馏
Although large-scale language models show their outstanding ability on various tasks, their enormous parameters require a significant amount of computational resources and make them difficult to deploy on real-life applications. Language model distillation [8, 11, 9, 10, 45, 46, 47, 48, 49] has recently drawn significant attention as one prevailing method to solve the foregoing problem. For example, [48] proposes DistilBERT to successfully halve the depth of the BERT model by matching the output between teacher and student model on pre-training and fine-tuning stages. TinyBERT [8] adds two additional points of distillation, the attention distribution and hidden representations, to improve distillation quality. To further boost the distillation, in addition to the pre-training and fine-tuning stages, ERNIE-Tiny [11] introduces two additional stages to ensure that the student captures the domain knowledge from the fine-tuned teacher. Unlike previous works that require a fine-tuned teacher for distillation, [9] proposes task-agnostic distillation that only performs distillation on the pre-trained teacher by mimicking self-attention and value relation.

尽管大型语言模型在各种任务上都表现出了出色的能力，但其庞大的参数需要大量的计算资源，使其难以部署到实际应用中。语言模型蒸馏 [8, 11, 9, 10, 45, 46, 47, 48, 49] 作为解决上述问题的一种主流方法，最近引起了极大的关注。 例如，[48] 提出 DistilBERT 通过在预训练和微调阶段匹配教师和学生模型之间的输出，成功地将 BERT 模型的深度减半。 TinyBERT [8] 添加了两个额外的蒸馏点，注意力分布和隐藏表示，以提高蒸馏质量。 为了进一步促进蒸馏，除了预训练和微调阶段外，ERNIE-Tiny [11] 还引入了两个额外的阶段，以确保学生从微调教师那里获取领域知识。 与以前需要微调教师进行蒸馏的工作不同，[9] 提出了与任务无关的蒸馏，它仅通过模仿自我注意和价值关系对预先训练的教师进行蒸馏。

Recent works [13, 14, 15, 17, 50] have observed that distillation from an oversized teacher to a significantly smaller student can cause an unexpected performance drop. [13] suggests that the performance drop is due to the capacity gap between the teacher and the student, and introduces additional student models named teacher assistants with size between teacher and student to alleviate this gap. [14] proposes to early stop the teacher to reduce the gap. [15] proposes to utilize unconverged teacher checkpoints to guide student’s learning process in a curriculum learning manner. [16] proposes joint training of teacher and student, allowing the teacher to be aware of the existence of the student and reducing the gap, although it deteriorates the performance of the teacher model. [17] suggests that the unexpected distillation performance drop is caused by the fact that the oversized model tends to be overconfident, and forcing the student to learn such overconfident output harms the distillation performance. It proposes to normalize the point of distillation to alleviate the overconfidence problem. 

最近的工作 [13, 14, 15, 17, 50] 观察到，从一个超大的老师到一个小得多的学生的提炼会导致意想不到的性能下降。 [13] 表明性能下降是由于教师和学生之间的能力差距造成的，并引入了额外的学生模型，称为教师助理，教师和学生之间的规模缩小了这一差距。 [14]建议提前停止教师以减少差距。 [15] 提出利用未融合的教师检查点以课程学习的方式指导学生的学习过程。 [16] 提出教师和学生的联合培训，让教师意识到学生的存在并减少差距，尽管这会降低教师模型的性能。 [17] 表明，意外的蒸馏性能下降是由于过大的模型往往过于自信，而强迫学生学习这种过度自信的输出会损害蒸馏性能。 它提出规范化蒸馏点以缓解过度自信问题。

## 3 ERNIE 3.0 Titan Model
Figure 1: The framework of ERNIE 3.0.

A significant improvement has been achieved on various natural language processing tasks for knowledge-enhanced pre-trained models with the base or large model size, such as ERNIE, ERNIE 2.0, and SpanBERT [51], in which the base/large model size represent 12/24 layers Transformer respectively. In order to explore the effectiveness of knowledge enhanced large-scale pre-trained model, a Continual Multi-Paradigms Unified Pre-training Framework named ERNIE 3.0 Framework is proposed in [2] to pre-train model on massive unsupervised corpus including plain texts and knowledge graphs. Specifically, ERNIE 3.0 Framework allows collaborative pre-training among multi-task paradigms, in which various types of pre-training tasks are incrementally deployed in the corresponding task paradigm to enable the model to learn different levels of knowledge, i.e., valuable lexical, syntactic and semantic information, more effectively. Benefiting from the superiority of ERNIE 3.0 Framework, ERNIE 3.0 has made astonishing improvements on abundant downstream tasks across natural language understanding and natural language generation. As a matter of course, ERNIE 3.0 Titan is built on ERNIE 3.0 Framework in this paper. The detail of ERNIE 3.0 Framework will be explained in the following sections.

对于具有基本或大模型尺寸的知识增强预训练模型，如 ERNIE、ERNIE 2.0 和 SpanBERT [51]，在各种自然语言处理任务上取得了显著改进，其中基本/大模型尺寸表示 分别为12/24层Transformer。 为了探索知识增强型大规模预训练模型的有效性，文献[2]提出了一种名为ERNIE 3.0 Framework的连续多范式统一预训练框架，用于在海量无监督语料库（包括纯文本和文本）上预训练模型。 知识图谱。 具体来说，ERNIE 3.0 Framework 允许在多任务范式之间进行协作预训练，其中各种类型的预训练任务被增量地部署在相应的任务范式中，使模型能够学习不同层次的知识，即有价值的词汇、句法 和语义信息，更有效。 得益于ERNIE 3.0 Framework的优势，ERNIE 3.0在自然语言理解和自然语言生成等丰富的下游任务上做出了惊人的改进。 当然，本文中的ERNIE 3.0 Titan是建立在ERNIE 3.0 Framework之上的。 ERNIE 3.0 Framework 的细节将在以下部分进行解释。

### 3.1 Overview of ERNIE 3.0 Framework
Until recently, the prevalent unified pre-training models trend to employ a shared Transformer network for different well-designed cloze tasks and utilize specific self-attention masks to control the context of the prediction conditions. Nevertheless, we believe that the different task paradigms of natural language processing consistently depend on identical underlying abstract features, such as lexical and syntactic information. However, the requirements of top-level concrete features are incompatible, in that the natural language understanding tasks have the disposition to learn the semantic coherence while natural language generation tasks expect further contextual information. Inspired by the classical model architecture of multi-task learning, in which the lower layers are shared across all tasks while the top layers are task-specific, [2] construct the ERNIE 3.0 Framework shown in Figure 1, which enable the different task paradigms to share the underlying abstract features learned in a shared network and utilizing the task-specific top-level concrete features learned in their task-specific network respectively. The backbone shared network and task-specific networks are referred to as the Universal Representation Module and Task-specific Representation Modules in ERNIE 3.0 Framework. Specifically, the universal representation network plays the role of a universal semantic features extractor (for example, a multi-layer Transformer). The parameters are shared across all kinds of task paradigms, including natural language understanding and natural language generation, and so on. And the task-specific representation networks undertake the function of extracting the task-specific semantic features, in which the parameters are learned by task-specific objectives. Furthermore, the continual multi-task learning framework introduced in ERNIE  2.0 [52] is integrated into ERNIE 3.0 Framework to help the model efficiently learn the lexical, syntactic, and semantic representations.

直到最近，流行的统一预训练模型还倾向于为不同的精心设计的完形填空任务采用共享的 Transformer 网络，并利用特定的自注意力掩码来控制预测条件的上下文。 尽管如此，我们认为自然语言处理的不同任务范式始终依赖于相同的底层抽象特征，例如词汇和句法信息。 然而，顶级具体特征的要求是不相容的，因为自然语言理解任务有学习语义连贯性的倾向，而自然语言生成任务需要更多的上下文信息。 受多任务学习经典模型架构的启发，其中较低层在所有任务之间共享，而顶层是特定于任务的，[2] 构建了图 1 所示的 ERNIE 3.0 框架，它支持不同的任务范式 共享在共享网络中学习到的底层抽象特征，并分别利用在其特定任务网络中学习到的特定任务顶层具体特征。 主干共享网络和任务特定网络在 ERNIE 3.0 框架中被称为通用表示模块和任务特定表示模块。 具体来说，通用表示网络扮演通用语义特征提取器的角色（例如，多层 Transformer）。 这些参数在各种任务范式之间共享，包括自然语言理解和自然语言生成等。 特定任务表示网络承担提取特定任务语义特征的功能，其中参数是通过特定任务目标学习的。 此外，ERNIE 2.0 [52] 中引入的持续多任务学习框架被集成到 ERNIE 3.0 框架中，以帮助模型有效地学习词汇、句法和语义表示。

Driven by the success of ERNIE 3.0 [2], ERNIE 3.0 Titan also employs the collaborative architecture of a Universal Representation Module and two Task-specific Representation Modules, namely natural language understanding (NLU) specific representation module and natural language generation (NLG) specific representation module. Details are as follows:
* Universal Representation Module. In likewise, a multi-layer Transformer-XL [53] is adopted as the backbone network like other pre-trained models such as ERNIE 3.0 [2], XLNet [54], Segatron [55] and ERNIE-Doc [56], in which Transformer-XL is similar to Transformer but introduces an auxiliary recurrence memory module to help modelling longer texts. Proverbially, the larger the scale of the Transformer model, the stronger its capacity to capture and store up various semantic information with different levels enabled by the self-attention mechanism. Therefore, ERNIE 3.0 Titan sets the universal representation module with a larger size (refer to the section 3.4) to enable the model to effectively capture universal lexical and syntactic information from training data by learning various pre-training tasks of different paradigms. And what needs special attention is that the memory module is only valid for natural language generation tasks while controlling the attention mask matrices.
* Task-specific Representation Modules. Instead of the multi-layer perceptron or shallow Transformer commonly used as task-specific representation networks in multi-task learning, ERNIE 3.0 Titan employs the Transformer-XL network with base model size as the task-specific representation modules to capture the top-level semantic representations for different task paradigms. Under this design, ERNIE 3.0 Titan achieves a triple-win scenario: the base Transformer has a stronger ability to capture semantic information than multilayer perceptron and shallow Transformer while not significantly increasing the parameters of the large-scale model; last but not least, a new available route that enables the realizable practical applications for large scale pre-trained model can be explored — only fine-tuning on the task-specific representation modules. ERNIE 3.0 Titan constructs two task-specific representation modules: the bi-directional modeling NLU-specific representation network and the uni-directional modeling NLG-specific representation network.

在 ERNIE 3.0 [2] 成功的推动下，ERNIE 3.0 Titan 还采用了通用表示模块和两个特定任务表示模块的协作架构，即自然语言理解（NLU）特定表示模块和自然语言生成（NLG）特定 表示模块。 详情如下：
* 通用表示模块。 同样，像 ERNIE 3.0 [2]、XLNet [54]、Segatron [55] 和 ERNIE-Doc [56] 等其他预训练模型一样，采用多层 Transformer-XL [53] 作为骨干网络， 其中 Transformer-XL 与 Transformer 类似，但引入了一个辅助递归记忆模块来帮助建模更长的文本。 众所周知，Transformer 模型的规模越大，其通过自注意力机制捕获和存储各种不同层次的语义信息的能力就越强。 因此，ERNIE 3.0 Titan 设置了更大尺寸的通用表示模块（参考 3.4 节），使模型能够通过学习不同范式的各种预训练任务，有效地从训练数据中捕获通用的词汇和句法信息。 需要特别注意的是，记忆模块在控制attention mask矩阵的同时，只对自然语言生成任务有效。
* 任务特定的表示模块。 ERNIE 3.0 Titan 不是在多任务学习中通常用作特定任务表示网络的多层感知器或浅层 Transformer，而是采用具有基本模型大小的 Transformer-XL 网络作为特定任务表示模块来捕获顶层 不同任务范式的语义表示。 在这种设计下，ERNIE 3.0 Titan 实现了三赢的场景：基础 Transformer 比多层感知器和浅层 Transformer 具有更强的语义信息捕获能力，同时不会显著增加大规模模型的参数； 最后但并非最不重要的一点是，可以探索一条新的可用路线，使大规模预训练模型的实际应用成为可能——仅对特定于任务的表示模块进行微调。 ERNIE 3.0 Titan 构建了两个特定于任务的表示模块：双向建模 NLU 特定表示网络和单向建模 NLG 特定表示网络。

### 3.2 Pre-training Tasks
In order to make the capacity of understanding, generation and reasoning available to ERNIE 3.0 Titan, we construct several tasks for various task paradigms to capture different aspects of information in the training corpora, including word-aware pre-training tasks, structure-aware pre-training tasks and knowledge-aware pre-training task introuced in ERNIE 3.0 [2]. Additionally, an innovative knowledge-aware pre-training task namely Credible and Controllable Generations is built to control the generation result and obtain the result factually consistent with the real world. 

为了让 ERNIE 3.0 Titan 具备理解、生成和推理的能力，我们为各种任务范式构建了几个任务，以捕获训练语料库中不同方面的信息，包括词感知预训练任务、结构感知预训练任务 -ERNIE 3.0 [2]中引入的训练任务和知识感知预训练任务。 此外，创新的知识感知预训练任务 可信可控生成被构建来控制生成结果并获得与现实世界事实一致的结果。

#### 3.2.1 Word-aware Pre-training Tasks 词感知预训练任务
Knowledge Masked Language Modeling ERNIE 1.0 [7] proposed an effective strategy to enhance representation through knowledge integration, namely Knowledge Integrated Masked Language Modeling task. It introduced phrase masking and named entity masking that predict the whole masked phrases and named entities to help the model learn the dependency information in both local contexts and global contexts.

Knowledge Masked Language Modeling ERNIE 1.0 [7]提出了一种通过知识整合来增强表示的有效策略，即知识整合Masked Language Modeling任务。 它引入了短语掩码和命名实体掩码来预测整个被掩码的短语和命名实体，以帮助模型学习本地上下文和全局上下文中的依赖信息。

Document Language Modeling As introduced in [2], document language modeling task is a special version of traditional language modeling task, which trains models on long text instead of the prevailing shorter segments of manageable size (at most 512 tokens). Enhanced Recurrence Memory Mechanism proposed in ERNIE-Doc [56] is introduced into ERNIE 3.0 Titan to heighten the capability of modeling a larger effective context length than traditional recurrence Transformer.

文档语言建模 正如 [2] 中介绍的那样，文档语言建模任务是传统语言建模任务的一个特殊版本，它在长文本上训练模型，而不是在可管理大小（最多 512 个标记）上流行的较短片段上训练模型。 ERNIE-Doc [56] 中提出的增强型递归记忆机制被引入 ERNIE 3.0 Titan，以提高建模比传统递归 Transformer 更大的有效上下文长度的能力。

#### 3.2.2 Structure-aware Pre-training Tasks 结构感知预训练任务
Sentence Reordering Sentence reordering task, which is introduced in ERNIE 2.0 [52], aims to train the model to learn the relationship between sentences by reorganizing permuted segments. At length, a given paragraph is randomly split into 1 to m segments during pre-training, and all of the combinations are shuffled by a randomly permuted order. Then, the pre-trained model is asked to reorganize these permuted segments, modeled as a k-class classification problem where k = $\sum^m_{n=1}n!$.

句子重新排序 ERNIE 2.0 [52] 中引入的句子重新排序任务旨在训练模型通过重组置换段来学习句子之间的关系。 最后，一个给定的段落在预训练期间被随机分成 1 到 m 个片段，并且所有的组合都按照随机排列的顺序进行打乱。 然后，要求预训练模型重新组织这些排列的片段，建模为 k 类分类问题，其中 k = $\sum^m_{n=1}n!$。

Sentence Distance Sentence distance task, an extension of the traditional next sentence prediction (NSP) task, is widely used in various pre-trained models to enhance their ability to learn the sentence-level information, which can be modeled as a 3-class classification problem. The three categories represent that the two sentences are adjacent, nonadjacent but in the same document and from two different documents, respectively.

Sentence Distance 句子距离任务是传统下一句预测（NSP）任务的扩展，广泛应用于各种预训练模型，以增强其学习句子级信息的能力，可以建模为3类分类 问题。 这三个类别分别表示两个句子相邻、不相邻但在同一文档中和来自两个不同的文档。

#### 3.2.3 Knowledge-aware Pre-training Task 知识感知预训练任务
Universal Knowledge-Text Prediction Universal knowledge-text prediction (UKTP) task, a particular masked language modeling that constructed on both unstructured texts and structured knowledge graphs, plays a pivotal role in incorporating world knowledge and commonsense knowledge into pre-trained model. Given a pair of a triple from the knowledge graph and the corresponding sentence from the encyclopedia, UKTP task randomly mask the relation in triple or the words in corresponding sentence. To predict the relation in the triple, the model needs to detect mentions of the head entity and the tail entity and determine their semantic relationship in the corresponding sentence. Another, to predict the words in the corresponding sentence, the model needs to consider the dependency information in the sentence and the logical relationship in the triple.

Universal Knowledge-Text Prediction Universal Knowledge-Text Prediction (UKTP) 任务是一种基于非结构化文本和结构化知识图的特殊掩码语言建模，在将世界知识和常识知识整合到预训练模型中起着关键作用。 给定知识图谱中的一对三元组和百科全书中的相应句子，UKTP 任务随机屏蔽三元组中的关系或相应句子中的单词。 为了预测三元组中的关系，模型需要检测头实体和尾实体的提及，并确定它们在相应句子中的语义关系。 另外，为了预测对应句子中的词，模型需要考虑句子中的依存信息和三元组中的逻辑关系。

Credible and Controllable Generations Controlling the generated texts based on desired attributes and improving their credibility is a key and practical feature we introduced in ERNIE 3.0 Titan. To achieve this, we design a selfsupervised adversarial loss and a controllable language modeling loss for generating credible and controllable texts, respectively. 

可信和可控的生成根据所需属性控制生成的文本并提高其可信度是我们在 ERNIE 3.0 Titan 中引入的一个关键且实用的功能。 为实现这一目标，我们设计了一种自我监督的对抗性损失和一种可控的语言建模损失，分别用于生成可信和可控的文本。

Figure 2: The framework of ERNIE 3.0 Titan on credible and controllable generations. 
图2：ERNIE 3.0 Titan关于可信可控生成的框架。


The self-supervised adversarial loss allows the model to distinguish whether a text is generated or the original one. As a result, it is easy for ERNIE 3.0 Titan to discard the low credibility generated texts with repeating words, unfluent and conflicting sentences. In detail, we formalize this as a binary classification problem experimented on our ERNIE 3.0 adversarial dataset $D_a$ = {$D_{original}, D_{generated}$} which is a subset of original ERNIE 3.0 Corpus $D_{original}$ with its adversarial samples $D_{generated}$ generated by ERNIE 3.0.

自监督对抗性损失允许模型区分文本是生成的还是原始文本。 因此，ERNIE 3.0 Titan 很容易丢弃可信度低、单词重复、句子不流畅和冲突的文本。 详细地说，我们将其形式化为在我们的 ERNIE 3.0 对抗数据集 $D_a$ = {$D_{original}, D_{generated}$} 上进行实验的二元分类问题，它是原始 ERNIE 3.0 语料库 $D_{original}$ 的子集 其对抗样本 $D_{generated}$ 由 ERNIE 3.0 生成。

$L_a(D_a) = − \sum^{|D_a|}_{n=1} log P_θ(y^n = I_{h^n_{[CLS]}}∈ D_{original} |h^n_{[CLS]})$ (1) 

where we trained ERNIE 3.0 Titan with parameters θ to minimize the cross-entropy loss, $I_{h^n_{[CLS]}}∈D_{original}$ indicates whether nth sample belongs to the original dataset $D_{original}$, and the output hidden state of the special token [CLS] is taken as input for binary classification.

其中我们使用参数 θ 训练 ERNIE 3.0 Titan 以最小化交叉熵损失，$I_{h^n_{[CLS]}}∈D_{original}$ 表示第 n 个样本是否属于原始数据集 $D_{original}$ ，并将特殊标记 [CLS] 的输出隐藏状态作为二进制分类的输入。

The controllable language modeling loss is a modified language modeling loss by conditioning on extra prompts for controlling the generated texts as follows:

可控语言建模损失是一种修改后的语言建模损失，它通过以额外提示为条件来控制生成的文本，如下所示：

$$ L_c(D_c)=
\begin{cases}

− \sum^{|Dc|}_{n=1} log P_θ(x^n_t |x^n < t) \ if \ prob.p ≤ 0.5 \\
−\sum^{|Dc|}_{n=1} log P_θ(x^n_t |x^n < t, prompts^n) \ if \ prob.p > 0.5


\end{cases}
$$  (2) 


where ERNIE 3.0 Titan is trained to minimize the negative log-likelihood loss on ERNIE 3.0 controllable dataset $D_c$ = {$x^1 , x^2 , . . . , x^{|Dc|}$}, t means the $t_th$ token of x. $x^n$ is associated with $prompts^n$ specifying the genre, topic, keywords, sentiment and length. The loss is switched to the normal language modeling loss with a pre-defined probability 0.5 to prevent the model from heavily depending on the prompts. Different from CTRL which convers a constrainted controllable attributes from the semi-structural raw texts, we enrich the controllable attributes set using task-specific supervised models on ERNIE 3.0 Corpus. As the ERNIE 3.0 Corpus in nature constructed from various sources including Web, QA, Novel, Knowledge graph and etc. (see 3.3), we assign soft prompts (learnable prompt embedding) for different datasets to better align the model within the genre of the target dataset.

其中 ERNIE 3.0 Titan 被训练以最小化 ERNIE 3.0 可控数据集 $D_c$ = {$x^1 , x^2 , . 上的负对数似然损失。 . . , x^{|Dc|}$}, t表示x的第$t_th$个token。 $x^n$ 与指定流派、主题、关键词、情绪和长度的 $prompts^n$ 相关联。 将损失切换为具有预定义概率 0.5 的正常语言建模损失，以防止模型严重依赖提示。 与从半结构化原始文本转换为受约束的可控属性的 CTRL 不同，我们在 ERNIE 3.0 语料库上使用特定于任务的监督模型来丰富可控属性集。 由于 ERNIE 3.0 语料库本质上是由 Web、QA、Novel、Knowledge graph 等各种来源构建的（见 3.3），我们为不同的数据集分配软提示（learnable prompt embedding），以更好地将模型与类型匹配 目标数据集。

### 3.3 Pre-training Data
To ensure the success of the pre-training of ERNIE 3.0 Titan, we utilize the ERNIE 3.0 Corpus [2], a large-scale, wide-variety, and high-quality Chinese text corpora amounting to 4TB storage size in 11 different categories. Two additional datasets, namely the ERNIE 3.0 adversarial dataset and ERNIE 3.0 controllable dataset, are constructed.

为确保 ERNIE 3.0 Titan 预训练的成功，我们使用了 ERNIE 3.0 语料库 [2]，这是一个规模大、种类多、质量高的中文文本语料库，存储量达 4TB，分为 11 个不同类别。 构建了另外两个数据集，即 ERNIE 3.0 对抗数据集和 ERNIE 3.0 可控数据集。

#### ERNIE 3.0 adversarial dataset: 
The adversarial dataset is constructed based on ERNIE 3.0 Corpus. The positive examples consist of 2M natural paragraphs sampled from ERNIE 3.0 Corpus, while for negative example generation, we randomly take the first 1~3 sentences from the original positive paragraph as the prefix input, and utilize ERNIE 3.0 to generate the rest part of the paragraph. The max length of generated paragraph is set to 512, and we discard the last incomplete sentence if the generation process is terminated by max-length excess.

ERNIE 3.0 对抗数据集：对抗数据集基于 ERNIE 3.0 语料库构建。 正例由从 ERNIE 3.0 语料库中采样的 2M 自然段落组成，而对于负例生成，我们随机取原始正例中的前 1~3 个句子作为前缀输入，并利用 ERNIE 3.0 生成其余部分 段落。 生成段落的最大长度设置为 512，如果生成过程因最大长度超出而终止，我们将丢弃最后一个不完整的句子。

####  ERNIE 3.0 controllable dataset: 
The controllable dataset is highly scalable to include more diverse user-defined attributes. Here, we introduce 5 different controllable attributes including genre, topic, keywords, sentiment and length as follows:
* Genre is assigned to samples based on the source the data collected from, including general (ERNIE 2.0 Corpus), Web, QA, Knowledge, Finance, Law, Medical, Novel, Couplet, Poet, Machine Translation. Each genre type is associated with pre-defined maximum soft prompt embeddings (64 in our experiment). In the pre-training phase, the number of soft prompt embeddings are sampled randomly between 0 and the maximum number.
* Topic is labeled using a topic model(2 https://ai.baidu.com/tech/nlp_apply/topictagger ) which can classify a document into 26 different topics such as international, sports, society, news, technology, digital, emotion, cars, education, fashion, games, travel, food, culture, healthy life, child and music.
* Keywords are extracted using a keyword extraction model(3 https://ai.baidu.com/tech/nlp_apply/doctagger ) which performs in-depth analysis of article titles and content, and outputs multi-dimensional tags that reflect the key information of the article, such as subject, entity, etc.  
* Sentiment is derived using a sentiment classification model (4 https://ai.baidu.com/tech/nlp_apply/sentiment_classify) . A positive, negative, or neutral label is assigned to each sample.
* Length is counted on the tokenized text. The length attribute can prompt the model to generate texts with the desired length to avoid harshly truncating.

ERNIE 3.0 可控数据集：可控数据集具有高度可扩展性，可以包含更多不同的用户定义属性。 在这里，我们介绍了 5 个不同的可控属性，包括类型、主题、关键字、情感和长度，如下所示：
* 根据收集的数据来源为样本分配类型，包括一般（ERNIE 2.0 语料库）、Web、QA、Knowledge、Finance、Law、Medical、Novel、Couplet、Poet、Machine Translation。 每种流派类型都与预定义的最大软提示嵌入（在我们的实验中为 64）相关联。 在预训练阶段，软提示嵌入的数量在 0 和最大数量之间随机采样。
* 使用主题模型（2 ...）标记主题，该模型可以将文档分为 26 个不同的主题，例如国际、体育、社会、新闻、科技、数字、情感 、汽车、教育、时尚、游戏、旅游、美食、文化、健康生活、儿童、音乐。
* 关键词提取采用关键词抽取模型(3 ... )对文章标题和内容进行深度分析，输出反映关键信息的多维标签 文章的主题，例如主题，实体等。
* 情绪是使用情绪分类模型(4 ... ) 得出的。 每个样本都分配有阳性、阴性或中性标签。
* 长度以标记化文本计算。 length 属性可以提示模型生成具有所需长度的文本，以避免过度截断。

In pre-training phase, we use the following input format for each sample: “[Genre-0], [Genre-1], · · · [Genere-N] [t] Topic texts [/t] [k] Keyword text0, Keyword text1, · · · [/k] [senti] Sentiment label text [/senti] [w] About L words [/w] Original text ” where [Genere-n] is nth soft prompt embedding for one of the genre type mentioned above, N ∈ [0, 64), L is the token number of the original text and [t], [/t], [k], [/k] ,[senti], [/senti], [w], [/w] are special tokens to seperate each attribute. For example, the input for the case in Figure. 2 can be “[News-0], [News-1], · · · [News-N] [t] Sports [/t] [k] Lampard, Chelsea, UCL [/k] [senti] Positive [/senti] [w] About 85 words [/w] Lampard said on the 4th that Chelsea... ”. Note that for each attribute, we randomly discard it with a pre-defined probability (0.5 in our experiment) to prevent the model from heavily depending on it.

在预训练阶段，我们对每个样本使用以下输入格式：“[Genre-0]，[Genre-1]，···[Genere-N] [t]主题文本[/t] [k]关键字 text0, Keyword text1, · · · [/k] [senti] 情感标签文本 [/senti] [w] 关于 L 个词 [/w] 原文” 其中 [Genere-n] 是其中一个的第 n 个软提示嵌入 上面提到的 genre type，N ∈ [0, 64)，L 是原文的 token 编号，[t], [/t], [k], [/k] ,[senti], [/senti], [w]、[/w] 是用于分隔每个属性的特殊标记。 例如，图中案例的输入。 2 可以是“[News-0], [News-1], ··· [News-N] [t] Sports [/t] [k] Lampard, Chelsea, UCL [/k] [senti] Positive [/ senti] [w] 大约 85 个字 [/w] 兰帕德在 4 日说切尔西......”。 请注意，对于每个属性，我们以预定义的概率（在我们的实验中为 0.5）随机丢弃它，以防止模型严重依赖它。

### 3.4 Pre-training Settings
Following the pre-training setting of ERNIE 3.0, ERNIE 3.0 Titan includes the universal representation module and the task-specific representation modules, which both use the Transformer-XL structure. We adopt a structure with 12 layers, 768 hidden units, and 12 heads for the task-specific representation modules. We adopt a structure with 48 layers, 12288 hidden units, and 192 heads for the universal representation modules. We found that continually increasing the hidden size would make it difficult to load the parameter of output embedding in a single machine with eight cards with 32GB memory. In order to further increase the model capacity, we choose to scale the parameters of the point-wise feed-forward network alternatively. The inner layer of the universal representation modules has a dimensional of 196608, which is 16 times the hidden size of the model. The total number of parameters of ERNIE 3.0 Titan is over 260 billion.

We use Adam [31] with learning rate of 1e-4, β1 = 0.9, β2 = 0.95, L2 weight decay of 0.1, We also clip the global norm of the gradient at 1.0 to improve the stability of pre-training. The maximum sequence length of context and the memory length of language generation is 512 and 128, respectively. We use progressive learning to speed up convergence in the first 4000 steps and linear decay of the learning rate. ERNIE 3.0 Titan is implemented on the

PaddlePaddle framework and uses parallelism techniques that facilitate the efficient training of large models that do not fit in the memory of a single GPU. We will give the detail of these in the next section. 

## 4 Efficient Training and Inference of ERNIE 3.0 Titan
### 4.1 Distributed Training
To train a huge model like GPT-3 faces severe challenges in memory consumption and computational efficiency. This model requires 2.1TB for parameter and optimizer states storage and 3.14E11 TeraFLOPS for training 300 billion tokens.

Given that a modern AI processor like Nvidia V100 GPU can only provide 32GB of memory and 125 TeraFLOPS 5 , it will take 28 days to train with 2048 GPU V100 cards even with a 50% percentage of theoretical peak FLOPS. There have been many related works to solve these problems, for example, Megatron-LM [57], Gpipe [32], Zero [58] , and so on. PaddlePaddle [3] has also proposed 4D hybrid parallelism with more sophisticated combination techniques 6.

To train the ERNIE 3.0 Titan model on heterogeneous clusters is faced with even more challenges than GPT-3. On the one hand, the ERNIE 3.0 model has a more sophisticated architecture than GPT-3, such as task-specific representation layers and memory mechanism [59]. Such structures are prone to unbalance and inefficient pipeline training on top of intensive memory consumption. On the other hand, different clusters are coupled with distinct software stacks, including operator implementations, communication libraries, schedulers, etc. For example, NPU performs worse than

GPU on the tensor computation of dynamic shapes and shows strength on FP16 calculation, posing new challenges like customized layers optimization, unbalanced pipeline, and training stability.

To achieve efficient and stable training with convergence guarantee, PaddlePaddle developed an end-to-end adaptive distributed training technology, including fine-grained parallelism, heterogeneous hardware-aware training, and fault tolerance mechanism to train the 260B model on both Nvidia V100 GPU and Ascend 910 NPU clusters. 


5 https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf 
6 https://ai.baidu.com/forum/topic/show/987996

#### 4.1.1 Fine-grained Hybrid Parallelism
The 4D hybrid parallelism includes data parallelism (DP) [60, 61], intra-layer tensor model parallelism (MP) [62, 57], inter-layer pipeline model parallelism (PP) [32, 63, 64, 65, 33], and the improved sharded data parallelism (Sharded) based on ZeRO [58]. DP is deployed to partition and distribute the data across devices and synchronize all their gradient.

PP is utilized to split model layers into multiple stages distributed across devices, where the throughput is directly related to load balancing and bubble fraction. MP is used to slice parameters and activation as distributed across devices.

Sharded commits to reducing the redundancy of optimizer states. Our improved version of Sharded, namely Group

Sharded, is designed to decouple data-parallel and Sharded flexibly.

Besides, we implemented distributed operators for large Embedding, FC, and FC_Softmax layers, supporting finegrained partition. These operators can utilize intra-node communication to further reduce memory occupation and redundant calculations. This strategy increases the overall throughput significantly on the ERNIE 3.0 Titan model’s unique architecture.



Figure 3: Resource-aware-partition.

#### 4.1.2 Heterogeneous hardware-aware training
We train ERNIE 3.0 Titan model on both NPU clusters at PengCheng Lab and GPU clusters in Baidu. Specifically, training large-scale models on NPU relies heavily on carefully treating the following problems: 1) interface design between deep learning framework with fast-evolving NPU software stack; 2) software-hardware collaborative NPU performance optimization 3) adjustment of load balance concerning NPU cluster throughput.

To take the best advantage of PaddlePaddle’s imperative programming and the 4D parallelism technology, we use Ascend Computing Language (ACL) 7 in our implementation instead of using Ascend Graph(GE) 8 . We found that NPU has strong performance using pure FP16 while showing weakness in dealing with smaller shapes and dynamic shapes commonly used in NLP models.

In the PaddlePaddle framework, the parallel strategy can be adjusted in a resource-ware manner to fully exploit the computing power of the hardware. Figure 3 shows that the Task-specific Representation layers of ERNIE 3.0 require more load balance adjustments on NPUs than that on GPUs because NPUs consume more time on small layers’ kernel launches. As shown in Table 1, the ERNIE 3.0 Titan model reaches 91.7% weak scalability with thousands of NPU cards and increases the throughput up to 2.1 times while the number of cards only increased by 22%. Furthermore, Figure 4 shows how we convert dynamic shape to static shape to utilize the ACL performance fully.

#### 4.1.3 Fault tolerance
During our experiments, we also encountered hardware problems, for example, a sudden drop of communication speed because of PCIe errors, GPU card errors such as Double Bit ECC Error (DBE) 9 . PaddlePaddle developed fault-tolerant 7 https://support.huaweicloud.com/asdevg-python-cann/atlaspython_01_0008.html 8 https://support.huawei.com/enterprise/en/doc/EDOC1100164817/753b4f6/introduction 9 https://docs.nvidia.com/deploy/dynamic-page-retirement/index.html 9 tokens masks max_seg_len=L max_pos = [1,4,5,7,10] (dynamic_shape) padding padding_mask_pos = [1,4,5,7,10,*,*,*,*,*,*,*](L) effective_mask_pos = [1,1,1,1,1,0,0,0,0,0,0,0](L), eff_ratio = 12/5 mlm_loss = paddle. mean(mlm_loss * effective_mask_pos) * eff_ratio

Figure 4: Convert dynamic shape to static shape. features that automatically replace faulty machines to reduce hardware waste and time consumption to resolve these problems. PaddlePaddle allows users to customize their need to store and restore states from the checkpoints.

Table 1: Comparison between the default configuration and the resource-aware configuration when training ERNIE 3.0 Titan on Ascend NPUs.


These problems encountered in practice prompt our work on end-to-end adaptive training. For details please refer to [66].

### 4.2 Distributed Inference
It becomes infeasible to perform inference using ERNIE 3.0 Titan model on common GPU devices such as Nvidia A100-SXM4 (40GB). Therefore, the model has to be split into multiple subgraphs deployed on multiple GPU cards. We implemented tensor model parallelism and pipeline model parallelism in Paddle Serving, PaddlePaddle’s online serving framework. In addition, we adopted methods such as unified memory optimization, op fusion, model IO optimization, and quantization-aware acceleration on a single card to improve the inference speed. 

## 5 Online Distillation Framework for ERNIE 3.0 Titan
We devise an online distillation framework concerning the computation overhead and carbon emissions during the training stage, where the teacher model will teach students and train itself simultaneously to utilize the computational resource more effectively. To describe the framework, we firstly introduce our main procedure and then discuss the two key techniques, On the Fly Distillation (OFD) and Auxiliary Layer Distillation (ALD), in detail.

### 5.1 Main Framework of Distillation
Unlike existing distillation methods, our proposed framework trains multiple students rather than one at once. Figure 6 shows the general structure of our proposed distillation framework. The blue rectangle on the left represents the teacher model, ERNIE 3.0 Titan, the green rectangle represents a 24-layer model, which we call teacher assistant (TA). Besides the TA model, we also introduce multiple smaller student models (shown in red) into this framework. Considering the 


Figure 5: Distributed Inference framework of Paddle Serving. large gap between the teacher model and those smaller students, we do not directly transfer the knowledge from teacher to students. Instead, we use the teacher assistant [13] as a distillation bridge to better transfer knowledge. As [9, 10] have shown that the attention probability of the teacher model’s last few layers is crucial to task-agnostic knowledge distillation, we follow this paradigm in our framework. Through this framework, three students models can be trained simultaneously, saving the complex process to train them one by one in traditional distillation methods. However, there are still two issues with the current procedure. The first one is that this distillation process does not utilize the training time computational resource effectively and requires additional forward propagation from the teacher for distillation, causing additional computation overhead. The second problem is that matching attention module [9, 10] in Transformer between teacher and students will leave the weights of feed forward network in the students’ last layer untrained, as one

Transformer layer is composed of an attention module and a feed forward network module [67]. To solve those two problems, we introduce two techniques called OFD and ALD which will be discussed in the following sections.

### 5.2 OFD: On the Fly Distillation
To better utilize the computational resource and for a more environmentally friendly distillation, we propose our online learning method: OFD. The learning process is that every time the teacher updates one step, the students update one step toward the teacher. Students’ learning targets (i.e. the teacher) change over time during the training process.

From this "moving target" perspective, this framework is similar to [15] where different teacher checkpoints during pre-training are selected as distillation targets. However, the teacher in this framework changes smoothly, whereas that in [15] changes discretely. OFD allows teacher training and distillation perform simultaneously. The benefit is that we can better utilize the forward propagation from the teacher during the teacher’s pre-training for distillation, unlike the existing knowledge distillation methods [8, 48] which requires additional forward propagation from the teacher to extract the knowledge for distillation. Note that the OFD will not influence the teacher’s training as the gradients from distillation loss will not flow from the TA or students back to the teacher.

### 5.3 ALD: Auxilliary Layer Distillation
As the knowledge being transferred during distillation is the attention probability distribution, the other module in a

Transformer block, the feed forward network, will not be trained during distillation. To show this problem more clearly, we will briefly describe the structure of the Transformer. 11

Figure 6: Online Distillation Framework for ERNIE 3.0 Titan.

In the Transformer architecture [67], each Transformer layer consists of two sub-modules, namely the multi-head self-attention (MHA) and position-wise feed-forward network (FFN). Transformer encodes contextual information for input tokens. The input embeddings {x}si=1 for sample x are packed together into H0 = [x1, · · · , xs] , where s denotes the input sequence length. Then stacked Transformer blocks iteratively compute the encoding vectors as

Hl = Transformerl (Hl−1), l ∈ [1, L], and the Transformer is computed as:

Al,a = MHAl,a(Hl−1WQ l,a, Hl−1WK l,a), (3)

H0l−1 = LN(Hl−1 + ( hka=1

Al,a(Hl−1WV l,a))WlO), (4)

Hl = LN  H0l−1 + FFN  H0l−1  , (5) where the previous layer’s output Hl−1 ∈ Rs×d is linearly projected to a triple of queries, keys and values using parameter matrices WQ l,a,WK l,a,WV l,a ∈ Rd×d0 , where d denotes the hidden size of Hl and d0 denotes the hidden size of each head’s dimension. Al,a ∈ Rs×s indicates the attention distributions for the a-th head in layer l, which is computed by the scaled dot-product of queries and keys respectively. h represents the number of self-attention heads. k denotes concatenate operator along the head dimension. WlO ∈ Rd×d denotes the linear transformer for the output of the attention module. FFN is composed of two linear transformation function including mapping the hidden size of

H0l−1 to df f and then mapping it back to d.

In our distillation framework, we use the Kullback–Leibler divergence of Al,a between teacher and students as the distillation objective. However, matching the attention in the last layer of the students will leave the FFN in the last layer untrained as the gradient only flows backward. To this end, we propose stacking an extra layer on the students during distillation to ensure that the gradient can flow through the entire network and that all the parameters are trained during distillation. This extra layer will be discarded when the students are fine-tuned on downstream tasks. 

## 6 Experiments
Three groups of experiments, including fine-tuning on natural language understanding tasks (in Sec. 6.2), few-shot learning (in Sec. 6.3), and zero-shot learning (in Sec. 6.4), are conducted on a variety of prevailing NLP tasks to evaluate the performance of ERNIE 3.0 Titan. All the previous state-of-the-art results for comparison come from the best public 12 single model reported that we could find.10. It is essential to mention that all experimental results of ERNIE 3.0 Titan are based on the insufficiently pre-trained model so far. ERNIE 3.0 Titan is still in training, and we believe that the model will become stronger as the pre-training progresses.

### 6.1 Evaluation Tasks
68 datasets belonging to 12 kinds of natural language processing tasks are used in our experiments, in which datasets marked with FC are from FewCLUE Benchmark. Significantly, several datasets are applied to the experiments of fine-tuning / few-shot learning and zero-shot learning simultaneously in different ways. In this paper, we treat duplicate datasets used in different evaluation methods independently. The details as follows:

我们的实验使用了属于 12 种自然语言处理任务的 68 个数据集，其中标有 FC 的数据集来自 FewCLUE Benchmark。 值得注意的是，几个数据集以不同的方式同时应用于微调/少样本学习和零样本学习的实验。 在本文中，我们独立处理不同评估方法中使用的重复数据集。 详情如下：

* Sentiment Analysis (SA): NLPCC2014-SC 11, SE-ABSA16_PHNS 12, SE-ABSA16_CAME, BDCI2019 13 , EPRSTMT [68].
* Opinion Extraction (OE): COTE-BD [69], COTE-MFW [69].
* Natural Language Inference (NLI): OCNLI [70], CMNLI [70], OCNLI-FC [68].
* Winograd Schema Challenge (WSC): CLUEWSC [70], CLUEWSC-FC [68].
* Relation Extraction (RE): FinRE [71], SanWen [72].
* Semantic Similarity (SS): AFQMC [70], LCQMC [73], PAWS-X [74], BQ Corpus [75], CSL [70], CSLFC [68], BUSTM [68].
* Text Classification (TC): TNEWS 14, TNEWS-FC [68], IFLYTEK [76], IFLYTEK-FC [68], THUCNEWS 15 , CNSE [77], CNSS [77], CSLDCP [68].
* Closed-Book Question Answering (CB-QA): NLPCC-DBQA 16, CHIP2019, cMedQA [78], cMedQA2 [79], CKBQA 17, WebQA [80].
* Cloze and Completion (Clz.&Compl.): PD&CFT [81], CMRC2017 [82], CMRC2019 [83], CHID [84], CHID-FC [68], WPLC [85].
* Machine Reading Comprehension (MRC): DRCD [86], DuReader [87], Dureaderrobust [88], Dureaderchecklist, Dureaderyesno 18, C3 [89], CMRC 2018 [90].
* Legal Documents Analysis (LDA): CAIL2018-Task1 [91], CAIL2018-Task2 [91].
* Cant Understanding (CU): DogWhistle Insider, DogWhistle Outsider [92].

### 6.2 Experiments on Fine-tuning Tasks
The results of natural language understanding tasks are reported in Table 2. As shown in Table 2,

Sentiment Analysis. Sentiment Analysis is a classification task aiming to determine whether a sentence is positive, negative, or neutral. We consider four datasets from different domains, including shopping (NLPCC2014-SC), electronics (SE-ABSA16_PHNS, SE-ABSA16_CAM), and financial (BDCI2019). ERNIE 3.0 Titan achieves state-ofthe-art results on all four datasets.

Opinion Extraction. Similar to the sentiment analysis task, opinion extraction requires the model to mine the opinion of a sentence. We use two sub-datasets from Chinese Customer Review (COTE). Experiment results show that ERNIE 3.0 Titan also outperforms the current SoTA system.
Relation Extraction. The relation extraction task is to identify the relationship between different entities like persons and organizations. We consider FinRE and SanWen – two relation extraction datasets for financial news and Chinese literature, respectively. ERNIE 3.0 Titan outperforms the previous SoTA model by a remarkable margin. 10The previous SoTA results of ERNIE 2.0 and RoBERTa-wwm-ext on corresponding datasets are reproduced by ourselves, except for the datasets that already have released pre-trained results. 

11 http://tcci.ccf.org.cn/conference/2014/pages/page04_dg.html 
12 http://alt.qcri.org/semeval2016/task5/ 
13 https://www.datafountain.cn/competitions/350 
14 https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset 
15 http://thuctc.thunlp.org/ 
16 http://tcci.ccf.org.cn/conference/2016/dldoc/evagline2.pdf 
17 https://github.com/pkumod/CKBQA 
18 https://aistudio.baidu.com/aistudio/competition/detail/49/?isFromLUGE=TRUE 

Semantic Similarity. Semantic Similarity is a classic NLP task that determines the similarity between various terms such as words, sentences, documents. In this work, we focus on sentence-level similarity tasks. We test ERNIE 3.0

Titan on several datasets in varied fields, including AFQMC, LCQMC, and BQ. Experiment results show that ERNIE

3.0 Titan outperforms the baseline models by a visible margin.
Text Classification. We also evaluate ERNIE 3.0 Titan on Text classification. We consider four datasets: app descriptions (IFLYTEK) and news stories (THUCNEWS, CNSE, CNSS). Under different types of classification tasks, ERNIE 3.0 Titan can consistently achieve better accuracy.

Closed-Book Question Answering. Closed-Book Question Answering aims to directly answer the questions without any additional references or knowledge. We select a general QA dataset NLPCC-DBQA and two medical field datasets – cMedQA and cMedQA2 to test the ability of ERNIE 3.0 Titan. Experiment results show that ERNIE 3.0 Titan performs better on all QA tasks. We believe knowledge-enhanced pre-training methods do bring benefits to the closed-book QA task.

Cant Understanding. Cant, also known as doublespeak, is an advanced language usage for humans. However, it is rather difficult for machines to understand this type of language. We test the cant understanding ability of ERNIE 3.0 Titan on DogWhistle – a dataset based on Decrypto game. The model is required to select the right answer with the guidance of the corresponding cant. ERNIE 3.0 Titan gets the best result and shows its potential for understanding more difficult languages.

Cloze and completion. Cloze tests require the ability to understand context and vocabulary in order to identify the correct language that belongs in the deleted passages. Benefiting from the knowledge-enhanced pre-training, ERNIE 3.0 Titan achieves the best score among baselines.
Machine Reading Comprehension. We comprehensively evaluate the ability of ERNIE 3.0 Titan on machine reading comprehension in different aspects, including span-predict reading comprehension (DuReader, DRCD, DuReaderchecklist), multiple-choice reading comprehension (C3, DuReaderyesno), and robustness test (Dureaderrobust).

With the help of knowledge-enhanced pre-training, ERNIE 3.0 Titan surpasses the baseline models with significant enhancements on all types of tasks.

Legal Documents Analysis. Next, two domain-specific tasks of law are chosen to evaluate the ability of ERNIE 3.0 Titan on document analysis. These two datasets from CAIL2018 are both multi-label document classification tasks.

ERNIE 3.0 Titan breaks the previous SoTA performance.

### 6.3 Experiments on Few-shot Learning
#### 6.3.1 Settings
In this section, we evaluate the few-shot performance of ERNIE 3.0 Titan on the FewCLUE benchmark. FewCLUE is a comprehensive few-shot evaluation benchmark in Chinese, including various of tasks. Flowing Section 6.1, we categorize these task into six different types including text classification, natural language understanding etc. Each task provides five training/evaluation splits and a corresponding union set (called train_all and dev_all). The number of the few-shot training samples is related to the number of the label categories. In other words, tasks with more classes have more training samples. We choose mT5-XXL, Yuan 13B-PLM, and ERNIE 3.0 as our baselines. For mT5-XXL, the results are produced based on the open-source codes with their default tuning method, and for Yuan 13B-PLM, we directly use the published results.

We tested three approaches to train the few-shot learners based on ERNIE 3.0 Titan. For different types of tasks, we use the corresponding task-specific training method:
* For text classification, sentiment analysis, and semantic similarity tasks, we use traditional fine-tuning methods.
* For reading comprehension (CHID-FC) and winograd schema challenge task, we utilize pattern exploiting training [99] to reformulate such tasks cloze-style questions and perform gradient-based tuning.
* For the natural language inference task, we exploit NSP-based prompt training [100]. Different labels are regarded as prompts to concatenate the two sentences, and the model is trained to select the label that makes the concatenated sentence the most coherent. The classification network of the sentence distance task (Section 3.2.2) in the pre-training phase can be seen as a good initialization to train the coherent ranker.
We use the union set to train the few-shot learner and report the results on the evaluation sets (dev_all) and public test sets to conduct the experiments. 


Table 2: Results on Natural Language Understanding Tasks. We compare ERNIE 3.0 Titan with ERNIE 3.0 and 10 previous SoTA baselines including CPM-2 [28], ERNIE 2.0 [52], SKEP [93], RoBERTa-wwm-ext-large [94] (marked as RoBERTa*), ALBERT [95], MacBERT [96], Zen 2.0 [97] and crossed BERT siamese BiGRU [98] (marked as BERT_BiGRU*). 15


Table 4: Results on zero-shot learning tasks. We reported the results on the dev-all set.

#### 6.3.2 Results
The main results of the few-shot learning tasks are illustrated in Table 3. ERNIE 3.0 Titan consistently outperforms baseline models, including ERNIE 3.0. Under vanilla fine-tuning methods, ERNIE 3.0 Titan still can surpass Yuan 1.0 4.58% average points on the text classification task, 1.56% on sentence similarity, and 0.62% on semantic similarity tasks. The task of reading comprehension and winograd schema challenge can be naturally reformulated to cloze-style tasks. Under prompt-based pattern exploiting, ERNIE 3.0 Titan outperforms Yuan 1.0 by a remarkable margin: 8.18% absolute improvement on CLUEWSC-FC and 7.92% on CHID-FC dataset. For natural language inference task, ERNIE3.0 Titan also achieves a significant improvement of 6.87 points.


Probability Form Implementation Example

P(x, y) arg maxi P lij=0 log P  ffill x0 ,yi j |ffill x0 ,yi <j li x0 = hypothesis?yi , premise. yi ∈ {Yes,No,Maybe} P(y|x) arg maxi P |yi| j=0 log P  yji |ffill x0 ,yi <j |yi| x0 = News:x. This news is about yi. yi ∈ {culture, sports, tech., etc.} P(x|y) arg maxi P |x0 | j=0 log P  x0 j |ffill  x0 <j , yi  x0 = This news is about yi . News:x. yi ∈ {culture, sports, tech., etc.} P (y|x) P (y) arg maxi P |yi| j=0 log P  yji |ffill x0 ,yi <j − P |yi| j=0 log P  yji |ffill ∅0 ,yi <j |yi| x0 = News:x. This news is about yi. ∅0 = News:∅. This news is about yi. yi ∈ {culture, sports, tech., etc.} P(True|x, y) arg maxi P  True|ffill  x0 , yi  x0 = [CLS] hypothesis? [SEP] yi , premise. yi ∈ {Yes,No,Maybe}

Table 5: Notations and scoring functions for zero-shot learning.

### 6.4 Experiments on Zero-shot Learning
This section conducts various types of tasks with the zero-shot setting where a model is applied without parameter updates. ERNIE 3.0 Titan achieves strong performance compared to recently proposed large-scale Chinese language models such as CPM-1 (2.6B), PanGu-α, Yuan 1.0 on all downstream tasks. On the CKBQA-sub dataset, which requires strong knowledge reasoning ability, ERNIE 3.0 Titan surpassed GPT-3 by over 8 point percent with respect to accuracy. In our case study (Sec. 6.4.3), we demonstrate the ability of ERNIE 3.0 Titan to generate controllable and credible results. Quantitatively, we evaluated ERNIE 3.0 with baselines on our manually collected 467 cases across 13 different tasks and showed that it could generate more coherent, natural, and accurate responses.

#### 6.4.1 Evaluation Methods
This section unified five probability forms of the scoring function for tasks with a limited label set, such as text classification, sentiment analysis, and cloze-style tasks. Based on the unified ERNIE 3.0 framework, the implementation of these five scoring functions can be task-specific. The ablation study in Sec. 6.6.1 shows the effect of different scoring functions where some can obtain stable performance gain over others on a certain task type.

In Table. 5, five forms of the scoring function are shown where fprompt(·) is the function to prompt the input x to x0 , ffill(·) is the function to fill the ith label into prompted text x0 , yi ∈ Y, Y is the label set and |yi|, li denote the tokens length for label yi and the filled prompt respectively. We will introduce these scoring functions as follows:
* P(x, y). The implementation of the joint probability P(x, y) is equivalent to the perplexity of the prompted text, meaning that the prompted text with the lowest perplexity score will be predicted as the correct answer.
* P(y|x). Given the input text x, we will choose the highest probability answer among all possible labels. To ease the effect of the label length bias, length normalization is commonly used. This method is also called the average log-likelihood used in GPT-3 [1].
* P(x|y) is the reverse version of P(y|x). The above two forms explicitly include the label’s probability, which ignores the effect of the label bias. Since the label is imbalanced distributed in the pre-training corpus, the model tends to assign a higher probability to labels common in the corpus. By conditioning on y, we assume the impact of the label bias will be flattened over the input tokens.
* P (y|x) P (y) is proportional to the pointwise mutual information (PMI) of x, y. PMI has been used for finding collocations and associations between words. Compared to P(y|x), we can think of P (y|x) P (y) as a way to eliminate the effect of label bias, since common labels with high probability will decrease the final score through division. In practice, it is better to restrain the P(y) in the target domain by prompting the text using a domain-specific prompt but taking as input an empty input which is used in [101].
* P(True|x, y) formalizes each possible answer with the input text as a binary classification task. In this way, the next sentence prediction (NSP) task 19 can be utilized, which has been pre-trained on the hidden state of 19In ERNIE framework, we use the sentence distance prediction task which is an extension of the traditional next sentence prediction (NSP) task (introduced in Sec. 3.2.2). 


Table 6: Prompts and evaluation methods of ERNIE 3.0 Titan used in zero-shot learning tasks. [CLS]. Intuitively, NSP task estimates the affinity score of two sentences. Thus, we can prompt the text into two sentences where one is filled with different labels.

For generative tasks such as machine reading comprehension, ERNIE 3.0 used a restrained beam search with a beam width of 8 for extractive MRC to ensure the generation is a span that occurred in the context. Though effective, the beam search is time-consuming for ERNIE 3.0 Titan. Since ERNIE 3.0 Titan is assumed to be more powerful, we used the top-1 sampling strategy for all generation tasks.

#### 6.4.2 Results
In Table. 6, we summarized the prompting functions, label verbalizers and evaluation methods we used for each task.

The detailed results will be introduced as follows:

Text Classification. For the TNEWS and IFLYTEK datasets, we randomly sample three candidates as negative labels for each sample. This sampling strategy is aligned with CPM-1’s, PanGu-α’s, and ERNIE 3.0’s to make fair comparisons. While for TNEWS-FC, IFLYTEK-FC, CSLDCP, the negative sampling strategy is discarded in order to compare with Yuan 1.0. ERNIE 3.0 outperforms competitive baselines on these tasks.

Sentiment analysis. On the EPRSTMT dataset, ERNIE 3.0 Titan achieves 88.75% w.r.t. accuracy on the zero-shot setting, meaning sentiment analysis is a simple task for large-scale pre-trained models.

Semantic Similarity. We consider AFQMC, CSL, CSL-FC, and BUSTM datasets. ERNIE 3.0 Titan outperforms baselines at a large margin. However, compared to ERNIE 3.0, the performance gain on AFQMC and CSL is marginal, and the minor improvement on CSL comes from the soft prompt tokens [webN]. It means that the soft prompt tokens appended before the original text help the model utilize the knowledge in a specific domain. On the other hand, it is hard for models to learn the semantic similarity based only on a language modeling loss. While on BUSTM, ERNIE

#3.0 Titan using the NSP task as the scoring function surpassed Yuan 1.0 by 5% point. We assume the inductive bias of
 the NSP task helps the model learn semantic similarity.

Natural Language Inference. ERNIE 3.0 Titan is evaluated on three NLI datasets, namely OCNLI, OCNLI-FC, and

CMNLI, and achieves the best performance. We used the P(True|x, y) scoring function for OCNLI-FC since we found that the NSP task shows some capability in modeling the semantic similarity on the BUSTM dataset. We tested different soft prompt tokens for CMNLI including [webN], [qaN] and [novelN]. Soft prompt tokens [novelN] achieves the highest score (51.70) compared to 49.87 with [webN] and 49.4 with [qaN]. And, there is still a large room for improvement for pre-trained models on zero-shot NLI tasks. 18

Type Task (# of cases) CPM-1 PLUG PanGu-α ERNIE 3.0 ERNIE 3.0 Titan


Table 7: The zero-shot generation performance manually evaluated on our collected 467 cases. (we reported the average score of coherence, fluency, and accuracy respectively on a scale of [0, 1, 2])

Winograd Schema Challenge: We formalize the CLUEWSC dataset as a multi-choice completion task where a pronoun is replaced with each candidate to calculate the scores. Since there are no candidates set for the CLUEWSC-FC dataset, we come up with a superior prompt by appending a complement after the target pronoun, and the model is required to judge the correctness of the complement. ERNIE 3.0 Titan surpassed Yuan 1.0 a lot on the CLUEWSC-FC dataset and achieved superior performance on CLUEWSC attributing to the power of scale.

Cloze and completion. We split a sample containing multiple blanks as multiple sentences to predict independently on the CHID dataset. The Hungarian algorithm [102] is used to ensure that two blanks in one sample have unique predictions. ERNIE 3.0 Titan achieves the best score among baselines, and the Hungarian algorithm contributes a lot improving from 77.32 to 86.21. For Chinese Word Prediction with Long Context (Chinese WPLC), a sample consists of a masked text and a correct word. Following PanGu-α, we replace the mask token with the correct word and calculate the perplexity score of a whole sentence. ERNIE 3.0 Titan achieves a much lower perplexity score (16.50) with the help of soft prompt tokens [novelN]. On the CMRC2019 dataset, we randomly sample three negative candidates for each blank from the original candidates, then beam search is applied to calculate the optimal path for a sample. We also formalize the PD, CFT, and CMRC2017 as multi-choice tasks where multiple choices are the words appearing in the given text. For efficiency, restricted generation [2] is used for these three datasets. ERNIE 3.0 Titan surpassed the baselines with a large margin.

Machine Reading Comprehension. We consider four MRC datasets. Due to the power of ERNIE 3.0 Titan, we simply utilized the top-1 sampling strategy to generate answers. The maximum generated length of completion is limited by a pre-defined number based on 95% percentile point of answers’ length on the dataset. The performance of ERNIE 3.0

Titan is superior, outperforming Yuan 1.0 with a comparable number of parameters by 3.03% point on average for the

CMRC2018 dataset.

Closed-book Question Answering. We evaluated ERNIE 3.0 on two Closed-book Question Answering datasets which require the model to generate answers using its inherent knowledge learned during pre-training. WebQA is a large-scale real-world QA dataset from Baidu Zhidao. CKBQA is a knowledge-based question answering task. We only provide

ERNIE 3.0 Titan with the question without additional evidence. In addition, we evaluated GPT-3 on a subset of the

CKBQA dataset where questions requiring the background knowledge of China are filtered out, and then questions are manually translated into English. The engine for GPT-3 is Davinci, and the object is text completion. ERNIE 3.0 Titan significantly outperforms baselines and exceeds GPT-3 by over 8% point, indicating that ERNIE 3.0 Titan is superior in learning and reasoning.

#### 6.4.3 Case Study
We manually collected 467 cases 20 to evaluate the zero-shot generation ability of current large-scale pre-trained models on 13 tasks from 5 different types including Question Answering, Interpretation, Dialogue, Text Generation and

Summarization. In human evaluation, the annotators are asked to score the generation quality on a scale of [0, 1, 2]. We reported the average score of coherence, fluency, and accuracy in Tab. 7, 8, and showed some controllable generations of ERNIE 3.0 Titan in Tab. 9. In addition, we construct a subset of above manually collected cases where cases requiring 20https://ernie-github.cdn.bcebos.com/cases.xlsx 19


Table 8: The zero-shot generation performance between GPT-3 and our model on a subset of manually collected cases. (Cases requiring background knowledge about Chinese history, geography, and culture have been removed. We reported the average score of coherence, fluency, and accuracy respectively on a scale of [0, 1, 2]).

Input (Genre:News) [News-0] [News-1] ... [News-63]

Generation 一、二级公立医疗服务体系建设的关键是建立起有效率的服务体系,即能够满足群众的 需要,而不是单纯的追求最大的经济利益。因此要重视医德 和业务素质的培养,要求医 生要有爱心,要有责任心,要有耐心。

The key to the construction of the first and second level public medical service system is to establish an efficient service system, that is, to meet the needs of the public, rather than simply pursuing the greatest economic benefits. Therefore, we should pay attention to the training of medical ethics and business quality, requiring doctors to be caring, responsible, and patient.

Input (Genre:QA) [QA-0] [QA-1] ... [QA-63]

Generation 如何让手机的图片变得好看？1.美图秀秀软件 2.photoshop软件 3.snapseed软件

How to make the pictures on your phone look good? 1. Meitu software 2. Photoshop software 3. Snapseed software
Input (Genre:Novel) [Novel-0][Novel-1] ... [Novel-63]

Generation “不过是一个人，还不至于如此小心。”李青阳看着手中的那颗丹药说道。 "It’s just one person, no need to be so careful." Qingyang Li looked at the pill in his hand and said.

Input (Topic:Digital) [t] 数码 [/t]

Generation iphone12pro max可以插两张卡吗? 可以。

Does iPhone 12 Pro Max support Dual SIM? Yes.

Input (Topic:Astrology Horoscope) [t] 星座运势 [/t]

Generation 天秤座的人，在12月底，财运上上签，运势如虹，财源广进。

The Libra, at the end of December, is on a financial upswing, with a wealthy and prosperous fortune.

Input (Keywords:Desert) [k] 沙漠 [/k]

Generation 沙漠之心在我的世界里面可是一个稀有物品啊，那可是只有在沙漠中才能找到。

The Desert Heart is a rare item in the Minecraft, which can only be found in the desert.

Input (Keywords:Beijing, Winter) [k] 北京,冬天 [/k]

Generation 北京的冬天太长了，这个星期都没有太阳，还不能晒被子，只能在宿舍里面躺着。

The winter in Beijing is too long, it’s cloudy this week. You can’t hang out the quilts, so just lounge around in the dorm.

Input (Genre:QA; Keywords:Beijing, Winter) [QA-0] [QA-1] ... [QA-63] [k] 北京,冬天 [/k]

Generation 北京冬天有哪些好吃的美食推荐呢？北京的冬天最好吃的当然是涮羊肉啦，羊肉的营 养价值可是很高的。

What are some of the best winter food recommendations in Beijing? The best thing to eat in

Beijing in winter is instant-boiled mutton slices, which has a very high nutritional value.

Input (Genre:Novel; Keywords:Beijing, Winter) [Novel-0][Novel-1] ... [Novel-63] [k] 北京,冬天 [/k]

Generation 北平的冬天很干燥，尤其是在城南，城南的一户人家，门口放着一架风车，风车吱吱 地转着，风里透着一股冷气。

It’ s a dry winter in Peking, especially in the south of the city. In a house in the south of the city, a windmill was placed at the door, and the windmill creaked and turned with a cold wind.

Input (Genre:Novel; Topic:Pet; Keywords:Beijing, Winter) [QA-0] [QA-1] ... [QA-63] [t] 宠物 [/t] [k] 北京,冬天 [/k]

Generation 北京哪里可以找一只狗做宠物，冬天太难熬，想找一只狗陪我度过寒冬。

Where to find a dog as a pet in Beijing, it’ s too hard to survive the winter, I want to find a dog to spend the winter with me.

Table 9: Illustrations of controllable generations from ERNIE 3.0 Titan. 

Table 10: Scoring details for zero-shot generation. background knowledge about Chinese history, geography, and culture have been removed. Overall, ERNIE 3.0 Titan can generate the most coherent, fluent and accurate texts on average as compared to CPM-1, PLUG, PanGu-α, ERNIE 3.0 and GPT-3 21, and users can combine different attributes to generate highly credible and controllable generations.
The introduction of three scoring metrics are listed as follows, and the scoring details are provided in Tab. 10.
* Coherence measures whether the generation is relevant and consistent with the context.
* Fluency evaluates whether the generated text is natural or readable. A fluent text should have no semantic contradiction among the generated text.
* Accuracy is a metric to evaluate whether the generated text is the same as the ground truth.

### 6.5 Experiments on Model Distillation
This section discusses the experiment for our task-agnostic model distillation framework for ERNIE 3.0 Titan. We evaluate our distilled ERNIE 3.0 Titan on five typical types of downstream tasks, including natural language inference (XNLI [103]), semantic analysis (ChnSentiCorp [104]), document question answering (NLPCC-DBQA22), semantic similarity (LCQMC [73]), and machine reading comprehension (CMRC2018 [105]).

To reduce the gap between the giant teacher model and the student models, we introduce a 24-layers TA model with a hidden size of 1024. To accelerate the distillation procedure, we also pre-train the TA model for 500K steps before starting distillation with the same pre-training setting as ERNIE 3.0 Titan. For ALD settings, we add one extra layer for each student. Then we jointly train the teacher and students with OFD. We introduce five students with different model sizes and parameters ranging from 14M to 110M. We measure the inference latency of these student models on a V100 GPU with PaddlePaddle and show the relative speedup with respect to BERT-Base in Table 11.

Figure 7: Perplexity variation of the mask language model task with respect to training steps.

We compare our downstream fine-tuning results with other compact PLMs. We choose the Chinese version BERT [6], TinyBERT [8], ERNIE 2.0 [52], and RoBERTa-wwm-ext [106] as baseline models.

BERT, RoBERTa-wwm-ext and ERNIE 2.0 are PLMs pre-trained from scratch without any distillation. TinyBERT is pre-trained via a multi-step distillation procedure using the task-specific distillation paradigm. Results are shown in Table 11, the experiments are reported with means over five random initialization.

As shown in Table 11, the 12L768H student model achieves SOTA results on all tasks. By taking a closer look at the EM and F1 for

CMRC2018, this student model surpasses the strong baseline ERNIE 2.0 by 4.78 and 2.82, respectively. Notably, among all 6-layer PLMs, the 6L768H version of distilled ERNIE 3.0 Titan performs the best and even outperforms the 12-layer BERT-Base on XNLI, LCQMC, and NLPCC-DBQA. Comparison between these 6-layer students and TinyBERT also demonstrate the effectiveness of our proposed framework. 

21 We use the implementation of CPM-1 in https://github.com/jm12138/CPM-Generate-Paddle, PLUG in https://nl p.aliyun.com/portal?/BigText_chinese#/BigText_chinese, PanGu-α in https://git.openi.org.cn/PCL-Platfor m.Intelligence/PanGu-Alpha, ERNIE 3.0 in https://wenxin.baidu.com/wenxin/ernie and GPT-3 in https://beta.o penai.com/docs/guides/completion 
22 http://tcci.ccf.org.cn/conference/2016/dldoc/evagline2.pdf 


Table 11: Results for model compression on Chinese tasks.

Table 12: Comparison of different scoring functions using ERNIE 3.0 Titan on zero-shot learning tasks. uni, bi means the unidirectional and bi-directional attention used to calculate the score.

### 6.6 Analysis
#### 6.6.1 The Effect of Different Scoring Functions
In Table. 12, we compare the performance of different scoring functions on five zero-shot tasks. Overall, P(y|x)/P(y)- bi is more amenable to text classification tasks, while semantic similarity and natural language inference tasks prefer P(True|x, y)-bi. We put forward a hypothesis that the effectiveness of P(y|x)/P(y)-bi is positively correlated with the size of the label set. On IFLYEK-ZC with 119 classes, ERNIE 3.0 Titan obtains 9.68% point (30.74 → 40.42) performance gain compared to 4.28% point (53.55 → 57.83) on TNEWS-FC with 15 classed when eliminating the label bias effect through the division of P(y). P(y|x)/P(y)-bi even fails on BUSTM with 2 classes and OCNLI-FC with 3 classes compared to the second-best scoring function P(x|y)-uni. Intuitively, the dataset with a larger label set is more likely to be affected by the label bias. The model prefers frequent answers in the pre-training dataset, conflicting with the balanced label distribution in the downstream dataset. P(True|x, y)-bi utilizes the inherently pre-trained

NSP-task, which is suitable for tasks that need to distinguish the semantic similarity between two sentences. The best performance always achieves using the scoring function with bidirectional attention. When comparing P(y|x)-uni with P(y|x)-bi, P(y|x)-bi could additionally utilize the information from the text behind the label resulting in better performance among all datasets.

#### 6.6.2 Adversarial Credibility Classification
We notice that the adversarial credibility classification method mentioned in 3.2.3 not only can filter out low credibility texts during the generation but also can speed up the pre-training convergence. We conduct a comparative experiment with the base model settings (12 layers, 768 dims, 12 attention heads) and report the results in Figure 7. The pre-training tasks of our baseline strictly follow the settings of the original ERNIE 3.0, while the contrast model has an additional self-supervised adversarial loss for credibility classification.

Figure 7 illustrates the perplexity variation of the masked language model task during the pre-training process. The model with the auxiliary adversarial loss reaches a higher convergence speed. We think this might because the adversarial loss of distinguishing whether a text is generated or the original one can help the model learn the distribution of true natural sentences.

Figure 8: The number of generated tokens controlled by the length attribute.

In ERNIE 3.0 Titan framework, we have introduced five different controllable attributes, including genre, topic, keywords, sentiment, and length in Sec. 3.3. By assembling different attributes, users can have more access to ERNIE 3.0 Titan to obtain diverse and controllable generations shown in Table. 9. Meanwhile, the genre soft prompts improve the performance on some zero-shot tasks such as CHID, CMNLI, and WPLC. We assume the improvement comes from the domain calibration. For example, [novelN] soft prompts are prefix of novel texts when optimizing the language modeling loss. When conditioning on the [novelN] soft prompts, the output distribution of the model will be shifted to the novel domain which results in the lower perplexity score on WPLC dataset. In addition, we test the effect of the length attribute showing that the number of actual generated tokens has a positive correlation with the expected generated tokens in Figure. 8. Also, we observe that the length attribute affects the genre of generations. ERNIE 3.0 Titan tends to generate texts constructed from knowledge graph (like The capital of China is Beijing.) when the length attribute is small and prefers novel and web texts when the length attribute is large. 

## 7 Conclusion
We pre-train a knowledge-enhanced language model with 260 billion parameters named ERNIE 3.0 Titan based on the ERNIE 3.0 framework. It is the largest Chinese dense pre-training model as far as we know. We have validated it on 68 datasets, and the results show that ERNIE 3.0 Titan achieves new state-of-the-art results. In addition, We propose a novel method for users to control the generation result and obtain the result factually consistent with the real world. We also devise an online distillation framework and conduct several distilled models of different sizes concerning the computation overhead of large-scale pre-training models. In the next stage, we will continually update ERNIE 3.0 Titan with more data to further explore the limit of the performance of large-scale pre-trained language models. We will also endeavor to explore the potential of knowledge-enhanced large-scale multi-modal models for more and various tasks.

我们基于 ERNIE 3.0 框架预训练了一个名为 ERNIE 3.0 Titan 的具有 2600 亿个参数的知识增强型语言模型。 据我们所知，它是最大的中文密集预训练模型。 我们已经在 68 个数据集上对其进行了验证，结果表明 ERNIE 3.0 Titan 取得了新的最先进的结果。 此外，我们提出了一种新的方法，供用户控制生成结果，并获得与现实世界事实一致的结果。 我们还设计了一个在线蒸馏框架，并针对大规模预训练模型的计算开销进行了几个不同大小的蒸馏模型。 下一阶段，我们将用更多的数据不断更新ERNIE 3.0 Titan，进一步探索大规模预训练语言模型的性能极限。 我们还将努力探索知识增强的大规模多模态模型在更多和各种任务中的潜力。

## References
1. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020.
2. Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, Weixin Liu, Zhihua Wu, Weibao Gong, Jianzhong Liang, Zhizhou Shang, Peng Sun, Wei Liu, Xuan Ouyang, Dianhai Yu, Hao Tian, Hua Wu, and Haifeng Wang. Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation. arXiv preprint arXiv:2107.02137, 2021.
3. Yanjun Ma, Dianhai Yu, Tian Wu, and Haifeng Wang. Paddlepaddle: An open-source deep learning platform from industrial practice. Frontiers of Data and Domputing, 1(1):105–115, 2019.
4. Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.
5. Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. URL https://s3-us-west-2.amazonaws.com/openai-assets/researchcovers/languageunsupervised/language understanding paper. pdf, 2018.
6. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
7. Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, and Hua Wu. Ernie: Enhanced representation through knowledge integration. arXiv preprint arXiv:1904.09223, 2019. 23 of generated tokens (actual)
8. Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. Tinybert: Distilling BERT for natural language understanding. In Trevor Cohn, Yulan He, and Yang Liu, editors, Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, volume EMNLP 2020 of Findings of ACL, pages 4163–4174. Association for Computational Linguistics, 2020.
9. Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.
10. Wenhui Wang, Hangbo Bao, Shaohan Huang, Li Dong, and Furu Wei. Minilmv2: Multi-head self-attention relation distillation for compressing pretrained transformers. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, volume ACL/IJCNLP 2021 of Findings of ACL, pages 2140–2151. Association for Computational Linguistics, 2021.
11. Weiyue Su, Xuyi Chen, Shikun Feng, Jiaxiang Liu, Weixin Liu, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Ernie-tiny : A progressive distillation framework for pretrained transformer compression. CoRR, abs/2106.02241, 2021.
12. David A. Patterson, Joseph Gonzalez, Quoc V. Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David R. So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. CoRR, abs/2104.10350, 2021.
13. Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant: Bridging the gap between student and teacher. CoRR, abs/1902.03393, 2019.
14. Jang Hyun Cho and Bharath Hariharan. On the efficacy of knowledge distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4794–4802, 2019.
15. Xiao Jin, Baoyun Peng, Yichao Wu, Yu Liu, Jiaheng Liu, Ding Liang, Junjie Yan, and Xiaolin Hu. Knowledge distillation via route constrained optimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1345–1354, 2019.
16. Wenxian Shi, Yuxuan Song, Hao Zhou, Bohan Li, and Lei Li. Follow your path: a progressive method for knowledge distillation. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 596–611. Springer, 2021.
17. Jia Guo, Minghao Chen, Yao Hu, Chen Zhu, Xiaofei He, and Deng Cai. Reducing the teacher-student gap via spherical knowledge disitllation. arXiv preprint arXiv:2010.07485, 2020.
18. Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. CoRR, abs/1909.08053, 2019.
19. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. CoRR, abs/1910.10683, 2019.
20. William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. CoRR, abs/2101.03961, 2021.
21. Jurassic-1: Technical details and evaluation. https://uploads-ssl.webflow.com/60fd4503684b466578 c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf.
22. Scaling language models: Methods, analysis & insights from training gopher. https://deepmind.com/resea rch/publications/2021/scaling-language-models-methods-analysis-insights-from-train ing-gopher.
23. Using deepspeed and megatron to train megatron-turing nlg 530b, the world’s largest and most powerful generative language mode. https://developer.nvidia.com/blog/using-deepspeed-and-megatron -to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative -language-model/.
24. Wei Zeng, Xiaozhe Ren, Teng Su, Hui Wang, Yi Liao, Zhiwei Wang, Xin Jiang, ZhenZhang Yang, Kaisheng Wang, Xiaoda Zhang, et al. Pangu-alpha: Large-scale autoregressive pretrained chinese language models with auto-parallel computation. arXiv preprint arXiv:2104.12369, 2021. 24
25. Shaohua Wu, Xudong Zhao, Tong Yu, Rongguo Zhang, Chong Shen, Hongli Liu, Feng Li, Hong Zhu, Jiangang Luo, Liang Xu, et al. Yuan 1.0: Large-scale pre-trained language model in zero-shot and few-shot learning. arXiv preprint arXiv:2110.04725, 2021.
26. Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. arXiv preprint arXiv:2103.16716, 2021.
27. Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, and Jason Weston. Hash layers for large sparse models. arXiv preprint arXiv:2106.04426, 2021.
28. Zhengyan Zhang, Yuxian Gu, Xu Han, Shengqi Chen, Chaojun Xiao, Zhenbo Sun, Yuan Yao, Fanchao Qi, Jian Guan, Pei Ke, et al. Cpm-2: Large-scale cost-effective pre-trained language models. arXiv preprint arXiv:2106.10715, 2021.
29. Junyang Lin, An Yang, Jinze Bai, Chang Zhou, Le Jiang, Xianyan Jia, Ang Wang, Jie Zhang, Yong Li, Wei Lin, et al. M6-10t: A sharing-delinking paradigm for efficient multi-trillion parameter pretraining. arXiv preprint arXiv:2110.03888, 2021.
30. Matthew Hutson. Robo-writers: the rise and risks of language-generating ai. Website, 2021. https://www.na ture.com/articles/d41586-021-00530-0.
31. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
32. Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng Chen. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.
33. Zhuohan Li, Siyuan Zhuang, Shiyuan Guo, Danyang Zhuo, Hao Zhang, Dawn Song, and Ion Stoica. TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models.
34. Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. Efficient large-scale language model training on gpu clusters. arXiv preprint arXiv:2104.04473, 2021.
35. Lili Yao, Nanyun Peng, Ralph Weischedel, Kevin Knight, Dongyan Zhao, and Rui Yan. Plan-and-write: Towards better automatic storytelling. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 7378–7385, 2019.
36. Zeqiu Wu, Michel Galley, Chris Brockett, Yizhe Zhang, Xiang Gao, Chris Quirk, Rik Koncel-Kedziorski, Jianfeng Gao, Hannaneh Hajishirzi, Mari Ostendorf, et al. A controllable model of grounded response generation. arXiv preprint arXiv:2005.00613, 2020.
37. Sarah Kreps, R Miles McCain, and Miles Brundage. All the news that’s fit to fabricate: Ai-generated text as a tool of media misinformation. Journal of Experimental Political Science, pages 1–14, 2020.
38. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Blog, 1(8), 2019.
39. Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, and Yejin Choi. Defending against neural fake news. arXiv preprint arXiv:1905.12616, 2019.
40. Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and Richard Socher. Ctrl: A conditional transformer language model for controllable generation. arXiv: Computation and Language, 2019.
41. Benjamin Schiller, Johannes Daxenberger, and Iryna Gurevych. Aspect-controlled neural argument generation. arXiv: Computation and Language, 2020.
42. Alexis Ross, Tongshuang Wu, Hao Peng, Matthew E. Peters, and Matt Gardner. Tailor: Generating and perturbing text with semantic controls. arXiv: Computation and Language, 2021.
43. Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. Plug and play language models: A simple approach to controlled text generation. arXiv: Computation and Language, 2019.
44. Alvin Chan, Yew-Soon Ong, Bill Pung, Aston Zhang, and Jie Fu. Cocon: A self-supervised approach for controlled text generation. arXiv: Computation and Language, 2020.
45. Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient knowledge distillation for BERT model compression. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4323–4332, Hong Kong, China, November 2019. Association for Computational Linguistics. 25
46. Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. MobileBERT: a compact task-agnostic BERT for resource-limited devices. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2158–2170, Online, July 2020. Association for Computational Linguistics.
47. Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-read students learn better: On the importance of pre-training compact models. arXiv preprint arXiv:1908.08962, 2019.
48. Xingkai Ren, Ronghua Shi, and Fangfang Li. Distill bert to traditional models in chinese machine reading comprehension (student abstract). In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 13901–13902, 2020.
49. Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 14014–14024, 2019.
50. Aref Jafari, Mehdi Rezagholizadeh, Pranav Sharma, and Ali Ghodsi. Annealing knowledge distillation. arXiv preprint arXiv:2104.07163, 2021.
51. Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke Zettlemoyer, and Omer Levy. Spanbert: Improving pre-training by representing and predicting spans. Transactions of the Association for Computational Linguistics, 8:64–77, 2020.
52. Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Hao Tian, Hua Wu, and Haifeng Wang. Ernie 2.0: A continual pre-training framework for language understanding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8968–8975, 2020.
53. Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.
54. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv:1906.08237, 2019.
55. He Bai, Peng Shi, Jimmy Lin, Luchen Tan, Kun Xiong, Wen Gao, and Ming Li. Segabert: Pre-training of segment-aware BERT for language understanding. CoRR, abs/2004.14996, 2020.
56. Siyu Ding, Junyuan Shang, Shuohuan Wang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Ernie-doc: The retrospective long-document modeling transformer. arXiv preprint arXiv:2012.15688, 2020.
57. Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.
58. Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.
59. Siyu Ding, Junyuan Shang, Shuohuan Wang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. ERNIE-DOC: the retrospective long-document modeling transformer. CoRR, abs/2012.15688, 2020.
60. Alexander Sergeev and Mike Del Balso. Horovod: fast and easy distributed deep learning in tensorflow. arXiv preprint arXiv:1802.05799, 2018.
61. Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, and Soumith Chintala. PyTorch distributed: Experiences on accelerating data parallel training. 13(12):3005–3018.
62. Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin Tran, Ashish Vaswani, Penporn Koanantakool, Peter Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff Young, et al. Mesh-tensorflow: Deep learning for supercomputers. arXiv preprint arXiv:1811.02084, 2018.
63. Aaron Harlap, Deepak Narayanan, Amar Phanishayee, Vivek Seshadri, Nikhil Devanur, Greg Ganger, and Phil Gibbons. Pipedream: Fast and efficient pipeline parallel dnn training. arXiv preprint arXiv:1806.03377, 2018.
64. Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, Phillip B. Gibbons, and Matei Zaharia. Pipedream: Generalized pipeline parallelism for dnn training. SOSP ’19, page 1–15, New York, NY, USA, 2019. Association for Computing Machinery.
65. Deepak Narayanan, Amar Phanishayee, Kaiyu Shi, Xie Chen, and Matei Zaharia. Memory-Efficient PipelineParallel DNN Training. 26
66. Yulong Ao, Zhihua Wu, Dianhai Yu, Weibao Gong, Zhiqing Kui, Minxu Zhang, Zilingfeng Ye, Liang Shen, Yanjun Ma, Tian Wu, Haifeng Wang, Wei Zeng, and Chao Yang. End-to-end adaptive distributed training on paddlepaddle, 2021.
67. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.
68. Liang Xu, Xiaojing Lu, Chenyang Yuan, Xuanwei Zhang, Huilin Xu, Hu Yuan, Guoao Wei, Xiang Pan, Xin Tian, Libo Qin, and Hu Hai. Fewclue: A chinese few-shot learning evaluation benchmark, 2021.
69. Yanzeng Li, Tingwen Liu, Diying Li, Quangang Li, Jinqiao Shi, and Yanqiu Wang. Character-based bilstm-crf incorporating pos and dictionaries for chinese opinion target extraction. In Asian Conference on Machine Learning, pages 518–533. PMLR, 2018.
70. Liang Xu, Hai Hu, Xuanwei Zhang, Lu Li, Chenjie Cao, Yudong Li, Yechen Xu, Kai Sun, Dian Yu, Cong Yu, et al. Clue: A chinese language understanding evaluation benchmark. arXiv preprint arXiv:2004.05986, 2020.
71. Ziran Li, Ning Ding, Zhiyuan Liu, Haitao Zheng, and Ying Shen. Chinese relation extraction with multi-grained information and external linguistic knowledge. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4377–4386, 2019.
72. Jingjing Xu, Ji Wen, Xu Sun, and Qi Su. A discourse-level named entity recognition and relation extraction dataset for chinese literature text. arXiv preprint arXiv:1711.07010, 2017.
73. Xin Liu, Qingcai Chen, Chong Deng, Huajun Zeng, Jing Chen, Dongfang Li, and Buzhou Tang. Lcqmc: A largescale chinese question matching corpus. In Proceedings of the 27th International Conference on Computational Linguistics, pages 1952–1962, 2018.
74. Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. Paws-x: A cross-lingual adversarial dataset for paraphrase identification. arXiv preprint arXiv:1908.11828, 2019.
75. Jing Chen, Qingcai Chen, Xin Liu, Haijun Yang, Daohe Lu, and Buzhou Tang. The bq corpus: A large-scale domain-specific chinese corpus for sentence semantic equivalence identification. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4946–4951, 2018.
76. LTD IFLYTEK CO. Iflytek: a multiple categories chinese text classifier. competition official website, 2019.
77. Bang Liu, Di Niu, Haojie Wei, Jinghong Lin, Yancheng He, Kunfeng Lai, and Yu Xu. Matching article pairs with graphical decomposition and convolutions. arXiv preprint arXiv:1802.07459, 2018.
78. Sheng Zhang, Xin Zhang, Hui Wang, Jiajun Cheng, Pei Li, and Zhaoyun Ding. Chinese medical question answer matching using end-to-end character-level multi-scale cnns. Applied Sciences, 7(8):767, 2017.
79. Sheng Zhang, Xin Zhang, Hui Wang, Lixiang Guo, and Shanshan Liu. Multi-scale attentive interaction networks for chinese medical question answer selection. IEEE Access, 6:74061–74071, 2018.
80. Peng Li, Wei Li, Zhengyan He, Xuguang Wang, Ying Cao, Jie Zhou, and Wei Xu. Dataset and neural recurrent sequence labeling model for open-domain factoid question answering. arXiv preprint arXiv:1607.06275, 2016.
81. Yiming Cui, Ting Liu, Zhipeng Chen, Shijin Wang, and Guoping Hu. Consensus attention-based neural networks for chinese reading comprehension. arXiv preprint arXiv:1607.02250, 2016.
82. Yiming Cui, Ting Liu, Zhipeng Chen, Wentao Ma, Shijin Wang, and Guoping Hu. Dataset for the first evaluation on chinese machine reading comprehension. arXiv preprint arXiv:1709.08299, 2017.
83. Yiming Cui, Ting Liu, Ziqing Yang, Zhipeng Chen, Wentao Ma, Wanxiang Che, Shijin Wang, and Guoping Hu. A sentence cloze dataset for chinese machine reading comprehension. arXiv preprint arXiv:2004.03116, 2020.
84. Chujie Zheng, Minlie Huang, and Aixin Sun. Chid: A large-scale chinese idiom dataset for cloze test. arXiv preprint arXiv:1906.01265, 2019.
85. Huibin Ge, Chenxi Sun, Deyi Xiong, and Qun Liu. Chinese wplc: A chinese dataset for evaluating pretrained language models on word prediction given long-range context. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3770–3778, 2021.
86. Chih Chieh Shao, Trois Liu, Yuting Lai, Yiying Tseng, and Sam Tsai. Drcd: a chinese machine reading comprehension dataset. arXiv preprint arXiv:1806.00920, 2018.
87. Wei He, Kai Liu, Jing Liu, Yajuan Lyu, Shiqi Zhao, Xinyan Xiao, Yuan Liu, Yizhong Wang, Hua Wu, Qiaoqiao She, et al. Dureader: a chinese machine reading comprehension dataset from real-world applications. arXiv preprint arXiv:1711.05073, 2017. 27
88. Hongxuan Tang, Jing Liu, Hongyu Li, Yu Hong, Hua Wu, and Haifeng Wang. Dureaderrobust: A chinese dataset towards evaluating the robustness of machine reading comprehension models. arXiv preprint arXiv:2004.11142, 2020.
89. Kai Sun, Dian Yu, Dong Yu, and Claire Cardie. Investigating prior knowledge for challenging chinese machine reading comprehension. Transactions of the Association for Computational Linguistics, 8:141–155, 2020.
90. Yiming Cui, Ting Liu, Li Xiao, Zhipeng Chen, Wentao Ma, Wanxiang Che, Shijin Wang, and Guoping Hu. A span-extraction dataset for chinese machine reading comprehension. CoRR, abs/1810.07366, 2018.
91. Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao Tu, Zhiyuan Liu, Maosong Sun, Yansong Feng, Xianpei Han, Zhen Hu, Heng Wang, et al. Cail2018: A large-scale legal dataset for judgment prediction. arXiv preprint arXiv:1807.02478, 2018.
92. Canwen Xu, Wangchunshu Zhou, Tao Ge, Ke Xu, Julian McAuley, and Furu Wei. Blow the dog whistle: A Chinese dataset for cant understanding with common sense and world knowledge. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2139–2145, Online, June 2021. Association for Computational Linguistics.
93. Hao Tian, Can Gao, Xinyan Xiao, Hao Liu, Bolei He, Hua Wu, Haifeng Wang, and Feng Wu. Skep: Sentiment knowledge enhanced pre-training for sentiment analysis. arXiv preprint arXiv:2005.05635, 2020.
94. Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang, Shijin Wang, and Guoping Hu. Pre-training with whole word masking for chinese bert. arXiv preprint arXiv:1906.08101, 2019.
95. Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942, 2019.
96. Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, and Guoping Hu. Revisiting pre-trained models for chinese natural language processing. arXiv preprint arXiv:2004.13922, 2020.
97. Yan Song, Tong Zhang, Yonggang Wang, and Kai-Fu Lee. Zen 2.0: Continue training and adaption for n-gram enhanced text encoders. arXiv preprint arXiv:2105.01279, 2021.
98. Xiongtao Cui and Jungang Han. Chinese medical question answer matching based on interactive sentence representation learning. arXiv preprint arXiv:2011.13573, 2020.
99. Timo Schick and Hinrich Schütze. Exploiting cloze questions for few shot text classification and natural language inference. arXiv preprint arXiv:2001.07676, 2020.
100. Yi Sun, Yu Zheng, Chao Hao, and Hangping Qiu. Nsp-bert: A prompt-based zero-shot learner through an original pre-training task–next sentence prediction. arXiv preprint arXiv:2109.03564, 2021.
101. Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, and Luke Zettlemoyer. Surface form competition: Why the highest probability answer isn’t always right. arXiv: Computation and Language, 2021.
102. Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955.
103. Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R Bowman, Holger Schwenk, and Veselin Stoyanov. Xnli: Evaluating cross-lingual sentence representations. arXiv preprint arXiv:1809.05053, 2018.
104. TAN Song-bo. Chnsenticorp.
105. Yiming Cui, Ting Liu, Wanxiang Che, Li Xiao, Zhipeng Chen, Wentao Ma, Shijin Wang, and Guoping Hu. A span-extraction dataset for chinese machine reading comprehension. arXiv preprint arXiv:1810.07366, 2018.
106. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach, 2019. cite arxiv:1907.11692. 28