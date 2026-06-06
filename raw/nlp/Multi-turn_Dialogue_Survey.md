# Advances in Multi-turn Dialogue Comprehension: A Survey
多回合对话理解的进展：一项调查 2021.3.4 https://arxiv.org/abs/2103.03125

## 阅读笔记
* 训练数据, 影视剧里的人物对话台词？

## Abstract
Training machines to understand natural language and interact with humans is an elusive and essential task of artificial intelligence. A diversity of dialogue systems has been designed with the rapid development of deep learning techniques, especially the recent pre-trained language models (PrLMs). Among these studies, the fundamental yet challenging type of task is dialogue comprehension whose role is to teach the machines to read and comprehend the dialogue context before responding. In this paper, we review the previous methods from the technical perspective of dialogue modeling for the dialogue comprehension task. We summarize the characteristics and challenges of dialogue comprehension in contrast to plain-text reading comprehension. Then, we discuss three typical patterns of dialogue modeling. In addition, we categorize dialogue-related pretraining techniques which are employed to enhance PrLMs in dialogue scenarios. Finally, we highlight the technical advances in recent years and point out the lessons from the empirical analysis and the prospects towards a new frontier of researches.

训练机器理解自然语言并与人类互动是人工智能一项难以捉摸的基本任务。 随着深度学习技术的快速发展, 特别是最近的预训练语言模型(PrLMs), 已经设计了多种对话系统。 在这些研究中, 基本但具有挑战性的任务类型是对话理解, 其作用是教机器在回应之前阅读和理解对话上下文。 在本文中, 我们从对话理解任务的对话建模技术角度回顾了以前的方法。 我们总结了对话理解与纯文本阅读理解相比的特点和挑战。 然后, 我们讨论三种典型的对话建模模式。 此外, 我们对对话相关的预训练技术进行了分类, 这些技术用于在对话场景中增强 PrLMs。 最后, 我们强调了近年来的技术进步, 并指出了实证分析的经验教训以及对新研究前沿的展望。

## 1 Introduction
Building an intelligent dialogue system that can naturally and meaningfully communicate with humans is a long-standing goal of artificial intelligence (AI) and has been drawing increasing interest from both academia and industry areas due to its potential impact and alluring commercial values. It is a classic topic of human-machine interaction that has a long history. Before computer science and artificial intelligence were categorized into various specific branches, dialogue has become a critical research topic with clear application scenario, as a phenomenon. Dialogue also serves as the important applications area for pragmatics (Leech, 2003) and the Turing Test (Turing and Haugeland, 1950).

建立一个可以与人类自然而有意义地交流的智能对话系统是人工智能(AI) 的长期目标, 并且由于其潜在的影响和诱人的商业价值, 已经引起了学术界和工业界越来越多的兴趣。 是一个历史悠久的人机交互经典话题。 在计算机科学和人工智能被划分为各个具体分支之前, 对话作为一种现象已经成为一个具有明确应用场景的关键研究课题。 对话也是语用学 (Leech, 2003) 和图灵测试 (Turing and Haugeland, 1950) 的重要应用领域。

Traditional methods were proposed to help users complete specific tasks with pre-defined hand-crafted templates after analyzing the scenario of the input utterances (Weizenbaum, 1966; Colby et al., 1971), which are recognized as rulebased methods (Chizhik and Zherebtsova, 2020). However, the growth in this area is hindered by the problem of data scarcity as these systems are expected to learn linguistic knowledge, decision making, and question answering from insufficient amounts of high-quality corpora (Zaib et al., 2020). To alleviate the scarcity, a variety of tasks have been proposed such as response selection (Lowe et al., 2015; Wu et al., 2017; Zhang et al., 2018b), conversation-based question answering (QA) (Sun et al., 2019; Reddy et al., 2019; Choi et al., 2018), decision making and question generation (Saeidi et al., 2018). Examples are shown in Figure 1.

在分析输入话语的场景后, 提出了传统方法来帮助用户使用预定义的手工模板完成特定任务(Weizenbaum, 1966; Colby et al., 1971), 这些方法被认为是基于规则的方法(Chizhik 和 Zherebtsova, 2020)。 然而, 这一领域的发展受到数据稀缺问题的阻碍, 因为这些系统需要从数量不足的高质量语料库中学习语言知识、决策制定和问题回答(Zaib et al., 2020)。 为了缓解稀缺性, 已经提出了多种任务, 例如响应选择(Lowe et al., 2015; Wu et al., 2017; Zhang et al., 2018b), 基于对话的问答(QA)(Sun et et al., 2019; Reddy et al., 2019; Choi et al., 2018), 决策制定和问题生成(Saeidi et al., 2018)。 样本如图1 所示。

![Figure 1](../images/MTD_S/fig_1.png)<br/>
Figure 1: Examples for various dialogue comprehension tasks including response selection, conversation-based QA, and conversational machine reading (w/ decision making and question generation).
图1：各种对话理解任务的样本, 包括响应选择、基于对话的 QA 和对话式机器阅读(带决策制定和问题生成)。

Recently, with the development of deep learning methods (Serban et al., 2016), especially the recent pre-trained language models (PrLMs) (Devlin et al., 2019; Liu et al., 2019; Yang et al., 2019; Lan et al., 2020; Clark et al., 2020), the capacity of neural models has been boosted dramatically. However, most studies focus on an individual task like response retrieval or generation. Stimulated by the interest towards building more generally effective and comprehensive systems to solve real-world problems, traditional natural language processing (NLP) tasks, including the dialogue-related tasks, have been undergoing a fast transformation, where those tasks tend to be crossed and unified in form (Zhang et al., 2020b). Therefore, we may view the major dialogue tasks in a general format of dialogue comprehension: given contexts, a system is required to understand the contexts, and then reply or answer questions. The reply can be derived from retrieval or generation. As a generic concept of measuring the ability to understand dialogues, as opposed to static texts, dialogue comprehension has broad inspirations to the NLP/AI community as shown in Figure 2.

最近, 随着深度学习方法的发展(Serban et al., 2016), 特别是最近的预训练语言模型(PrLMs)(Devlin et al., 2019; Liu et al., 2019; Yang et al., 2019; Lan et al., 2020; Clark et al., 2020), 神经模型的能力得到了显著提升。 然而, 大多数研究都侧重于单个任务, 如响应检索或生成。 在建立更普遍有效和更全面的系统来解决现实世界问题的兴趣的刺激下, 传统的自然语言处理(NLP) 任务, 包括与对话相关的任务, 一直在经历快速转变, 这些任务往往是交叉的, 形式统一(Zhang et al., 2020b)。 因此, 我们可以用对话理解的一般形式来看待主要的对话任务：给定语境, 系统需要理解语境, 然后回复或回答问题。 回复可以来自检索或生成。 作为衡量理解对话能力的通用概念, 与静态文本相反, 对话理解对 NLP/AI 社区具有广泛的启发, 如图2 所示。

![Figure 2](../images/MTD_S/fig_2.png)<br/>
Figure 2: Overview of dialogue comprehension as phenomenon. The left part illustrates the task-related skills for dialogue comprehension. The right part presents the typical techniques in view of dialogue modeling. 
图 2：对话理解现象概述。 左侧部分说明了对话理解的任务相关技能。 右边部分介绍了对话建模的典型技术。

Among the dialogue comprehension studies, the basic technique is dialogue modeling which focuses on how to encode the dialogue context effectively and efficiently to solve the tasks, thus we regard dialogue modeling as the technical aspect of dialogue comprehension. Early techniques mainly focus on the matching mechanisms between the pairwise sequence of dialogue context and candidate response or question (Wu et al., 2017; Zhang et al., 2018b; Huang et al., 2019a). Recently, PrLMs have shown impressive evaluation results for various downstream NLP tasks (Devlin et al., 2019) including dialogue comprehension. They handle the whole texts as a linear sequence of successive tokens and capture the contextualized representations of those tokens through self-attention (Qu et al., 2019; Liu et al., 2020a; Gu et al., 2020a; Xu et al., 2021a). The word embeddings derived by these language models are pre-trained on large corpora. Providing fine-grained contextual embedding, these pre-trained models could be either easily applied to downstream models as the encoder or used for fine-tuning. Besides employing PrLMs for fine-tuning, there also emerges interest in designing dialogue-motivated self-supervised tasks for pre-training.

在对话理解研究中, 最基本的技术是对话建模, 它关注的是如何有效地编码对话上下文以解决任务, 因此我们将对话建模视为对话理解的技术方面。 早期技术主要关注对话上下文的成对序列与候选响应或问题之间的匹配机制(Wu et al., 2017; Zhang et al., 2018b; Huang et al., 2019a)。 最近, PrLM 对包括对话理解在内的各种下游 NLP 任务(Devlin et al., 2019)显示出令人印象深刻的评估结果。 他们将整个文本处理为连续标记的线性序列, 并通过自注意力捕获这些标记的上下文表示(Qu et al., 2019; Liu et al., 2020a; Gu et al., 2020a; Xu et al., 2021a). 由这些语言模型导出的词嵌入是在大型语料库上进行预训练的。 提供细粒度的上下文嵌入, 这些预训练模型可以作为编码器轻松应用于下游模型或用于微调。 除了使用 PrLM 进行微调外, 人们还对设计以对话为动机的自监督任务进行预训练产生了兴趣。

In this survey, we review the previous studies of dialogue comprehension in the perspective of modeling the dialogue tasks as a two-stage Encoder-Decoder framework inspired by the advance of PrLMs and machine reading comprehension (Zhang et al., 2020b), in which way we bridge the gap between the dialogue modeling and comprehension, and hopefully benefit the future researches with the cutting-edge PrLMs. In detail, we will discuss both sides of architecture designs and the pre-training strategies. We summarize the technical advances in recent years and highlight the lessons we can learn from the empirical analysis and the prospects towards a new frontier of researches. Compared with existing surveys that focus on specific dialogue tasks (Zaib et al., 2020; Huang et al., 2020; Fan et al., 2020; Qin et al., 2021b), this work is task-agnostic that discusses the common patterns and trends of dialogue systems in the scope of dialogue comprehension, in order to bridge the gap between different tasks so that those research lines can learn the highlights from each other. 

在本次调查中, 我们从将对话任务建模为两阶段编码器-解码器框架的角度回顾了之前的对话理解研究, 其灵感来自 PrLM 和机器阅读理解的进步 (Zhang et al., 2020b), 其中 我们弥合对话建模和理解之间差距的方式, 并希望通过前沿的 PrLMs 对未来的研究有所帮助。 详细地, 我们将讨论架构设计和预训练策略的两个方面。 我们总结了近年来的技术进步, 并强调了我们可以从实证分析中吸取的教训以及对新研究前沿的展望。 与专注于特定对话任务的现有调查相比(Zaib et al., 2020; Huang et al., 2020; Fan et al., 2020; Qin et al., 2021b), 这项工作与任务无关, 讨论了共同点 对话理解范围内对话系统的模式和趋势, 以弥合不同任务之间的差距, 使这些研究线可以相互学习亮点。

## 2 Characteristics of Dialogues 对话的特点
In contrast to plain-text reading comprehension like SQuAD (Rajpurkar et al., 2016), a multi-turn conversation is intuitively associated with spoken (as opposed to written) language and also is also interactive that involves multiple speakers, intentions, topics, thus the utterances are full of transitions. 
1. Speaker interaction. The transition of speakers in conversations is in a random order, breaking the continuity as that in common nondialogue texts due to the presence of crossing dependencies which are commonplace in a multiparty chat. 
2. Topic Transition. There may be multiple dialogue topics happening simultaneously within one dialogue history and topic drift is common and hard to detect in spoken conversations. Therefore, the multi-party dialogue appears discourse dependency relations between non-adjacent utterances, which leads up to a complex discourse structure.
3. Colloquialism. Dialogue is colloquial, and it takes fewer efforts to speak than to write, resulting in the dialogue context rich in component ellipsis and information redundancy. However, during a conversation, the speakers cannot retract what has been said, which easily leads to self-contradiction, requiring more context, especially clarifications to fully understand the dialogue. 
4. Timeliness. The importance of each utterance towards the expected reply is different, which makes the utterances contribute to the final response in dramatic diversity. Therefore, the order of utterance influences the dialogue modeling. In general, the latest utterances would be more critical (Zhang et al., 2018b, 2021a). 

与像 SQuAD (Rajpurkar et al., 2016) 这样的纯文本阅读理解相比, 多轮对话直观地与口头(而不是书面)语言相关联, 并且也是交互式的, 涉及多个说话者、意图、主题、 因此, 话语充满了过渡。
1. 演讲者互动。 对话中说话者的转换是随机的, 由于存在多方聊天中常见的交叉依赖关系, 打破了普通非对话文本中的连续性。
2. 话题转移。 在一个对话历史中可能会同时发生多个对话主题, 并且主题漂移很常见并且在口语对话中很难检测到。 因此, 多方对话在非相邻话语之间出现话语依存关系, 从而导致复杂的话语结构。
3. 口语化。 对话是口语化的, 说起来比写起来省力, 导致对话语境中含有丰富的成分省略和信息冗余。 但是, 在谈话过程中, 说话者不能收回已经说过的话, 这很容易导致自相矛盾, 需要更多的上下文, 特别是澄清才能完全理解对话。
4. 及时性。 每个话语对预期回复的重要性是不同的, 这使得话语对最终反应的贡献具有戏剧性的多样性。 因此, 话语的顺序会影响对话建模。 一般来说, 最新的言论会更具批判性(Zhang et al., 2018b, 2021a)。


## 3 Methodology
### 3.1 Problem Formulation 问题表述
Although existing studies of dialogue comprehension commonly design independent systems for each downstream tasks, we find that dialogue systems can be generally formulated as an EncoderDecoder framework where an encoder is employed to understand the dialogue context and the decoder is for giving the response. The backbone of the encoder can be either a recurrent neural network such as LSTM (Hochreiter and Schmidhuber, 1997), or a pre-trained language model such as BERT (Devlin et al., 2019). The decoder can be simple as a dense layer for discriminative tasks such as candidate response classification, or part of the seq2seq architecture (Bahdanau et al., 2015) for question or response generation. We could reach the view that dialogue comprehension tasks, especially dialogue generation, share essential similarity with machine translation (Sennrich et al., 2016), from which such unified modeling view can be used to help develop better translation and dialogue generation models from either side of advances.

尽管现有的对话理解研究通常为每个下游任务设计独立的系统, 但我们发现对话系统通常可以表述为 Encoder-Decoder 框架, 其中使用编码器来理解对话上下文, 而解码器用于给出响应。 编码器的主干可以是循环神经网络, 例如 LSTM (Hochreiter and Schmidhuber, 1997), 也可以是预训练语言模型, 例如 BERT (Devlin et al., 2019)。 解码器可以是简单的密集层, 用于诸如候选响应分类之类的判别任务, 也可以是用于问题或响应生成的 seq2seq 架构的一部分(Bahdanau et al., 2015)。 我们可以得出这样的观点, 即对话理解任务, 尤其是对话生成, 与机器翻译具有本质的相似性 (Sennrich et al., 2016), 从中可以使用这种统一的建模视图来帮助从任何一方开发更好的翻译和对话生成模型 预付款。

Here we take two typical dialogue comprehension tasks, i.e., response selection (Lowe et al., 2015; Wu et al., 2017; Zhang et al., 2018b; Cui et al., 2020) and conversation-based QA (Sun et al., 2019; Reddy et al., 2019; Choi et al., 2018), as examples to show the general technical patterns to gain insights, which would also hopefully facilitate other dialogue-related tasks.(1Actually, there are other dialogue comprehension tasks such as decision making and question generation that share similar formations as the fundamental part is still the dialogue context modeling. We only elaborate on the two examples to save space.)

这里我们采用两个典型的对话理解任务, 即响应选择(Lowe et al., 2015; Wu et al., 2017; Zhang et al., 2018b; Cui et al., 2020)和基于对话的问答(Sun et al., 2019; Reddy et al., 2019; Choi et al., 2018), 作为展示一般技术模式以获得洞察力的样本, 这也有望促进其他与对话相关的任务。(1实际上, 还有其他对话 决策和问题生成等理解任务具有相似的形式, 因为基础部分仍然是对话上下文建模。为了节省篇幅, 我们仅详细说明这两个样本)

Suppose that we have a dataset $D = \{(C_i, X_i; Y_i)\}^N_{i=1}$, where $Ci = \{u_{i,1}, ..., u_{i,n_i}\}$ represents the dialogue context with $\{u_i,k\}^{n_i}_{k=1}$ as utterances. $X_i$ is a task-specific paired input, which can be either the candidate response R for response selection, or the question Q for conversation-based QA. $Y_i$ denotes the model prediction. 

假设我们有一个数据集 $D = \{(C_i, X_i; Y_i)\}^N_{i=1}$, 其中 $Ci = \{u_{i,1}, ..., u_{i, n_i}\}$ 表示以 $\{u_i,k\}^{n_i}_{k=1}$ 为话语的对话上下文。 $X_i$ 是特定于任务的成对输入, 可以是用于响应选择的候选响应 R, 也可以是用于基于对话的 QA 的问题 Q。 $Y_i$ 表示模型预测。


<i>Response Selection</i> involves the pairwise input with R as a candidate response. The goal is to learn a discriminator g(·, ·) from D, and at the inference phase, given the context C and response R, we use the discriminator to calculate Y = g(C, R) as their matching score.

响应选择涉及以 R 作为候选响应的成对输入。 目标是从 D 中学习判别器 g(·,·), 在推理阶段, 给定上下文 C 和响应 R, 我们使用判别器计算 Y = g(C, R) 作为它们的匹配分数。

<i>Conversation-based QA</i> aims to answer questions given the dialogue context. Let Q denotes the question Q. The goal is to learn a discriminator g(C, Q) from D to extract the answer span from the context or select the right option from a candidate answer set.

基于对话的 QA 旨在回答给定对话上下文的问题。 令 Q 表示问题Q。目标是从 D 中学习鉴别器 g(C, Q), 以从上下文中提取答案范围或从候选答案集中选择正确的选项。

In the input encoding perspective, since both of the tasks share the paired inputs of either {C; R} or {C; Q}, we simplify the formulation by focusing the response selection task, i.e., replacing R with Q can directly transform into the QA task and without the response as input, the framework is then applicable to dialogue generation tasks.

从输入编码的角度来看, 由于这两个任务都共享 {C; R}或{C; Q}, 我们通过关注响应选择任务来简化公式, 即用 Q 替换 R 可以直接转换为 QA 任务, 无需响应作为输入, 该框架就适用于对话生成任务。

### 3.2 Dialogue Modeling Framework
As shown in Figure 3, we review the previous studies of dialogue comprehension in the perspective of modeling the dialogue tasks as a two-stage Encoder-Decoder framework. The methods of dialogue modeling can be categorized into three patterns: 
1. concatenated matching; 
2. separate interaction; 
3. PrLM-based interaction.

如图3 所示, 我们从将对话任务建模为两阶段编码器-解码器框架的角度回顾了以往对对话理解的研究。 对话建模的方法可以分为三种模式：
1. 串联匹配; 
2. 单独交互; 
3. 基于PrLM 的交互。

![Figure 3](../images/MTD_S/fig_3.png)<br/>
Figure 3: Dialogue modeling framework. The dispensable parts are marked in dashed lines. Without the response as input, the framework is then applicable to dialogue generation tasks.
图 3：对话建模框架。 可有可无的部分用虚线标出。 在没有响应作为输入的情况下, 该框架适用于对话生成任务。


#### Framework 1: Concatenated Matching  框架1：级联匹配
The early methods (Kadlec et al., 2015) treated the dialogue context as a whole by concatenating all previous utterances and last utterance as the context representation and then computed the matching degree score based on the context representation to encode candidate response (Lowe et al., 2015):

早期的方法 (Kadlec et al., 2015) 通过将所有先前的话语和最后的话语连接起来作为上下文表示, 将对话上下文视为一个整体, 然后根据上下文表示计算匹配度得分以编码候选响应 (Lowe et al ., 2015):

EC = Encoder(C); 

ER = Encoder(R); 

Y = Decoder(EC; ER); (1) 

where Encoder is used to encode the raw texts into contextualized representations. Decoder is the module that transforms the contextualized representations to model predictions (Y), which depends on the tasks. For response selection, it can be the attention-based module that calculate the matching score between EC and ER.

其中编码器用于将原始文本编码为上下文表示。 解码器是将上下文表示转换为模型预测 (Y) 的模块, 这取决于任务。 对于响应选择, 可以是基于注意力的模块计算 EC 和 ER 之间的匹配分数。

#### Framework 2: Separate Interaction  框架2：单独交互
With the bloom of attention-based pairwise matching mechanisms, researchers soon find it effective by calculating different levels of interactions between the dialogue context and response. The major research topic is how to improve the semantic matching between the dialogue context and candidate response. For example, Zhou et al. (2016) performed context-response matching with a multi-view model on both word-level and utterance level. Wu et al. (2017) proposed to capture utterances relationship and contextual information by matching a response with each utterance in the context. Those methods can be unified by the view similar to the above concatenated matching:

随着基于注意力的成对匹配机制的兴起, 研究人员很快发现它通过计算对话上下文和响应之间不同级别的交互是有效的。 主要研究课题是如何提高对话上下文和候选响应之间的语义匹配。 例如, Zhou et al. (2016) 在单词级别和话语级别上使用多视图模型执行上下文响应匹配。 Wu et al. (2017) 提出通过将响应与上下文中的每个话语匹配来捕获话语关系和上下文信息。 这些方法可以通过类似于上面的串联匹配的视图来统一：

E$U_i$ = Encoder($u_i$); 

ER = Encoder(R); 

I = ATT([$EU_1, . . . , EU_n$]; ER); 

Y = Decoder(I); (2) 

where ATT denotes the attention-based interactions, which can be pairwise attention, self attention, or the combinations.

其中 ATT 表示基于注意力的交互, 可以是成对注意力、自注意力或组合。

#### Framework 3: PrLM-based Interaction 框架3：基于PrLM 的交互
PrLMs handle the whole input text as a linear sequence of successive tokens and implicitly capture the contextualized representations of those tokens through self-attention (Devlin et al., 2019). Given the context C and response R, we concatenate all utterances in the context and the response candidate as a single, consecutive token sequence with special tokens separating them, and then encode the text sequence by a PrLM:

PrLM 将整个输入文本处理为连续标记的线性序列, 并通过自注意力隐式捕获这些标记的上下文表示(Devlin et al., 2019)。 给定上下文 C 和响应 R, 我们将上下文中的所有话语和候选响应连接为单个连续的标记序列, 并用特殊标记将它们分开, 然后通过 PrLM 对文本序列进行编码：

EC = Encoder([CLS]C[SEP]R[SEP]); 

Y = Decoder(EC); (3) 

where [CLS] and [SEP] are special tokens.

其中 [CLS] 和 [SEP] 是特殊标记。

#### Comparison of Three Frameworks  三种框架的比较
In the early stages of studies that lack computational sources, concatenated matching has the advantage of efficiency, which encodes the context as a whole with a simple structure and directly feeds it to the decoder. With the rapid spread of attention mechanisms, separate interaction has become mainstream and is generally better than concatenated matching because the relationships between utterances and between utterances and response can be sufficiently captured after the fine-grained attention-based interaction. PrLM-based models further extend the advantage of interaction by conducting multi-layer word-by-word interaction over the context and the response. With another benefit from pre-training on large-scale corpora through self-supervised tasks, PrLM-based models significantly outperform the conventional two frameworks. However, the latter two interaction-based methods would be less efficient for real-world applications due to the cost of the heavy computation. Fortunately, it is possible to keep the effectiveness and efficiency at the same time. Inspired by the recent studies of dense retrieval (Seo et al., 2019; Karpukhin et al., 2020; Zhang et al., 2021b) and the fact that dialogue histories are often used repeatedly, a potential solution is to pre-compute their representations for latter indexing, which allows for fast real-time inference in a production setup, giving an improved trade-off between accuracy and speed (Humeau et al., 2020).

在缺乏计算源的研究的早期阶段, 级联匹配具有效率优势, 它以简单的结构将上下文作为一个整体进行编码, 并直接将其提供给解码器。 随着注意力机制的迅速普及, 分离交互已成为主流, 并且通常优于串联匹配, 因为在细粒度的基于注意力的交互之后, 可以充分捕获话语之间以及话语与响应之间的关系。 基于PrLM 的模型通过在上下文和响应上进行多层逐字交互, 进一步扩展了交互的优势。 通过自监督任务对大规模语料库进行预训练的另一个好处是, 基于PrLM 的模型明显优于传统的两个框架。 然而, 由于巨大的计算成本, 后两种基于交互的方法在实际应用中效率较低。 幸运的是, 可以同时保持有效性和效率。 受最近密集检索研究的启发(Seo et al., 2019; Karpukhin et al., 2020; Zhang et al., 2021b)以及对话历史经常被重复使用的事实, 一个潜在的解决方案是预先计算他们的对话历史 后者索引的表示形式, 允许在生产设置中进行快速实时推理, 从而改善准确性和速度之间的权衡(Humeau et al., 2020)。

### 3.3 Dialogue-related Pre-training 对话相关预训练
Although the PrLMs demonstrate superior performance due to their strong representation ability from self-supervised pre-training, it is still challenging to effectively capture task-related knowledge during the detailed task-specific training (Gururangan et al., 2020). Generally, directly using PrLMs would be suboptimal to model dialogue tasks which holds exclusive text features that plain text for PrLM training may hardly embody. Besides, pre-training on general corpora has critical limitations if task datasets are highly domainspecific (Whang et al., 2019), which cannot be sufficiently and accurately covered by the learned universal language representation.

尽管 PrLMs 由于其强大的自监督预训练表示能力而表现出卓越的性能, 但在详细的特定任务训练期间有效地获取与任务相关的知识仍然具有挑战性(Gururangan et al., 2020)。 一般来说, 直接使用 PrLMs 对对话任务建模不是最优的, 对话任务拥有专有文本特征, 而 PrLM 训练的纯文本可能很难体现这些特征。 此外, 如果任务数据集是高度特定于领域的(Whang et al., 2019), 那么对一般语料库的预训练具有严重的局限性, 而学习到的通用语言表示不能充分和准确地涵盖这一点。

Therefore, some researchers have tried to further pre-train PrLMs with general language modeling (LM) objectives on in-domain dialogue texts. The notable examples are BioBERT (Lee et al., 2020), SciBERT (Beltagy et al., 2019), ClinicalBERT (Huang et al., 2019b), and DialoGPT (Zhang et al., 2020a). As our work emphasizes dialogue comprehension tasks, which are more complex than other forms of texts like sentence pairs or essays, the corresponding training objective should be very carefully designed to fit the important elements of dialogues. As shown in Figure 4, there are three kinds of dialogue-related language modeling strategies, namely, generalpurpose pre-training, domain-aware pre-training, and task-oriented pre-training, among which selfsupervised methods do not require additional annotation and can be easily applied into existing approaches(2Though general-purpose pre-training is not our major focus, we describe it here as the basic knowledge for completeness). Typical examples are compared in Table 1.

因此, 一些研究人员尝试在域内对话文本上进一步预训练具有通用语言建模 (LM) 目标的 PrLMs。 著名的例子是 BioBERT(Lee et al., 2020)、SciBERT(Beltagy et al., 2019)、ClinicalBERT(Huang et al., 2019b)和 DialoGPT(Zhang et al., 2020a)。 由于我们的工作强调对话理解任务, 这比句子对或论文等其他形式的文本更复杂, 因此应非常仔细地设计相应的训练目标以适应对话的重要元素。 如图4所示, 对话相关的语言建模策略共有三种, 即通用预训练、领域感知预训练和面向任务的预训练, 其中自监督方法不需要额外的标注和 可以很容易地应用到现有的方法中(2虽然通用预训练不是我们的主要关注点, 但我们在这里将其描述为完整性的基础知识)。 典型样本在表1 中进行了比较。

![Figure 4](../images/MTD_S/fig_4.png)<br/>
Figure 4: Dialogue-related pre-training. There are three kinds of dialogue-related language modeling strategies, namely, general-purpose pre-training, domain-aware pre-training, and task-oriented pre-training. 
图 4：对话相关的预训练。 对话相关的语言建模策略分为三种, 即通用预训练、领域感知预训练和面向任务的预训练。


![Table 1](../images/MTD_S/tab_1.png)<br/>
Table 1: Dialogue-related pre-training methods. Application task shows the evaluated task as reported in the corresponding literature. Note that some methods are applicable for both tasks though only evaluated on one of them. 
表 1：对话相关的预训练方法。 应用任务显示相应文献中报告的评估任务。 请注意, 某些方法适用于这两项任务, 但仅针对其中一项进行评估。

#### General-purpose Pre-training  通用预训练
As the standard pre-training procedure, PrLMs are pre-trained on large-scale domain-free texts and then used for fine-tuning according to the specific task needs. There are token-level and sentence-level objectives used in the general-purpose pre-training. BERT (Devlin et al., 2019) adopts Masked Language Modeling (MLM) as its pre-training objective. It first masks out some tokens from the input sentences and then trains the model to predict them by the rest of the tokens. There are derivatives of MLM like Permuted Language Modeling (PLM) in XLNet (Yang et al., 2019) and Sequence-to-Sequence MLM (Seq2Seq MLM) in MASS (Song et al., 2019) and T5 (Raffel et al., 2019). Next Sentence Prediction (NSP) is another widely used pre-training objective, which trains the model to distinguish whether two input sentences are continuous segments from the training corpus. Sentence Order Prediction (SOP) is one of the replacements of NSP. It requires models to tell whether two consecutive sentences are swapped or not and is first used in ALBERT (Lan et al., 2020). 

作为标准的预训练程序, PrLMs 在大规模无域文本上进行预训练, 然后根据特定任务需要进行微调。 通用预训练中使用了 token-level 和 sentence-level 目标。 BERT (Devlin et al., 2019) 采用掩码语言建模 (MLM) 作为其预训练目标。 它首先从输入句子中屏蔽掉一些令牌, 然后训练模型通过其余令牌预测它们。 有 MLM 的衍生物, 例如 XLNet(Yang et al., 2019)中的置换语言建模(PLM)和 MASS(Song et al., 2019)和 T5(Raffel et al., 2019)中的序列到序列 MLM(Seq2Seq MLM)。 , 2019). 下一句预测(NSP)是另一个广泛使用的预训练目标, 它训练模型区分两个输入句子是否是来自训练语料库的连续片段。 句序预测(SOP)是NSP的替代品之一。 它需要模型来判断两个连续的句子是否被交换, 并且首先用于 ALBERT(Lan et al., 2020)。

#### Domain-aware Pre-training 域感知预训练
The PrLMs are pre-trained on a large text corpus to learn general language representations. To incorporate specific in-domain knowledge, adaptation on indomain corpora, also known as domain-aware pretraining, is designed, which directly post-trains the original PrLMs using the dialogue-domain corpus (Whang et al., 2020b; Wu et al., 2020). The most widely-used PrLM for domain-adaption in the dialogue field is BERT (Devlin et al., 2019), whose pre-training is based on MLM and NSP loss functions. Although NSP has been shown trivial in RoBERTa (Liu et al., 2019) during generalpurpose pre-training, it yields surprising gains in dialogue scenarios (Li et al., 2020c). The most plausible reason is that dialogue emphasizes the relevance between dialogue context and the subsequent response, which shares a similar goal with NSP. Notably, there are Seq2Seq (also known as Text2Text) Transformers pre-trained on massive conversational datasets that serve as the backbone of conversation systems. Though those methods have the advantage of fluency, however, there are criticisms that they often suffer from factual incorrectness and hallucination of knowledge (Roller et al., 2021). A potential solution would be retrieving relevant knowledge and conditioning on dialogue turns (Shuster et al., 2021).

PrLMs 在大型文本语料库上进行预训练以学习通用语言表示。 为了整合特定的领域内知识, 设计了对领域内语料库的适应, 也称为领域感知预训练, 它使用对话域语料库直接对原始 PrLM 进行后训练(Whang et al., 2020b; Wu et al., 2020). 对话领域中使用最广泛的领域自适应 PrLM 是 BERT (Devlin et al., 2019), 其预训练基于 MLM 和 NSP 损失函数。 尽管在通用预训练期间, NSP 在 RoBERTa(Liu et al., 2019)中被证明是微不足道的, 但它在对话场景中产生了惊人的收益(Li et al., 2020c)。 最合理的原因是对话强调对话上下文与后续响应之间的相关性, 这与 NSP 具有相似的目标。 值得注意的是, 有 Seq2Seq(也称为Text2Text) Transformers 在作为对话系统支柱的大量对话数据集上进行了预训练。 尽管这些方法具有流畅性的优势, 但也有人批评它们经常存在事实错误和知识幻觉(Roller et al., 2021)。 一个潜在的解决方案是检索相关知识并调节对话轮次(Shuster et al., 2021)。

#### Task-oriented Pre-training  面向任务的预训练
In contrast to the plain-text modeling as the focus of the PrLMs, dialogue texts involve multiple speakers and re- flect special characteristics such as topic transitions and structural utterance dependencies as discussed in Section 2. Inspired by such characteristics to imitate the real-world dialogues, recent studies are pondering the dialogue-specific selfsupervised training objectives to model dialoguerelated features. There are two categories of studies from the sides of cross-utterance and innerutterance. 

与作为 PrLMs 重点的纯文本建模相比, 对话文本涉及多个说话者并反映特殊特征, 例如第2 节中讨论的主题转换和结构话语依赖性。受这些特征的启发, 模仿现实世界 对话, 最近的研究正在思考对话特定的自监督训练目标, 以模拟对话相关的特征。 从交叉话语和内部话语两方面进行研究。

1. Cross-utterance. Prior works have indicated that the order information would be important in the text representation, and the well-known NSP and SOP can be viewed as special cases of order prediction. Especially in the dialogue scenario, predicting the word order of utterance, as well as the utterance order in the context, has shown effectiveness in the dialogue modeling task (Kumar et al., 2020; Gu et al., 2020b), where the utterance order information is well restored from shuffled dialogue context. Li et al. (2021b) designed a variant of NSP called next utterance prediction as a pre-training scheme to adapt BERT to accommodate the inherent context continuity underlying the multi-turn dialogue. Whang et al. (2021) proposed various utterance manipulation strategies including utterance insertion, deletion, and search to maintain dialog coherence. Similarly, Xu et al.  (2021a) introduced four self-supervised tasks to explicitly model the cross-utterance relationships to improve coherence and consistency between the utterances, including next session prediction, utterance restoration, incoherence detection, and consistency discrimination. 

1 交叉话语。 先前的工作表明顺序信息在文本表示中很重要, 众所周知的 NSP 和 SOP 可以看作是顺序预测的特例。 特别是在对话场景中, 预测话语的词序以及上下文中的话语顺序在对话建模任务中显示出有效性 (Kumar et al., 2020; Gu et al., 2020b), 其中话语 从打乱的对话上下文中可以很好地恢复顺序信息。 Li et al. (2021b) 设计了一种称为下一次话语预测的 NSP 变体作为预训练方案, 以适应 BERT 以适应多轮对话背后的固有上下文连续性。 Whang et al. (2021) 提出了各种话语操纵策略, 包括话语插入、删除和搜索, 以保持对话的连贯性。 同样, Xu et al. (2021a)引入了四个自监督任务来显式地建模交叉话语关系, 以提高话语之间的连贯性和一致性, 包括下一会话预测、话语恢复、不连贯检测和一致性辨别。

2. Inner-utterance . The other type of objectives is proposed as inner-utterance modeling, which has not attracted much attention. The intuition is to model the fact and events inside an utterance. Zhang and Zhao (2021) introduced a sentence backbone regularization task as regularization to improve the factual correctness of summarized subject-verb-object triplets. Zhao et al. (2020) proposed word order recovery and masked word recovery to enhance understanding of the sequential dependency among words and encourage more attention to the semantics of words to find better representations of words. 

2 内部的话语。 另一种类型的目标被提出作为内部话语建模, 这并没有引起太多关注。 直觉是对话语中的事实和事件建模。 Zhang 和 Zhao (2021) 引入了句子主干正则化任务作为正则化, 以提高概括的主谓宾三元组的事实正确性。 Zhao et al. (2020) 提出了词序恢复和掩码词恢复, 以增强对单词之间顺序依赖性的理解, 并鼓励更多地关注单词的语义以找到更好的单词表示。

## 4 Dialogue Comprehension with Explicit Knowledge 对话理解与显性知识
Dialogue contexts are colloquial and full of incomplete information, which requires a machine to refine the key information while reflecting the relevant details. Without background knowledge as a reference, the machine may merely capture limited information from the surface text of the dialogue context or query. Such knowledge is not limited to topics, emotions, and multimodal grounding. These sources can provide extra information beyond the textual dialogue context to enhance dialogue comprehension.

对话上下文是口语化的, 充满了不完整的信息, 这需要机器在反映相关细节的同时提炼关键信息。 在没有背景知识作为参考的情况下, 机器可能只能从对话上下文或查询的表面文本中捕获有限的信息。 这些知识不限于主题、情感和多模态基础。 这些来源可以提供文本对话上下文之外的额外信息, 以增强对话理解。

### 4.1 Auxiliary Knowledge Grounding  辅助知识接地
Linguistic knowledge has been shown important for dialogue modeling, such as syntax (Wang et al., 2015; Eshghi et al., 2017) and discourse information (Galley et al., 2003; Ouyang et al., 2020). In addition, Various kinds of background knowledge can be adaptively grounded in dialogue modeling according to task requirements, including commonsense items (Zhang et al., 2021a), speaker relationships (Liu et al., 2020b), domain knowledge (Li et al., 2020a), from knowledge graphs to strengthen reasoning ability, personabased attributes such as speaker identity, dialogue topic, speaker sentiments, to enrich the dialogue context (Olabiyi et al., 2019), scenario information to provide the dialogue background (Ouyang et al., 2020), etc.

语言知识已被证明对对话建模很重要, 例如句法 (Wang et al., 2015; Eshghi et al., 2017) 和话语信息 (Galley et al., 2003; Ouyang et al., 2020)。 此外, 各种背景知识可以根据任务要求自适应地建立在对话建模中, 包括常识项(Zhang et al., 2021a)、说话人关系(Liu et al., 2020b)、领域知识(Li et al., 2021a), 从知识图加强推理能力, 基于角色的属性, 如说话人身份、对话主题、说话人情感, 丰富对话上下文 (Olabiyi et al., 2019), 场景信息提供对话背景 (Ouyang et al., 2020)等。

### 4.2 Emotional Promotion
Emotional feeling or sentiment is a critical characteristic to distinguish people from machines (Hsu et al., 2018). People’s emotions are complex, and there are various complex emotional features such as metaphor and irony in the dialogue. Besides, the same expressions may have different meanings in different situations. Not only should the dialogue systems capture the user intents, topics transitions, and dialogue structures, but also they should be able to perceive the sentiment changes and even adjust the tone and guide the conversation according to the user emotional states, to provide a more friendly, acceptable, and empathetic conversations, which would be especially useful for building social bots and automatic e-commerce marketing.

情绪感受或情感是区分人与机器的关键特征(Hsu et al., 2018)。 人的情感是复杂的, 对话中存在隐喻、反讽等各种复杂的情感特征。 此外, 相同的表达在不同的情况下可能有不同的含义。 对话系统不仅要捕捉用户的意图、话题转换和对话结构, 还要能够感知情绪变化, 甚至根据用户情绪状态调整语气和引导对话, 提供更友好的对话、可接受和有同理心的对话, 这对于构建社交机器人和自动电子商务营销特别有用。

### 4.3 Multilingual and Multimodal Dialogue 多语言多模式对话
As a natural interface of human and machine interaction, a dialogue system would be beneficial for people in different language backgrounds to communicate with each other. Besides the natural language texts, visual and audio sources are also effective carriers are can be incorporated with texts for comprehensive and immersed conversations. With the rapid development of multilingual and multimodal researches (Qin et al., 2020a; Firdaus et al., 2021), building an intelligent dialogue system is not elusive in the future. 

作为人机交互的自然界面, 对话系统有利于不同语言背景的人相互交流。 除了自然语言文本, 视觉和音频资源也是有效的载体, 可以与文本结合, 进行全面和沉浸式对话。 随着多语言和多模态研究的快速发展(Qin et al., 2020a; Firdaus et al., 2021), 构建智能对话系统在未来并非遥不可及。

## 5 Empirical Analysis 实证分析
### 5.1 Dataset
We analyze three kinds of dialogue comprehension tasks, 1) response selection: Ubuntu Dialogue Corpus (Ubuntu) (Lowe et al., 2015), Douban Conversation Corpus (Douban) (Wu et al., 2017), E-commerce Dialogue Corpus (ECD) (Zhang et al., 2018b), Multi-Turn Dialogue Reasoning (MuTual) (Cui et al., 2020)(3Because the test set of MuTual is not publicly available, we conducted the comparison with our baselines on the Dev set for convenience) ; 2) conversationbased QA: DREAM (Sun et al., 2019); 3) conversational machine reading: ShARC (Saeidi et al., 2018).(4We only use these typical datasets to save space. Table 2 presents a collection of widely-used datasets for the reference of interested readers.)

我们分析了三种对话理解任务, 
1. 响应选择：Ubuntu对话语料库(Ubuntu)(Lowe et al., 2015)、豆瓣对话语料库(Douban)(Wu et al., 2017)、电子商务对话语料库( ECD) (Zhang et al., 2018b), 多回合对话推理(MuTual) (Cui et al., 2020);  (3由于MuTual的测试集不公开, 为了方便, 我们在Dev set上与我们的baselines进行了比较。)
2. 基于对话的QA：DREAM(Sun et al., 2019);  
3. 会话式机器阅读：ShARC(Saeidi et al., 2018)。(4为了节省篇幅, 我们只使用这些典型的数据集。表2展示了一组广泛使用的数据集, 供有兴趣的读者参考。)

![Table 2](../images/MTD_S/tab_2.png)<br/>
Table 2: Widely-used datasets for dialogue comprehension tasks. Response Form shows the way to provide the response according to the official evaluation metrics in the corresponding literature. Size indicates the size of the whole dataset, including training, development, and test sets. Manually indicates that human writing of the question or answers is involved in the data annotation process. Since dialogue corpus can often be used for both response selection and generation tasks, we use the term "next utterance prediction". 
表 2：广泛用于对话理解任务的数据集。 Response Form 显示了根据相应文献中的官方评估指标提供响应的方式。 Size表示整个数据集的大小, 包括训练集、开发集和测试集。 手动表示数据注释过程中涉及人工编写的问题或答案。 由于对话语料库通常可用于响应选择和生成任务, 因此我们使用术语“下一个话语预测”。


#### 5.1.1 Response Selection 响应选择
<strong>Ubuntu</strong> consists of English multi-turn conversations about technical support collected from chat logs of the Ubuntu forum. The dataset contains 1 million context-response pairs, 0.5 million for validation, and 0.5 million for testing. In the training set, each context has one positive response generated by humans and one negative response sampled randomly. In the validation and test sets, for each context, there are 9 negative responses and 1 positive response.

Ubuntu 由从 Ubuntu 论坛的聊天记录中收集的有关技术支持的英语多轮对话组成。 该数据集包含 100 万个上下文响应对, 50 万个用于验证, 50 万个用于测试。 在训练集中, 每个上下文都有一个由人类生成的正面响应和一个随机采样的负面响应。 在验证集和测试集中, 对于每个上下文, 有 9 个否定响应和 1 个肯定响应。

<strong>Douban</strong> is different from Ubuntu in the following ways. First, it is an open domain where dialogues are extracted from the Douban Group. Second, response candidates on the test set are collected by using the last turn as the query to retrieve 10 response candidates and labeled by humans. Third, there could be more than one correct response for a context.

豆瓣在以下方面与 Ubuntu 不同。 首先, 它是一个开放域, 其中的对话是从豆瓣组中提取的。 其次, 通过使用最后一轮作为查询来收集测试集上的响应候选, 以检索 10 个响应候选并由人工标记。 第三, 对于一个上下文可能有不止一个正确的响应。

<strong>ECD</strong> dataset is extracted from conversations between customer and service staff on E-commerce platforms. It contains over 5 types of conversations based on over 20 commodities. There are also 1 million context-response pairs in the training set, 0.5 million in the validation set, and 0.5 million in the test set.

ECD 数据集是从电子商务平台上客户和服务人员之间的对话中提取的。 它包含基于 20 多种商品的 5 多种类型的对话。 训练集中还有 100 万个上下文响应对, 验证集中有 50 万个, 测试集中有 50 万个。

<strong>MuTual</strong> consists of 8,860 manually annotated dialogues based on Chinese student English listening comprehension exams(5MuTual Leaderboard https://nealcly.github.io/MuTual-leaderboard/). For each context, there is one positive response and three negative responses. The difference compared to the above three datasets is that only MuTual is reasoningbased. There are more than 6 types of reasoning abilities reflected in MuTual. 

MuTual 包含 8,860 个基于中国学生英语听力考试的手动注释对话(5 MuTual Leaderboard https://nealcly.github.io/MuTual-leaderboard/)。 对于每个上下文, 都有一个正面响应和三个负面响应。 与上述三个数据集相比的不同之处在于, 只有 MuTual 是基于推理的。 MuTual 中体现的推理能力有 6 种以上。

![Table 3](../images/MTD_S/tab_3.png)<br/>
Table 3: Results on Ubuntu, Douban, and E-commerce datasets.
表 3：Ubuntu、豆瓣和电子商务数据集的结果。

#### 5.1.2 Conversation-based QA 基于对话的质量保证
DREAM is a dialogue-based multi-choice reading comprehension dataset, which is collected from English exams(6DREAM Leaderboard https://dataset.org/dream/). Each dialogue, as the given context, has multiple questions, and each question has three response options. In total, it contains 6,444 dialogues and 10,197 questions. The most important feature of the dataset is that more than 80% of the questions are non-extractive, and more than a third of the given questions involve commonsense knowledge. As a result, the dataset is small but quite challenging. 

DREAM 是一个基于对话的多选阅读理解数据集, 收集自英语考试(6DREAM Leaderboard https://dataset.org/dream/)。 每个对话, 作为给定的上下文, 有多个问题, 每个问题有三个回答选项。 它总共包含 6,444 个对话和 10,197 个问题。 该数据集最重要的特征是超过 80% 的问题是非提取性的, 超过三分之一的给定问题涉及常识性知识。 因此, 数据集很小但很有挑战性。

#### 5.1.3 Conversational Machine Reading 对话式机器阅读
ShARC is the current CMR benchmark(7 ShARC Leaderboard: https://sharc-data.github.io/leaderboard.html), which contains two subtasks: decision making and question generation, as an example shown Table 1 (c). For the first subtask, the machine needs to decide "Yes", "No", "Inquire" and "Irrelevant" given a document consisting of rule conditions, initial question, user scenario, and dialog history for each turn. "Yes/No" gives a definite answer to the initial question. "Irrelevant" means that the question cannot be answered with such knowledge base text. If the information provided so far is insufficient for the machine to decide, an "Inquire" decision will be made and we may step into the second subtask and the machine will ask a corresponding question using the under-specified rule span to fill the gap of information. The dataset contains up to 948 dialog trees clawed from government websites. Those dialog trees are then flattened into 32,436 examples. The sizes of train, dev, and test are 21,890, 2,270, and 8,276 respectively.

ShARC是目前CMR的benchmark(7 ShARC Leaderboard: https://sharc-data.github.io/leaderboard.html), 它包含两个子任务：决策制定和问题生成, 如表1(c)所示。 对于第一个子任务, 机器需要根据每轮的规则条件、初始问题、用户场景和对话历史记录组成的文档来决定“是”、“否”、“询问”和“不相关”。 “是/否”对最初的问题给出了明确的答案。 “无关”是指该问题不能用这样的知识库文本来回答。 如果到目前为止提供的信息不足以让机器做出决定, 则会做出“询问”决定, 我们可能会进入第二个子任务, 机器会使用未指定的规则跨度提出相应的问题, 以填补空白 信息。 该数据集包含从政府网站抓取的多达 948 个对话树。 然后将这些对话树展平为 32,436 个样本。 train、dev 和 test 的大小分别为 21,890、2,270 和 8,276。

![Table 4](../images/MTD_S/tab_4.png)<br/>
Table 4: Results on MuTual dataset. The upper and lower blocks present the models w/o and w/ PrLMs, respectively. 
表 4：MuTual 数据集的结果。 上部和下部模块分别展示了不带 PrLM 和不带 PrLM 的模型。

### 5.2 Evaluation Metrics
Following Lowe et al. (2015), we calculate the proportion of true positive response among the top-k selected responses from the list of n available candidates for one context, denoted as Rn@k. For our tasks, the number of candidate responses n is 10, so we write the metric as R@k to save space. For the conversation-based QA task, DREAM, the official metric is accuracy (Acc). Concerning the CMRC task, ShARC evaluates the Micro- and Macro- Acc. for the decision-making subtask. If both the decision is Inquire, BLEU (Papineni et al., 2002) score (particularly BLEU1 and BLEU4) will be evaluated on the follow-up question generation subtask.

继 Lowe et al. (2015)之后, 我们计算了从一个上下文的 n 个可用候选列表中选择的前 k 个响应中真实响应的比例, 表示为 Rn@k。 对于我们的任务, 候选响应的数量 n 为 10, 因此我们将指标写为 R@k 以节省空间。 对于基于对话的 QA 任务 DREAM, 官方指标是准确性 (Acc)。 关于 CMRC 任务, ShARC 评估 Micro- 和 Macro-Acc。 用于决策子任务。 如果两个决定都是询问, BLEU (Papineni et al., 2002) 分数(特别是 BLEU1 和 BLEU4)将在后续问题生成子任务上进行评估。

### 5.3 Observations
Tables 3-6 present the benchmark results on six typical dialogue comprehension tasks, including three response selection tasks, Ubuntu (Lowe et al., 2015), Douban (Wu et al., 2017), ECD (Zhang et al., 2018b), Mutual (Cui et al., 2020), one conversation-based QA task, DREAM (Sun et al., 2019), and one conversation machine reading task consisting of decsion making and question generation, ShARC (Saeidi et al., 2018), from which we summarize the following observations:(8The evaluation results are collected from published literature (Zhang et al., 2021a; Whang et al., 2021; Xu et al., 2021a; Lin et al., 2020; Lowe et al., 2015; Liu et al., 2020b, 2021; Li et al., 2020c; Sun et al., 2019; Wan, 2020).)
1. <strong>Interaction methods generally yield better performance than single-turn models.</strong> In the early stage without PrLMs, separate interaction (Framework 2) commonly achieves better performance than the simple concatenated matching (Framework 1), verifying the effectiveness of attention-based pairwise matching. However, multi-turn matching networks (separate interaction) perform worse than PrLMs-based ones (Framework 3), illustrating the power of contextualized representations in context-sensitive dialogue modeling. 
2. <strong>Dialogue-related pre-training helps make PrLM better suitable for dialogue comprehension.</strong> Compared with general-purpose PrLMs, dialogue-aware pre-training (e.g., BERT-VFT (Whang et al., 2020a), SA-BERT (Gu et al., 2020a), PoDS (Zhang et al., 2021a)) can further improve the results by a large margin. In addition, task-oriented pre-training (e.g., DCM (Li et al., 2021b), UMS (Whang et al., 2021), BERT-SL (Xu et al., 2021a)) even shows superiority among the pre-training techniques. 
3. <strong>Discriminative modeling beats generative methods.</strong> Empirically, for the concerned dialogue comprehension tasks, retrieval-based or discriminative methods commonly show better performance than generative models such as GPT (Radford et al., 2018). 
4. <strong>Data augmentation from negative sampling has attracted interests to enlarge corpus size and improve response quality. </strong>Among the models, G-MSN (Lin et al., 2020), BERT-SS-DA (Lu et al., 2020), ELECTRA+DAPO (Li et al., 2020c) show that training/pre-training data construction, especially negative sampling is a critical influence factor to the model performance. 
5. <strong>Context Disentanglment helps discover the essential dialogue structure.</strong> SA-BERT (Gu et al., 2020a) and MDFN (Liu et al., 2021) indicate that modeling the speaker information is effective for dialogue modeling. Further, EMT (Gao et al., 2020a), Discern (Gao et al., 2020b), and DGM (Ouyang et al., 2020) indicate that decoupling the dialogue context into elementary discourse units (EDUs) and model the graph-like relationships between EDUs would be effective to capture inner discourse structures of complex dialogues. 
6. <strong>Extra knowledge injection further improves dialogue modeling.</strong> From Table 5, we see that external knowledge like commonsense would be beneficial for dialogue modeling of dialogue systems (Li et al., 2020c). Recent studies also show that interactive open-retrieval of relevant knowledge is useful for reducing the hallucination in conversation (Shuster et al., 2021). 

表 3-6 给出了六个典型对话理解任务的基准结果, 包括三个响应选择任务, Ubuntu (Lowe et al., 2015)、豆瓣 (Wu et al., 2017)、ECD (Zhang et al., 2018b) , Mutual (Cui et al., 2020), 一项基于对话的 QA 任务 DREAM (Sun et al., 2019), 以及一项由决策制定和问题生成组成的对话机器阅读任务 ShARC (Saeidi et al., 2018) ), 我们从中总结了以下观察结果：(8 评估结果收集自已发表的文献(Zhang et al., 2021a; Whang et al., 2021; Xu et al., 2021a; Lin et al., 2020; Lowe et al. al., 2015; Liu et al., 2020b, 2021; Li et al., 2020c; Sun et al., 2019; Wan, 2020)。
1. <strong>交互方法通常比单轮模型产生更好的性能。 </strong> 在没有 PrLMs 的早期阶段, 单独交互(框架 2)通常比简单的连接匹配(框架 1)获得更好的性能, 验证了基于注意力的成对匹配的有效性。 然而, 多轮匹配网络(单独的交互)比基于PrLM 的网络(框架 3)表现更差, 说明了上下文相关的对话建模中上下文表示的力量。
2. <strong>对话相关的预训练有助于让PrLM更适合对话理解。</strong>  与通用 PrLM 相比, 对话感知预训练(例如, BERT-VFT(Whang et al., 2020a)、SA-BERT(Gu et al., 2020a)、PoDS(Zhang et al., 2021a)) 可以进一步大幅度提高结果。 此外, 面向任务的预训练(例如 DCM (Li et al., 2021b)、UMS (Whang et al., 2021)、BERT-SL (Xu et al., 2021a))甚至在预训练中表现出优势 -训练技巧。
3. <strong>判别建模胜过生成方法。</strong>  从经验上讲, 对于相关的对话理解任务, 基于检索或判别方法通常表现出比 GPT 等生成模型更好的性能(Radford et al., 2018)。
4. <strong>负采样的数据扩充引起了人们对扩大语料库规模和提高响应质量的兴趣。</strong>  在这些模型中, G-MSN (Lin et al., 2020)、BERT-SS-DA (Lu et al., 2020)、ELECTRA+DAPO (Li et al., 2020c) 表明训练/预训练数据构建 , 尤其是负采样是模型性能的关键影响因素。
5. <strong>上下文分离有助于发现基本的对话结构。</strong>  SA-BERT (Gu et al., 2020a) 和 MDFN (Liu et al., 2021) 表明对说话人信息进行建模对于对话建模是有效的。 此外, EMT (Gao et al., 2020a)、Discern (Gao et al., 2020b) 和 DGM (Ouyang et al., 2020) 表明, 将对话上下文解耦为基本话语单元 (EDU) 并为图形建模 EDU 之间的相似关系将有效地捕捉复杂对话的内部话语结构。
6. <strong>额外的知识注入进一步改进了对话建模。</strong>  从表 5 中, 我们看到像常识这样的外部知识将有利于对话系统的对话建模(Li et al., 2020c)。 最近的研究还表明, 相关知识的交互式开放检索有助于减少对话中的幻觉(Shuster et al., 2021)。


![Table 5](../images/MTD_S/tab_5.png)<br/>
Table 5: Results (%) on DREAM dataset. The upper and lower blocks present the models w/o and w/PrLMs, respectively. 
表 5：DREAM 数据集的结果 (%)。 上部和下部块分别显示了不带 PrLM 和带 PrLM 的模型。

![Table 6](../images/MTD_S/tab_6.png)<br/>
Table 6: Results on the the dev set and blind held-out test set of ShARC tasks for decision making and question generation. Micro and Macro stand for Micro Accuracy and Macro Accuracy. The upper and lower blocks present the models w/o and w/ PrLMs, respectively. 
表 6：用于决策制定和问题生成的 ShARC 任务的开发集和盲目保留测试集的结果。 微观和宏观代表微观准确性和宏观准确性。 上部和下部模块分别展示了不带 PrLM 和不带 PrLM 的模型。

## 6 Frontiers of Training Dialogue Comprehension Models 训练对话理解模型的前沿
### 6.1 Dialogue Disentanglment Learning 对话解纠结学习
Recent widely-used PrLM-based models deal with the whole dialogue(9Because PrLMs are interaction-based methods, thus encoding the context as a whole achieves better performance than encoding the utterances individually.), which results in entangled information that originally belongs to different parts and is not optimal for dialogue modeling, especially for multi-party dialogues (Yang and Choi, 2019b; Li and Choi, 2020; Li et al., 2020b). Sequence decoupling is a strategy to tackle this problem by explicitly separating the context into different parts and further constructing the relationships between those parts to yield better fine-grained representations. One possible solution is splitting the context into several topic blocks (Xu et al., 2021b; Lu et al., 2020). However, there existing topic crossing, which would hinder the segmentation effect. Another scheme is to employ a masking mechanism inside self-attention network (Liu et al., 2021), to limit the focus of each word only on the related ones, such as those from the same utterance, or the same speaker, to model the local dependencies, in complement with the global contextualized representation from PrLMs. Further, recent studies show that explicitly modeling discourse structures and action triples (Gao et al., 2020c; Ouyang et al., 2020; Chen and Yang, 2021; Li et al., 2021a; Feng et al., 2020) would be effective for improving dialogue comprehension. 

最近广泛使用的基于PrLM 的模型处理整个对话(9 因为 PrLM 是基于交互的方法, 因此将上下文作为一个整体进行编码比单独对话语进行编码具有更好的性能),  这导致原本属于不同部分的信息纠缠不清, 对于对话建模来说并不是最优的, 特别是对于多方对话(Yang and Choi, 2019b; Li and Choi , 2020; Li et al., 2020b). 序列解耦是解决这个问题的一种策略, 它通过明确地将上下文分成不同的部分并进一步构建这些部分之间的关系以产生更好的细粒度表示。 一种可能的解决方案是将上下文分成几个主题块(Xu et al., 2021b; Lu et al., 2020)。 然而, 存在主题交叉, 这会阻碍分割效果。 另一种方案是在 self-attention 网络中采用掩码机制 (Liu et al., 2021), 将每个词的焦点限制在相关词上, 例如来自相同话语或相同说话者的词, 以进行建模 局部依赖性, 与 PrLMs 的全局上下文表示相辅相成。 此外, 最近的研究表明, 对话语结构和动作三元组进行显式建模(Gao et al., 2020c; Ouyang et al., 2020; Chen and Yang, 2021; Li et al., 2021a; Feng et al., 2020) 有效提高对话理解能力。 

### 6.2 Dialogue-aware Language Modeling 对话感知语言建模
Recent studies have indicated that dialoguerelated language modeling can enhance dialogue comprehension substantially (Gu et al., 2020a; Li et al., 2021b; Zhang et al., 2021a; Whang et al., 2021; Xu et al., 2021a). However, these methods rely on the dialogue-style corpus for the pretraining, which is not always available in general application scenarios. Given the massive freeform and domain-free data from the internet, how to simulate the conversation, e.g., in an adversarial way, with the general-purpose and general-domain data, is a promising research direction. Besides transferring from general-purpose to dialogueaware modeling, multi-domain adaption is another important topic that is effective to reduce the annotation cost and achieve robust and scalable dialogue systems (Qin et al., 2020b).

最近的研究表明, 与对话相关的语言建模可以显著提高对话理解能力(Gu et al., 2020a; Li et al., 2021b; Zhang et al., 2021a; Whang et al., 2021; Xu et al., 2021a)。 然而, 这些方法依赖于对话式语料库进行预训练, 这在一般应用场景中并不总是可用。 鉴于来自互联网的海量自由形式和无域数据, 如何使用通用和通用域数据模拟对话, 例如以对抗方式, 是一个有前途的研究方向。 除了从通用到对话感知建模的转变之外, 多领域适应是另一个重要主题, 它可以有效降低注释成本并实现健壮且可扩展的对话系统(Qin et al., 2020b)。

### 6.3 High-quality Negative Sampling 高质量负采样
Most prior works train the dialogue comprehension models with training data constructed by a simple heuristic. They treated human-written responses as positive examples and randomly sampled responses from other dialogue contexts as equally bad negative examples, i.e., inappropriate responses (Lowe et al., 2015; Wu et al., 2017; Zhang et al., 2018b). As discussed in Section 5, data construction is also critical to the model capacity. The randomly sampled negative responses are often too trivial, making the model unable to handle strong distractors for dialogue comprehension. Intending to train a more effective and reliable model, there is an emerging interest in mining better training data (Lin et al., 2020; Li et al., 2020c; Su et al., 2020). 

大多数先前的工作使用通过简单启发式构建的训练数据来训练对话理解模型。 他们将人写的回复视为正面样本, 并将来自其他对话上下文的随机抽样回复视为同样糟糕的负面样本, 即不恰当的回复(Lowe et al., 2015; Wu et al., 2017; Zhang et al., 2018b)。 正如第 5 节中所讨论的, 数据构建对于模型容量也很关键。 随机抽样的负面反应往往太微不足道, 使模型无法处理对话理解的强烈干扰因素。 为了训练更有效、更可靠的模型, 人们开始对挖掘更好的训练数据产生兴趣(Lin et al., 2020; Li et al., 2020c; Su et al., 2020)。

## 7 Open Challenges 公开挑战
Though there are extensive efforts that have been made, with impressive results obtained on many benchmarks for dialogue comprehension, there are still various open challenges.

尽管已经做出了广泛的努力, 在许多对话理解基准上取得了令人瞩目的成果, 但仍然存在各种开放的挑战。

### Temporal Reasoning 时间推理
Daily dialogues are rich in events, which in turn requires understanding temporal commonsense concepts interwoven with those events, such as duration, frequency, and order. There are preliminary attempts to investigate temporal features like utterance order and topic flow. However, such features are too shallow to reveal the reasoning chain of events. Qin et al. (2021a) showed that the best dominant variant of PrLM like T5-large with in-domain training still struggles on temporal reasoning in dialogue which relies on superficial cues based on existing temporal patterns in context.

日常对话中包含丰富的事件, 这反过来又需要理解与这些事件交织在一起的时间常识性概念, 例如持续时间、频率和顺序。 初步尝试调查话语顺序和主题流等时间特征。 然而, 这些特征太浅, 无法揭示事件的推理链。 秦 et al. (2021a) 表明, 具有域内训练的 PrLM 的最佳主导变体(如 T5-large)仍然在对话中的时间推理上挣扎, 对话依赖于基于上下文中现有时间模式的表面线索。

### Logic Consistency 逻辑一致性
Logic is vital for dialogue systems which not only guarantees the consistency and meaningfulness of responses but also strengthens the mode with logical reasoning abilities. Existing interaction-based commonly focus on capturing the semantic relevance between the dialogue context and the response but usually neglect the logical consistency during the dialogue that is a critical issue reflected in dialogue models (Cui et al., 2020). The widely-used backbone PrLM models are trained from plain texts with simple LM objectives, which has shown to suffer from adversarial attacks (Liu et al., 2020c) easily and lack the specific requirements for dialogue systems such as logic reasoning.

逻辑对于对话系统至关重要, 它不仅保证了响应的一致性和有意义性, 而且增强了逻辑推理能力的模式。 现有的基于交互的方法通常侧重于捕获对话上下文和响应之间的语义相关性, 但通常忽略对话过程中的逻辑一致性, 这是对话模型中反映的一个关键问题(Cui et al., 2020)。 广泛使用的骨干 PrLM 模型是从具有简单 LM 目标的纯文本中训练出来的, 这表明它很容易受到对抗性攻击 (Liu et al., 2020c), 并且缺乏对逻辑推理等对话系统的特定要求。

### Large-scale Open-retrieval 大规模开放检索
The current mainstream dialogue tasks often assume that the dialogue context or background information is provided for the user query. In real-world scenarios, a system would be required to retrieve various types of relevant information such as similar conversation history from a large corpus or necessary supporting evidence from a knowledge base to respond to queries interactively. Therefore, how to retrieve accurate, consistent, and semantically meaningful evidence is critical. Compared with the open-domain QA tasks, open-retrieval dialogues raise new challenges of both efficiency and effectiveness mostly due to the human-machine interaction features.

目前主流的对话任务往往假设提供对话上下文或背景信息供用户查询。 在现实场景中, 系统需要从大型语料库中检索各种类型的相关信息, 例如类似的对话历史记录, 或者从知识库中检索必要的支持证据, 以交互方式响应查询。 因此, 如何检索准确、一致且具有语义意义的证据至关重要。 与开放域 QA 任务相比, 开放式检索对话对效率和有效性提出了新的挑战, 这主要是由于人机交互的特点。

### Dialogue for Social Good 社会公益对话
To effectively solve dialogue tasks, we are often equipped with specific and accurate information, from which the datasets are often crawled from real-world dialogue histories that require many human efforts. Domain transfer is a long-standing problem for the practical utility of dialogue systems that is far from being solved. As reflected in Table 2, dialogue corpora are often restricted in specific domains. There are many domains that have not been paid litter attention to due to the lack of commercial interests or lack of annotated data. Besides, few attention has been paid to low-resource communities, and high-quality dialogue corpus is generally scarce beyond English communities.

为了有效地解决对话任务, 我们通常配备了具体而准确的信息, 这些数据集通常是从需要许多人工努力的现实世界对话历史中抓取的。 域转移是对话系统实用性的一个长期存在的问题, 远未得到解决。 如表 2 所示, 对话语料库通常仅限于特定领域。 由于缺乏商业利益或缺乏注释数据, 有许多领域没有受到重视。 此外, 很少有人关注低资源社区, 高质量的对话语料库在英语社区之外普遍稀缺。

## References
* Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings. 
* Iz Beltagy, Kyle Lo, and Arman Cohan. 2019. SciBERT: A pretrained language model for scientific text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3615–3620, Hong Kong, China. Association for Computational Linguistics. 
* Jiaao Chen and Diyi Yang. 2021. Structure-aware abstractive conversation summarization via discourse and action graphs. In 2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2021). 
* Anna Chizhik and Yulia Zherebtsova. 2020. Challenges of building an intelligent chatbot. In IMS, pages 277–287. 
* Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. 2018. QuAC: Question answering in context. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2174–2184, Brussels, Belgium. Association for Computational Linguistics. 
* Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. 2020. ELECTRA: pre-training text encoders as discriminators rather than generators. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26- 30, 2020. OpenReview.net. 
* Kenneth Mark Colby, Sylvia Weber, and Franklin Dennis Hilf. 1971. Artificial paranoia. Artificial Intelligence, 2(1):1–25. 
* Leyang Cui, Yu Wu, Shujie Liu, Yue Zhang, and Ming Zhou. 2020. MuTual: A dataset for multiturn dialogue reasoning. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1406–1416, Online. Association for Computational Linguistics. 
* Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. 
* Arash Eshghi, Igor Shalyminov, and Oliver Lemon. 2017. Bootstrapping incremental dialogue systems from minimal data: the generalisation power of dialogue grammars. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 2220–2230, Copenhagen, Denmark. Association for Computational Linguistics. 
* Yifan Fan, Xudong Luo, and Pingping Lin. 2020. A survey of response generation of dialogue systems. International Journal of Computer and Information Engineering, 14(12):461–472. 
* Xiachong Feng, Xiaocheng Feng, Bing Qin, Xinwei Geng, and Ting Liu. 2020. Dialogue discourse-aware graph convolutional networks for abstractive meeting summarization. arXiv preprint arXiv:2012.03502. 
* Mauajama Firdaus, Nidhi Thakur, and Asif Ekbal. 2021. Aspect-aware response generation for multimodal dialogue system. ACM Transactions on Intelligent Systems and Technology (TIST), 12(2):1–33. 
* Michel Galley, Kathleen R. McKeown, Eric Fosler-Lussier, and Hongyan Jing. 2003. Discourse segmentation of multi-party conversation. In Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics, pages 562–569, Sapporo, Japan. Association for Computational Linguistics. 
* Yifan Gao, Chien-Sheng Wu, Shafiq Joty, Caiming Xiong, Richard Socher, Irwin King, Michael Lyu, and Steven C.H. Hoi. 2020a. Explicit memory tracker with coarse-to-fine reasoning for conversational machine reading. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 935–945, Online. Association for Computational Linguistics. 
* Yifan Gao, Chien-Sheng Wu, Jingjing Li, Shafiq Joty, Steven C.H. Hoi, Caiming Xiong, Irwin King, and Michael Lyu. 2020b. Discern: Discourse-aware entailment reasoning network for conversational machine reading. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2439–2449, Online. Association for Computational Linguistics. 
* Yifan Gao, Chien-Sheng Wu, Jingjing Li, Shafiq Joty, Steven C.H. Hoi, Caiming Xiong, Irwin King, and Michael Lyu. 2020c. Discern: Discourse-aware entailment reasoning network for conversational machine reading. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2439–2449, Online. Association for Computational Linguistics. 
* Jia-Chen Gu, Tianda Li, Quan Liu, Zhen-Hua Ling, Zhiming Su, Si Wei, and Xiaodan Zhu. 2020a. Speaker-aware BERT for multi-turn response selection in retrieval-based chatbots. In CIKM ’20: The 29th ACM International Conference on Information and Knowledge Management, Virtual Event, Ireland, October 19-23, 2020, pages 2041–2044. ACM. 
* Jia-Chen Gu, Zhen-Hua Ling, and Quan Liu.  2019. Interactive matching network for multiturn response selection in retrieval-based chatbots. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, CIKM 2019, Beijing, China, November 3-7, 2019, pages 2321–2324. ACM. 
* Xiaodong Gu, Kang Min Yoo, and Jung-Woo Ha. 2020b. Dialogbert: Discourse-aware response generation via learning to recover and rank utterances. arXiv:2012.01775. 
* Suchin Gururangan, Ana Marasovi´c, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. 2020. Don’t stop pretraining: Adapt language models to domains and tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8342–8360, Online. Association for Computational Linguistics. 
* Sepp Hochreiter and Jürgen Schmidhuber. 1997. 
* Long short-term memory. Neural computation, 9(8):1735–1780. 
* Chao-Chun Hsu, Sheng-Yeh Chen, Chuan-Chun Kuo, Ting-Hao Huang, and Lun-Wei Ku.  2018. EmotionLines: An emotion corpus of multi-party conversations. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA). 
* Hsin-Yuan Huang, Eunsol Choi, and Wen-tau Yih. 2019a. Flowqa: Grasping flow in history for conversational machine comprehension. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net. 
* Kexin Huang, Jaan Altosaar, and R. Ranganath. 2019b. Clinicalbert: Modeling clinical notes and predicting hospital readmission. arXiv:1904.05342. 
* Minlie Huang, Xiaoyan Zhu, and Jianfeng Gao. 2020. Challenges in building intelligent opendomain dialog systems. ACM Transactions on Information Systems (TOIS), 38(3):1–32. 
* Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Polyencoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net. 
* Rudolf Kadlec, Martin Schmid, and Jan Kleindienst. 2015. Improved deep learning baselines for ubuntu corpus dialogs. NIPS Workshop. 
* Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online. Association for Computational Linguistics. 
* Pawan Kumar, Dhanajit Brahma, Harish Karnick, and Piyush Rai. 2020. Deep attentive ranking networks for learning to order sentences. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 8115–8122. AAAI Press. 
* Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2020. ALBERT: A lite BERT for self-supervised learning of language representations. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net. 
* Carolin Lawrence, Bhushan Kotnis, and Mathias Niepert. 2019. Attending to future tokens for bidirectional sequence generation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1– 10, Hong Kong, China. Association for Computational Linguistics. 
* Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, D. Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2020. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics. 
* Geoffrey Leech. 2003. Pragmatics and dialogue. In The Oxford handbook of computational linguistics. 
* Changmao Li and Jinho D. Choi. 2020. Transformers to learn hierarchical contexts in multiparty dialogue for span-based question answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5709–5714, Online. Association for Computational Linguistics. 
* Feng-Lin Li, Hehong Chen, Guohai Xu, Tian Qiu, Feng Ji, Ji Zhang, and Haiqing Chen. 2020a.  Alimekg: Domain knowledge graph construction and application in e-commerce. In CIKM ’20: The 29th ACM International Conference on Information and Knowledge Management, Virtual Event, Ireland, October 19-23, 2020, pages 2581–2588. ACM. 
* Jiaqi Li, Ming Liu, Min-Yen Kan, Zihao Zheng, Zekun Wang, Wenqiang Lei, Ting Liu, and Bing Qin. 2020b. Molweni: A challenge multiparty dialogues-based machine reading comprehension dataset with discourse structure. In Proceedings of the 28th International Conference on Computational Linguistics, pages 2642–2652, Barcelona, Spain (Online). International Committee on Computational Linguistics. 
* Jiaqi Li, Ming Liu, Zihao Zheng, Heng Zhang, Bing Qin, Min-Yen Kan, and Ting Liu. 2021a. Dadgraph: A discourse-aware dialogue graph neural network for multiparty dialogue machine reading comprehension. arXiv:2104.12377. 
* Junlong Li, Zhuosheng Zhang, Hai Zhao, Xi Zhou, and Xiang Zhou. 2020c. Task-specific Objectives of Pre-trained Language Models for Dialogue Adaptation. arXiv: 2009.04984. 
* Lu Li, Chenliang Li, and Donghong Ji. 2021b. Deep context modeling for multi-turn response selection in dialogue systems. Information Processing & Management, 58(1):102415. 
* Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. DailyDialog: A manually labelled multi-turn dialogue dataset. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 986–995, Taipei, Taiwan. Asian Federation of Natural Language Processing. 
* Zibo Lin, Deng Cai, Yan Wang, Xiaojiang Liu, Haitao Zheng, and Shuming Shi. 2020. The world is not binary: Learning to rank with grayscale data for dialogue response selection. 
* In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9220–9229, Online. Association for Computational Linguistics. 
* Chuang Liu, Deyi Xiong, Yuxiang Jia, Hongying Zan, and Changjian Hu. 2020a. Hisbert for conversational reading comprehension. In 2020 International Conference on Asian Language Processing (IALP), pages 147–152. IEEE. 
* Jian Liu, Dianbo Sui, Kang Liu, and Jun Zhao. 2020b. Graph-based knowledge integration for question answering over dialogue. In Proceedings of the 28th International Conference on Computational Linguistics, pages 2425–2435, Barcelona, Spain (Online). International Committee on Computational Linguistics. 
* Kai Liu, Xin Liu, An Yang, Jing Liu, Jinsong Su, Sujian Li, and Qiaoqiao She. 2020c. A robust adversarial training approach to machine reading comprehension. In The ThirtyFourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 8392–8400. AAAI Press. 
* Longxiang Liu, Zhuosheng Zhang, , Hai Zhao, Xi Zhou, and Xiang Zhou. 2021. Filling the Gap of Utterance-aware and Speaker-aware Representation for Multi-turn Dialogue. In The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21). 
* Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A Robustly Optimized BERT Pretraining Approach. arXiv: 1907.11692. 
* Ryan Lowe, Nissan Pow, Iulian Serban, and Joelle Pineau. 2015. The Ubuntu dialogue corpus: A large dataset for research in unstructured multiturn dialogue systems. In Proceedings of the 16th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 285– 294, Prague, Czech Republic. Association for Computational Linguistics. 
* Junyu Lu, Xiancong Ren, Yazhou Ren, Ao Liu, and Zenglin Xu. 2020. Improving contextual language models for response retrieval in multi-turn conversation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, pages 1805–1808. ACM. 
* Oluwatobi Olabiyi, Anish Khazane, Alan Salimov, and Erik Mueller. 2019. An adversarial learning framework for a persona-based multiturn dialogue model. In Proceedings of the Workshop on Methods for Optimizing and Evaluating Neural Language Generation, pages 1– 10, Minneapolis, Minnesota. Association for Computational Linguistics. 
* Siru Ouyang, Zhuosheng Zhang, and Hai Zhao. 2020. Dialogue graph modeling for conversational machine reading. arXiv:2012.14827. 
* Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. 
* Lianhui Qin, Aditya Gupta, Shyam Upadhyay, Luheng He, Yejin Choi, and Manaal Faruqui. 2021a. TIMEDIAL: Temporal commonsense reasoning in dialog. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7066– 7076, Online. Association for Computational Linguistics. 
* Libo Qin, Minheng Ni, Yue Zhang, and Wanxiang Che. 2020a. Cosda-ml: Multi-lingual code-switching data augmentation for zero-shot cross-lingual NLP. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI 2020, pages 3853– 3860. ijcai.org. 
* Libo Qin, Tianbao Xie, Wanxiang Che, and Ting Liu. 2021b. A survey on spoken language understanding: Recent advances and new frontiers. In the 30th International Joint Conference on Artificial Intelligence (IJCAI-21: Survey Track). 
* Libo Qin, Xiao Xu, Wanxiang Che, Yue Zhang, and Ting Liu. 2020b. Dynamic fusion network for multi-domain end-to-end task-oriented dialog. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6344–6354, Online. Association for Computational Linguistics. 
* Chen Qu, Liu Yang, Minghui Qiu, W. Bruce Croft, Yongfeng Zhang, and Mohit Iyyer. 2019. BERT with history answer embedding for conversational question answering. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2019, Paris, France, July 21- 25, 2019, pages 1133–1136. ACM. 
* Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding by generative pre-training. 
* Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, W. Li, and Peter J. Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv: 1910.10683. 
* Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. 
* In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics. 
* Siva Reddy, Danqi Chen, and Christopher D. Manning. 2019. CoQA: A conversational question answering challenge. Transactions of the Association for Computational Linguistics, 7:249– 266. 
* Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Eric Michael Smith, Y-Lan Boureau, et al. 2021. Recipes for building an opendomain chatbot. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 300–325. 
* Marzieh Saeidi, Max Bartolo, Patrick Lewis, Sameer Singh, Tim Rocktäschel, Mike Sheldon, Guillaume Bouchard, and Sebastian Riedel. 2018. Interpretation of natural language rules in conversational machine reading. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2087–2097, Brussels, Belgium. Association for Computational Linguistics. 
* Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany. Association for Computational Linguistics. 
* Minjoon Seo, Jinhyuk Lee, Tom Kwiatkowski, Ankur Parikh, Ali Farhadi, and Hannaneh Hajishirzi. 2019. Real-time open-domain question answering with dense-sparse phrase index. 
* In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4430–4441, Florence, Italy. Association for Computational Linguistics. 
* Iulian Vlad Serban, Alessandro Sordoni, Yoshua Bengio, Aaron C. Courville, and Joelle Pineau. 2016. Building end-to-end dialogue systems using generative hierarchical neural network models. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, February 12-17, 2016, Phoenix, Arizona, USA, pages 3776–3784. AAAI Press. 
* Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021. Retrieval augmentation reduces hallucination in conversation. arXiv preprint arXiv:2104.07567. 
* Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2019. MASS: masked sequence to sequence pre-training for language generation. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pages 5926–5936. PMLR. 
* Yixuan Su, Deng Cai, Qingyu Zhou, Zibo Lin, Simon Baker, Yunbo Cao, Shuming Shi, Nigel Collier, and Yan Wang. 2020. Dialogue response selection with hierarchical curriculum learning. arXiv preprint arXiv:2012.14756. 
* Kai Sun, Dian Yu, Jianshu Chen, Dong Yu, Yejin Choi, and Claire Cardie. 2019. DREAM: A challenge data set and models for dialoguebased reading comprehension. Transactions of the Association for Computational Linguistics, 7:217–231. 
* Chongyang Tao, Wei Wu, Can Xu, Wenpeng Hu, Dongyan Zhao, and Rui Yan. 2019a. Multirepresentation fusion network for multi-turn response selection in retrieval-based chatbots. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining, WSDM 2019, Melbourne, VIC, Australia, February 11-15, 2019, pages 267–275. ACM. 
* Chongyang Tao, Wei Wu, Can Xu, Wenpeng Hu, Dongyan Zhao, and Rui Yan. 2019b. One time of interaction may not be enough: Go deep with an interaction-over-interaction network for response selection in dialogues. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1–11, Florence, Italy. Association for Computational Linguistics. 
* Alan M Turing and J Haugeland. 1950. Computing machinery and intelligence. MIT Press Cambridge, MA. 
* Nikhil Verma, Abhishek Sharma, Dhiraj Madan, Danish Contractor, Harshit Kumar, and Sachindra Joshi. 2020. Neural conversational QA: Learning to reason vs exploiting patterns. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7263–7269, Online. Association for Computational Linguistics. 
* Hui Wan. 2020. Multi-task learning with multihead attention for multi-choice reading comprehension. arXiv:2003.04992. 
* Shengxian Wan, Yanyan Lan, Jun Xu, Jiafeng Guo, Liang Pang, and Xueqi Cheng. 2016. 
* Match-srnn: Modeling the recursive matching structure with spatial RNN. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI 2016, New York, NY, USA, 9-15 July 2016, pages 2922– 2928. IJCAI/AAAI Press. 
* Mingxuan Wang, Zhengdong Lu, Hang Li, and Qun Liu. 2015. Syntax-based deep matching of short texts. In Proceedings of the TwentyFourth International Joint Conference on Arti- ficial Intelligence, IJCAI 2015, Buenos Aires, Argentina, July 25-31, 2015, pages 1354–1361. 
* AAAI Press. 
* Shuohang Wang and Jing Jiang. 2016. Learning natural language inference with LSTM. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1442–1451, San Diego, California. Association for Computational Linguistics. 
* Joseph Weizenbaum. 1966. Eliza—a computer program for the study of natural language communication between man and machine. Communications of the ACM, 9(1):36–45. 
* Sean Welleck, Jason Weston, Arthur Szlam, and Kyunghyun Cho. 2019. Dialogue natural language inference. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3731–3741, Florence, Italy. Association for Computational Linguistics. 
* T. Whang, Dongyub Lee, C. Lee, Kisu Yang, Dongsuk Oh, and Heuiseok Lim. 2019. An effective domain adaptive post-training method for bert in response selection. In INTERSPEECH. 
* Taesun Whang, Dongyub Lee, Chanhee Lee, Kisu Yang, Dongsuk Oh, and HeuiSeok Lim. 2020a. An effective domain adaptive posttraining method for bert in response selection. 
* In INTERSPEECH. 
* Taesun Whang, Dongyub Lee, Chanhee Lee, Kisu Yang, Dongsuk Oh, and Heuiseok Lim. 2020b. An effective domain adaptive posttraining method for bert in response selection. 
* INTERSPEECH. 
* Taesun Whang, Dongyub Lee, Dongsuk Oh, Chanhee Lee, Kijong Han, Dong-hun Lee, and Saebyeok Lee. 2021. Do response selection models really know what’s next? utterance manipulation strategies for multi-turn response selection. In The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21). 
* Chien-Sheng Wu, Steven C.H. Hoi, Richard Socher, and Caiming Xiong. 2020. TODBERT: Pre-trained natural language understanding for task-oriented dialogue. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 917–929, Online. Association for Computational Linguistics. 
* Yu Wu, Wei Wu, Chen Xing, Ming Zhou, and Zhoujun Li. 2017. Sequential matching network: A new architecture for multi-turn response selection in retrieval-based chatbots. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 496–505, Vancouver, Canada. Association for Computational Linguistics. 
* Ruijian Xu, Chongyang Tao, Daxin Jiang, Xueliang Zhao, Dongyan Zhao, and Rui Yan. 2021a. Learning an effective context-response matching model with self-supervised tasks for retrieval-based dialogues. In The ThirtyFifth AAAI Conference on Artificial Intelligence (AAAI-21). 
* Yi Xu, Hai Zhao, and Zhuosheng Zhang. 2021b. 
* Topic-aware multi-turn dialogue modeling. In The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21). 
* Rui Yan, Yiping Song, and Hua Wu. 2016. Learning to respond with deep neural networks for retrieval-based human-computer conversation system. In Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval, SIGIR 2016, Pisa, Italy, July 17-21, 2016, pages 55– 64. ACM. 
* Zhengzhe Yang and Jinho D. Choi. 2019a. FriendsQA: Open-domain question answering on TV show transcripts. In Proceedings of the 20th Annual SIGdial Meeting on Discourse and Dialogue, pages 188–197, Stockholm, Sweden. Association for Computational Linguistics. 
* Zhengzhe Yang and Jinho D. Choi. 2019b. FriendsQA: Open-domain question answering on TV show transcripts. In Proceedings of the 20th Annual SIGdial Meeting on Discourse and Dialogue, pages 188–197, Stockholm, Sweden. Association for Computational Linguistics. 
* Zhilin Yang, Zihang Dai, Yiming Yang, Jaime G. 
* Carbonell, Ruslan Salakhutdinov, and Quoc V. 
* Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 5754–5764. 
* Chunyuan Yuan, Wei Zhou, Mingming Li, Shangwen Lv, Fuqing Zhu, Jizhong Han, and Songlin Hu. 2019. Multi-hop selector network for multi-turn response selection in retrievalbased chatbots. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 111–120, Hong Kong, China. Association for Computational Linguistics. 
* Munazza Zaib, Quan Z Sheng, and Wei Emma Zhang. 2020. A short survey of pretrained language models for conversational aia new age in nlp. In Proceedings of the Australasian Computer Science Week Multiconference, pages 1–4. 
* Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. 2018a. Personalizing dialogue agents: I have a dog, do you have pets too? In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2204–2213, Melbourne, Australia. 
* Association for Computational Linguistics. 
* Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, and Bill Dolan. 2020a. DIALOGPT : Large-scale generative pre-training for conversational response generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 270–278, Online. 
* Association for Computational Linguistics. 
* Zhuosheng Zhang, Jiangtong Li, Pengfei Zhu, Hai Zhao, and Gongshen Liu. 2018b. Modeling multi-turn conversation with deep utterance aggregation. In Proceedings of the 27th International Conference on Computational Linguistics, pages 3740–3752, Santa Fe, New Mexico, USA. Association for Computational Linguistics. 
* Zhuosheng Zhang, Junlong Li, and Hai Zhao. 2021a. Multi-turn dialogue reading comprehension with pivot turns and knowledge. 
* IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:1161–1173. 
* Zhuosheng Zhang, Siru Ouyang, Hai Zhao, Masao Utiyama, and Eiichiro Sumita. 2021b. Smoothing dialogue states for open conversational ma- chine reading. In The 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021). 
* Zhuosheng Zhang and Hai Zhao. 2021. Structural pre-training for dialogue comprehension. 
* In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5134–5145, Online. Association for Computational Linguistics. 
* Zhuosheng Zhang, Hai Zhao, and Rui Wang. 2020b. Machine reading comprehension: The role of contextualized language models and beyond. arXiv:2005.06249. 
* Yufan Zhao, Can Xu, and Wei Wu. 2020. Learning a simple and effective model for multi-turn response generation with auxiliary tasks. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3472–3483, Online. Association for Computational Linguistics. 
* Victor Zhong and Luke Zettlemoyer. 2019. E3: Entailment-driven extracting and editing for conversational machine reading. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 2310–2320, Florence, Italy. Association for Computational Linguistics. 
* Xiangyang Zhou, Daxiang Dong, Hua Wu, Shiqi Zhao, Dianhai Yu, Hao Tian, Xuan Liu, and Rui Yan. 2016. Multi-view response selection for human-computer conversation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 372– 381, Austin, Texas. Association for Computational Linguistics. 
* Xiangyang Zhou, Lu Li, Daxiang Dong, Yi Liu, Ying Chen, Wayne Xin Zhao, Dianhai Yu, and Hua Wu. 2018. Multi-turn response selection for chatbots with deep attention matching network. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1118– 1127, Melbourne, Australia. Association for Computational Linguistics.