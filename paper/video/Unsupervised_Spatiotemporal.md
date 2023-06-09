# A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning
无监督时空表征学习的大规模研究 https://arxiv.org/abs/2104.14558

## Abstract
We present a large-scale study on unsupervised spatiotemporal representation learning from videos. With a unified perspective on four recent image-based frameworks, we study a simple objective that can easily generalize all these methods to space-time. Our objective encourages temporallypersistent features in the same video, and in spite of its simplicity, it works surprisingly well across: (i) different unsupervised frameworks, (ii) pre-training datasets, (iii) downstream datasets, and (iv) backbone architectures. We draw a series of intriguing observations from this study, e.g., we discover that encouraging long-spanned persistency can be effective even if the timespan is 60 seconds. In addition to state-of-the-art results in multiple benchmarks, we report a few promising cases in which unsupervised pre-training can outperform its supervised counterpart. Code is made available at https://github.com/facebookresearch/SlowFast.

我们提出了一项关于从视频中学习无监督时空表征的大规模研究。 通过对最近四个基于图像的框架的视角统一，我们研究了一个简单的目标，可以轻松地将所有这些方法推广到时空。 我们的目标鼓励同一视频中的时间持久性特征，尽管它很简单，但它在以下方面的表现出奇地好：(i)不同的无监督框架，(ii)预训练数据集，(iii)下游数据集，以及(iv)骨干架构 . 我们从这项研究中得出了一系列有趣的观察结果，例如，我们发现，即使时间跨度为 60 秒，鼓励长时间的持续性也是有效的。 除了在多个基准测试中获得最先进的结果外，我们还报告了一些有前途的案例，其中的无监督预训练可以优于有监督的预训练。

## 1. Introduction
A series of recent methods on unsupervised representation learning from images [36, 12, 32, 9] are based on maximizing a similarity objective for different views of the same image under data augmentations [18, 89]. In addition to the artificial augmentations on images, videos can provide natural augmentations of visual content under various changing factors, such as motion, deformation, occlusion, and illumination. This work aims to generalize these image-based methods [36, 12, 32, 9] into space-time.

最近一系列关于从图像 [36、12、32、9] 进行无监督表示学习的方法基于在数据增广下最大化同一图像的不同视图的相似性目标 [18、89]。 除了对图像进行人工增广外，视频还可以在各种变化因素(例如运动、变形、遮挡和照明)下提供视觉内容的自然增广。 这项工作旨在将这些基于图像的方法 [36、12、32、9] 推广到时空。

We study a simple objective that can be easily incorporated into these image-based methods. Our hypothesis is that the visual content is often temporally-persistent along a timespan in the video. This persistency may involve an action (e.g., a person dancing), an object (e.g., an individual person, who transitions from running to walking), and a scene (e.g., a room with people moving), covering short to long spans, with different levels of visual invariance (action, object, scene). Our objective simply encourages the visual representations in different clips of the same video to be similar. We empirically find that this objective works well across different unsupervised frameworks (MoCo [36], SimCLR [12], BYOL [32], SwAV [9]), either with or without using dissimilar (negative) samples.

我们研究了一个简单的目标，它可以很容易地融入这些基于图像的方法中。 我们的假设是视觉内容通常在视频中的时间跨度内是暂时持续的。 这种持久性可能涉及一个动作(例如，一个人在跳舞)、一个物体(例如，一个人，从跑步过渡到步行)和一个场景(例如，一个有人在移动的房间)，涵盖从短到长的跨度， 具有不同级别的视觉不变性(动作、对象、场景)。 我们的目标只是鼓励同一视频的不同剪辑中的视觉表示相似。 我们根据经验发现，无论是否使用不同的(负)样本，该目标在不同的无监督框架(MoCo [36]、SimCLR [12]、BYOL [32]、SwAV [9])中都能很好地工作。

Figure 1. Learning to maximize the similarity between different temporal clips of the same video encourages feature persistency over time. A query clip (q) is matched to multiple key clips ($k_1 , k_2$ , . . .) that are temporally shifted. This method can be incorporated into several unsupervised learning frameworks (MoCo [36], SimCLR [12], BYOL [32], SwAV [9]). The figure on the top shows that increasing the number (ρ) of temporal clips improves representation quality for all these frameworks. 
图 1. 学习最大化同一视频的不同时间片段之间的相似性可以促进特征随时间的持久性。 查询剪辑 (q) 与时间上移位的多个关键剪辑 ($k_1 , k_2$、...) 相匹配。 这种方法可以合并到几个无监督学习框架中(MoCo [36]、SimCLR [12]、BYOL [32]、SwAV [9])。 上图显示，增加时间剪辑的数量 (ρ) 可以提高所有这些框架的表示质量。
<!-- clip 剪辑，是单帧还是多帧 -->


Our objective is a natural generalization of crops in images [18, 89] to clips in videos. This allows us to make use of the recent unsupervised learning frameworks with minimal modifications. We aim to learn a high-level representation of the categorical semantics present in a video by enforcing persistency of the representation over space-time. We investigate factors such as the effective timespan, t, between positives, and number of temporal clips, ρ, to find that longer timespans (up to a minute) and multiple samples are beneficial for downstream performance (Fig. 1).

我们的目标是将图像 [18、89] 中的裁剪自然地泛化为视频中的剪辑。 这使我们能够以最少的修改利用最近的无监督学习框架。 我们的目标是通过强制表示在时空上的持久性来学习视频中存在的分类语义的高级表示。 我们调查了有效时间跨度 t、阳性之间的时间和时间片段的数量 ρ 等因素，发现更长的时间跨度(最多一分钟)和多个样本有利于下游性能(图 1)。

Our unsupervised training is performed on large-scale data, including Kinetics [47] (240k videos) and three versions of million-scale Instagram sets. In addition to standard linear probing, we evaluate representation quality on multiple classification and detection downstream datasets, e.g., Charades [75], Something-Something [31], and AVA [33]. 

我们的无监督训练是在大规模数据上进行的，包括 Kinetics [47](240k 视频)和百万级 Instagram 集的三个版本。 除了标准线性探测之外，我们还评估了多个分类和检测下游数据集的表示质量，例如 Charades [75]、Something-Something [31] 和 AVA [33]。

Our results suggest that unsupervised pre-training can achieve competitive performance in videos, and it can surpass the supervised pre-training counterparts in a few cases. Finally, our study also reveals room for improvement along multiple directions.

我们的结果表明，无监督预训练可以在视频中取得有竞争力的表现，并且在少数情况下可以超越有监督的预训练。 最后，我们的研究还揭示了多个方向的改进空间。

In summary, our large-scale study involves the following five aspects: 
1. Four unsupervised learning frameworks (MoCo [36], SimCLR [12], BYOL [32], SwAV [9]) viewed from a unified perspective, and incorporated with a simple temporal persistency objective; 
2. Three pre-training datasets, including the relatively well-controlled Kinetics [47] and the relatively “in-the-wild” Instagram sets at million-scale; 
3. Six downstream datasets/tasks for evaluating representation quality; 
4. Ablation experiments on different factors, such as temporal samples, contrastive objective, momentum encoders, training duration, backbones, data augmentation, curated vs. uncurated, trimmed vs. untrimmed, etc.; 
5. State-of-the-art results of unsupervised video representation learning on established benchmarks, UCF-101 [77], HMDB51 [50] and Kinetics-400 [47] .

综上所述，我们的大规模研究涉及以下五个方面：
1. 从统一角度来看四个无监督学习框架(MoCo [36]、SimCLR [12]、BYOL [32]、SwAV [9])，并结合了简单的时间持久性目标;  <!-- 时间持久性目标 ?-->
2. 三个预训练数据集，包括相对控制较好的Kinetics[47]和相对“in-the-wild”的百万级Instagram集; 
3. 六个用于评估表示质量的下游数据集/任务; 
4. 不同因素的消融实验，例如时间样本、对比目标、动量编码器、训练持续时间、主干、数据增广、策划与未策划、修剪与未修剪等; 
5. 无监督视频表示学习在既定基准 UCF-101 [77]、HMDB51 [50] 和 Kinetics-400 [47] 上的最新成果。

## 2. Related Work
### Unsupervised learning in images 
has been actively researched recently with approaches focusing on various pretext tasks related to coloror patch-based processing [67, 94, 17, 64], instance discrimination with contrastive objectives [18, 89, 83, 40, 41, 46, 36, 95, 12, 81] and ones that focus on positive pairs [8, 9, 32].

最近对图像中的无监督学习进行了积极的研究，其方法侧重于与基于颜色或分块的处理相关的各种前置任务 [67、94、17、64]、具有对比目标的实例辨别 [18、89、83、40、41、46, 36, 95, 12, 81] 和那些专注于正对的 [8, 9, 32]。

### Unsupervised learning in videos 
has followed a similar trajectory with earlier methods focusing on predictive tasks based on motion, color and spatiotemporal ordering [29,43,1, 44, 78, 85, 60, 84, 58, 57, 21, 51, 86, 66, 22, 48, 91, 16, 87, 70, 45], and contrastive objectives with visual [74, 79, 34, 53, 28, 92] and audio-visual input [65, 4, 5, 49, 3, 68, 69].

视频中的无监督学习遵循类似的轨迹，早期的方法侧重于基于运动、颜色和时空排序的预测任务 [29,43,1,44,78,85,60,84,58,57,21,51,86 , 66, 22, 48, 91, 16, 87, 70, 45]，以及具有视觉 [74, 79, 34, 53, 28, 92] 和视听输入 [65, 4, 5, 49, 3, 68, 69]。

Several recent ones [28, 35, 3, 68, 2, 71, 92, 62] relate to image-based approaches [36, 8, 12, 89]. With some of them using additional modalities of optical-flow [81, 35], audio [3, 68, 2, 62] and text [79, 2] to transfer supervision from one modality to another.

最近的几个 [28、35、3、68、2、71、92、62] 涉及基于图像的方法 [36、8、12、89]。 其中一些使用光流 [81, 35]、音频 [3, 68, 2, 62] 和文本 [79, 2] 等其他模式将监督从一种模态转移到另一种模态。

In relation to these previous efforts, our work studies purely visual unsupervised learning from video and tries to compare the meta-methodologies on common ground.

关于这些先前的努力，我们的工作研究了从视频中进行的纯视觉无监督学习，并试图在共同点上比较元方法。

### Evaluation protocols and backbones 
in most imagebased approaches have converged to ResNet-50 [39] encoders with ImageNet linear-classification protocol, and several smaller downstream tasks [36, 12, 32, 9] for evaluation. In video understanding research, the field has not yet converged and is using different backbones with focus on finetuning performance on two relatively small datasets [77, 50]. We investigate this aspect by looking at different encoders and 6 different downstream benchmarks for evaluation.

大多数基于图像的方法中的评估协议和骨干网已经融合到具有 ImageNet 线性分类协议的 ResNet-50 [39] 编码器，以及用于评估的几个较小的下游任务 [36、12、32、9]。 在视频理解研究中，该领域尚未融合，并且正在使用不同的主干，重点是在两个相对较小的数据集上微调性能 [77、50]。 我们通过查看不同的编码器和 6 个不同的下游基准进行评估来研究这方面。

## 3. Approach
The objective of this work is to study several recent unsupervised representation learning methodologies to train a spatiotemporal encoder $f_θ$, exploring implementation details and comparing them on a common ground to measure their efficacy in video understanding. We focus on two contrastive approaches using positive and negative samples: SimCLR [12] and MoCo [36], as well as two approaches that solely rely on positives, BYOL [32] and SwAV [9] (Sec. 3.2).

这项工作的目的是研究最近的几种无监督表示学习方法来训练时空编码器 $f_θ$，探索实现细节并在共同点上比较它们以衡量它们在视频理解中的功效。 我们关注两种使用正样本和负样本的对比方法：SimCLR [12] 和 MoCo [36]，以及两种完全依赖正样本的方法，BYOL [32] 和 SwAV [9](第 3.2 节)。

These approaches were originally presented for learning image representations, and they all share the objective of learning invariant features across different views (crops/augmentations) of the spatial image input. In this paper, this idea is extended to the temporal domain. Our core idea is to learn an encoder $f_θ$ that produces embeddings which are persistent in space-time, over multiple (ρ) temporally distant clips of the same video. This is related to Slow Feature Analysis [88] where the objective is to minimize the representations’ temporal derivative over the input. The general idea of learning temporally persistent features is not new and has been proposed in the past with similar motivation e.g., [6, 61, 29].

这些方法最初是为学习图像表示而提出的，它们都共享跨空间图像输入的不同视图(裁剪/增广)学习不变特征的目标。 在本文中，这个想法被扩展到时空输入。 我们的核心思想是学习一个编码器 $f_θ$，它产生在同一视频的多个(ρ)时间上遥远的片段上在时空上持久的嵌入。 这与慢特征分析 [88] 有关，其目标是最小化表示对输入的时间导数。 学习时间持久性特征的一般思想并不新鲜，过去曾以类似的动机提出，例如 [6, 61, 29]。<!-- Slow Feature Analysis 时间持久性特征?-->

### 3.1. Persistent temporal feature learning 持续时间特征学习
Our framework takes different augmented clips x of an unlabeled video and passes them through an encoder $f_θ$ with weights θ to obtain corresponding embeddings $q = f_θ(x)$. The encoder is spatiotemporal ConvNet, by default a ResNet-50 (R-50) [39], Slow-only pathway of SlowFast Networks [20], which is a 3D ResNet-50 [39] without temporal pooling in convolutional feature maps, followed by an MLP projection head, that produces and output of dimension d.

我们的框架采用未标记视频的不同增广剪辑 x，并将它们传递给权重为 θ 的编码器 $f_θ$，以获得相应的嵌入 $q = f_θ(x)$。 编码器是时空 ConvNet，默认情况下是 ResNet-50 (R-50) [39]，SlowFast Networks [20] 的 Slow-only 通路，这是一个 3D ResNet-50 [39]，在卷积特征图中没有时间池， 接下来是一个 MLP 投影头，它产生和输出维度 d。

The input clips are stacks of RGB frames of size 3 × T × $S^2$ for temporal × spatial dimensions, which are sampled with temporal stride τ , i.e., the encoder processes only one out of τ frames of the raw video. Therefore, T × τ define the timespan and resolution of the encoder.

输入剪辑是大小为 3 × T × $S^2$ 的 RGB 帧堆栈，用于时间 × 空间维度，以时间步长 τ 采样，即编码器仅处理原始视频的 τ 帧中的一个。 因此，T × τ 定义了编码器的时间跨度和分辨率。

Given a minibatch of B videos, our framework creates a set of ρB positive examples by sampling ρ clips from the videos. The learning methodologies studied in this section maximize similarity of a “query” sample q with a set of positive “key” samples {$k^+$} that are encoded versions of different clips of the same video as q is computed from. Fig. 1 illustrates an example where ρ=3 clips are used.

给定一小批视频 B ，我们的框架通过从视频中采样 ρ 个片段来创建一组 ρB 个正例。 本节研究的学习方法最大化了“查询”样本 q 与一组正“关键”样本 {$k^+$} 的相似性，这些正“关键”样本是同一视频的不同剪辑的编码版本，作为 q 的计算来源。 图 1 说明了使用 ρ = 3 个片段的样本。

The next section describes how the contrastive and noncontrastive unsupervised representation learning methodologies are exemplified.

下一节将描述如何举例说明对比和非对比无监督表示学习方法。

Figure 2. Conceptual comparison of four unsupervised learning mechanisms applied to video. The inputs consist of ρ=2 clips from B videos. Each clip is a stack of T frames with temporal stride τ and spatial resolution $S^2$. Each method trains encoder weights θ by computing a positive loss component w.r.t. to the other clips of the same video. SimCLR (a) and MoCo (b) use a contrastive loss with negatives coming from different videos in the batch or a a queue. respectively. MoCo (b) and BYOL (c) use extra momentum encoders with weights $θ_m$ being moving averages of the trained θ. SwAV (d) uses a Sinkhorn-Knop (SK) transform to generate the positive targets.
图 2. 应用于视频的四种无监督学习机制的概念比较。 输入由来自 B 视频的 ρ=2 个片段组成。 每个剪辑都是 T 帧的堆栈，具有时间步长 τ 和空间分辨率 $S^2$。 每种方法都通过计算正损失分量 w.r.t 来训练编码器权重 θ。 到同一视频的其他剪辑。 SimCLR (a) 和 MoCo (b) 使用来自批次或队列中不同视频的负样本的对比损失。 分别。 MoCo (b) 和 BYOL (c) 使用额外的动量编码器，其权重 $θ_m$ 是经过训练的 θ 的移动平均值。 SwAV (d) 使用 Sinkhorn-Knop (SK) 变换生成正目标。

图2:应用于视频的四种无监督学习机制的概念比较。输入由来自B视频的ρ=2个剪辑组成。每个剪辑是T帧的堆栈，具有时间步长τ和空间分辨率$S^2$。每种方法通过计算同一视频的其他剪辑的正损失分量w.r.t.来训练编码器权重θ。SimCLR(a)和MoCo(b)使用来自批处理或队列中不同视频的负样本对比损失。分别地MoCo(b)和BYOL(c)使用额外的动量编码器，权重$θ_m$是训练θ的移动平均值。SwAV(d)使用Sinkhorn Knop(SK)变换来生成正目标。<!-- positive loss component 正损失分量 动量编码器？  negatives？-->

### 3.2. Unsupervised learning frameworks
Contrastive learning maximizes the similarity of a sample q with positive ones {$k^+$} and minimizes similarity to negative ones {$k^−$}. The contrastive approaches in this paper use the InfoNCE [83] objective, 

对比学习最大化样本 q 与正样本 {$k^+$} 的相似性，并最小化与负样本 {$k^−$} 的相似性。 本文中的对比方法使用 InfoNCE [83] 目标，

k∈{$k^+$} exp(sim(q,k)/α)

Lq = −log k∈{$k^+$,k−}exp(sim(q,k)/α), (1) 

with α being a temperature hyper-parameter for scaling and {$k^+$} are embedded clips of the same video as q. All the embeddings are $l_2$ normalized and dot product (cosine) similarity is used to compare them sim(q, k) = $q^⊤k/∥q∥∥k∥$.

其中 α 是用于缩放的温度超参数，{$k^+$} 是与 q 相同视频的嵌入剪辑。 所有嵌入都经过 $l_2$ 归一化，并使用点积(余弦)相似度来比较它们 sim(q, k) = $q^⊤k/∥q∥∥k∥$。

SimCLR [12] (Fig. 2a) uses the embeddings of clips of other videos in the minibatch as negatives {$k^−$}.

SimCLR [12](图 2a)使用小批量中其他视频剪辑的嵌入作为底片 {$k^−$}。

MoCo [36] (Fig. 2b) is a method that uses an explicit momentum encoder which parameters, $θ_m$, are a moving average $θ_m ← mθ_k + (1 − m)θ $ with m a momentum parameter. In eq. (1) MoCo uses this encoder to compute the positive embeddings {$k^+$} from clips of the same video as q, and negative embeddings {k−} are taken from a queue that stores embeddings of clips from previous iterations. There is no backpropagation into the momentum-encoder weights $θ_m$.

MoCo [36](图 2b)是一种使用显式动量编码器的方法，其参数 $θ_m$ 是移动平均值 $θ_m ← mθ_k + (1 − m)θ $ 其中 m 是动量参数。 在等式中。 (1) MoCo 使用此编码器计算来自与 q 相同视频的剪辑的正嵌入 {$k^+$}，负嵌入 {k−} 取自存储先前迭代剪辑嵌入的队列。 没有反向传播到动量编码器权重 $θ_m$。

BYOL [32] (Fig. 2c) can be viewed as a form of MoCo that does not use negative samples, but an extra predictor MLP with weights θp, which is stacked on top of $f_θ$’s MLP head. For a sample q = $f_θ$p ($f_θ$ (x)), BYOL minimizes negative cosine similarity,

BYOL [32](图 2c)可以看作是一种不使用负样本的 MoCo 形式，而是一个额外的具有权重 θp 的预测器 MLP，它堆叠在 $f_θ$ 的 MLP 头部之上。 对于样本 q = $f_θ$p ($f_θ$ (x))，BYOL 最小化负余弦相似度，

Lq = − 􏰁 sim(q, k) = − 􏰁 q⊤$k^+$/∥q∥∥k∥, (2) 

with {k^+ =f_θm (x+)} being embedded clips x+ from the same video as q, encoded with momentum weights $θ_m$.

其中 {k^+ =f_θm (x+)} 是来自与 q 相同视频的嵌入剪辑 x+，使用动量权重 $θ_m$ 编码。

SwAV [9] (Fig. 2d) can be viewed as a form of SimCLR that does not use negative samples. SwAV first performs a linear mapping of the positive embeddings q, $k^+$ to learned prototypes q ̃,k ̃+ and then transforms the targets with an extra Sinkhorn-Knopp (SK) step. Then the SwAV loss is

SwAV [9](图 2d)可以看作是一种不使用负样本的 SimCLR 形式。 SwAV 首先执行正嵌入 q, $k^+$ 到学习原型 q ̃,k ̃+ 的线性映射，然后使用额外的 Sinkhorn-Knopp (SK) 步骤转换目标。 那么 SwAV 损失是

Lq = DKL(q ̃∥SK(k ̃+)), (3) 

where DKL is the The Kullback-Leibler divergence and gradients are not back-propagated through the SK operation.

其中 DKL 是 Kullback-Leibler 散度，梯度不通过 SK 操作反向传播。

Compared to SimCLR and MoCo, in BYOL and SwAV, q and k are not typical “query” and “key” samples (but rather “source” and “target” samples); however, for consistency we use q, k terminology in notation for all methods.

与 SimCLR 和 MoCo 相比，在 BYOL 和 SwAV 中，q 和 k 不是典型的“查询”和“关键”样本(而是“源”和“目标”样本);  但是，为了保持一致性，我们在所有方法的符号中使用 q、k 术语。

Implementation specifics. We implement the methods with a symmetric loss, as in original SimCLR, BYOL and SwAV, where every input clip is used to produce a loss (and gradient) signal. For each of the ρ ≥ 2 clips, we compute q, while all other ρ−1 clips of the same video are used as {$k^+$} to evaluate sub-loss Lq and the symmetric loss is the average over all ρ sub-losses. Thus, for MoCo and BYOL, every input clip is processed by both encoders.

实施细节。 我们使用对称损失实现这些方法，如原始 SimCLR、BYOL 和 SwAV，其中每个输入剪辑都用于产生损失(和梯度)信号。 对于 ρ ≥ 2 个剪辑中的每一个，我们计算 q，而同一视频的所有其他 ρ−1 个剪辑用作 {$k^+$} 来评估子损失 Lq，对称损失是所有 ρ 的平均值 子损失。 因此，对于 MoCo 和 BYOL，每个输入剪辑都由两个编码器处理。

For MoCo and BYOL, our symmetric loss is aggregated sequentially which implies that memory consumption for ρ > 2 equals to a single clips’ forward and backward pass, since these methods do not backpropagate through the momentum encoder. For SimCLR and SwAV the overall loss is evaluated in parallel across all clips and therefore memory consumption grows linearly with the number of clips used. All details on implementation and pre-training are in §B.1.

对于 MoCo 和 BYOL，我们的对称损失是按顺序聚合的，这意味着 ρ > 2 的内存消耗等于单个剪辑的前向和后向传递，因为这些方法不通过动量编码器反向传播。 对于 SimCLR 和 SwAV，整体损失是在所有剪辑中并行评估的，因此内存消耗随着使用的剪辑数量线性增长。 有关实施和预训练的所有细节都在§B.1 中。

## 4. Experiments
Datasets. Unlessotherwisenoted,we perform unsupervised pre-training on Kinetics-400 [47] (K400) with ∼240k training videos in 400 human action categories. 

数据集。除非另有说明，我们在Kinetics-400[47](K400)上进行了监督预训练，在400个人类动作类别中播放了约240k的训练视频。

To study learning from “in-the-wild” videos from the web, we pre-train the methods on Instagram videos: IG-Curated [24], a dataset with hashtags similar to K400 classes; IG-Uncurated which has videos taken randomly from Instagram; and IG-Uncurated-Short which is similar, but has constrained duration. Each dataset has 1M videos.

为了研究从网络的“野外”视频中学习，我们在 Instagram 视频上预训练方法：IG-Curated [24]，一个带有类似于 K400 类的标签的数据集;  IG-Uncurated，其中有从 Instagram 随机样本的视频;  和 IG-Uncurated-Short 相似，但持续时间有限。 每个数据集有 100 万个视频。

Table 1. Pre-training data statistics with timings in seconds.
表1。预训练数据统计，计时单位为秒。

Table 1 shows dataset statistics of all datasets used for unsupervised pre-training. Most of Kinetics videos are of 10 seconds in duration. IG-Curated is a dataset with Instagram videos that have an average duration $t_{mean}$ of 26.3 seconds and a standard deviation $t_{std}$ of 29.8 seconds. The maximum duration $t_{max}$ is 60s. IG-Uncurated contains videos taken randomly from Instagram, with larger deviation in length and maximum duration of 10 minutes (600s). IG-UncuratedShort is a dataset consisting of random Instagram videos that have a duration between 10 and 16 seconds, to study the effect of a fixed duration and the assumption that short videos may hold more useful information for pre-training.

表1 显示了用于无监督预训练的所有数据集的数据集统计信息。 大多数 Kinetics 视频的时长为 10 秒。 IG-Curated 是一个包含 Instagram 视频的数据集，平均时长 $t_{mean}$ 为 26.3 秒，标准差 $t_{std}$ 为 29.8 秒。 最大持续时间 $t_{max}$ 为 60s。 IG-Uncurated 包含从 Instagram 随机截取的视频，长度偏差较大，最长时长为 10 分钟(600 秒)。 IG-UncuratedShort 是一个由随机 Instagram 视频组成的数据集，持续时间在 10 到 16 秒之间，用于研究固定持续时间的效果以及短视频可能包含更多有用信息以进行预训练的假设。

Evaluation protocols. For evaluation we use two protocols. The first one is common to evaluate unsupervised image representations [36, 12]. It validates the linear classifier performance based on frozen encoder features that are taken from the global average pooling layer. We report top-1 classification accuracy (%) on the K400 validation set.

评估协议。 我们使用两种协议进行评估。 第一个通常用于评估无监督图像表示 [36、12]。 它基于从全局平均池化层获取的冻结编码器特征来验证线性分类器的性能。 我们报告了 K400 验证集上的 top-1 分类精度 (%)。

The second protocol reports finetuning accuracy on the first split of the UCF101 dataset [77] which contains 13k videos in 101 human action classes; this is a common procedure used to evaluate unsupervised video representations. Finally, we also report finetuning accuracy on AVA [33], Charades [75], Something-Something [31] and HMDB51 [50].

第二个协议报告了 UCF101 数据集 [77] 第一次拆分的微调精度，该数据集包含 101 个人类行为类别中的 13k 个视频;  这是用于评估无监督视频表示的常用程序。 最后，我们还报告了 AVA [33]、Charades [75]、Something-Something [31] 和 HMDB51 [50] 的微调精度。

Architecture. By default, we use a R-50 [39] following the Slow pathway in [20] with clips of T =8 frames sampled with stride τ=8 from 64 raw-frames of video. The supervised performance for training 200, 400, 800 epochs on K400 is 74.7%, 74.3% and 72.7%, respectively, and does not improve for training longer due to overfitting.

架构。默认情况下，我们使用 R-50 [39] 遵循 [20] 中的慢速路径，其中 T = 8 帧的剪辑从 64 个原始视频帧中以步长 τ = 8 采样。 在 K400 上训练 200、400、800 个 epoch 的监督性能分别为 74.7%、74.3% 和 72.7%，并且由于过度拟合而不会随着训练时间的延长而提高。

Implementation details. We follow default settings in video classification [20]. Specifics on the approaches, their training and evaluation and the impact of implementation on performance are provided in §B and §A.3, respectively.

实施细节。 我们遵循视频分类中的默认设置 [20]。 §B 和 §A.3 分别提供了方法的具体细节、它们的培训和评估以及实施对绩效的影响。

### 4.1. Persistent temporal learning 持续时间学习
Here, we investigate the impact of learning spatiotemporal vs. only spatial persistent features. Table 2 shows the accuracy of the four methods when trained for 200 epochs on K400, and evaluated on K400 (linear) and UCF101 (finetuned), i.e. our default setting. 

在这里，我们研究了学习时空与仅空间持久特征的影响。 表 2 显示了四种方法在 K400 上训练 200 个时期并在 K400(线性)和 UCF101(微调)(即我们的默认设置)上进行评估时的准确性。

Table 2. Number of temporal clips ρ. Data: K400, 200 epochs. Learning temporally persistent features (ρ ≥ 2) is effective.
表 2. 时间剪辑 ρ 的数量。 数据：K400，200 个纪元。 学习时间持久性特征 (ρ ≥ 2) 是有效的。

Temporal augmentation. The first row in Table 2, ρ=1, uses two spatial crops at the same temporal instance, while the ρ=2 row uses clips at different temporal locations as positives; therefore, learns persistent features in time. This difference has a large impact on performance, especially for SimCLR (60.5 → 36.1) and SwAV (61.6 → 38.6) performance degrades significantly when sampling positives from the same temporal instance (ρ=1).

时间增强。 表 2 中的第一行，ρ=1，使用同一时间实例的两个空间裁剪，而 ρ=2 行使用不同时间位置的剪辑作为正例;  因此，及时学习持久性特征。 这种差异对性能有很大影响，尤其是对于 SimCLR (60.5 → 36.1) 和 SwAV (61.6 → 38.6)，当从同一时间实例 (ρ = 1) 采样正样本时，性能会显著下降。

More clips are beneficial. The remaining rows in Table 2 show that accuracy is further increasing with the number of temporal samples per video, e.g. at ρ=4 the best accuracy is achieved with BYOL at 68.9% K400 and 93.8% UCF101.

更多剪辑是有益的。 表 2 中的其余行表明，准确性随着每个视频的时间样本数量而进一步增加，例如 在 ρ = 4 时，使用 BYOL 可达到 68.9% K400 和 93.8% UCF101 的最佳准确度。

Negatives do not help but momentum encoders do. When comparing the methods in Table 2, we see that: 
1. There is no clear performance difference between contrastive/non-contrastive methods. This indicates that learning space-time persistence within a video is key for the methods, but learning in-persistence across videos is not. 
2. There is a clear difference of ∼4% on K400 between methods that employ momentum encoders (MoCo, BYOL), vs. these that do not (SimCLR, SwAV).

负样本没有帮助，但动量编码器有帮助。 比较表 2 中的方法时，我们看到：
1. 对比/非对比方法之间没有明显的性能差异。 这表明在视频中学习时空持久性是这些方法的关键，但跨视频学习非持久性不是。
2. 在使用动量编码器(MoCo、BYOL)的方法与不使用动量编码器的方法(SimCLR、SwAV)之间，K400 上存在明显差异 ~4%。

Table 3. Training duration in epochs (ep): Dataset: K400, ρ=2. Training longer brings consistent gains for all methods up to 400 epochs and saturates for K400 but not for UCF101 at 800ep. SwAV is the strongest performer for short training (50ep).
表 3. 训练持续时间(ep)：数据集：K400，ρ=2。 更长时间的训练为所有方法带来了一致的收益，最多 400 个 epochs，并且 K400 饱和，但 UCF101 在 800ep 时不饱和。 SwAV 是短期训练 (50ep) 中表现最好的。

Increasing the number of clips per training iteration increases training cost, so it is reasonable to compare it to training more epochs. Table 3 is studying the base case ρ=2 for various number of epochs (ep).

增加每次训练迭代的剪辑数量会增加训练成本，因此将其与训练更多时期进行比较是合理的。 表3 正在研究不同时期数 (ep) 的基本情况 ρ=2。

Overall, the results show that there is a clear gain for training longer which has been also observed in image-related tasks [12, 36, 32, 9]. BYOL performs the worst when training short durations. This might be related to hyper-parameter settings which we do not adjust for this experiment (the original implementation [32] uses different hyper-parameters for different number of training epochs). 

总体而言，结果表明，在与图像相关的任务中也观察到更长时间的训练有明显的收益 [12、36、32、9]。 BYOL 在短时间训练时表现最差。 这可能与我们没有为此实验调整的超参数设置有关(原始实现 [32] 针对不同数量的训练时期使用不同的超参数)。

### 4.2. Timespan between positives 阳性之间的时间跨度
All experiments with ρ≥2 so far were using global temporal sampling of positives, which means that the clips can be sampled at unconstrained temporal locations from the input video. This might be counter-productive because if there is a long duration that has passed between a pair of positive clips they might no longer share the same semantic context for learning high-level features corresponding in time. 

到目前为止，所有 ρ≥2 的实验都使用全局时间采样的正样本，这意味着可以在输入视频的不受约束的时间位置对剪辑进行采样。 这可能会适得其反，因为如果在一对正片段之间经过了很长的持续时间，它们可能不再共享相同的语义上下文来学习及时对应的高级特征。

Table 4. Maximum frame distance for positives. Method: BYOL, ρ = 2. Training is surprisingly robust with increasing accuracy for increased distance between samples. Accuracy only (mildly) degrades when sampling positives that are more than 36 seconds apart when using uncurated (random) videos.
表 4. 阳性的最大帧距离。 方法：BYOL，ρ = 2。随着样本之间距离的增加，训练的准确性出奇地稳健。 在使用未经策划(随机)的视频时，如果对间隔超过 36 秒的正样本进行采样，准确性只会(轻微)降低。

This experiment is concerned with the maximum distance between the positive training samples. We use BYOL pretraining on K400, IG-Curated-1M and IG-Uncurated-1M and report 400 linear readout accuracy in Table 4.

这个实验关注的是正训练样本之间的最大距离。 我们在 K400、IG-Curated-1M 和 IG-Uncurated-1M 上使用 BYOL 预训练，并在表 4 中报告了 400 线性读出精度。

Table 4a shows performance for increasing the maximum temporal distance between positives in K400 pre-training. It can be seen that using positives from the same time ($t_{max}$=0) degrades perforance b ∼5% but other than that performance is relatively robust up to global sampling of positive clips from the whole video ($t_{max}$=10s). This is interesting as it seems that a long-temporal correspondence objectives does not hurt performance (but also does not boost it).

表 4a 显示了在 K400 预训练中增加阳性之间最大时间距离的性能。 可以看出，同时使用正片($t_{max}$=0)会降低性能 b ∼5%，但除此之外，性能相对稳健，直到对整个视频中的正片进行全局采样($t_{ 最大值}$=10s)。 这很有趣，因为看起来长期对应目标不会损害性能(但也不会提高性能)。

Table 4b shows performance for increasing the temporal distance between positive samples on IG-Curated-1M. This dataset has a maximum duration of 60 seconds; statistics are in Table 1. Table 4b reveals that increasing the maximum duration between positive pairs is beneficial for performance and unrestricted sampling of positives is the best with 64.1% top-1 accuracy for evaluation on K400. This is especially interesting, as it shows that even longer videos benefit from global sampling. There is no benefit from restricting the time window of positives, which can be interpreted as the objective of learning extremely-slow features [88] that do not change over 60 seconds of video. Long-temporal-distance samples might also increase robustness of the model by providing “hard-positive” samples for learning. Note that here the videos are still sampled according to hashtags related to K400 classes [24]; therefore, the conjecture might be biased.

表 4b 显示了在 IG-Curated-1M 上增加阳性样本之间时间距离的性能。 该数据集的最长持续时间为 60 秒;  统计数据在表 1 中。表 4b 表明，增加正对之间的最大持续时间有利于性能，并且对正的无限制采样是最好的，在 K400 上的评估精度为 64.1% top-1。 这特别有趣，因为它表明甚至更长的视频都可以从全局采样中受益。 限制阳性的时间窗口没有任何好处，这可以解释为学习极慢特征的目标 [88]，这些特征不会超过 60 秒的视频发生变化。 长时间距离样本还可以通过为学习提供“硬阳性”样本来提高模型的稳健性。 请注意，这里的视频仍然是根据与 K400 类 [24] 相关的主题标签进行采样的;  因此，推测可能有失偏颇。

Finally, we are looking at the IG-Uncurated-1M dataset which consists of a random sampling of 1M videos from Instagram. These videos can be between 0.5s and 10 minutes of duration. Most of the videos however are much shorter than 10 minutes, with a mean duration of 35.3 seconds and a standard deviation of 38.4 seconds (Table 1). For this data, Table 4c shows the results of progressively increasing the maximum timespan between positive samples. It can be observed that increasing the maximum distance between positives up to 36 seconds is beneficial and beyond that performance decreases, but only slightly, even when performing global sampling of positives (the default).

最后，我们正在研究 IG-Uncurated-1M 数据集，它由来自 Instagram 的 1M 视频的随机抽样组成。 这些视频的持续时间可以在 0.5 秒到 10 分钟之间。 然而，大多数视频都短于 10 分钟，平均持续时间为 35.3 秒，标准差为 38.4 秒(表 1)。 对于此数据，表 4c 显示了逐渐增加阳性样本之间的最大时间跨度的结果。 可以观察到，将正例之间的最大距离增加到 36 秒是有益的，超过此性能会降低，但只是轻微的，即使在执行正例的全局采样(默认值)时也是如此。

### 4.3. Backbone architectures
So far all experiments were using a R-50, 8×8 Slow pathway [39, 20] as backbone. The next set of ablations studies different architectures for the spatiotemporal encoder. 

到目前为止，所有实验都使用 R-50、8×8 慢速通路 [39、20] 作为骨架。 下一组消融研究时空编码器的不同架构。

Table 5. Backbone comparison. The ResNet [39] backbone (Slow pathway [20]) is used with different depth (R-18, R-50, R-101), input frames T and stride τ. R2+1D [82] and S3D-G [90] are commonly used backbones for unsupervised video representation learning with downstream evaluation on UCF101.
表 5. 骨干比较。 ResNet [39] 主干(慢路径 [20])用于不同深度(R-18、R-50、R-101)、输入帧 T 和步幅 τ。 R2+1D [82] 和 S3D-G [90] 是无监督视频表示学习的常用主干，在 UCF101 上进行下游评估。

Table 5 compares different backbones for usage with MoCo in our default setting (ρ=2, 200 epoch pre-training on K400). From left to right, the table shows the input duration T , sampling-rate τ , FLOPs (at 2242 spatial resolution) and parameters of these backbones, as well as the average duration for training one iteration of the MoCo algorithm (measured on a single machine with 8 V100 GPUs in PySlowFast [19] and torchvision decoder), the supervised performance on K400 and UCF101 (finetuned from K400), as well as the downstream performance for K400 linear evaluation and UCF101 finetuning.

表 5 比较了在我们的默认设置(ρ=2、K400 上的 200 个纪元预训练)中使用 MoCo 的不同骨干网。 从左到右，该表显示了输入持续时间 T、采样率 τ、FLOP(在 2242 空间分辨率下)和这些主干的参数，以及训练 MoCo 算法一次迭代的平均持续时间(在单个 在 PySlowFast [19] 和 torchvision 解码器中配备 8 个 V100 GPU 的机器)，K400 和 UCF101(从 K400 微调)的监督性能，以及 K400 线性评估和 UCF101 微调的下游性能。

The first observation in Table 5 is that for the Slow architecture [20], using shallower (R-18) or deeper (R-101) networks can influence supervised and downstream performance in a sizable manner, with MoCo, K400 evaluation benefiting from more parameters. Doubling the input framerate (8×8 → 16×4) boosts accuracy on UCF101.

表 5 中的第一个观察结果是，对于慢速架构 [20]，使用更浅的 (R-18) 或更深的 (R-101) 网络可以以相当大的方式影响监督和下游性能，MoCo、K400 评估受益于更多 参数。 将输入帧速率加倍 (8×8 → 16×4) 可提高 UCF101 的精度。

The second observation is that R2+1D [82] has a large gap on Kinetics (71.7% supervised vs. 57.2% unsupervised), while being remarkably strong on UCF101 (93.7%). This gap is also observed for S3D-G [90]. The reason for this might be that UCF101 is a small dataset which is easy to overfit and can benefit from fewer parameters. 

第二个观察结果是 R2+1D [82] 在动力学上有很大差距(71.7% 监督 vs. 57.2% 无监督)，同时在 UCF101 (93.7%) 上非常强大。 S3D-G [90] 也观察到了这种差距。 这样做的原因可能是 UCF101 是一个小数据集，很容易过度拟合并且可以受益于更少的参数。

### 4.4. Uncurated data and video duration 未经整理的数据和视频时长
Table 6. Training on curated (a), uncurated (b) and short duration video (c) data from the web. Longer training degrades performance for BYOL, possibly due to suboptimal hyper-parameters. ρ=2.
表 6. 对来自网络的精选 (a)、未精选 (b) 和短时视频 (c) 数据进行培训。 较长的训练会降低 BYOL 的性能，这可能是由于次优的超参数造成的。 ρ=2。

In Table 6 we show the performance of all four methodologies on IG-Curated-1M (a), IG-Uncurated-1M (b) and IG-Uncurated-Short-1M (c) for pre-training with 50 and 200 epochs. We make the following observations: 
1. Among the methods MoCo performs the best with e.g. 69.0% vs. second-best 64.3% of SwAV on curated data (a). 
2. MoCo and SwAV scale the best for training longer, gaining roughly 3-4% for 200ep vs. 50ep. 
3. On uncurated data, MoCo and SwAV perform ∼1% better on the unconstrained duration videos in Table 6b. 
4. BYOL and SimCLR show better performance on IG-Uncurated-Short (10-16s videos) in Table 6c, seemingly benefiting from shorter videos, but there is no clear benefit from either longer or shorter duration among all methods. 
5. BYOL degrades performance for training longer which might be due to the requirement of different hyperparameters for different schedules (as noted in Sec. 4.1).

在表 6 中，我们展示了所有四种方法在 IG-Curated-1M (a)、IG-Uncurated-1M (b) 和 IG-Uncurated-Short-1M (c) 上用于 50 和 200 个 epoch 的预训练的性能。 我们做出以下观察：
1. 在 MoCo 等方法中表现最好 69.0% 对比 SwAV 在精选数据上的第二好 64.3% (a)。
2. MoCo 和 SwAV 在训练时间更长的情况下规模最大，200ep 与 50ep 相比增加了大约 3-4%。
3. 在未整理的数据上，MoCo 和 SwAV 在表 6b 中的无约束时长视频上的表现要好 ∼1%。
4. BYOL 和 SimCLR 在表 6c 中的 IG-Uncurated-Short(10-16 秒视频)上显示出更好的性能，似乎受益于较短的视频，但在所有方法中，较长或较短的持续时间都没有明显的好处。
5. BYOL 会降低训练时间的性能，这可能是由于不同的计划需要不同的超参数(如第 4.1 节所述)。

We will return to this point in §A.1, where we show that increasing clips-size ρ can overcome this issue in BYOL, along with further studies on the trade-off against training more epochs, and dataset scale.

我们将在 §A.1 中回到这一点，我们将展示增加剪辑大小 ρ 可以克服 BYOL 中的这个问题，以及对训练更多时期和数据集规模的权衡的进一步研究。

### 4.5. Data augmentations
Importance of augmentations. Augmentations can have a major impact on visual unsupervised feature learning [12, 14]. In Fig. 3, we ablate spatial cropping (S), temporal clipping (T) and radiometric color (C) augmentations from the four unsupervised learning methods (e.g. “T S C” are the baselines using all augmentations and removing “S C” equals ρ=1 in Table 2). We make three main observations: 
1. Among the methods, MoCo and BYOL perform most robust for using fewer augmentations; their advantage over SimCLR and SwAV might be related to the momentum encoder which can provide extra augmentation in training.
2. When minimizing the augmentations by resizing the shorter size of the video to the input size of 224 and only cropping along the long side of the video (Base in Fig. 3), MoCo still provides 42.2% K400 linear accuracy, over BYOLs’ 32.4%, showing an advantage of the contrastive loss in a weak augmentation scenario. 
3. Among the augmentations, learning temporal (T) persistency, has the largest impact on performance, except for MoCo which benefits more from color (C) (incl. grayscale) augmentations. Especially SimCLR and SwAV show significant drops in performance when removing T, i.e. when extracting positive clips from the same instance in time.

增强的重要性。 增强可以对视觉无监督特征学习产生重大影响 [12, 14]。 在图 3 中，我们消融了四种无监督学习方法的空间裁剪 (S)、时间裁剪 (T) 和辐射颜色 (C) 增强(例如，“T S C”是使用所有增强的基线，删除“S C”等于 ρ= 表 2 中的 1)。 我们主要观察三点：
1. 在这些方法中，MoCo 和 BYOL 因使用较少的增强而表现最稳健;  它们优于 SimCLR 和 SwAV 的优势可能与可以在训练中提供额外增强的动量编码器有关。
2. 当通过将视频的较短尺寸调整为 224 的输入尺寸并仅沿视频的长边裁剪(图 3 中的基线)来最小化增强时，MoCo 仍然提供 42.2% 的 K400 线性精度，超过 BYOL 的 32.4%，显示出弱增强场景中对比损失的优势。
3. 在增强中，学习时间 (T) 持久性对性能的影响最大，MoCo 除外，它更多地受益于颜色 (C)(包括灰度)增强。 特别是 SimCLR 和 SwAV 在移除 T 时表现出显著的性能下降，即在及时从同一实例中提取正片段时。

Figure 3. Ablating augmentations. We explore temporal (T), spatial (S), and color (C) augmentations to learn persistent features. 
图 3.消融增强。 我们探索时间 (T)、空间 (S) 和颜色 (C) 增强来学习持久性特征。

In the remainder of this section, we explore using stronger augmentations than the default ones in previous experiments. We perform the ablations with MoCo in the basic setting of ρ = 2, 200 epochs K400 pre-training. 

在本节的其余部分，我们探索使用比之前实验中的默认增强更强的增强。 我们在 ρ = 2、200 个时期 K400 预训练的基本设置中使用 MoCo 进行消融。

Table 7. Radiometric augmentation. Method: MoCo, 200 epochs, ρ = 2. Dataset: K400. Stronger color augmentation in K400 pre-training can especially benefit UCF101 (+1.3%).
表 7. 辐射增强。 方法：MoCo，200 个时期，ρ = 2。数据集：K400。 K400 预训练中更强的颜色增强尤其有利于 UCF101 (+1.3%)。

Stronger color augmentation. In Table 7 color strength of 0.5 indicates the default one for MoCo [14], 0.75 and 1.0 increase the strength of randomly jittering brightness, contrast, saturation and hue proportionally.

更强的颜色增强。 在表 7 中，颜色强度 0.5 表示 MoCo [14] 的默认颜色强度，0.75 和 1.0 按比例增加随机抖动亮度、对比度、饱和度和色调的强度。

Table 7 shows that increasing it to 0.75 can improve K400/UCF101 accuracy. Increasing the random grayscale probability from 0.2 to 0.4 does not provide an improvement on either of the datasets. However, using a temporaldifference augmentation which randomly (with probability 0.2) first converts the frames to grayscale and then subtracts them across time, can increase K400 accuracy by 0.4%. Finally, using frame-rate jittering of ±50% of the original frame-rate does not improve K400 but UCF101 slightly.  

表 7 显示将其增加到 0.75 可以提高 K400/UCF101 的精度。 将随机灰度概率从 0.2 增加到 0.4 不会对任何一个数据集提供改进。 然而，使用随机(概率为 0.2)首先将帧转换为灰度然后跨时间减去它们的时间差异增强可以将 K400 精度提高 0.4%。 最后，使用原始帧率 ±50% 的帧率抖动不会改善 K400，但会略微改善 UCF101。

Table 8. Cropping augmentation. Method: MoCo, 200 epochs, ρ = 2. Dataset: K400. Stronger cropping and aspect ratio augmentation can be beneficial by +1.0% (K400) and 0.7% UCF101.

表 8. 种植增加。 方法：MoCo，200 个时期，ρ = 2。数据集：K400。 更强的裁剪和纵横比增加可以带来 +1.0% (K400) 和 0.7% UCF101 的好处。

Spatial cropping. Our default implementation uses VGGstyle [76, 39] cropping that randomly resizes the shorter spatial side of a video between [256, 320] pixels and takes a random 2242 crop extended over time to extract a clip [20].

空间裁剪。 我们的默认实现使用 VGGstyle [76, 39] 裁剪，它在 [256, 320] 像素之间随机调整视频的较短空间边的大小，并随时间扩展随机 2242 裁剪以提取剪辑 [20]。

Since unsupervised learning might benefit from more aggressive cropping, we explore Inception-style [80] cropping with aspect ratio augmentation that is commonly used in unsupervised learning from images [36, 12, 32, 9]. This cropping procedure randomly resizes the input area between a minimum scale and a maximum scale and jitters aspect ratio between 3/4 to 4/3, before taking a 2242 crop.

由于无监督学习可能受益于更积极的裁剪，我们探索了 Inception 风格 [80] 裁剪和宽高比增加，这通常用于无监督图像学习 [36, 12, 32, 9]。 在进行 2242 裁剪之前，此裁剪过程会在最小比例和最大比例之间随机调整输入区域的大小，并在 3/4 到 4/3 之间调整纵横比。

We do not change the cropping for downstream training, as this can drop accuracy significantly (by ∼2% on K400).

我们不更改下游训练的裁剪，因为这会显著降低准确性(在 K400 上降低约 2%)。

In Table 8 we ablate this approach for MoCo (the augmentation in the downstream evaluators are unchanged).

在表 8 中，我们为 MoCo 消融了这种方法(下游评估器中的增强没有改变)。

The first ablation shows the comparison of default cropping [76, 39] with a similar version that randomly crops a fraction between [0.49, 0.76] = [2242 /3202 , 2242 /2562 ] of the original area, instead of the short-side. The performance degrades by 1% on K400 linear evaluation. Randomly cropping based on area favors larger crops over the short-side resizing and we observe lower training error for this variant.

第一个消融显示了默认裁剪 [76, 39] 与随机裁剪原始区域 [0.49, 0.76] = [2242 /3202 , 2242 /2562 ] 之间的一小部分的类似版本的比较，而不是短边 . 在 K400 线性评估中，性能下降了 1%。 基于区域的随机裁剪比短边大小调整有利于更大的裁剪，我们观察到该变体的训练误差较低。

Next, adding aspect ratio augmentation can recover some of this performance (65.4%), and using a smaller minimum area of 0.2, with the maximum area of 0.76 leads to best performance of 66.8%. Using the default values for Inception [80] training, [0.08, 1.00], appears to be too aggressive.

接下来，添加纵横比增强可以恢复部分性能 (65.4%)，使用较小的最小面积 0.2，最大面积 0.76 可获得 66.8% 的最佳性能。 使用 Inception [80] 训练的默认值 [0.08, 1.00] 似乎过于激进。

Table 9. Stronger augmentations. Data: K400, 200 epochs. “aug+’’ combines the best color and cropping augmentations from Table 7 and Table 8, respectively.
表 9. 更强的增强。 数据：K400，200 个纪元。 “aug+”分别结合了表 7 和表 8 中最好的颜色和裁剪增强。

Combined augmentations. We pull together the best color and cropping augmentations in Tables 7 & 8, and train MoCo and BYOL with ρ=4 for 200ep on K400. The result shown as “aug+” in Table 9 can increase performance on K400 by ∼1%. Training the linear classifier of BYOL (ρ=4) for 100ep instead of 60ep leads to our best accuracy of 70.0% on K400, which is 4.7% below the supervised R-50, Slow 8×8 accuracy of 74.7%. 

联合增强。 我们将表 7 和表 8 中的最佳颜色和裁剪增强结合在一起，并在 K400 上训练 200ep 的 MoCo 和 BYOL，其中 ρ=4。 表 9 中显示为“aug+”的结果可以将 K400 上的性能提高约 1%。 为 100ep 而不是 60ep 训练 BYOL (ρ=4) 的线性分类器导致我们在 K400 上的最佳精度为 70.0%，比受监督的 R-50、Slow 8×8 精度低 4.7% 74.7%。

### 4.6. Alternative downstream tasks  替代下游任务
The gap between K400 and UCF101 accuracy in Sec. 4.3 question if solely looking at typical evaluation of UCF101 (or the smaller HMDB51) is enough to identify and rank approaches for unsupervised learning in video.

K400 和 UCF101 精度之间的差距。 4.3 问题是否仅查看 UCF101(或较小的 HMDB51)的典型评估就足以识别和排名视频中无监督学习的方法。

Table 10 studies several new downstream tasks for unsupervised representation learning in video. We use our MoCo, SimCLR, BYOL and SwAV models trained with ρ=3 for 200 epochs on K400 and evaluate their performance by finetuning on Charades [75], AVA [33], or SomethingSomething [31] (in addition to the K400 linear readout performance and UCF101 performance reported in Table 2). Details on implementation are given in §B.

表 10 研究了视频中无监督表示学习的几个新的下游任务。 我们使用我们的 MoCo、SimCLR、BYOL 和 SwAV 模型，在 K400 上用 ρ=3 训练了 200 个时期，并通过在 Charades [75]、AVA [33] 或 SomethingSomething [31] 上进行微调来评估它们的性能(除了 K400 线性 表 2 中报告的读出性能和 UCF101 性能)。 §B 中给出了有关实施的详情。

The first two rows in Table 10 show the two main competitors for this evaluation: (i) training from scratch on the datasets and (ii) K400 pre-training. First, we observe that the supervised pre-trained backbones outperform the trainfrom-scratch counterpart significantly, as expected.

表 10 中的前两行显示了此评估的两个主要竞争对手：(i) 在数据集上从头开始训练和 (ii) K400 预训练。 首先，我们观察到受监督的预训练主干明显优于从头开始训练的主干，正如预期的那样。

Downstream datasets. For K400 pre-training and linear evaluation, its supervised counterpart has an advantage between 12.7% and 6.4% top-1 accuracy among the methods.

下游数据集。 对于 K400 预训练和线性评估，其受监督的对应方法在这些方法中具有 12.7% 和 6.4% 的 top-1 精度优势。

On UCF101 unsupervised pre-training is only 1% lower than the supervised counterpart for BYOL (the strongest).

在 UCF101 上，无监督预训练仅比 BYOL 的有监督预训练低 1%(最强)。

On AVA short-term action detection we observe that the BYOL pre-trained model is able to outperform the supervised counterpart by +1.2% mAP, when using the same, fixed region proposals [20]. This result is significant, as e.g. switching from K400 to K600 (nearly double the size of K400) pre-training on AVA leads to a smaller gains in performance [20]. Overall this is a surprising result as the tasks in K400 and AVA are similar [52], only that the temporal granularity of the actions in AVA is finer while their semantic granularity is coarser; e.g. “shoot” in AVA vs. “playing paintball” in Kinetics, which might be better captured by the BYOL objective which solely works on positive temporal samples of a video, without contrasting them to other videos (“shoot” might be a positive appearing in many different videos and contrasting them could be harmful to downstream performance). This line of thinking is supported with MoCo’s (contrastive objective) performance that is 3.1% worse than BYOL on AVA. Similarly, SimCLR (contrastive) is worse than SwAV (non-contrastive) when benchmarked on AVA.

在 AVA 短期动作检测中，我们观察到 BYOL 预训练模型在使用相同的固定区域提议 [20] 时能够比受监督模型高出 +1.2% mAP。 这个结果很重要，例如 从 K400 切换到 K600(几乎是 K400 大小的两倍)，AVA 预训练导致性能提升较小 [20]。 总体而言，这是一个令人惊讶的结果，因为 K400 和 AVA 中的任务相似 [52]，只是 AVA 中动作的时间粒度更细，而语义粒度更粗;  例如 AVA 中的“shoot”与 Kinetics 中的“play paintball”，BYOL objective 可能会更好地捕捉到这一点，BYOL objective 仅适用于视频的正时间样本，而不将它们与其他视频进行对比(“shoot”可能是出现在 许多不同的视频和对比它们可能对下游性能有害)。 这种思路得到了 MoCo(对比目标)性能的支持，该性能比 AVA 上的 BYOL 差 3.1%。 同样，在 AVA 上进行基准测试时，SimCLR(对比)比 SwAV(非对比)差。

On Charades, long-term action classification, we observe the opposite. Here, the contrastive MoCo is clearly the best performer with 33.5% mAP (close to the supervised pre-training performance of 34.7% mAP), while the noncontrastive BYOL is 12.5% lower. Similarly, now SimCLR (contrastive) is better than SwAV (non-contrastive). Compared to AVA, Charades is a temporally less localized dataset containing activities that need to be recognized from a longer temporal range video, for which contrastive pre-training appears to be outperforming the non-contrastive variants.

在 Charades 长期行动分类上，我们观察到相反的情况。 在这里，对比 MoCo 显然表现最好，具有 33.5% mAP(接近监督预训练性能 34.7% mAP)，而非对比 BYOL 低 12.5%。 同样，现在 SimCLR(对比)优于 SwAV(非对比)。 与 AVA 相比，Charades 是一个时间上较少局部化的数据集，其中包含需要从较长时间范围的视频中识别的活动，对比预训练似乎优于非对比变体。

Table 10. Downstream benchmarks: We use linear evaluation on K400 and finetuning accuracy on the other datasets. 200 epochs. ρ=3.  
表 10. 下游基准：我们在 K400 上使用线性评估，在其他数据集上使用微调精度。 200 个纪元。 ρ=3。

On Something-Something v2 (SSv2 in Table 10), all the methods perform strong, with BYOL pre-training showing the largest gain of +3% over supervised pre-training on Kinetics (55.8% vs. 52.8% top-1 accuracy).

在 Something-Something v2(表 10 中的 SSv2)上，所有方法都表现出色，BYOL 预训练显示出超过 Kinetics 监督预训练的最大增益 +3%(55.8% vs. 52.8% top-1 准确度) .

Pre-training sets: Kinetics vs. IG. Next, we experiment with pre-training on videos from the web. We first investigate IG-Curated-1M [24], which is a dataset that has been collected with hashtags that are similar to Kinetics labels. This data is a 1M subset of the original 65M introduced in [24]. Using this data (penultimate row in Table 10) can excel the performance of MoCo with K400 pre-training, which has a training set of 240K samples (roughly 4.2× smaller), and surprisingly even outperforms pre-training on K400 linear readout itself (69.9% vs. 67.3% accuracy).

预训练集：Kinetics vs. IG。 接下来，我们对来自网络的视频进行预训练实验。 我们首先研究 IG-Curated-1M [24]，这是一个使用类似于 Kinetics 标签的主题标签收集的数据集。 该数据是 [24] 中引入的原始 65M 的 1M 子集。 使用此数据(表 10 中的倒数第二行)可以超越 MoCo 与 K400 预训练的性能，它具有 240K 样本的训练集(大约小 4.2 倍)，并且令人惊讶的是甚至优于 K400 线性读数本身的预训练( 69.9% 与 67.3% 的精度)。

Second, we ablate the effect of using uncurated videos, with IG-Uncurated-1M which are purely random videos taken from the web. On most downstream tasks performance shown in the last row of Table 10 is equal or only slightly lower than pre-training on K400. Specifically, MoCo changes by -1.3% on K400 (as expected), +0.1% on UCF, +0.2% on AVA, -2.2% on Charades and -1.2% on SomethingSomething v2. This is an encouraging result for unsupervised learning, as only ∼4.2×the number of videos but random ones are required to match the performance of supervised K400 pre-training on the UCF101 and AVA.

其次，我们使用 IG-Uncurated-1M 消除了使用未经策划的视频的影响，这些视频是从网络上截取的纯随机视频。 在表 10 最后一行显示的大多数下游任务中，性能等于或仅略低于 K400 上的预训练。 具体来说，MoCo 在 K400 上变化 -1.3%(如预期)，在 UCF 上变化 +0.1%，在 AVA 上变化 +0.2%，在 Charades 上变化 -2.2%，在 SomethingSomething v2 上变化 -1.2%。 这对于无监督学习来说是一个令人鼓舞的结果，因为只需要大约 4.2 倍的视频数量，但需要随机视频来匹配 UCF101 和 AVA 上有监督的 K400 预训练的性能。

Overall, our results indicate that unsupervised pretraining can be a new paradigm for all of these downstream tasks, for which supervised pre-training is the de-facto standard to achieve best performance. Further, the large difference in performance for pre-training methodologies and objectives (e.g. contrastive/non-contrastive) revealed in the light of these benchmarks signals large room for future work.

总的来说，我们的结果表明，无监督预训练可以成为所有这些下游任务的新范例，而有监督预训练是实现最佳性能的实际标准。 此外，根据这些基准所揭示的预训练方法和目标(例如对比/非对比)在性能上的巨大差异预示着未来工作的巨大空间。

### 4.7. Comparison to previous work 与之前工作的比较
In a final experiment we take the best model from Table 9 and compare it with the state-of-the-art using the commonly used protocols on UCF101 and HMDB51 (across all 3 train/val splits) and K400. In Table 11 we show the results.

在最终实验中，我们采用表 9 中的最佳模型，并使用 UCF101 和 HMDB51(跨越所有 3 个训练/验证拆分)和 K400 上的常用协议将其与最先进的模型进行比较。 在表 11 中，我们显示了结果。

The strongest previous approaches are using multi-modal input, Vision “V”, Audio “A”, Text “T”, to train a contrastive objective across modalities; XDC [3] performs DeepCluster [8] on (V+A), CVRL [71], GDT [68] and MMV [2] use an objective similar to SimCLR on (V), (V+A), and (V+A+T), with the latter training on a Audioset (AS) [23] and HowTo100M (HT) [59], and CoCLR [35] can be seen as a variant of MoCo on rgb and optical-flow input.

以前最强大的方法是使用多模态输入、视觉“V”、音频“A”、文本“T”来训练跨模态的对比目标;  XDC [3] 在 (V+A) 上执行 DeepCluster [8]，CVRL [71]、GDT [68] 和 MMV [2] 在 (V)、(V+A) 和 (V) 上使用类似于 SimCLR 的目标 +A+T)，后者在 Audioset (AS) [23] 和 HowTo100M (HT) [59] 上进行训练，而 CoCLR [35] 可以看作是 MoCo 在 rgb 和光流输入上的变体。

Table 11. Comparison with state-of-the-art. “param” indicates the number of parameters, T inference frames, in the backbone. “V” is Vision, “A” is Audio, “T” Text modality. ρBYOL is our best model trained with temporal persistency of ρ=4. We report fine-tuning accuracy on UCF/HMDB and linear accuracy on K400. 
表 11. 与最先进技术的比较。 “param”表示主干中参数的数量，T 个推理帧。 “V”是视觉，“A”是音频，“T”是文本模态。 ρBYOL 是我们用 ρ=4 的时间持久性训练的最佳模型。 我们报告了 UCF/HMDB 的微调精度和 K400 的线性精度。

In comparisons, our best performing model ρBYOL, which is BYOL trained with temporal persistency over ρ=4 clips, (cf. Tables 2 & 9), provides a substantial performance gain over the best published method [35]: +5.7% and +12.1% top-1 accuracy on UCF101 and HMDB51 (using identical backbone and pre-training data).

相比之下，我们性能最好的模型 ρBYOL 是在 ρ=4 个片段上进行时间持久性训练的 BYOL(参见表 2 和 9)，与已发布的最佳方法 [35] 相比，性能显著提高：+5.7% 和 + UCF101 和 HMDB51 上的 top-1 精度为 12.1%(使用相同的主干和预训练数据)。

On K400 linear evaluation with the same data and R-50, Slow pathway [20] as backbone, our approach outperforms the concurrent CVRL [71] by +5.4% accuracy.

在使用相同数据和 R-50 的 K400 线性评估中，慢速通路 [20] 作为主干，我们的方法优于并发 CVRL [71] + 5.4% 的准确度。

## 5. Conclusion
This paper has studied four meta-methodologies for unsupervised learning from video. Our findings include that it is beneficial to sample positives with longer timespans between them, contrastive objectives are less influential than momentum encoders, and training duration, backbones, video augmentation and curation are all critical for good performance. Our resulting models which learn persistent features across augmented spacetime clips set a new state-of-the-art.

本文研究了四种用于视频无监督学习的元方法。 我们的发现包括，对具有较长时间跨度的正样本进行采样是有益的，对比目标比动量编码器影响更小，并且训练持续时间、主干、视频增强和管理对于良好的性能都是至关重要的。 我们生成的模型在增强时空片段中学习持久性特征，设置了一个新的最先进的技术。

We observed that linear readout on Kinetics is a good indicator of the performance on other datasets and that unsupervised pre-training can compete with the supervised counterpart on several datasets, but there is room for improvement. We hope that our baselines will foster research and provide common ground for future comparisons. 

我们观察到 Kinetics 上的线性读数是其他数据集性能的良好指标，无监督预训练可以在多个数据集上与有监督的预训练竞争，但仍有改进的空间。 我们希望我们的基线能够促进研究并为未来的比较提供共同点。

## Appendix
This appendix provides additional material: §A contains further results on “in-the-wild” data (§A.1), Kinetics-600 (K600) [10] and Kinetics-700 (K700) [11] data (§A.2) and on the effect of key implementation details (§A.3). §B contains additional implementation details for: Unsupervised pre-training (§B.1), and downstream evaluation in Kinetics (§B.2), AVA (§B.3), Charades (§B.4), SomethingSomething V2 (§B.5), UCF101 (§B.6), HMDB51 (§B.7).

本附录提供了额外的材料：§A 包含关于“野外”数据 (§A.1)、Kinetics-600 (K600) [10] 和 Kinetics-700 (K700) [11] 数据 (§A) 的进一步结果 .2) 以及关键实施细节的影响 (§A.3)。 §B 包含以下方面的其他实施细节：无监督预训练 (§B.1)，以及动力学中的下游评估 (§B.2)、AVA (§B.3)、Charades (§B.4)、SomethingSomething V2 ( §B.5)、UCF101 (§B.6)、HMDB51 (§B.7)。

### A. Additional Results
#### A.1. Scaling “in-the-wild” data
As a follow-up experiment Table 12 compares training BYOL longer (200ep) to increasing its clips-size ρ but not training longer (50ep). For both (a) curated and (b) random data, this results in a significant gain of performance.

作为后续实验，表 12 将 BYOL 训练更长时间 (200ep) 与增加其剪辑大小 ρ 但不训练更长时间 (50ep) 进行了比较。 对于 (a) 精选数据和 (b) 随机数据，这都会显著提高性能。

Table 12. More epochs (ep) vs. more clips (ρ), Longer training degrades performance for BYOL, but increasing ρ does not.
表 12. 更多时期 (ep) 与更多剪辑 (ρ)，更长的训练会降低 BYOL 的性能，但增加 ρ 不会。

We also explore an experiment for increasing the clipsize in MoCo and training longer (as MoCo works stable for more epochs). Table 13 shows the results. It can be observed that increasing the number of clips from ρ=2 to ρ=3 can increase the results by 1.6%/0.9% K400 and 0.4%/1% on UCF101 for 100/200ep training. Going to ρ = 4 brings further gain. In terms of efficiency, increasing ρ is both more accurate and faster than increasing the number of epochs, e.g. training MoCo (ρ=3, 100ep) takes only 63% of the duration that MoCo (ρ=2, 200ep) requires.

我们还探索了一个实验，用于增加 MoCo 中的 clipsize 和更长时间的训练(因为 MoCo 在更多时期内工作稳定)。 表 13 显示了结果。 可以观察到，将剪辑数量从 ρ=2 增加到 ρ=3 可以将结果增加 1.6%/0.9% K400 和 0.4%/1% 在 UCF101 上进行 100/200ep 训练。 达到 ρ = 4 会带来更多收益。 在效率方面，增加 ρ 比增加 epoch 的数量更准确和更快，例如 训练 MoCo (ρ=3, 100ep) 只需要 MoCo (ρ=2, 200ep) 所需时间的 63%。

Table 13. More epochs (ep) vs. more clips (ρ): Dataset: IGCurated-1M, ρ=2. Training longer is less effective than increasing the number of temporal clips per iteration (ρ).
表 13. 更多时期 (ep) 与更多剪辑 (ρ)：数据集：IGCurated-1M，ρ=2。 与增加每次迭代的时间剪辑数量 (ρ) 相比，训练更长时间的效果更差。

Finally, we remark that the IG-Curated-1M is subsampled such that the hastags are uniformly distributed (roughly balanced). Therefore this dataset is matching K400 in terms of content and distribution. We revisit this point next by investigating the effect of scale, curation and balancing of the video data.

最后，我们注意到 IG-Curated-1M 被二次采样，使得 hastags 均匀分布(大致平衡)。 因此，该数据集在内容和分布方面与 K400 匹配。 我们接下来通过调查视频数据的规模、管理和平衡的影响来重新审视这一点。

In this experiment, we increase the scale of the data from 128K to 1M distinct videos. We increase dataset size (number of videos) for IG-Curated [24], IG-CuratedUnbalanced [24] (which has random class distribution), and IG-Uncurated (which are random IG videos). The experiment with 200-epoch MoCo with ρ=2, linear protocol downstream evaluation on K400 is shown in Fig. 4 and reveals: 
1. Comparing the curation axis: At 240K training samples, the four data sources provide 65.8%, 63.2%, 63.1%, 60.6% top-1 accuracy for K400, IG-Curated, IG-CuratedUnbalanced and IG-Uncurated, respectively. The decay from the heavily curated K400 to IG-Curated (2.6%) is similar to the one from IG-Curated to IG-Uncurated (2.5%), while the class balancing seems to have a minor effect on accuracy. 
2. Comparing the scale axis: Doubling the data scale (number of videos) roughly linearly increases the accuracy across all datasets. With 1M uncurated videos the performance approaches 65.4% which is similar to the 65.8% produced by using K400 pre-training. The experiment indicates that it is possible to approach unsupervised Kinetics pretraining when using 4×more (1M vs. 240K in Kinetics), but random, videos when evaluating on Kinetics.

在这个实验中，我们将数据规模从 128K 增加到 1M 不同的视频。 我们增加了 IG-Curated [24]、IG-CuratedUnbalanced [24](具有随机类分布)和 IG-Uncurated(随机 IG 视频)的数据集大小(视频数量)。 图 4 显示了 200-epoch MoCo 的实验，其中 ρ = 2，K400 上的线性协议下游评估显示：
1. curation 轴比较：在 240K 训练样本时，四个数据源分别为 K400、IG-Curated、IG-CuratedUnbalanced 和 IG-Uncurated 提供 65.8%、63.2%、63.1%、60.6% 的 top-1 精度。 从精心策划的 K400 到 IG-Curated (2.6%) 的衰减与从 IG-Curated 到 IG-Uncurated (2.5%) 的衰减相似，而类平衡似乎对准确性影响很小。
2. 比较比例轴：将数据比例(视频数量)加倍大致呈线性增加所有数据集的准确性。 对于 100 万未经处理的视频，性能接近 65.4%，这与使用 K400 预训练产生的 65.8% 相似。 实验表明，在使用 4 倍以上(1M vs. 240K 的 Kinetics)但随机的视频进行 Kinetics 评估时，可以接近无监督 Kinetics 预训练。

Figure 4. Data scale and curation. We increase dataset size (number of videos) for IG-Curated, IG-Curated-Unbalanced, and IGUncurated. By using 4× the number of videos, IG-Uncurated approaches the heavily curated Kinetics (K400) pre-training on K400 linear evaluation protocol. The dotted line represents a linear trend. Method: MoCo, 200 epochs, ρ=2.
图 4. 数据规模和管理。 我们增加了 IG-Curated、IG-Curated-Unbalanced 和 IGUncurated 的数据集大小(视频数量)。 通过使用 4 倍的视频数量，IG-Uncurated 在 K400 线性评估协议上接近了精心策划的 Kinetics (K400) 预训练。 虚线表示线性趋势。 方法：MoCo，200 个周期，ρ=2。

#### A.2. Scaling Kinetics data
As referenced in Sec. 4 of the main paper, Table 14 shows a series of extra results for pre-training on the larger-scale Kinetics-600 (K600) [10] and Kinetics-700 (K700) [11] datasets, and is analyzed next: The first row of the table shows supervised training on the respective datasets, where UCF101 has two entries, one for training-from-scratch and one for using K400 as pre-training.

正如在第二节中所引用的那样。 主论文的第 4 节，表 14 显示了在更大规模的 Kinetics-600 (K600) [10] 和 Kinetics-700 (K700) [11] 数据集上进行预训练的一系列额外结果，接下来进行分析： 表的第一行显示了对各个数据集的监督训练，其中 UCF101 有两个条目，一个用于从头开始训练，一个用于使用 K400 作为预训练。

For the experiments we focus on our temporally persistent MoCo algorithm and, as in the main paper, evaluate Kinetics with the linear classification protocol and UCF101 by finetuning all weights. The first unsupervised row in Table 14 shows our best K400 pre-trained MoCo (ρ=4) model, achieving 69.0%, 70.0%, 54.2% and 93.6% on K400, K600, K700 and UCF101, respectively (this is the model with strong augmentations from Table 10 of the main paper).

对于实验，我们专注于我们的时间持久性 MoCo 算法，并且与主要论文中一样，通过微调所有权重，使用线性分类协议和 UCF101 评估动力学。 表 14 中的第一行无监督行显示了我们最好的 K400 预训练 MoCo (ρ=4) 模型，在 K400、K600、K700 和 UCF101 上分别达到 69.0%、70.0%、54.2% 和 93.6%(这是具有 来自主要论文表 10 的强大增强)。

Table 14. Dataset scale: Configuration: backbone: R-50, Slow 8 × 8, 200 epochs. Our approach, MoCo (ρ=4), is able to approach supervised pre-training on the popular UCF101 evaluation protocol, but there remains a gap for the linear protocol on K400, K600 and K700.

表 14. 数据集规模：配置：主干：R-50，慢速 8 × 8，200 个纪元。 我们的方法 MoCo (ρ=4) 能够在流行的 UCF101 评估协议上接近监督预训练，但在 K400、K600 和 K700 上的线性协议仍然存在差距。

The next row shows MoCo trained on K600 with a temporal persistency objective across two clips, ρ=2. This version is able to slightly outperform the K400 pre-trained variant on all datasets, except UCF101. Directly comparing this version with learning temporal persistency across ρ=4 clips can significantly increase accuracy on all datasets by ∼2%.

下一行显示了在 K600 上训练的 MoCo，具有跨越两个片段的时间持久性目标，ρ=2。 除了 UCF101 之外，该版本在所有数据集上的表现都略优于 K400 预训练变体。 直接比较这个版本与跨 ρ=4 个剪辑的学习时间持久性可以显著提高所有数据集的准确性 ~2%。

The final two rows of Table 14, show the same two models when pre-trained on K700. Here, we see that going from K400 to K700 increases accuracy by 2.7%, 3.2% and 3.9%, 1.2% on K400, K600, K700 and UCF101, respectively.

表 14 的最后两行显示了在 K700 上预训练时相同的两个模型。 在这里，我们看到从 K400 到 K700，在 K400、K600、K700 和 UCF101 上，精度分别提高了 2.7%、3.2% 和 3.9%，分别提高了 1.2%。

Overall the experiments suggest clear benefits of using larger-scale datasets for unsupervised pre-training and room for improvement under the linear classification protocol, especially when evaluated on larger datasets.

总体而言，实验表明使用大规模数据集进行无监督预训练的明显好处以及线性分类协议下的改进空间，尤其是在对较大数据集进行评估时。

#### A.3. Key implementation specifics
While the full implementation details of all four metamethodologies are provided in §B.1, we want to discuss the most impactful ones, which we found critical to achieve good performance in their realizations, throughout this section.

虽然 §B.1 中提供了所有四种元方法的完整实施细节，但我们希望在本节中讨论最具影响力的方法，我们发现这些方法对于在其实现中实现良好性能至关重要。

Table 15. Momentum annealing for MoCo. Dataset: K400, 200 epochs, ρ= 2. Using cosine-annealing of the momentum brings gains of ∼1% accuracy. We use 0.994 as default for MoCo.
表 15. MoCo 的动量退火。 数据集：K400，200 个 epochs，ρ= 2。使用动量的余弦退火带来 ∼1% 的精度增益。 我们使用 0.994 作为 MoCo 的默认值。

Momentum annealing. BYOL is using an annealing of the rate at which parameters of the momentum encoder $θ_m$, that are a moving average, with momentum m, of the trained encoder θ. During training BYOL starts with a momentum of mbase=0.996 and increases it to 1 with a cosine annealing m = 1 − (1 − mbase) · (cos(πk/K) + 1)/2 with k the current iteration and K the maximum number of training iterations [32] (this is unrelated to the learning rate decay).

动量退火。 BYOL 使用动量编码器参数 $θ_m$ 的速率退火，即经过训练的编码器 θ 的动量 m 的移动平均值。 在训练期间，BYOL 以 mbase=0.996 的动量开始，并通过余弦退火将其增加到 1 m = 1 − (1 − mbase) · (cos(πk/K) + 1)/2 其中 k 是当前迭代，K 是 最大训练迭代次数 [32](这与学习率衰减无关)。

By default MoCo, is using a fixed momentum of m = 0.999 during training. In Table 15, we ablate the positive (or negative) effect of using momentum annealing with different starting rates mbase for MoCo. We observe that not using any annealing (N/A) produces 64.5% accuracy and using momentum annealing can boost this performance by ∼1%, while being relatively stable for different values of mbase. Consequently, we are using momentum annealing with mbase = 0.994 for all our MoCo experiments. 

默认情况下，MoCo 在训练期间使用 m = 0.999 的固定动量。 在表 15 中，我们消除了对 MoCo 使用具有不同起始速率 mbase 的动量退火的正面(或负面)影响。 我们观察到，不使用任何退火 (N/A) 会产生 64.5% 的准确度，而使用动量退火可以将此性能提高 ∼1%，同时对于不同的 mbase 值相对稳定。 因此，我们对所有 MoCo 实验都使用 mbase = 0.994 的动量退火。

Figure 5. Key implementation specifics. BYOL, SimCLR, SwAV heavily rely on LARS, SyncBN, and BN in the MLP (MLP-BN), MoCo does not require these, but does not benefit of having them.
图 5. 关键实施细节。 BYOL、SimCLR、SwAV 严重依赖 MLP (MLP-BN) 中的 LARS、SyncBN 和 BN，MoCo 不需要这些，但拥有它们也无益。

Normalization and optimization. Here, we present normalization specifics that we found critical to achieve good performance in the underlying implementation of the methods: SimCLR, BYOL and SwAV are using synchronized Batch-Normalization (BN) [42] statistics (SyncBN) across 8 GPUs during training, batch-normalization after every MLP layer (MLP-BN), and a large-batch optimizer (LARS) [93]. LARS adaptively scales the learning rate for each individual parameter by using the ratio between gradient and parameter magnitudes. MoCo is not using these components (None) by default. In Fig. 5 we illustrate the results. It shows accuracy on K400 linear readout, if step-by-step adding these specifics to the methods. We make the following observations: 
1. Using None of the augmentations provides best performance for MoCo (its default) but significantly degrades BYOL, SimCLR and SwAV. Here, it is worth noting that BYOL provides decent accuracy of 32.9% without SyncBN, LARS and any BN in the MLP. 
2. Adding LARS optimizer reduces performance in MoCo and BYOL, while having a boost of around 10% for both SimCLR and SwAV. It is interesting, that solely using a more advanced optimizer, which adapts the learning rates of the weights according to their gradient magnitudes, decreases performance in methods using a momentum encoder (MoCo, BYOL), but boosts it without (SimCLR, SwAV). 
3. further adding SyncBN and MLP-BN increases BYOL performance dramatically; this related to recent studies [73] which suggest that normalization is important to achieve good performance using BYOL. 
4. While BYOL, SimCLR and SwAV do show further gains for adding SyncBN and MLP-BN, MoCo shows no significant change for using SyncBN, and degrades drastically in performance for using BN in the MLP-head.

归一化和优化。 在这里，我们展示了我们发现对于在方法的底层实现中实现良好性能至关重要的归一化细节：SimCLR、BYOL 和 SwAV 在训练、批处理期间跨 8 个 GPU 使用同步批处理归一化 (BN) [42] 统计信息 (SyncBN) - 在每个 MLP 层 (MLP-BN) 和大批量优化器 (LARS) [93] 之后进行归一化。 LARS 通过使用梯度和参数大小之间的比率自适应地缩放每个单独参数的学习率。 默认情况下，MoCo 不使用这些组件(无)。 在图 5 中，我们说明了结果。 如果逐步将这些细节添加到方法中，它会显示 K400 线性读数的准确性。 我们做出以下观察：
1. 不使用任何增强为 MoCo(其默认值)提供最佳性能，但显著降低 BYOL、SimCLR 和 SwAV。 在这里，值得注意的是，在 MLP 中没有 SyncBN、LARS 和任何 BN 的情况下，BYOL 提供了 32.9% 的不错准确性。
2. 添加 LARS 优化器会降低 MoCo 和 BYOL 的性能，同时将 SimCLR 和 SwAV 的性能提高约 10%。 有趣的是，仅使用更高级的优化器(根据权重的梯度大小调整权重的学习率)会降低使用动量编码器(MoCo、BYOL)的方法的性能，但在没有动量编码器(SimCLR、SwAV)的情况下会提高性能。
3. 进一步添加 SyncBN 和 MLP-BN 可显著提高 BYOL 性能;  这与最近的研究 [73] 有关，这些研究表明标准化对于使用 BYOL 实现良好性能很重要。
4. 虽然 BYOL、SimCLR 和 SwAV 确实显示出添加 SyncBN 和 MLP-BN 会带来更多收益，但 MoCo 显示使用 SyncBN 没有显著变化，并且在 MLP 头中使用 BN 会显著降低性能。

Projection MLP. It has been shown that using a deeper projection MLP in pre-training can increase the accuracy of the resulting representations for image classification [12, 14, 13]. Here, we investigate the effect of more hidden layers for video classification, across all four meta architectures. The results are shown in Table 16 and discussed next. 
1. MoCo achieves a significant gain of 1.2% on K400 for using a 3-layer (2 hidden layers) MLP vs. using a 2layer MLP and there is no gain for using a 4th layer. UCF performance appears stable to this modification. The gain is in line with results in image classification [14]. 
2. For BYOL, which has an additional Predictor MLP, with weights θp (see Fig. 2c), we ablate two dimensions: increasing the projection depth, and the prediction depth. Our results show that using 3-layer projection vs. 2-layer does not affect performance on K400, and has a decay of -0.7% on UCF101. Increasing also the depth of the predictor from our default value of 2 to 3 layers will lead to a significant decrease of -2.2% and -2.5% on both K400 and UCF101. 
3. SimCLR, shows similar behavior as MoCo: A consistent gain for using 3 projection layers (+1.5% on K400, +0.5% on UCF101), and no further gain for a 4-layer MLP. 
4. SwAV shows continuing gains on K400 for adding more MLP layers, +1.3% for going from 2 to 3 and another +0.4% for 4-layer MLP; however, its UCF-101 performance is decaying with more projection layers.

投影 MLP。 已经表明，在预训练中使用更深的投影 MLP 可以提高图像分类结果表示的准确性 [12、14、13]。 在这里，我们研究了所有四种元架构中更多隐藏层对视频分类的影响。 结果显示在表 16 中并在接下来进行讨论。
1. MoCo 在 K400 上使用 3 层(2 个隐藏层)MLP 与使用 2 层 MLP 相比获得了 1.2% 的显著增益，而使用第 4 层则没有增益。 UCF 性能似乎对此修改稳定。 增益与图像分类 [14] 中的结果一致。
2. 对于 BYOL，它有一个额外的预测器 MLP，权重为 θp(见图 2c)，我们消融了两个维度：增加投影深度和预测深度。 我们的结果表明，使用 3 层投影与 2 层投影不会影响 K400 的性能，并且在 UCF101 上有 -0.7% 的衰减。 将预测器的深度从我们的默认值 2 层增加到 3 层将导致 K400 和 UCF101 上的 -2.2% 和 -2.5% 显著下降。
3. SimCLR，表现出与 MoCo 类似的行为：使用 3 个投影层获得一致的增益(K400 上 +1.5%，UCF101 上 +0.5%)，4 层 MLP 没有进一步增益。
4. SwAV 显示在 K400 上持续增加更多的 MLP 层，从 2 层到 3 层增加 1.3%，4 层 MLP 增加 0.4%;  然而，它的 UCF-101 性能随着投影层的增加而衰减。

Overall, Table 16 suggests that K400 linear evaluation accuracy gernally benefits from deeper projection heads, while the performance for fine-tuned UCF101 downstream performance is relatively unchanged and rather shows a decaying effect for deeper MLPs. When studying the training complexity for pre-training, which we measure as floating point operations (FLOPs) and Parameters for the full training architecture (encoders + MLPs), Table 16 shows that FLOPs are mostly unchanged by deeper MLPs (as they operate on feature maps of size 1×1×1), but parameters increase leading to large models especially for momentum encoder based approaches (MoCo and BYOL).

总体而言，表 16 表明 K400 线性评估精度通常受益于更深的投影头，而微调的 UCF101 下游性能的性能相对未变，而是显示更深 MLP 的衰减效应。 在研究预训练的训练复杂性时，我们将其测量为浮点运算 (FLOP) 和完整训练架构(编码器 + MLP)的参数，表 16 显示 FLOPs 大部分没有被更深的 MLPs 改变(因为它们对特征进行操作 大小为 1×1×1 的地图)，但参数增加导致大型模型，特别是对于基于动量编码器的方法(MoCo 和 BYOL)。

### B. Additional Implementation Details
#### B.1. Unsupervised pre-training
Training details. We use the initialization outlined in [38]. The projection and prediction MLP weights are initialized with [27]. We optimize with synchronized SGD training on 64 GPUs with a mini-batch size of 8 clips per GPU; therefore, the total mini-batch size is 512. We train with Batch Normalization (BN) [42], and the BN statistics are computed within each 8 clips for MoCo and 64 clips by method MoCo synchronizing across 8 GPUs (SyncBN) for BYOL, SimCLR and SwAV. We adopt a half-period cosine schedule [56] of learning rate decaying: the learning rate at the n-th iteration is η·0.5[cos( n π)+1], where n is the maximum training nmax max iterations and the base learning rate η is set for each method to ηMoCo = 0.4, and ηSimCLR = ηBYOL = ηSwAV = 4.8. We apply (LARS) [93] (except for bias and BN parameters [32]), with trust coefficient of 0.001, for BYOL, SimCLR, and SwAV training. The SGD weight decay is 10−4 for MoCo and 10−6 for for BYOL, SimCLR and SwAV. The temperature parameter α = 0.1 for MoCo, SimCLR and SwAV. The projection MLP output dimensions are dMoCo = dSimCLR = ηSwAV = 128, and dBYOL = 256, as in their original publications [36, 12, 9, 32].

训练细节。 我们使用 [38] 中概述的初始化。 投影和预测 MLP 权重用 [27] 初始化。 我们通过 64 个 GPU 上的同步 SGD 训练进行优化，每个 GPU 的小批量大小为 8 个剪辑;  因此，总的小批量大小为 512。我们使用批量归一化 (BN) [42] 进行训练，BN 统计数据是在 MoCo 的每 8 个剪辑和 64 个剪辑中通过 MoCo 跨 8 个 GPU 同步 (SyncBN) 的 BYOL 方法计算的 、SimCLR 和 SwAV。 我们采用学习率衰减的半周期余弦时间表 [56]：第 n 次迭代的学习率为 η·0.5[cos( n π)+1]，其中 n 是最大训练 nmax 最大迭代次数， 将每种方法的基础学习率 η 设置为 ηMoCo = 0.4，以及 ηSimCLR = ηBYOL = ηSwAV = 4.8。 我们应用 (LARS) [93](偏差和 BN 参数 [32] 除外)，信任系数为 0.001，用于 BYOL、SimCLR 和 SwAV 训练。 MoCo 的 SGD 权重衰减为 10−4，BYOL、SimCLR 和 SwAV 的 SGD 权重衰减为 10−6。 MoCo、SimCLR 和 SwAV 的温度参数 α = 0.1。 投影 MLP 输出维度为 dMoCo = dSimCLR = ηSwAV = 128 和 dBYOL = 256，与其原始出版物 [36、12、9、32] 中一样。

Table 16. Varying depth of MLPs. Dataset: K400, 200 epochs, ρ=2. Training complexity is measured in floating point operations (FLOPs) and Parameters. Accuracy is reported as linear evaluation (K400) and fine-tuning (UCF101) of the backbone without MLPs. 
表 16. 不同深度的 MLP。 数据集：K400，200 个时期，ρ=2。 训练复杂性以浮点运算 (FLOP) 和参数来衡量。 准确性被报告为没有 MLP 的主干的线性评估 (K400) 和微调 (UCF101)。

MoCo details. We use a queue storing 65536 negatives and shuffling BN to avoid intra-batch communication among samples [36]. We use a 3-layer (2 hidden layers, ablation in Table 6 of the main paper) projection MLP with hidden dimension 2048, ReLU activation [63] and no BN. Other hyperparameters are as in [36, 14]. The momentum encoder weights $θ_m$ are updated with an annealed momentum m = 1 − (1 − mbase) · (cos(πk/K) + 1)/2 with k the current iteration and K the maximum number of training iterations [32], starting with mbase = 0.994. The corresponding ablation is in Table 3 of the main paper.

MoCo细节。 我们使用一个队列存储 65536 个底片并洗牌 BN 以避免样本之间的批内通信 [36]。 我们使用 3 层(2 个隐藏层，主论文表 6 中的消融)投影 MLP，隐藏维度为 2048，ReLU 激活 [63] 且没有 BN。 其他超参数如 [36, 14] 中所述。 动量编码器权重 $θ_m$ 用退火动量 m = 1 − (1 − mbase) · (cos(πk/K) + 1)/2 更新，其中 k 是当前迭代，K 是最大训练迭代次数 [32 ], 从 mbase = 0.994 开始。 相应的消融在主要论文的表 3 中。

BYOL details. Our BYOL implementation uses a momentum annealing starting from mbase = 0.996. We minimize the negative cosine similarity in equation (2) of the main paper multiplied by 2 which is equivalent to BYOL’s MSE of l2-normalized vectors [32]. The projection and prediction MLPs have 2 layers (one hidden layer with dimension 4096) and use BN following the original publication [32].

BYOL 详情。 我们的 BYOL 实施使用从 mbase = 0.996 开始的动量退火。 我们最小化主要论文的等式 (2) 中的负余弦相似度乘以 2，这相当于 BYOL 的 l2-归一化向量的 MSE [32]。 投影和预测 MLP 有 2 层(一个隐藏层，维度为 4096)并按照原始出版物 [32] 使用 BN。

SimCLR details. We follow the default implementation [12]. We use a 3-layer projection MLP with a hidden dimension of 2048, ReLU and BN. The loss in equation (1) of the main paper is computed synchronized over the full batch size. 

SimCLR 详情。 我们遵循默认实现 [12]。 我们使用隐藏维度为 2048、ReLU 和 BN 的 3 层投影 MLP。 主要论文的等式 (1) 中的损失是在整个批量大小上同步计算的。

SwAV details. We follow the default implementation [9], using 3 Sinkhorn-Knopp iterations [15] and freezing the prototypes for the first epoch. The Sinkhorn regularization parameter is set to 0.05. As in the default implementation [9], the matrix normalization statistics of the Sinkhorn-Knopp algorithm are computed synchronized over the full training batch. The projection MLP uses ReLU and BN and is identical to the one used in [9], only that we use a 3-layer MLP instead of 2 (ablations are in Table 6 of the main paper).

SwAV 细节。 我们遵循默认实现 [9]，使用 3 次 Sinkhorn-Knopp 迭代 [15] 并冻结第一个时期的原型。 Sinkhorn 正则化参数设置为 0.05。 与默认实现 [9] 一样，Sinkhorn-Knopp 算法的矩阵归一化统计在整个训练批次上同步计算。 投影 MLP 使用 ReLU 和 BN，与 [9] 中使用的相同，只是我们使用 3 层 MLP 而不是 2(消融在主要论文的表 6 中)。

Encoder details. Our default encoder, $f_θ$, is a R-50 Slow model [20], i.e. a ResNet-50 [39] with a temporal dimension of size T and sample rate τ . We perform all ablations with default T ×τ of 8×8. We show the architecture in Table 17.

编码器细节。 我们的默认编码器 $f_θ$ 是一个 R-50 慢速模型 [20]，即时间维度为 T 和采样率 τ 的 ResNet-50 [39]。 我们执行所有消融，默认 T×τ 为 8×8。 我们在表 17 中展示了架构。

Augmentation details. We perform video decoding and data augmentation using PyTorch’s torchvision package.

增强细节。 我们使用 PyTorch 的 torchvision 包执行视频解码和数据增广。

We obtain different clips from a video by the following procedure. For the temporal dimension, we randomly sample a clip (of T ×τ frames) from the full-length video, and the input to the ResNet encoder are T frames subsampled from the raw clip with a stride of τ; for the spatial dimension, we randomly crop 224×224 pixels from a video, or its horizontal flip, with a shorter side randomly sampled in [256, 320] pixels [20] (VGG-style [76, 39] spatial cropping, a comparison to Inception-style [80] cropping, which we use for results in §4.5, is given in Table 9 of the main paper).

我们通过以下过程从视频中获取不同的剪辑。 对于时间维度，我们从全长视频中随机采样一个片段(T×τ 帧)，ResNet 编码器的输入是从原始片段中二次采样的 T 帧，步幅为 τ;  对于空间维度，我们从视频或其水平翻转中随机裁剪 224×224 像素，短边在 [256, 320] 像素 [20] 中随机采样(VGG 样式 [76, 39] 空间裁剪，a 与我们在 §4.5 中用于结果的初始样式 [80] 裁剪的比较在主要论文的表 9 中给出)。

To each clip, we apply a random horizontal flip, color distortion and Gaussian blur following the SimCLR and MoCo v2 implementation [12, 14]. For color augmentation we use the ColorJitter (probability 0.8) and RandomGrayscale (probability 0.2) method from torchvision.transforms module of PyTorch with the color strength parameter s: {brightness, contrast, saturation, hue} = {0.4s, 0.4s, 0.4s, 0.1s} By default s=0.5. Ablations are given in Table 8 of the main paper. For Gaussian blur we use a spatial kernel with standard-deviation ∈ [0.1, 2.0] applied with probability of 0.5.

对于每个剪辑，我们按照 SimCLR 和 MoCo v2 实施 [12、14] 应用随机水平翻转、颜色失真和高斯模糊。 对于颜色增强，我们使用 PyTorch 的 torchvision.transforms 模块中的 ColorJitter(概率 0.8)和 RandomGrayscale(概率 0.2)方法，颜色强度参数 s：{brightness, contrast, saturation, hue} = {0.4s, 0.4s, 0.4 s, 0.1s} 默认 s=0.5。 主论文的表 8 中给出了消融。 对于高斯模糊，我们使用标准差 ∈ [0.1, 2.0] 的空间核，概率为 0.5。

#### B.2. Details: Kinetics Action Classification
Datasets. Kinetics-400 [47] consists of ∼240k training videos and 20k validation videos in 400 human action categories. Kinetics-600 [10] has ∼392k training videos and 30k validation videos in 600 classes. Kinetics-700 [11] has ∼523k training videos and 35k validation videos in 600 classes.

数据集。 Kinetics-400 [47] 由 400 个人类行为类别中的 240k 训练视频和 20k 验证视频组成。 Kinetics-600 [10] 在 600 个类别中有大约 392k 个训练视频和 30k 个验证视频。 Kinetics-700 [11] 在 600 个类别中有大约 523k 个训练视频和 35k 个验证视频。

Linear classification protocol. We validate the methods by linear classification on frozen features, following the common protocol in image classification [36]. After unsupervised pre-training on Kinetics, we freeze the features of the encoder and train a linear classifier on top of the last layer features (e.g. pool5 in Table 17). For all ablations in the paper the classifier is trained for 60 epochs (using 100 epochs will increase accuracy by ∼0.2%) using the same cosine schedule as for pre-training (Sec. B.1) with a base learning rate of η = 4.0 (10×higher than in pre-training), linear warm-up in the first 8 epochs, and weight decay of 0.

线性分类协议。 我们遵循图像分类中的通用协议 [36]，通过对冻结特征进行线性分类来验证这些方法。 在对 Kinetics 进行无监督预训练后，我们冻结了编码器的特征，并在最后一层特征(例如表 17 中的 pool5)之上训练了一个线性分类器。 对于本文中的所有消融，分类器使用与预训练(第 B.1 节)相同的余弦时间表训练 60 个时期(使用 100 个时期将提高精度约 0.2%)，基本学习率为 η = 4.0(比预训练高 10 倍)，前 8 个 epoch 线性预热，权重衰减为 0。

Table 17. R-50, Slow pathway [20]. The dimensions of kernels are denoted by {T×$S^2$, C} for temporal, spatial, and channel sizes. Strides are denoted as {temporal stride, spatial stride2}. Non-degenerate temporal filters are underlined. Residual blocks are in brackets. Temporal pooling is only performed at the last layer, collapsing spacetime dimensions. By default T ×τ = 8×8. 
表 17. R-50，慢速通路 [20]。 内核的维度用 {T×$S^2$, C} 表示，表示时间、空间和通道大小。 步幅表示为 {temporal stride, spatial stride2}。 非退化的时间过滤器带有下划线。 剩余块在括号中。 时间池仅在最后一层执行，折叠时空维度。 默认 T ×τ = 8×8。

Training augmentation. We use the default training augmentation [20]. We randomly sample a clip (of T ×τ frames) from the full-length video and randomly crop 224×224 pixels from a video, or its horizontal flip, with a shorter side randomly sampled in [256, 320] pixels.

训练增强。 我们使用默认的训练增强 [20]。 我们从全长视频中随机采样一个片段(T×τ 帧)，并从视频中随机裁剪 224×224 像素，或其水平翻转，较短的边随机采样 [256, 320] 像素。

Inference. Following common practice, in video classification [20], we report 30-view, top-1 classification accuracy on the Kinetics validation set. We uniformly sample 10 clips from a video along its temporal axis. For each clip, we scale the shorter spatial side to 256 pixels and take 3 crops of 256×256 to cover the spatial dimensions. We average the sof$t_{max}$ scores for prediction.

推理。 按照惯例，在视频分类 [20] 中，我们报告了 Kinetics 验证集上的 30 视图、top-1 分类精度。 我们沿时间轴从视频中统一采样 10 个片段。 对于每个剪辑，我们将较短的空间边缩放到 256 像素，并采用 3 次 256×256 的裁剪来覆盖空间维度。 我们平均预测的 sof$t_{max}$ 分数。

#### B.3. Details: AVA Action Detection
Dataset. The AVA dataset [33] has bounding box annotations for spatiotemporal localization of (possibly multiple) human actions. It has 211k training and 57k validation video segments. We follow the standard protocol reporting mean Average Precision (mAP) on 60 classes [33] on AVA v2.2.

数据集。 AVA 数据集 [33] 具有用于(可能多个)人类行为的时空定位的边界框注释。 它有 211k 训练和 57k 验证视频片段。 我们遵循 AVA v2.2 上 60 个类 [33] 的标准协议报告平均精度 (mAP)。

Detection architecture. We exactly follow the detection architecture in [20] to allow direct comparison of the pretrained models used as a backbone for the AVA task [33]. The detector is similar to Faster R-CNN [72] with minimal modifications adapted for video. Region-of-interest (RoI) features [26] are extracted at the last feature map of res5 (cf. Table 17) by extending a 2D proposal at a frame into a 3D RoI by replicating it along the temporal axis, followed by application of frame-wise RoIAlign [37] and temporal global average pooling. We set the spatial stride of res5 to 1 (instead of 2), and use a dilation of 2 for its filters [20]. This increases the spatial resolution of res5 by 2×. The RoI features are then max-pooled and fed to a per-class sigmoid classifier for prediction.

检测架构。 我们完全遵循 [20] 中的检测架构，以允许直接比较用作 AVA 任务 [33] 主干的预训练模型。 该检测器类似于 Faster R-CNN [72]，只对视频进行了最少的修改。 感兴趣区域 (RoI) 特征 [26] 在 res5 的最后一个特征图上提取(参见表 17)，方法是通过沿时间轴复制它，将帧中的 2D 建议扩展到 3D RoI，然后应用 逐帧 RoIAlign [37] 和时间全局平均池化。 我们将 res5 的空间步幅设置为 1(而不是 2)，并为其过滤器使用 2 的扩张 [20]。 这将 res5 的空间分辨率提高了 2 倍。 然后将 RoI 特征最大池化并馈送到每类 s 型分类器进行预测。

Training. Fordirectcomparison,thetrainingprocedureand hyper-parameters for AVA follow [20] without modification. The network weights are initialized from the Kinetics models and we use step-wise learning rate decay, that is reduced by 10× after 16, 24 and 28 epochs. We train for 32 epochs on ∼211k data, with linear warm-up [30] for the first 5 epochs and use a weight decay of 10−7, as in [20]. For 8 GPU training, we use a batch-size of 64, a learning rate of 0.05 for the supervised pre-trained Kinetics models and 0.3 for the unsupervised ones, as this gives the best result for each of them.

训练。 为了直接比较，AVA 的训练过程和超参数遵循 [20]，没有修改。 网络权重从动力学模型初始化，我们使用逐步学习率衰减，在 16、24 和 28 个时期后减少 10 倍。 我们在 ∼211k 数据上训练 32 个时期，前 5 个时期使用线性预热 [30]，并使用 10−7 的权重衰减，如 [20] 中所示。 对于 8 GPU 训练，我们使用 64 的批量大小，监督预训练动力学模型的学习率为 0.05，无监督模型的学习率为 0.3，因为这为每个模型提供了最佳结果。

The region proposal extraction also follows [20] and is summarized here for completeness. Our region proposals are computed by an off-the-shelf person detector, i.e., that is not jointly trained with the action detection models. We adopt a person-detection model trained with Detectron [25]. It is a Faster R-CNN with a ResNeXt-101-FPN backbone. It is pre-trained on ImageNet and the COCO human keypoint images [55]. We fine-tune this detector on AVA for person (actor) detection. The person detector produces 93.9 AP@50 on the AVA validation set. Then, the region proposals for action detection are detected person boxes with a confidence of > 0.8, which has a recall of 91.1% and a precision of 90.7% for the person class.

区域提案提取也遵循 [20]，并在此处进行总结以确保完整性。 我们的区域建议由现成的人员检测器计算，即未与动作检测模型联合训练。 我们采用用 Detectron [25] 训练的人体检测模型。 它是具有 ResNeXt-101-FPN 主干的 Faster R-CNN。 它在 ImageNet 和 COCO 人体关键点图像 [55] 上进行了预训练。 我们在 AVA 上微调此检测器以进行人员(演员)检测。 人体检测器在 AVA 验证集上产生 93.9 AP@50。 然后，用于动作检测的区域建议是置信度 > 0.8 的检测人员框，其召回率为 91.1%，人员类别的精度为 90.7%。

Inference. We perform inference on a single clip with 8 frames sampled with stride 8 centered at the frame that is to be evaluated.
推理。 我们对具有 8 帧采样的单个剪辑执行推理，步长 8 以要评估的帧为中心。

#### B.4. Details: Charades Action Classification
Dataset. Charades [75] has ∼9.8k training videos and 1.8k validation videos in 157 classes in a multi-label classification setting of longer activities spanning ∼30 seconds on average. Performance is measured in mean Average Precision (mAP).

数据集。 Charades [75] 在平均跨越 30 秒的较长活动的多标签分类设置中，拥有 157 个类别的 9.8k 训练视频和 18k 验证视频。 性能以平均精度 (mAP) 衡量。

Training. For Charades, we fine-tune the Kinetics models, but extend their duration by 2× (T ×τ = 16×8) to account for the long-term nature of the dataset. This increase accuracy of all models by ∼3 mAP. Our training augmentation is the same as as in §B.2. A per-class sigmoid output is used for mutli-class prediction. We train for 60 epochs using a batch size of 64 and a base learning rate of 0.2 (for 8 GPUs) with 10× step-wise decay at epoch 40 and 50, after warm-up in the first 5 epochs. We use weight decay of 10-4 and dropout of 0.5. Other training details are analogous to Kinetics.

训练。 对于 Charades，我们微调了动力学模型，但将它们的持续时间延长了 2 倍(T ×τ = 16 × 8)以考虑数据集的长期性质。 这将所有模型的准确性提高了 ∼3 mAP。 我们的训练增强与§B.2 中的相同。 每类 sigmoid 输出用于多类预测。 我们使用 64 的批量大小和 0.2 的基础学习率(对于 8 个 GPU)训练 60 个 epoch，在前 5 个 epoch 预热后，在第 40 和 50 个 epoch 进行 10× 逐步衰减。 我们使用 10-4 的权重衰减和 0.5 的 dropout。 其他训练细节类似于动力学。

Inference. This is as for Kinetics (§B.2), but to infer the actions over a single video, we spatiotemporally max-pool prediction scores in testing [20].

推理。 这与动力学 (§B.2) 一样，但为了推断单个视频的动作，我们在测试 [20] 中时空最大池预测分数。

#### B.5. Details: Something-Something V2 (SSv2)
Dataset. The Something-Something V2 dataset [31] contains 169k training, and 25k validation videos. The videos show human-object interactions to be classified into 174 classes. We report top-1 accuracy on the validation set.

数据集。 Something-Something V2 数据集 [31] 包含 169k 训练和 25k 验证视频。 这些视频显示人与物体的交互被分为 174 个类别。 我们报告验证集上的 top-1 准确性。

Training. We fine-tune the pre-trained Kinetics models. We train for 22 epochs using a batch size of 64 and a base learning rate of 0.12 (for 8 GPUs) with 10× step-wise decay at epoch 14 and 18. Weight decay is set to 10−6 and dropout 0.5. Our training augmentation is the same as in §B.2, but as Something-Something V2 requires distinguishing between directions, we disable random flipping during training. We use segment-based input frame sampling [54] that splits each video into segments, and from each of them, we sample one frame to form a clip.

训练。 我们微调预训练的动力学模型。 我们使用 64 的批量大小和 0.12 的基础学习率(对于 8 个 GPU)训练 22 个时期，在时期 14 和 18 处进行 10× 逐步衰减。权重衰减设置为 10-6，dropout 0.5。 我们的训练增强与§B.2 中的相同，但由于 Something-Something V2 需要区分方向，因此我们在训练期间禁用随机翻转。 我们使用基于片段的输入帧采样 [54]，将每个视频分成多个片段，我们从每个片段中采样一个帧以形成一个剪辑。

Inference. We perform single center clip testing to form predictions over a single video.

推理。 我们执行单中心剪辑测试以形成对单个视频的预测。

#### B.6. Details: UCF-101 Action Classification
Dataset. UCF101 [77] has 13320 human action videos in 101 categories. Our ablations are performed on the first train/val split, and for the comparison to prior work we report the mean average accuracy over the three splits.

数据集。 UCF101 [77] 有 101 个类别的 13320 个人类动作视频。 我们的消融是在第一次训练/验证拆分上进行的，为了与之前的工作进行比较，我们报告了三个拆分的平均准确度。

Training. We fine-tune the pre-trained Kinetics models and use the same augmentation as for Kinetics. We train for 200 epochs using a batch size of 64 and a base learning rate of 0.025 (for 8 GPUs) with 10× step-wise decay at epoch 60, 120 and 180. Weight decay is set to 0 and dropout to 0.8.

训练。 我们微调预训练的动力学模型并使用与动力学相同的增强。 我们使用 64 的批量大小和 0.025 的基础学习率(对于 8 个 GPU)进行 200 个时期的训练，在时期 60、120 和 180 处进行 10 倍的逐步衰减。权重衰减设置为 0，dropout 设置为 0.8。

Inference. WeusethesameprocedureasinKinetics(§B.2). 

推理。 我们使用与动力学相同的程序(§B.2)。

#### B.7. Details: HMDB-51 Action Classification
Dataset. HMDB51 [50] contains 6766 videos that have been annotated for 51 actions. Our evaluation follows the protocol for UCF101.

数据集。 HMDB51 [50] 包含 6766 个视频，已标注 51 个动作。 我们的评估遵循 UCF101 的协议。

Training and Inference. Our settings are identical to the ones used for UCF101 and we expect further tuning of hyperparameters to increase its downstream performance.

训练和推理。 我们的设置与用于 UCF101 的设置相同，我们希望进一步调整超参数以提高其下游性能。

## References
1. Pulkit Agrawal, Joa ̃o Carreira, and Jitendra Malik. Learning to see by moving. In Proc. ICCV, pages 37–45. IEEE, 2015. 2
2. Jean-Baptiste Alayrac, Adria` Recasens, Rosalia Schneider, Relja Arandjelovic ́, Jason Ramapuram, Jeffrey De Fauw, Lucas Smaira, Sander Dieleman, and Andrew Zisserman. Selfsupervised multimodal versatile networks. In Proc. NeurIPS, 2020. 2, 8 13 
3. Humam Alwassel, Dhruv Mahajan, Lorenzo Torresani, Bernard Ghanem, and Du Tran. Self-supervised learning by cross-modal audio-video clustering. In Proc. NeurIPS, 2020. 2, 8
4. Relja Arandjelovic ́ and Andrew Zisserman. Look, listen and learn. In Proc. ICCV, 2017. 2
5. Relja Arandjelovic ́ and Andrew Zisserman. Objects that sound. In Proc. ECCV, 2018. 2
6. Suzanna Becker. Learning temporally persistent hierarchical representations. In Proc. NeurIPS, 1997. 2
7. Sagie Benaim, Ariel Ephrat, Oran Lang, Inbar Mosseri, William T. Freeman, Michael Rubinstein, Michal Irani, and Tali Dekel. SpeedNet: Learning the Speediness in Videos. In Proc. CVPR, 2020. 8
8. Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In Proc. ECCV, 2018. 2, 8
9. Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. arXiv preprint arXiv:2006.09882, 2020. 1, 2, 3, 4, 7, 11, 12
10. Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics600. arXiv preprint arXiv:1808.01340, 2018. 9, 12
11. Joa ̃o Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019. 9, 12
12. Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709, 2020. 1,2,3,4,6,7,11,12
13. Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. Big self-supervised models are strong semi-supervised learners. In Proc. NeurIPS, 2020. 11
14. Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020. 6, 11, 12
15. Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Proc. NeurIPS, 2013. 12
16. AliDiba,VivekSharma,LucVanGool,andRainerStiefelhagen. DynamoNet: Dynamic Action and Motion Network. In Proc. ICCV, 2019. 2
17. Carl Doersch, Abhinav Gupta, and Alexei Efros. Unsupervised visual representation learning by context prediction. In Proc. ICCV, 2015. 2
18. A.Dosovitskiy,P.Fischer,J.T.Springenberg,M.Riedmiller, and T. Brox. Discriminative unsupervised feature learning with exemplar convolutional neural networks. IEEE PAMI, 38(9):1734–1747, Sept 2016. 1, 2
19. Haoqi Fan, Yanghao Li, Bo Xiong, Wan-Yen Lo, and Christoph Feichtenhofer. PySlowFast. https://github. com/facebookresearch/slowfast, 2020. 5
20. Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. SlowFast Networks for Video Recognition. In Proc. ICCV, 2019. 2, 4, 5, 7, 8, 12, 13
21. Basura Fernando, Hakan Bilen, Efstratios Gavves, and Stephen Gould. Self-supervised video representation learning with odd-one-out networks. In Proc. ICCV, 2017. 2
22. Chuang Gan, Boqing Gong, Kun Liu, Hao Su, and Leonidas J Guibas. Geometry guided convolutional neural networks for self-supervised video representation learning. In Proc. CVPR, 2018. 2
23. Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017. 8
24. Deepti Ghadiyaram, Matt Feiszli, Du Tran, Xueting Yan, Heng Wang, and Dhruv Mahajan. Large-scale weaklysupervised pre-training for video action recognition. In Proc. CVPR, 2019. 4, 5, 8, 9
25. Ross Girshick, Ilija Radosavovic, Georgia Gkioxari, Piotr Dolla ́r, and Kaiming He. Detectron. https://github. com/facebookresearch/detectron, 2018. 13
26. R. B. Girshick. Fast R-CNN. In Proc. ICCV, 2015. 12
27. Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 315–323, 2011. 11
28. Daniel Gordon, Kiana Ehsani, Dieter Fox, and Ali Farhadi. Watching the world go by: Representation learning from unlabeled videos. arXiv preprint arXiv:2003.07990, 2020. 2 [29] Ross Goroshin, Joan Bruna, Jonathan Tompson, David Eigen, and Yann LeCun. Unsupervised learning of spatiotemporally coherent metrics. In Proc. ICCV, 2015. 2 [30]PriyaGoyal,PiotrDolla ́r,RossGirshick,PieterNoordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: training ImageNet in 1 hour. arXiv:1706.02677, 2017. 13
31. Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz MuellerFreitag, et al. The “Something Something” video database for learning and evaluating visual common sense. In ICCV, 2017. 1, 4, 7, 13
32. Jean-Bastien Grill, Florian Strub, Florent Altche ́, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Re ́mi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In NeurIPS, 2020. 1, 2, 3, 4, 7, 10, 11
33. ChunhuiGu,ChenSun,DavidA.Ross,CarlVondrick,Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, Cordelia Schmid, and Jitendra Malik. AVA: A video dataset of spatiotemporally localized atomic visual actions. In Proc. CVPR, 2018. 1, 4, 7, 12
34. Tengda Han, Weidi Xie, and Andrew Zisserman. Video representation learning by dense predictive coding. In Workshop on Large Scale Holistic Video Understanding, ICCV, 2019. 2 14 
35. Tengda Han, Weidi Xie, and Andrew Zisserman. Selfsupervised co-training for video representation learning. In Proc. NeurIPS, 2020. 2, 8
36. Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proc. CVPR, 2020. 1, 2, 3, 4, 7, 11, 12
37. Kaiming He, Georgia Gkioxari, Piotr Dolla ́r, and Ross Girshick. Mask R-CNN. In Proc. ICCV, 2017. 13
38. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proc. ICCV, 2015. 11
39. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. CVPR, 2016. 2, 4, 5, 7, 12
40. Olivier J. He ́naff, Ali Razavi, Carl Doersch, S. M. Ali Eslami, and Aa ̈ron van den Oord. Data-efficient image recognition with contrastive predictive coding. arXiv preprint arXiv:1905.09272, 2019. 2
41. RDevonHjelm,AlexFedorov,SamuelLavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. In Proc. ICLR, 2019. 2
42. S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proc. ICML, 2015. 10, 11
43. Phillip Isola, Daniel Zoran, Dilip Krishnan, and Edward H Adelson. Learning visual groups from co-occurrences in space and time. In Proc. ICLR, 2015. 2
44. Dinesh Jayaraman and Kristen Grauman. Learning image representations tied to ego-motion. In Proc. ICCV, 2015. 2
45. Simon Jenni, Givi Meishvili, and Paolo Favaro. Video representation learning by recognizing temporal transformations. arXiv preprint arXiv:2007.10730, 2020. 2
46. Xu Ji, Joa ̃o F. Henriques, and Andrea Vedaldi. Invariant information clustering for unsupervised image classification and segmentation. In Proc. ICCV, pages 9865–9874, 2019. 2
47. Will Kay, Joa ̃o Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017. 1, 2, 3, 4, 12
48. Dahun Kim, Donghyeon Cho, and In So Kweon. Selfsupervised video representation learning with space-time cubic puzzles. In AAAI, 2019. 2
49. Bruno Korbar, Du Tran, and Lorenzo Torresani. Cooperative learning of audio and video models from self-supervised synchronization. In Proc. NeurIPS, 2018. 2
50. H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. HMDB: A large video database for human motion recognition. In Proc. ICCV, pages 2556–2563, 2011. 2, 4, 13
51. Hsin-Ying Lee, Jia-Bin Huang, Maneesh Singh, and MingHsuan Yang. Unsupervised representation learning by sorting sequence. In Proc. ICCV, 2017. 2
52. Ang Li, Meghana Thotakuri, David A Ross, Joa ̃o Carreira, Alexander Vostrikov, and Andrew Zisserman. The avakinetics localized human actions video dataset. arXiv preprint arXiv:2005.00214, 2020. 7
53. Tianhao Li and Limin Wang. Learning spatiotemporal features via video and text pair discrimination. arXiv preprint arXiv:2001.05691, 2020. 2
54. Ji Lin, Chuang Gan, and Song Han. Temporal shift module for efficient video understanding. In ICCV, 2019. 13
55. Tsung-YiLin,MichaelMaire,SergeBelongie,JamesHays, Pietro Perona, Deva Ramanan, Piotr Dolla ́r, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In Proc. ECCV, 2014. 13
56. IlyaLoshchilovandFrankHutter.SGDR:Stochasticgradient descent with warm restarts. arXiv:1608.03983, 2016. 11
57. William Lotter, Gabriel Kreiman, and David Cox. Deep predictive coding networks for video prediction and unsupervised learning. In Proc. ICLR, 2017. 2
58. Michael Mathieu, Camille Couprie, and Yann LeCun. Deep multi-scale video prediction beyond mean square error. In ICLR, 2016. 2
59. Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proc. CVPR, 2019. 8
60. Ishan Misra, C. Lawrence Zitnick, and Martial Hebert. Shuffle and learn: Unsupervised learning using temporal order verification. In Proc. ECCV, 2016. 2
61. Hossein Mobahi, Ronan Collobert, and Jason Weston. Deep learning from temporal coherence in video. In Proc. ICML, pages 737–744, 2009. 2
62. Pedro Morgado, Nuno Vasconcelos, and Ishan Misra. Audiovisual instance discrimination with cross-modal agreement. arXiv preprint arXiv:2004.12943, 2020. 2
63. V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In Proc. ICML, 2010. 11
64. Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In Proc. ECCV, pages 69–84. Springer, 2016. 2
65. AndrewOwens,PhillipIsola,JoshH.McDermott,Antonio Torralba, Edward H. Adelson, and William T. Freeman. Visually indicated sounds. In Proc. CVPR, pages 2405–2413, 2016. 2
66. DeepakPathak,RossGirshick,PiotrDolla ́r,TrevorDarrell, and Bharath Hariharan. Learning features by watching objects move. In Proc. CVPR, 2017. 2
67. Deepak Pathak, Philipp Kra ̈henbu ̈hl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context encoders: Feature learning by inpainting. In Proc. CVPR, 2016. 2
68. Mandela Patrick, Yuki M. Asano, Ruth Fong, Joa ̃o F. Henriques, Geoffrey Zweig, and Andrea Vedaldi. Multi-modal self-supervision from generalized data transformations. arXiv preprint arXiv:2003.04298, 2020. 2, 8
69. AJ Piergiovanni, Anelia Angelova, and Michael S. Ryoo. Evolving losses for unsupervised video representation learning. In Proc. CVPR, 2020. 2
70. Senthil Purushwalkam and Abhinav Gupta. Demystifying contrastive self-supervised learning: Invariances, augmentations and dataset biases. arXiv preprint arXiv:2007.13916, 2020. 2 15 
71. Rui Qian, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, and Yin Cui. Spatiotemporal contrastive video representation learning. arXiv preprint arXiv:2008.03800, 2020. 2, 8
72. S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In Proc. NeurIPS, 2016. 12
73. Pierre H Richemond, Jean-Bastien Grill, Florent Altche ́, Corentin Tallec, Florian Strub, Andrew Brock, Samuel Smith, Soham De, Razvan Pascanu, Bilal Piot, et al. Byol works even without batch statistics. arXiv preprint arXiv:2010.10241, 2020. 10
74. Pierre Sermanet et al. Time-contrastive networks: Selfsupervised learning from video. In Proc. Intl. Conf. on Robotics and Automation, 2018. 2
75. Gunnar A Sigurdsson, Gu ̈l Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. In ECCV, 2016. 1, 4, 7, 13
76. Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Proc. ICLR, 2015. 7, 12
77. Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 2, 4, 13
78. N. Srivastava, E. Mansimov, and R. Salakhudinov. Unsupervised learning of video representations using lstms. In Proc. ICML, 2015. 2
79. Chen Sun, Fabien Baradel, Kevin Murphy, and Cordelia Schmid. Contrastive bidirectional transformer for temporal representation learning. arXiv preprint arXiv:1906.05743, 2019. 2
80. ChristianSzegedy,WeiLiu,YangqingJia,PierreSermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proc. CVPR, 2015. 7, 12
81. Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In Proc. ECCV, 2020. 2
82. Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In Proc. CVPR, 2018. 5
83. Aa ̈ron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 2, 3
84. Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Anticipating visual representations from unlabelled video. In Proc. CVPR, 2016. 2
85. Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using videos. In Proc. ICCV, 2015. 2
86. Xiaolong Wang, Kaiming He, and Abhinav Gupta. Transitive invariance for self-supervised visual representation learning. In Proc. ICCV, 2017. 2
87. Xiaolong Wang, Allan Jabri, and Alexei A. Efros. Learning correspondence from the cycle-consistency of time. In Proc. CVPR, 2019. 2
88. Laurenz Wiskott and Terrence Sejnowski. Slow feature analysis: Unsupervised learning of invariances. In Neural Computation, 2002. 2, 5
89. Zhirong Wu, Yuanjun Xiong, Stella Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance-level discrimination. In Proc. CVPR, volume abs/1805.01978, 2018. 1, 2
90. Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking spatiotemporal feature learning for video understanding. In Proc. ECCV, 2018. 5
91. Dejing Xu, Jun Xiao, Zhou Zhao, Jian Shao, Di Xie, and Yueting Zhuang. Self-supervised spatiotemporal learning via video clip order prediction. In Proc. CVPR, 2019. 2
92. Ceyuan Yang, Yinghao Xu, Bo Dai, and Bolei Zhou. Video representation learning with visual tempo consistency. arXiv preprint arXiv:2006.15489, 2020. 2, 8
93. Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. arXiv preprint arXiv:1708.03888, 2017. 10, 11
94. Richard Zhang, Phillip Isola, and Alexei A. Efros. Colorful image colorization. In Proc. ECCV, 2016. 2
95. Chengxu Zhuang, Alex Lin Zhai, and Daniel Yamins. Local aggregation for unsupervised learning of visual embeddings. In Proc. ICCV, 2019. 2 16
