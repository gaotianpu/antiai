# Multimodal Token Fusion for Vision Transformers
ViT的多模态令牌融合 https://arxiv.org/abs/2204.08721

## Abstract
Many adaptations of transformers have emerged to address the single-modal vision tasks, where self-attention modules are stacked to handle input sources like images. Intuitively, feeding multiple modalities of data to vision transformers could improve the performance, yet the innermodal attentive weights may also be diluted, which could thus undermine the final performance. In this paper, we propose a multimodal token fusion method (TokenFusion), tailored for transformer-based vision tasks. To effectively fuse multiple modalities, TokenFusion dynamically detects uninformative tokens and substitutes these tokens with projected and aggregated inter-modal features. Residual positional alignment is also adopted to enable explicit utilization of the inter-modal alignments after fusion. The design of TokenFusion allows the transformer to learn correlations among multimodal features, while the single-modal transformer architecture remains largely intact. Extensive experiments are conducted on a variety of homogeneous and heterogeneous modalities and demonstrate that TokenFusion surpasses state-of-the-art methods in three typical vision tasks: multimodal image-to-image translation, RGBdepth semantic segmentation, and 3D object detection with point cloud and images. Our code is available at https://github.com/yikaiw/TokenFusion.

为了解决单模态视觉任务，已经出现了许多transformer的修改，其中自注意模块被堆叠以处理图像等输入源。直观地说，将多个模态的数据输入到ViT可以提高性能，但内模态注意力权重也可能被稀释，从而破坏最终性能。在本文中，我们提出了一种多模态令牌融合方法(TokenFusion)，专门针对基于transformer的视觉任务。为了有效地融合多种模态，TokenFusion动态检测无信息令牌，并用投影和聚合的模态间特征替换这些令牌。还采用残余位置对准，以实现融合后模态间对准的显式利用。TokenFusion的设计允许transformer学习多模态特征之间的相关性，而单模态transformer架构在很大程度上保持不变。在各种同质和异质模态上进行了广泛的实验，并证明TokenFusion在三个典型的视觉任务中超越了最先进的方法：多模态图像到图像转换、RGB深度语义分割以及使用点云和图像的3D目标检测。<!-innermodal attentive -->

## 1. Introduction
Transformer is initially widely studied in the natural language community as a non-recurrent sequence model [40] and it is soon extended to benefit vision-language tasks. Recently, numerous studies have further adopted transformers for computer vision tasks with well-adapted architectures and optimization schedules. As a result, vision transformer variants have shown great potential in many singlemodal vision tasks, such as classification [6, 21], segmentation [44, 47], detection [3, 8, 22, 48], image generation [16].

Transformer最初作为一种非递归序列模型在自然语言社区被广泛研究[40]，并很快被扩展到有利于视觉语言任务。最近，许多研究进一步采用了具有良好适应的架构和优化调度的计算机视觉任务transformer。因此，ViT变体在许多单模态视觉任务中显示出巨大的潜力，例如分类[6，21]、分割[44，47]、检测[3，8，22，48]、图像生成[16]。

Yet up until the date of this work, the attempt of extending vision transformers to handle multimodal data remains scarce. When multimodal data with complicated alignment relations are introduced, it poses great challenges in designing the fusion scheme for model architectures. The key question to answer is how and where the interaction of features from different modalities should take place. There have been a few methods for transformer-based visionlanguage fusion, e.g., VL-BERT [37] and ViLT [17]. In these methods, vision and language tokens are directly concatenated before each transformer layer, making the overall architecture very similar to the original transformer. Such fusion is usually alignment-agnostic, which indicates the inter-modal alignments are not explicitly utilized. We also try to apply similar fusion methods on multimodal vision tasks (Sec. 4). Unfortunately, this intuitive transformer fusion cannot bring promising gains or may even result in worse performance than the single-modal counterpart, which is mainly due to the fact that the inter-modal interaction is not fully exploited. There are also several attempts for fusing multiple vision modalities. For example, TransFuser [26] leverages transformer modules to connect CNN backbones of images and LiDAR points. Different from exising trials, our work aims to seek an effective and general method to combine multiple single-modal transformers while inserting inter-modal alignments into the models.

然而，在这项工作开展之前，将ViT扩展到处理多模态数据的尝试仍然很少。当引入具有复杂对齐关系的多模态数据时，在设计模型架构的融合方案方面提出了巨大的挑战。要回答的关键问题是，来自不同模态的特征之间的交互应如何以及在何处发生。已有几种基于transformer的视觉语言融合方法，例如VL-BERT[37]和ViLT[17]。在这些方法中，视觉和语言令牌直接连接在每个transformer层之前，使整体架构与原始transformer非常相似。这种融合通常是对齐不可知的，这表明模态间对齐没有被明确利用。我们还尝试将类似的融合方法应用于多模态视觉任务(第4节)。不幸的是，这种直观的transformer融合不能带来有希望的收益，甚至可能导致比单模态对应物更差的性能，这主要是由于没有充分利用模态间的相互作用。还有几种融合多种视觉模态的尝试。例如，TransFuser[26]利用transformer模块连接CNN图像主干和LiDAR点。与现有的试验不同，我们的工作旨在寻求一种有效和通用的方法，在将模态间对齐插入模型的同时组合多个单模态transformer。

This work benefits the learning process by multimodal data while leveraging inter-modal alignments. Such alignments are naturally available in many vision tasks, e.g., with camera intrinsics/extrinsics, world-space points could be projected and correspond to pixels on the camera plane. Unlike the alignment-agnostic fusion (Sec. 3.1), the alignmentaware fusion explicitly involves the alignment relations of different modalities. Yet, since inter-modal projections are introduced to the transformer, alignment-aware fusion may greatly alter the original model structure and data flow, which potentially undermines the success of single-modal architecture designs or learned attention during pretraining. Thus, one may have to determine the “correct” layers/tokens/channels for multimodal projection and fusion, and also re-design the architecture or re-tune optimization settings for the new model. To avoid dealing with these challenging matters and inherit the majority of the original single-modal design, we propose multimodal token fusion, termed TokenFusion, which adaptively and effectively fuses multiple single-modal transformers.

这项工作在利用模态间对齐的同时，通过多模态数据使学习过程受益。这样的对齐在许多视觉任务中自然可用，例如，使用相机内部/外部，世界空间点可以被投影并对应于相机平面上的像素。与对齐不可知融合(第3.1节)不同，对齐感知融合明确涉及不同模态的对齐关系。然而，由于模态间投影被引入到transformer中，对齐感知融合可能会极大地改变原始模型结构和数据流，这可能会破坏单模态架构设计的成功或预训练期间的学习注意力。因此，可能需要确定用于多模态投影和融合的“正确”层/令牌/信道，还需要重新设计新模型的架构或重新调整优化设置。为了避免处理这些具有挑战性的问题并继承大多数原始的单模态设计，我们提出了多模态令牌融合，称为TokenFusion，它自适应地有效地融合多个单模态transformer。

The basic idea of our TokenFusion is to prune multiple single-modal transformers and then re-utilize pruned units for multimodal fusion. We apply individual pruning to each single-modal transformer and each pruned unit is substituted by projected alignment features from other modalities. This fusion scheme is assumed to have a limited impact on the original single-modal transformers, as it maintains the relative attention relations of the important units. TokenFusion also turns out to be superior in allowing multimodal transformers to inherit the parameters from single-modal pretraining, e.g., on ImageNet.

我们的TokenFusion的基本思想是修剪多个单模态transformer，然后重新利用修剪后的单元进行多模态融合。我们对每个单模态transformer应用单独的修剪，每个修剪单元由其他模态的投影对齐特征代替。假设该融合方案对原始单模态transformer的影响有限，因为它保持了重要单元的相对注意力关系。TokenFusion在允许多模态transformer从单模态预训练(例如在ImageNet上)继承参数方面也表现出了优势。

To demonstrate the advantage of the proposed method, we consider extensive tasks including multimodal image translation, RGB-depth semantic segmentation, and 3D object detection based on images and point clouds, covering up to four public datasets and seven different modalities. TokenFusion obtains state-of-the-art performance on these extensive tasks, demonstrating its great effectiveness and generality. Specifically, TokenFusion achieves 64.9% and 70.8% mAP@0.25 for 3D object detection on the challenging SUN RGB-D and ScanNetV2 benchmarks, respectively.

为了证明所提出方法的优势，我们考虑了广泛的任务，包括多模态图像翻译、RGB深度语义分割和基于图像和点云的3D目标检测，涵盖了多达四个公共数据集和七种不同的模态。TokenFusion在这些广泛的任务中获得了最先进的性能，证明了其巨大的有效性和通用性。具体而言，TokenFusion达到64.9%和70.8%mAP@0.25分别在具有挑战性的SUN RGB-D和ScanNetV2基准上进行3D目标检测。
<!-多模态图像翻译,RGB深度语义分割, 基于图像和点云的3D目标检测 -->

## 2. Related Work
### Transformers in computer vision. 
Transformer is originally designed for NLP research fields [40], which stacking multi-head self-attention and feed-forward MLP layers to capture the long-term correlation between words. Recently, vision transformer (ViT) [6] reveals the great potential of transformer-based models in large-scale image classification. As a result, transformer has soon achieved profound impacts in many other computer vision tasks such as segmentation [44, 47], detection [3, 8, 22, 48], image generation [16], video processing [20], etc.

计算机视觉中的Transformers。Transformer最初是为NLP研究领域设计的[40]，它堆叠了多头自主意和前馈MLP层，以捕获单词之间的长期相关性。最近，视觉transformer(ViT)[6]揭示了基于transformer的模型在大规模图像分类中的巨大潜力。因此，transformer很快在许多其他计算机视觉任务中取得了深远的影响，如分割[44，47]、检测[3，8，22，48]、图像生成[16]、视频处理[20]等。

### Fusion for vision transformers. 
Deep fusion with multimodal data has been an essential topic which potentially boosts the performance by leveraging multiple sources of inputs, and it may also unleash the power of transformers further. Yet it is challenging to combine multiple offthe-rack single transformers while guaranteeing that such combination will not impact their elaborate singe-modal designs. [2] and [20] process consecutive video frames with transformers for spatial-temporal alignments and capturing fine-grained patterns by correlating multiple frames. Regarding multimodal data, [26, 41] utilize the dynamic property of transformer modules to combine CNN backbones for fusing infrared/visible images or LiDAR points. [9] extends the coarse-to-fine experience from CNN fusion methods to transformers for image processing tasks. [14] adopts transformers to combine hyperspectral images by the simple feature concatenation. [24] inserts intermediate tokens between image patches and audio spectrogram patches as bottlenecks to implicitly learn inter-modal alignments. These works, however, differ from ours since we would like to build a general fusion pipeline for combing off-the-rack vision transformers without the need of re-designing their structures or re-tuning their optimization settings, while explicitly leveraging inter-modal alignment relations.

ViT的融合。与多模态数据的深度融合一直是一个重要课题，它可以通过利用多个输入源来提高性能，还可以进一步释放transformer的能量。然而，在保证这种组合不会影响其精心设计的单模态设计的同时，将多个现成的单transformer组合起来是一个挑战。[2] 以及[20]使用transformer处理连续视频帧以进行时空对齐，并通过关联多个帧来捕获细粒度图案。关于多模态数据，[26，41]利用transformer模块的动态特性来组合CNN主干，以融合红外/可见图像或激光雷达点。[9] 将CNN融合方法的粗到细经验扩展到图像处理任务的transformer。[14] 采用transformer通过简单的特征拼接来组合高光谱图像。[24]在图像块和音频谱图块之间插入中间令牌，作为隐式学习模态间对齐的瓶颈。然而，这些工作与我们的不同，因为我们希望构建一个通用的融合管道，用于组合机架外ViT，而无需重新设计其结构或调整其优化设置，同时明确利用模态间对齐关系。

## 3. Methodology
This part intends to provide a full landscape of the proposed methodology. We first introduce two na¨ıve multimodal fusion methods for vision transformers in Sec. 3.1. Given the limitations of both intuitive methods, we then propose multimodal token fusion in Sec. 3.2. We elaborate the fusion designs for both homogeneous and heterogeneous modalities to evaluate the effectiveness and generality of our method in Sec. 3.4 and Sec. 3.5, respectively.

本部分旨在全面介绍提议的方法。在第3.1节中，我们首先介绍了视觉变换器的两种新颖的多模态融合方法。鉴于这两种直观方法的局限性，我们在第3.2节中提出了多模态令牌融合。我们分别在第3.4节和第3.5节中阐述了同质和异质模态的融合设计，以评估我们方法的有效性和通用性。

### 3.1. Basic Fusion for Vision Transformers
Suppose we have the i-th input data $x^{(i)}$ that contains M modalities: $x^{(i)} = \{x^{(i)}_m ∈ R^{N×C} \}^M_{m=1} $, where N and C denote the number of tokens and input channels respectively. For simplicity, we will omit the subscript (i) in the upcoming sections. The goal of deep multimodal fusion is to determine a multi-layer model f(x), and its output is expected to close to the target y as much as possible. Specifically in this work, f(x) is approximated by a transformerbased network architecture. Suppose the model contains L layers in total, we represent the input token feature of the lth layer (l = 1, . . . , L) as $e^l = \{e^l_m ∈ R^{N×C'} \}^M_{m=1}$, where C' denotes the number of feature channels of the layer in scope. Initially, $e^l_m$ is obtained using a linear projection of $x_m$, which is a widely adopted approach to vectorize the input tokens (e.g. image patches), so that the first transformer layer can accept tokens as input.

假设我们有第i个输入数据$x^{(i)}$，它包含M个模态：$x^{(i)} = \{x^{(i)}_m ∈ R^{N×C} \}^M_{m=1} $，其中N和C分别表示令牌和输入通道的数量。为了简单起见，我们将在接下来的部分中省略下标(i)。深度多模态融合的目标是确定多层模型f(x)，其输出预计将尽可能接近目标y。特别是在这项工作中，f(x)通过基于变换器的网络架构来近似。假设模型总共包含L个层，我们将第L层(L=1，…，L)的输入令牌特征表示为$e^l = \{e^l_m ∈ R^{N×C'} \}^M_{m=1}$，其中C'表示作用域中层的特征通道数。最初，$e^l_m$是使用$x_m$的线性投影获得的，这是一种广泛采用的对输入令牌进行矢量化的方法(例如图像块)，以便第一变换器层可以接受令牌作为输入。

We use different transformers for input modalities and denote $f_m(x) = e^{L+1}_m$ as the final prediction of the m-th transformer. Given the token feature $e^l_m$ of the m-th modality, the l-th layer computes 

我们对输入模态使用不同的变换器，并将$f_m(x)=e^{L+1}_m$表示为第m个变换器的最终预测。给定第m模态的令牌特征$e^l_m$，第l层计算

$\hat e ^l_m = MSA\big(LN(e^l_m)\big) , e^{l+1}_m = MLP\big(LN(\hat e ^l_m)\big) $, (1) 

where MSA, MLP, and LN denote the multi-head selfattention, multi-layer perception, and layer normalization, receptively. $\hat e ^l_m$ represents the output of MSA.

其中MSA、MLP和LN表示接受性的多头自主意、多层感知和层标准化$\hat e^l_m$表示MSA的输出。

During multimodal fusion for vision tasks, the alignment relations of different modalities may be explicitly available. For example, pixel positions are often used to determine the image-depth correlation; and camera intrinsics/extrinsics are important in projecting 3D points to images. Based on the involvement of alignment information, we consider two kinds of transformer fusion methods as below.

在视觉任务的多模态融合过程中，不同模态的对齐关系可能是明确可用的。例如，经常使用像素位置来确定图像深度相关性; 并且相机内在/外在在将3D点投影到图像中是重要的。基于对准信息的参与，我们考虑以下两种transformer融合方法。

#### Alignment-agnostic fusion 
does not explicitly use the alignment relations among modalities. It expects the alignment may be implicitly learned from large amount of data. A common method of the alignment-agnostic fusion is to directly concatenate multimodal input tokens, which is widely applied in vision-language models. Similarly, the input feature $e_l$ for the l-th layer is also the token-wise concatenation of different modalities. Although the alignment-agnostic fusion is simple and may have minimal modification to the original transformer model, it is hard to directly benefit from the known multimodal alignment relations.

对齐不可知融合 没有明确使用模态之间的对齐关系。它期望可以从大量数据中隐式学习对齐。对齐不可知融合的一种常见方法是直接连接多模态输入令牌，这在视觉语言模型中得到了广泛应用。类似地，第l层的输入特性$e_l$也是不同模态的符号串联。尽管对准不可知融合很简单，对原始transformer模型的修改可能很小，但很难从已知的多模态对准关系中直接受益。

#### Alignment-aware fusion 
explicitly utilizes inter-modal alignments. For instance, this can be achieved by selecting tokens that correspond to the same pixel or 3D coordinate. Suppose $x_m[n]$ is the n-th token of the m-th modality input $x_m$, where n = 1, · · · , $N_m$. We define the “token projection” from the m-th modality to the m' -th modality as

对齐感知融合 明确地利用了模态间对齐。例如，这可以通过选择对应于相同像素或3D坐标的令牌来实现。假设$x_m[n]$是第m个模态输入$x_m$的第n个令牌，其中n=1，··，$n_m$。我们将第m模态到第m模态的“令牌投影”定义为

$Proj^T_{m'} (x_m[n_m]) = h(x_{m'} [n_{m'} ])$ , (2) 

where h could simply be an identity function (for homogeneous modalities) or a shallow multi-layer perception (for heterogeneous modalities). And when considering the entire N tokens, we can conveniently define the “modality projection” as the concatenation of token projections:

其中h可以简单地是同一函数(对于同质模态)或浅层多层感知(对于异质模态)。当考虑整个N个令牌时，我们可以方便地将“模态投影”定义为令牌投影的级联：

$Proj^T_{m'} (x_m) = \bigg[ Proj^T_{m'} (x_m[1]) \ ; \ · · · \ ; Proj^T_{m'} (x_m[N]) \bigg]$. (3)

Eq. (3) only depicts the fusion strategy on the input side. We can also perform middle-layer or multi-layer fusion across different modality-specific models, by projecting and aggregating feature embeddings $e_m$ which possibly enables more diversified and accurate feature interactions. However, with the growing complexity of transformer-based models, searching for optimal fusion strategies (e.g. layers and tokens to apply projection and aggregation) for merely two modalities (e.g. 2D and 3D detection transformers) can grow into an extremely hard problem to solve. To tackle this issue, we propose multimodal token fusion in Sec. 3.2.

等式(3)仅描述了输入侧的融合策略。我们还可以通过投影和聚合特征嵌入$e_m$来跨不同模态特定模型执行中间层或多层融合，这可能会实现更加多样化和准确的特征交互。然而，随着基于变换器的模型越来越复杂，仅为两种模态(例如2D和3D检测变换器)搜索最优融合策略(例如，应用投影和聚合的层和令牌)可能会成为一个极其难以解决的问题。为了解决这个问题，我们在第3.2节中提出了多模态令牌融合。

### 3.2. Multimodal Token Fusion
As described in Sec. 1, multimodal token fusion (TokenFusion) first prunes single-modal transformers and further re-utilizes the pruned units for fusion. In this way, the informative units of original single-modal transformers are assumed to be preserved to a large extent, while multimodal interactions could be involved for boosting performance.

如第1节所述，多模态令牌融合(TokenFusion)首先修剪单模态变换器，并进一步重新利用修剪后的单元进行融合。通过这种方式，假设在很大程度上保留了原始单模态transformer的信息单元，而多模态相互作用可以用于提高性能。

As previously shown in [32], tokens of vision transformers could be pruned in a hierarchical manner while maintaining the performance. Similarly, we can select less informative tokens by adopting a scoring function $s^l(e^l ) = MLP(e^l) ∈ [0, 1]^N$ , which dynamically predicts the importance of tokens for the l-th layer and the m-th modality. To enable the back propagation on $s^l(e^l)$, we re-formulate the MSA output $\hat e^l_m$ in Eq. (1) as 

如前[32]所示，视觉变换器的令牌可以在保持性能的同时以分层方式进行修剪。类似地，我们可以通过采用评分函数$s^l(e^l)=MLP(e^l)∈[0，1]^N$来选择信息量较少的令牌，该函数动态预测令牌对于第l层和第m模态的重要性。为了在$s^l(e^l)$上实现反向传播，我们将等式(1)中的MSA输出$\hat e^l_m$重新公式化为

$ \hat e^l_m = MSA ( LN(e^l_m) \ · \ s^l(e^l_m) ) $. (4)

We use $L_m$ to denote the task-specific loss for the m-th modality. To prune uninformative tokens, we further add a token-wise pruning loss (an $l_1-norm$) on $s^l(e^l_m)$. Thus the overall loss function for optimization is derived as

我们使用$L_m$表示第m个模态的任务特定损失。为了修剪非信息令牌，我们进一步在$s^l(e^l_m)$上添加一个基于令牌的修剪损失($l_1-norm$)。因此，优化的总损失函数如下

$ L = \sum^M_{m=1} \bigg(  L_m + λ \sum^L_{l=1} | s^l(e^l_m) | \bigg) $ , (5) 

where λ is a hyper-parameter for balancing different losses.

其中λ是用于平衡不同损耗的超参数。

For the feature $e^l_m ∈ R^{N×C'}$ , token-wise pruning dynamically detects unimportant tokens from all N tokens. Mutating unimportant tokens or substituting them with other embeddings are expected to have limited impacts on other informative tokens. We thus propose a token fusion process for multimodal transformers, which substitute unimportant tokens with their token projections (defined in Sec. 3.1) from other modalities. Since the pruning process is dynamic, i.e., conditioned on the input features, the fusion process is also dynamic. This process performs token substitution before each transformer layer, thus the input feature of the l-th layer, i.e., $e^l_m$, is re-formulated as 

对于特征$e^l_m∈R^{N×C'}$，令牌式修剪从所有N个令牌中动态检测不重要的令牌。突变不重要的令牌或用其他嵌入替换它们，预计对其他信息性令牌的影响有限。因此，我们提出了一种用于多模态变换器的令牌融合过程，该过程将不重要的令牌替换为来自其他模态的令牌投影(定义见第3.1节)。由于修剪过程是动态的，即，取决于输入特征，因此融合过程也是动态的。此过程在每个转换器层之前执行令牌替换，因此第l层的输入特征，即$e^l_m$，被重新公式化为

$e^l_m = e^l_m \odot I_{s^l(e^l_m)≥θ} + Proj^M_{m'} (e^l_m) \odot I_{s^l(e^l_m)<θ}$, (6) 

<!--markdown 特殊符号 https://blog.nowcoder.net/n/7d5d9ff47af74c288d19ba29e88c5643-->

where I is an indicator asserting the subscript condition, therefore it outputs a mask tensor $∈\{0，1\}^N$ ; the parameter θ is a small threshold (we adopt $10^{−2}$ in our experiments); and the operator $\odot$ resents the element-wise multiplication.

其中I是断言下标条件的指示符，因此它输出掩码张量$∈\{0，1\}^N$; 参数θ是一个小阈值(我们在实验中采用$10^{−2}$); 运算符$\odot$表示元素乘法。

In Eq. (6), if there are only two modalities as input, m' will simply be the other modality other than m. With more than two modalities, we pre-allocate the tokens into M − 1 parts, each of which is bound with one of the other modalities than itself. More details of this pre-allocation will be described in Sec. 3.4.

在等式(6)中，如果只有两个模态作为输入，m’将只是m以外的另一模态。对于两个以上的模态，我们将令牌预分配到m−1个部分，每个部分与其他模态之一绑定，而不是与自身绑定。有关预分配的更多详情，请参见第3.4节。

### 3.3. Residual Positional Alignment
Directly substituting tokens will risk completely undermining their original positional information. Hence, the model can still be ignorant of the alignment of the projected features from another modality. To mitigate this problem, we adopt Residual Positional Alignment (RPA) that leverages Positional Embeddings (PEs) for the multimodal alignment. As depicted in Fig. 1 and Fig. 2 which will be detailed later, the key idea of RPA lies in injecting equivalent PEs to subsequent layers. Moreover, the back propagation of PEs stops after the first layer, which means only the gradients of PEs at the first layer are retained while for the rest of the layers are frozen throughout the training. In this way, PEs serve a purpose of aligning multimodal tokens despite the substitution status of the original token. In summary, even if a token is substituted, we still reserve its original PEs that are added to the projected feature from another modality.

直接替换令牌将有完全破坏其原始位置信息的风险。因此，模型仍然可以忽略来自另一模态的投影特征的对齐。为了缓解这个问题，我们采用了利用位置嵌入(PE)进行多模态对齐的残余位置对齐(RPA)。如稍后将详细描述的图1和图2所示，RPA的关键思想在于向后续层注入等效PE。此外，PE的反向传播在第一层之后停止，这意味着只有第一层的PE的梯度被保留，而其余层在整个训练过程中被冻结。通过这种方式，PE实现了对齐多模态令牌的目的，而不管原始令牌的替换状态如何。总之，即使令牌被替换，我们仍然保留其原始PE，这些PE从另一模态添加到投影特征。

Figure 1. Framework of TokenFusion for homogeneous modalities with RGB and depth as an example. Both modalities are sent to a shared transformer with also shared positional embeddings. 
图1。以RGB和深度为例的同质模态TokenFusion框架。这两种模态都被发送到一个共享的转换器，该转换器还具有共享的位置嵌入。

### 3.4. Homogeneous Modalities 同质模态
In the common setup of either a generation task (multimodal image-to-image translation) or a regression task (RGB-depth semantic segmentation), the homogeneous vision modalities $x_1, x_2, · · · , x_M$ are typically aligned with pixels, such that the pixels located at the same position in RGB or depth input should share the same label. We also expect that such property allows the transformer-based models to benefit from joint learning. Hence, we adopt shared parameters in both MSA and MLP layers for different modalities; yet rely on modality-specific layer normalizations to uncouple the normalization process, since different modalities may vary drastically in their statistical means and variances by nature. In this scenario, we simply set function h in Eq. (6) as an identity function, and we also let $n_{m'} = n_m$, which means we always substitute each pruned token with the token sharing the same position.

在生成任务(多模态图像到图像转换)或回归任务(RGB深度语义分割)的常见设置中，同质视觉模态$x_1、x_2、··、x_M$通常与像素对齐，使得位于RGB或深度输入中相同位置的像素应共享同一标签。我们还期望这种特性允许基于transformer的模型从联合学习中受益。因此，我们在不同模态的MSA和MLP层中采用共享参数; 然而，依赖于模态特定的层归一化来解除归一化过程，因为不同模态的统计均值和方差本质上可能会有很大的差异。在这种情况下，我们简单地将等式(6)中的函数h设置为恒等函数，并且我们还让$n_{m'}=n_m$，这意味着我们总是用共享相同位置的令牌替换每个修剪的令牌。

An overall illustration of TokenFusion for fusing homogeneous modalities is depicted in Fig. 1. Regarding two input modalities, we adopt bi-directional projection and apply token-wise pruning on both modalities respectively. Then the token substitution process is performed according to Eq. (6). When there are M > 2 modalities, we also apply the token-wise pruning on all modalities with an additional pre-allocation strategy that selects m' in based on m according to Eq. (6). To be specific, for the m-th modality, we randomly pre-allocate N tokens into M − 1 groups with equal group sizes. This pre-allocation is carried out prior to the commence of training procedure, and the obtained groups will be fixed throughout the training. We denote the group allocation as am' (m) ∈ {0, 1}N , where am' (m)[n] = 1 indicates that if the n-th token of the m-th modaltity is pruned, it will be substituted by the corresponding token of the m' th modality, otherwise am' (m)[n] = 0. Having obtained the pre-allocation strategy for M > 2 modalties, Eq. (6) can be further developed into a more specific form: 

图1描述了用于融合同质模态的TokenFusion的总体说明。对于两种输入模态，我们采用双向投影，并分别对两种模态应用令牌式修剪。然后根据等式(6)执行令牌替换过程。当有M＞2个模态时，我们还对所有模态应用令牌式修剪，并根据等式(6)使用基于M选择M’in的额外预分配策略。具体而言，对于第m个模态，我们将N个令牌随机预分配到m−1个具有相同组大小的组中。该预分配在培训程序开始之前进行，获得的小组将在整个培训过程中固定。我们将组分配表示为am’(m)∈｛0，1｝N，其中am‘(m)[N]=1表示如果第m模态的第N个令牌被修剪，它将被第m模态对应的令牌替换，否则am‘(m)[N]=0。在获得M>2模态的预分配策略后，方程(6)可以进一步发展为更具体的形式：

$e^l_m = e^l_m  \odot I_{s^l(e^l_m)≥θ} + \sum^X_{m' =1 \ \ m' \neq m } a_{m'} (m) \odot  Proj^M_{m'} (e^l_m) \odot  I_{s^l(e^l_m)<θ} $. (7)

### 3.5. Heterogeneous Modalities 异构模态
In this section, we further explore how TokenFusion handles heterogeneous modalities, in which input modalities exhibit quite different data formats and large structural discrepancies, e.g., different number of layers or embedding dimensions for the transformer architectures. A concrete example would be to learn 3D object detection (based on point cloud) and 2D object detection (based on images) simultaneously with different transformers. Although there are already specific transformer-based models designed for 3D or 2D object detection respectively, there still lacks a fast and effective method to combine these models and tasks.

在本节中，我们将进一步探讨TokenFusion如何处理异构模态，其中输入模态表现出截然不同的数据格式和巨大的结构差异，例如，transformer架构的不同层数或嵌入维度。一个具体的例子是使用不同的变换器同时学习3D目标检测(基于点云)和2D目标检测(根据图像)。尽管已经有专门的基于变换器的模型分别设计用于3D或2D目标检测，但仍然缺乏一种快速有效的方法来组合这些模型和任务。

An overall structure of TokenFusion for fusing heterogeneous modalities is depicted in Fig. 2. Different from the homogeneous case, we approximate the token projection function h in Eq. (2) with a shallow multi-layer perception (MLP), since transformers for these heterogeneous modalities may have different hidden embedding dimensions. For the case of 3D object detection with 3D point cloud and 2D image, we project each point to the corresponding image based on camera intrinsics and extrinsics. Likewise, we also project 3D object labels to the images for obtaining the corresponding 2D object labels. We train two standalone transformers with unshared parameters in an end-to-end manner. Regarding the 3D object detection with point cloud as input, we follow the architecture used in Group-Free [22], where $N_{point}$ sampled seed points and $K_{point}$ learned proposal points are considered as input tokens, which are sent to the transformer for predicting $K_{point}$ 3D bounding boxes and object categories. For the 2D object detection with images as input, we follow the framework in YOLOS [8] which sends $N_{img}$ image patches and $K_{img}$ object queries to the transformer to predict $K_{img}$ 2D bounding boxes together with their associated object categories.

图2描述了用于融合异构模态的TokenFusion的总体结构。与同质情况不同，我们使用浅层多层感知(MLP)近似方程(2)中的令牌投影函数h，因为这些异质模态的变换器可能具有不同的隐藏嵌入维度。对于使用3D点云和2D图像进行3D目标检测的情况，我们基于相机内部和外部将每个点投影到相应的图像。同样，我们还将3D目标标签投影到图像以获得相应的2D目标标签。我们以端到端的方式训练两个具有非共享参数的独立transformer。关于以点云为输入的3D目标检测，我们遵循Group Free[22]中使用的架构，其中$N_{point}$采样的种子点和$K_{point}$学习的建议点被视为输入令牌，这些令牌被发送到转换器，用于预测$K_点}$3D边界框和目标类别。对于以图像为输入的2D目标检测，我们遵循YOLOS[8]中的框架，该框架将$N_{img}$图像补丁和$K_{img}$目标查询发送到转换器，以预测$K_{img}$2D边界框及其关联的目标类别。

Figure 2. Framework of TokenFusion for heterogeneous modalities with point clouds and images. Both modalities are sent to individual transformer modules with also individual positional embeddings. Additional inter-modal projections (Proj) are needed which is different from the fusion for homogeneous modalities. 
图2:用于具有点云和图像的异构模态的TokenFusion框架。两种模态都被发送到具有单独位置嵌入的单独transformer模块。需要额外的模态间投影(Proj)，这不同于同质模态的融合。


The inter-modal projection maps seed points to image patches, i.e., an $N_{point}-to-N_{img}$ mapping. Specifically, the token-wise pruning is applied on the $N_{point}$ seed point tokens. Once a certain token obtains a low importance score, we project the 3D coordinate of this token to a 2D pixel on the corresponding image input. It is now viable to locate the specific image patch based on the 2D pixel. Suppose this projection obtains the  $n_{img}$-th image patch based on the $n_{point}-th$ seed point which is pruned. We substitute m and m' in Eq. (2) with the subscripts “point” and “img” respectively, i.e., $Proj^T_{img}(x_{point}[n_{point}]) = h(x_{img}[n_{img}])$. Thus the relation between  $n_{point}$ and  $n_{img}$ captured by the token projection satisfies  

模态间投影将种子点映射到图像块，即$N_{point}-到N_{img}$映射。具体而言，对$N_｛点｝$seed点令牌应用令牌式修剪。一旦某个令牌获得低重要度分数，我们将该令牌的3D坐标投影到相应图像输入上的2D像素。现在，基于2D像素定位特定图像块是可行的。假设该投影基于被修剪的第$n_{point}-个$seed点获得第$n_{img}-个图像块。我们分别用下标“point”和“img”替换等式(2)中的m和m'，即$Proj^T_{img}(x_{point}[n_{point}]) = h(x_{img}[n_{img}])$。因此，令牌投影捕获的$n_{point}$和$n_{img}$之间的关系满足

$[u, v, z]^T  = K · R_t · [ x_{n_{point}} , y_{n_{point}} , z_{n_{point}}, 1 ]^T $, (8) 

$n_{img}= [ \frac{[v/z]}{P}  × \frac{W}{P} ] + [ \frac{[u/z]}{P} ]$ , (9) 

where K ∈ $R^{4×4}$ and $R_t ∈ R^{4×4}$ are camera intrinsic and extrinsic matrices, respectively; $[x_{n_{point}} , y_{n_{point}} , z_{n_{point}} ]$ denotes the 3D coordinate of the  $n_{point}-th$ point; u, v, z are temporary variables with  [ [u/z] , [v/z] ] actually being the projected pixel coordinate of the image; P is the patch size of the vision transformer and W denotes the image width.

其中K∈$R^{4×4}$和$R_t∈R^{4×4}$分别是相机内矩阵和外矩阵$[x_{n_{point}} , y_{n_{point}} , z_{n_{point}} ]$表示第$n_{个}点的3D坐标; u、 v，z是临时变量，[[u/z]，[v/z]]实际上是图像的投影像素坐标; P是视觉变换器的面片大小，W表示图像宽度。

## 4. Experiments
To evaluate the effectiveness of the proposed TokenFusion, we conduct comprehensive experiments towards both homogeneous and heterogeneous modalities with state-ofthe-art (SOTA) methods. Experiments are conducted on totally seven different modalities and four application scenarios, implemented with PyTorch [25] and MindSpore [15].

为了评估所提出的TokenFusion的有效性，我们使用最先进的(SOTA)方法对同质和异质模态进行了全面的实验。使用PyTorch[25]和MindSpore[15]对总共七种不同的模式和四种应用场景进行了实验。

### 4.1. Multimodal Image-to-Image Translation 多模态图像到图像转换
The task of multimodal image-to-image translation aims at generating a target image modality based on different image modalities as input (e.g. Normal+Depth→RGB). We evaluate TokenFusion in this task using the Taskonomy [45] dataset, which is a large-scale indoor scene dataset containing about 4 million indoor images captured from 600 buildings. Taskonomy provides over 10 multimodal representations in addition to each RGB image, such as depth (euclidean or z-buffering), normal, shade, texture, edge, principal curvature, etc. The resolution of each representation is 512 × 512. To facilitate comparison with the existing fusion methods, we adopt the same sampling strategy as [42], resulting in 1,000 high-quality multimodal images for training, and 500 for validation.

多模态图像到图像转换的任务旨在基于作为输入的不同图像模态(例如正常+深度)生成目标图像模态→RGB)。我们使用Taskonomy[45]数据集评估了该任务中的TokenFusion，该数据集是一个大型室内场景数据集，包含从600栋建筑中捕获的约400万张室内图像。除了每个RGB图像之外，Taskonomy还提供了超过10种多模态表示，例如深度(欧氏或z缓冲)、法线、阴影、纹理、边缘、主曲率等。每个表示的分辨率为512×512。为了便于与现有融合方法进行比较，我们采用了与[42]相同的采样策略，生成了1000张用于训练的高质量多模态图像，500张用于验证。

Our implementation contains two transformers as the generator and discriminator respectively. We provide configuration details in our supplementary materials. The resolution of the generator/discriminator input or the generator prediction is 256 × 256. We adopt two kinds of architecture settings, the tiny (Ti) version with 10 layers and the small (S) version with 20 layers, and both settings are only different in layer numbers. The learning rates of both transformers are set to 2 × 10−4 . We adopt overlapped patches in both transformers inspired by [44].

我们的实现包含两个变压器，分别作为生成器和鉴别器。我们在补充材料中提供了配置详情。生成器/鉴别器输入或发生器预测的分辨率为256×256。我们采用两种架构设置，10层的微型(Ti)版本和20层的小型(S)版本，两种设置仅在层数上不同。两台变压器的学习率均设置为2×10−4。我们在两个变压器中采用重叠补丁，灵感来自[44]。

In our experiments for this task, we adopt shared transformers for all input modalities with individual layer normalizations (LNs) that individually compute the means and variances of different modalities. Specifically, parameters in the linear projection on patches, all linear projections (e.g. for key, queries, etc) in MSA, and MLP are shared for different modalities. Such a mechanism largely reduces the total model size which as discussed in the supplementary materials, even achieves better performance than using individual transformers. In addition, we also adopt shared positional embeddings for different modalities. We let the sparsity weight 

在这项任务的实验中，我们为所有输入模态采用共享变换器，并使用单独的层归一化(LN)来单独计算不同模态的均值和方差。具体而言，贴片上的线性投影、MSA中的所有线性投影(例如，用于键、查询等)和MLP中的参数对于不同的模态是共享的。这种机制大大减小了模型的总尺寸，如补充材料中所述，甚至比使用单个变压器实现更好的性能。此外，我们还为不同的模态采用了共享的位置嵌入。对于所有这些实验，我们让方程(10)中的稀疏权重λ=10−4，方程(7)中的阈值θ=2×10−2。

λ = 10−4 in Eq. (10) 

and the threshold 

θ = 2 × 10−2 in Eq. (7) 

for all these experiments.



Our evaluation metrics include FID/KID for RGB predictions and MAE/MSE for other predictions. These metrics are introduced in the supplementary materials.

我们的评估指标包括RGB预测的FID/KID和其他预测的MAE/MSE。补充材料中介绍了这些指标。

Results. In Table 1, we provide comparisons with extensive baseline methods and a SOTA method [42] with the same data settings. All methods adopt the learned ensemble over the two predictions which are corresponded to the two modality branches. In addition, all predictions have the same resolution 256×256 for a fair comparison. Since most existing methods are based on CNNs, we further provide two baselines for transformer-based models including the baseline without feature fusion (only uses ensemble for the late fusion) and the feature fusion method. By comparison, our TokenFusion surpasses all the other methods with large margins. For example, in the Shade+Texture→RGB task, our TokenFusion (S) achieves 43.92/0.94 FID/KID scores, remarkably better than the current SOTA method CEN [42] with 29.8% relative FID metric decrease.

结果在表1中，我们提供了与广泛基线方法和具有相同数据设置的SOTA方法[42]的比较。所有方法都在对应于两个模态分支的两个预测上采用学习的集合。此外，为了公平比较，所有预测都具有相同的分辨率256×256。由于大多数现有方法都基于神经网络，我们进一步为基于变压器的模型提供了两种基线，包括无特征融合的基线(仅使用集成进行后期融合)和特征融合方法。相比之下，我们的TokenFusion以巨大的利润超过了所有其他方法。例如，在“着色+纹理”中→RGB任务，我们的TokenFusion(S)实现了43.92/0.94 FID/KID分数，显著优于当前的SOTA方法CEN[42]，相对FID度量降低了29.8%。

Figure 3. Comparison on the validation data split for image-to-image translation (Texture+Shade→RGB). The resolution of all input/output images is 256×256. The third/forth column is predicted by the single modality, and the following three columns are predicted by CEN [42], the intuitive transformer fusion by feature concatenation, and our TokenFusion, respectively. Best view in color and zoom in. 
图3。图像到图像转换(纹理+阴影)的验证数据分割比较→RGB)。所有输入/输出图像的分辨率为256×256。第三列/第四列由单一模态预测，以下三列分别由CEN[42]、通过特征级联的直观变换器融合和我们的TokenFusion预测。最佳的颜色和放大视图。

In supplementary materials, we consider more modality inputs up to 4 which evaluates our group allocation strategy.

在补充材料中，我们考虑了最多4个模态输入，以评估我们的组分配策略。

Visualization and analysis. We provide qualitative results in Fig. 3, where we choose tough samples for comparison. The predictions with our TokenFusion obtain better natural patterns and are also richer in colors and details. In Fig. 4, we further visualize the process of TokenFusion of which tokens are learned to be fused under our l1 sparsity constraints. We observe that the tokens for fusion follow specific regularities. For example, the texture modality tends to preserve its advantage of detailed boundaries, and meanwhile seek facial tokens from the shade modality. In this sense, TokenFusion combines complementary properties of different modalities.

可视化和分析。我们在图3中提供了定性结果，其中我们选择了坚韧的样本进行比较。使用我们的TokenFusion进行的预测获得了更好的自然图案，颜色和细节也更丰富。在图4中，我们进一步可视化了TokenFusion的过程，其中令牌在我们的l1稀疏性约束下被学习融合。我们观察到融合的标记遵循特定的规则。例如，纹理模态倾向于保留其细节边界的优势，同时从阴影模态中寻找面部标记。在这个意义上，TokenFusion结合了不同模式的互补属性。

### 4.2. RGB-Depth Semantic Segmentation RGB-Depth语义分割
We then evaluate TokenFusion on another homogeneous scenario, semantic segmentation with RGB and depth as input, which is a very common multimodal task and numerous methods have been proposed towards better performance. We choose the typical indoor datasets, NYUDv2 [33] and SUN RGB-D [34]. For NYUDv2, we follow the standard 795/654 images for train/test splits to predict the standard 40 classes [10]. SUN RGB-D is one of the most challenging large-scale indoor datasets, and we adopt the standard 5,285/5,050 images for train/test of 37 semantic classes.

然后，我们在另一个同质场景(以RGB和深度为输入的语义分割)上评估TokenFusion，这是一个非常常见的多模式任务，已经提出了许多方法来提高性能。我们选择了典型的室内数据集NYUDv2[33]和SUN RGB-D[34]。对于NYUDv2，我们遵循标准795/654图像进行训练/测试分割，以预测标准40类[10]。SUN RGB-D是最具挑战性的大型室内数据集之一，我们采用标准5285/5050图像对37个语义类进行训练/测试。

Table 1. Results on Taskonomy for multimodal image-to-image translation. Evaluation metrics are FID/KID (×10−2 ) for RGB predictions and MAE (×10−1)/MSE (×10−1 ) for other predictions. Lower values indicate better performance for all the metrics. 
表1。多模态图像到图像翻译的任务经济学结果。RGB预测的评估指标为FID/KID(×10−2)，其他预测的评估标准为MAE(×10–1)/MSE(×10-1)。值越低，表示所有指标的性能越好。


Our models include TokenFusion (tiny) and TokenFusion (small), of which the single-modal backbones follow B2 and B3 settings of SegFormer [44]. Both tiny and small versions adopt the pretrained parameters on ImageNet-1k for initialization following [44]. Similar to our implementation in Sec. 4.1, we also adopt shared transformers and positional embeddings for RGB and depth inputs with individual LNs. We let the sparsity weight λ = 10−3 in Eq. (10) and the threshold θ = 2 × 10−2 in Eq. (7) for all these experiments.

我们的模型包括TokenFusion(微型)和TokenFusi(小型)，其中单模主干遵循SegFormer的B2和B3设置[44]。微型和小型版本都采用ImageNet-1k上的预训练参数进行初始化[44]。与我们在第4.1节中的实现类似，我们还为RGB和深度输入采用了共享的转换器和位置嵌入，以及单独的LNs。对于所有这些实验，我们让方程(10)中的稀疏权重λ=10−3，方程(7)中的阈值θ=2×10−2。

Results. Results provided in Table 2 conclude that current transformer-based models equipped with our TokenFusion surpass SOTA models using CNNs. Note that we choose relatively light backbone settings (B1 and B2 as mentioned in Sec. 4.2). We expect that using larger backbones (e.g., B5) would yield better performance.

结果表2中提供的结果表明，配备了我们的TokenFusion的基于电流互感器的模型超过了使用CNN的SOTA模型。请注意，我们选择了相对较轻的主干设置(如第4.2节所述的B1和B2)。我们希望使用较大的主干(如B5)将产生更好的性能。

Figure 4. Illustrations of which tokens are fused in our TokenFusion, performed on the validation data split. We provide two cases including Texture+Shade→RGB (first row) and Shade+RGB→Normal (second row). The resolution of all images is 256 × 256. We choose the last layers in the first and second transformer stages respectively. Best view in color and zoom in.
图4。在验证数据分割上执行的TokenFusion中融合了哪些令牌的说明。我们提供两种情况，包括纹理+阴影→RGB(第一行)和着色+RGB→正常(第二行)。所有图像的分辨率为256×256。我们分别选择第一和第二变压器级中的最后一层。最佳的颜色和放大视图。

Figure 5. Results visualization on the validation data split for heterogeneous modalities including point clouds and images, where 3D object detection and 2D object detection are learned simultaneously. We compare the performance without (w/o) or with our TokenFusion. Our TokenFusion mainly benefits 3D object detection results.
图5。对异构模态(包括点云和图像)的验证数据分割结果进行可视化，其中同时学习3D目标检测和2D目标检测。我们比较了没有(w/o)或使用TokenFusion的性能。我们的TokenFusion主要受益于3D目标检测结果。

Table 2. Comparison results on the NYUDv2 and SUN RGB-D datasets with SOTAs for RGB and depth (D) semantic segmentation. Evaluation metrics include pixel accuracy (Pixel Acc.) (%), mean accuracy (mAcc.) (%), and mean IoU (mIoU) (%).
表2。NYUDv2和SUN RGB-D数据集与SOTA的RGB和深度(D)语义分割的比较结果。评估指标包括像素精度(pixel Acc.)(%)、平均精度(mAcc.)(%，以及平均IoU(mIoU)(%)。

### 4.3. Vision and Point Cloud 3D Object Detection 视觉和点云3D目标检测
We further apply TokenFusion for fusing heterogeneous modalities, specifically, the 3D object detection task which has received great attention. We leverage 3D point clouds and 2D images to learn 3D and 2D detections, respectively, and both processes are learned simultaneously. We expect the involvement of 2D learning boosts the 3D counterpart.

我们进一步将TokenFusion应用于融合异构模态，特别是备受关注的3D目标检测任务。我们利用3D点云和2D图像分别学习3D和2D检测，同时学习这两个过程。我们预计2D学习的参与会促进3D学习。

We adopt SUN RGB-D [35] and ScanNetV2 [5] datasets. For SUN RGB-D, we follow the same train/test splits as in Sec. 4.2 and detect the 10 most common classes. For ScanNetV2, we adopt the 1,201/312 scans as train/test splits to detect the 18 object classes. All these settings (splits and detected target classes) follow current works [22,28] for a fair comparison. Note that different from SUN RGB-D, ScanNetV2 provides multi-view images for each scene alongside the point cloud. We randomly sample 10 frames per scene from the scannet-frames-25k samples provided in [5].

我们采用SUN RGB-D[35]和ScanNetV2[5]数据集。对于SUN RGB-D，我们遵循与第4.2节相同的训练/测试分割，并检测10个最常见的类别。对于ScanNetV2，我们采用1201/312扫描作为训练/测试分割，以检测18个目标类。所有这些设置(分割和检测到的目标类)遵循当前工作[22，28]，以进行公平比较。请注意，与SUN RGB-D不同，ScanNetV2为点云旁的每个场景提供多视图图像。我们从[5]中提供的扫描网帧-25k个样本中随机抽取每个场景10个帧。

Our architectures for 3D detection and 2D detection follow GF [22] and YOLOS [8], respectively. We adopt the “L6, O256” or “L12, O512” versions of GF for the 3D detection branch. We combine GF with the tiny (Ti) and small (S) versions of YOLOS, respectively, and adopt mAP@0.25 and mAP@0.5 as evaluation metrics following [22, 28].

我们的3D检测和2D检测架构分别遵循GF[22]和YOLOS[8]。我们采用GF的“L6，O256”或“L12，O512”版本作为3D检测分支。我们分别将GF与YOLOS的微型(Ti)和小型(S)版本相结合，并采用mAP@0.25和mAP@0.5作为[22，28]之后的评估指标。

Table 3. Comparison on SUN RGB-D with SOTAs for 3D object detection, including best results and average results in brackets. * indicates appending RGB to the points as described in Sec. 4.3.
表3。SUN RGB-D与SOTA的3D物体检测对比，包括括号中的最佳结果和平均结果。*表示将RGB附加到第4.3节所述的点。

Results. We provide results comparison in Table 3 and Table 4. The main comparison is based on the best results of five experiments between different methods, and numbers within the brackets are the average results. Besides, we perform intuitive multimodal experiments by appending the 3-channel RGB vectors to the sampled points after PointNet++ [30]. Such intuitive experiments are marked by the subscript * in both tables. We observe, however, that simply appending RGB information even leads to the performance drop, indicating the difficulty of such a heterogeneous fusion task. By comparison, our TokenFusion achieves new records on both datasets, which are remarkably superior to previous CNN/transformer models in terms of both metrics. For example, with TokenFusion, YOLOS-Ti can be utilized to boost the performance of GF by further 2.4 mAP@0.25 improvements, and using YOLOS-S brings further gains.

后果我们在表3和表4中提供了结果比较。主要的比较是基于不同方法之间的五个实验的最佳结果，括号内的数字是平均结果。此外，我们通过将3通道RGB向量附加到PointNet++[30]之后的采样点来执行直观的多模态实验。这两个表中的下标*标记了这种直观的实验。然而，我们观察到，简单地附加RGB信息甚至会导致性能下降，这表明这种异构融合任务的难度。相比之下，我们的TokenFusion在两个数据集上都取得了新的记录，在这两个指标方面都明显优于之前的CNN/transformer模型。例如，使用TokenFusion，YOLOS Ti可以进一步提高GF的性能2.4mAP@0.25改进，并使用YOLOS-S带来更多收益。

Visualizations. Fig. 5 illustrates the comparison of detection results when using TokenFusion for multimodal interactions against individual learning. We observe that TokenFusion benefits the 3D detection part. For example, with the help of images, models with TokenFusion can locate 3D objects even with sparse or missing point data (second row). In addition, using images also benefits when the points of two objects are largely overlapped (first row). These observations demonstrate the advantages of our TokenFusion.

可视化。图5说明了使用TokenFusion进行多模态交互与个体学习时检测结果的比较。我们观察到TokenFusion有利于3D检测部分。例如，借助图像，使用TokenFusion的模型可以定位3D目标，即使有稀疏或缺失的点数据(第二行)。此外，当两个目标的点在很大程度上重叠(第一行)时，使用图像也有好处。这些观察结果证明了我们TokenFusion的优势。

## 5. Ablation Study
l1-norm and token fusion. In Table 5, we demonstrate the advantages of l1-norm and token fusion. We additionally conduct experiments with random token fusion. We observe that applying l1-norm itself has little effect on the performance yet it is essential to reveal tokens for fusion. Our token fusion together with l1-norm achieves much better performance than the random fusion baselines.

l1范数和令牌融合。在表5中，我们展示了l1范数和令牌融合的优点。我们还进行了随机令牌融合实验。我们观察到，应用l1范数本身对性能几乎没有影响，但揭示融合的令牌至关重要。我们的令牌融合与l1范数一起实现了比随机融合基线更好的性能。

Evaluation of RPA. Table 6 evaluates RPA proposed in Sec. 3.3. Results indicate that only using RPA without token fusion does not noticeably affect the performance, but is important when combined with the token fusion process for alignments, especially for the 3D detection task.

RPA评估。表6评估了第3.3节中提出的RPA。结果表明，仅使用RPA而不使用令牌融合不会显著影响性能，但当结合令牌融合过程进行比对时，这一点非常重要，尤其是对于3D检测任务。

Table 4. Comparison on ScanNetV2 with SOTAs for 3D object detection, including best results and average results in brackets.
表4。ScanNetV2与SOTA的3D物体检测对比，包括括号内的最佳结果和平均结果。

Table 5. Effectiveness of l1-norm and token fusion. Experiments include RGB-depth segmentation (seg.) on NYUDv2 and 3D detection (det.) with images and points on SUN RGB-D. 
表5。l1范数和令牌融合的有效性。实验包括NYUDv2上的RGB深度分割(seg.)和SUN RGB-D上的图像和点的3D检测(det.)。

Table 6. Effectiveness of RPA proposed in Sec. 3.4. Experimental tasks and datasets follow Table 5.
表6。第3.4节中建议的RPA的有效性。实验任务和数据集见表5。

## 6. Conclusion
We propose TokenFusion, an adaptive method generally applicable for fusing vision transformers with homogeneous or heterogeneous modalities. TokenFusion exploits uninformative tokens and re-utilizes these tokens to strengthen the interaction of other informative multimodal tokens. Alignment relations of different modalities can be explicitly utilized due to our residual positional alignment and inter-modal projection. TokenFusion surpasses stateof-the-art methods on a variety of tasks, demonstrating its superiority and generality for multimodal fusion.

我们提出了TokenFusion，这是一种通常适用于融合具有同质或异质模态的视觉变换器的自适应方法。TokenFusion利用非信息令牌，并重新利用这些令牌来加强其他信息型多模态令牌的交互。由于我们的残余位置对准和模态间投影，可以明确利用不同模态的对准关系。TokenFusion在各种任务上超越了最先进的方法，证明了其在多模态融合中的优势和通用性。

## Acknowledgement
This work is funded by Major Project of the New Generation of Artificial Intelligence (No. 2018AAA0102900) and the Sino-German Collaborative Research Project Crossmodal Learning (NSFC 62061136001/DFG TRR169). We gratefully acknowledge the support of MindSpore, CANN and Ascend AI Processor used for this research.

这项工作得到了新一代人工智能重大项目(编号：2018AAA0102900)和中德合作研究项目“跨模式学习”(国家自然科学基金62061136001/DFG TRR169)的资助。我们非常感谢MindSpore、CANN和Ascend AI处理器对本研究的支持。

## Appendix
### A. Additional Results
Multiple input modalities. In Table 7, we further evaluate our TokenFusion with more modality inputs from 1 to 4. When the number of input modalities is larger than 2, we adopt the group allocation strategy as proposed in Sec. 3.4 of our main paper. By comparison, the performance is consistently improved when using more modalities, and TokenFusion is again noticeably better than CEN [42], suggesting the ability to absorb information from more modalities.

Table 7. Results on the Taskonomy dataset for multimodal imageto-image translation (to RGB) with 1 ∼ 4 modalities.

Network sharing. As mentioned in Sec. 3.4 of our main paper, we adopt shared parameters in both Multi-head SelfAttention (MSA) and Multi-Layer Perception (MLP) for the fusion with homogeneous modalities, and rely on modalityspecific Layer Normalization (LN) layers to uncouple the normalization process. Such network sharing technique is evaluated by our experiments including multimodal imageto-image translation (in Sec. 4.1) and RGB-depth semantic segmentation (in Sec. 4.2), which largely reduces the model size, and also enables the reuse of attention weights for different modalities. In Table 8, we further conduct ablation studies to demonstrate the effectiveness of our network sharing scheme. Fortunately, the comparison indicates that our default setting (i.e., Shared MSA and MLP, individual

LN) achieves a win-win scenario: apart from the advantage on storage efficiency, also achieves better results than using individual MSA and MLP on both tasks. Note that further sharing LN layers leads to the performance drop, especially on the image-to-image translation task. In addition, we adopt shared Positional Embeddings (PEs) by default for the fusion with homogeneous modalities, and we observe that sharing/unsharing PEs can achieve comparable performance in practice.

Combining TokenFusion with channel-wise fusion.

Our TokenFusion detects uninformative tokens and reutilizes these tokens for multimodal fusion. We may further combine TokenFusion with an orthogonal method by channel-wise pruning which automatically detects uninformative channels. Different from the token-wise fusion method in TokenFusion, the channel-wise fusion is not conditional on input features. Inspired by CEN [42], we leverImage translation Seg. 

Table 8. Results comparison when using different network sharing schemes for image-to-image translation (Shade+Texture→RGB) on Taskonomy and RGB-depth segmentation (seg.) on NYUDv2.

Lower FID or KID values indicate better performance.



Table 9. RGB-depth segmentation results on the NYUDv2 dataset when combining our TokenFusion with the channel-wise fusion. 3D det. (ScanNetV2) Input image frames Model mAP@0.25 mAP@0.5

Seconds per 100 scenes 0 Ours (L6, O256; Ti) 67.3 49.0 4.7 5 Ours (L6, O256; Ti) 67.9 50.5 5.9 10 Ours (L6, O256; Ti) 68.8 51.9 7.0

Table 10. Comparison of practical inference speed on ScanNetV2. age the scaling factors γ of layer normalization (LN) to perform channel-wise pruning, and apply sparsity constraints on γ. LN in transformers performs normalization on its input xm,l.

To prune uninformative channels, we add a channel-wise pruning loss P Mm=1

P

Ll=1 |γlm| to the main loss in Eq. (5) (main paper). The overall loss function is

L = MXm=1 

Lm + λ1 LXl=1   sl(e^l_m) + λ2 LXl=1 |γlm| , (10) where λ1, λ2 are hyper-parameters for balancing different losses; γlm is a vector with the length C, representing the scaling factor of LN at the l-th layer of the m-th modality.

We let λ1 = λ2 = 10−3 for RGB-depth segmentation experiments. Results provided in Table 9 demonstrate that our TokenFusion can be combined with the channel-wise fusion to obtain a further improved performance. For example, the segmentation on NYUDv2 with both token-wise and channel-wise fusion achieves an additional 0.5 mIoU gain than TokenFusion. More detailed studies of such combined framework, the relation between the overall pruning rate and fusion performance gain, and the extension to fuse heterogeneous modalities are left to be the future works.

Additional visualizations. In Fig. 6, we provide another group of visualizations that depict the fused tokens under the l1 sparsity constraints during training. We observe that fused tokens follow the regularities mentioned in our main paper, e.g., the texture modality preserves its ad-



Figure 6. Additional illustrations of the token fusion process as a supplement to Fig. 4 (main paper), performed on the validation data split of Taskonomy. We provide two cases: Texture+Shade→RGB (first row) and Shade+RGB→Normal (second row). The resolution of all images is 256 × 256. We choose the last layers in the first and second transformer stages respectively. Best view in color and zoom in. vantage at boundaries while seeking facial tokens from the shade modality.

Inference speed. In Table 10, we test the real inference speed (single V100, 256G RAM) with different numbers of input frames for 3D detection. We observe that additional time costs are mild, which is partly because the added YOLOS-Ti is a light model (with only three multi-heads).

### B. More Details of Image Translation
In this part, we discuss the implementation details for our image-to-image translation task. Our implementation contains two transformers as the generator and discriminator respectively. The resolution of the generator/discriminator input or the generator prediction is 256×256. Specifically, the discriminator of our model is similar to [16], which adopts five stages with two layers for each, where the embedding dimensions and head numbers gradually double from 32 to 512 and from 1 to 16 respectively. The generator is composed of nine stages where the first five have the same configurations with the discriminator, and the last four stages have reverse configurations of its first four stages.

We adopt four kinds of evaluation metrics including Mean Square Error (MSE), Mean Absolute Error (MAE), Fr´echet-Inception-Distance (FID), and Kernel-InceptionDistance (KID). Here we briefly introduce FID and KID scores. FID, proposed by [13], contrasts the statistics of generated samples against real samples. The FID fits a Gaussian distribution to the hidden activations of InceptionNet for each compared image set and then computes the Fr´echet distance (also known as the Wasserstein-2 distance) between those Gaussians. Lower FID is better, corresponding to generated images more similar to the real. KID, developed by [1], is a metric similar to the FID but uses the squared Maximum-Mean-Discrepancy (MMD) between Inception representations with a polynomial kernel. Unlike FID, KID has a simple unbiased estimator, making it more reliable especially when there are much more inception features channels than image numbers. Lower KID indicates more visual similarity between real and generated images.

Regarding our implementation of KID, the hidden representations are derived from the Inception-v3 [38] pool3 layer.

## References
1. Mikolaj Binkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD gans. In ICLR, 2018. 10
2. Aljaz Bozic, Pablo R. Palafox, Justus Thies, Angela Dai, and Matthias Nießner. Transformerfusion: Monocular RGB scene reconstruction using transformers. In NeurIPS, 2021. 2
3. Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020. 1, 2
4. Jintai Chen, Biwen Lei, Qingyu Song, Haochao Ying, Danny Z Chen, and Jian Wu. A hierarchical graph network for 3d object detection on point clouds. In CVPR, 2020. 8
5. Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas A. Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017. 7
6. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020. 1, 2
7. Francis Engelmann, Martin Bokeloh, Alireza Fathi, Bastian Leibe, and Matthias Nießner. 3d-mpa: Multi-proposal aggregation for 3d semantic instance segmentation. In CVPR, 2020. 8
8. Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, and Wenyu Liu. You only look at one sequence: Rethinking transformer in vision through object detection. arXiv preprint arXiv:2106.00666, 2021. 1, 2, 4, 7
9. Yu Fu, TianYang Xu, XiaoJun Wu, and Josef Kittler. Ppt fusion: Pyramid patch transformerfor a case study in image fusion. arXiv preprint arXiv:2107.13967, 2021. 2
10. Saurabh Gupta, Pablo Arbelaez, and Jitendra Malik. Perceptual organization and recognition of indoor scenes from RGB-D images. In CVPR, 2013. 6
11. JunYoung Gwak, Christopher Choy, and Silvio Savarese. Generative sparse detection networks for 3d single-shot object detection. arXiv preprint arXiv:2006.12356, 2020. 8
12. Caner Hazirbas, Lingni Ma, Csaba Domokos, and Daniel Cremers. Fusenet: Incorporating depth into semantic segmentation via fusion-based CNN architecture. In ACCV, 2016. 7
13. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NIPS, 2017. 10
14. Jin-Fan Hu, Ting-Zhu Huang, and Liang-Jian Deng. Fusformer: A transformer-based fusion approach for hyperspectral image super-resolution. arXiv preprint arXiv:2109.02079, 2021. 2
15. Huawei. Mindspore. https://www.mindspore.cn/, 2020. 5
16. Yifan Jiang, Shiyu Chang, and Zhangyang Wang. Transgan: Two pure transformers can make one strong gan, and that can scale up. In NeurIPS, 2021. 1, 2, 10
17. Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Visionand-language transformer without convolution or region supervision. arXiv preprint arXiv:2102.03334, 2021. 1
18. Seungyong Lee, Seong-Jin Park, and Ki-Sang Hong. Rdfnet: RGB-D multi-level residual feature fusion for indoor semantic segmentation. In ICCV, 2017. 7
19. Guosheng Lin, Fayao Liu, Anton Milan, Chunhua Shen, and Ian Reid. Refinenet: Multi-path refinement networks for dense prediction. In IEEE Trans. PAMI, 2019. 7
20. Rui Liu, Hanming Deng, Yangyi Huang, Xiaoyu Shi, Lewei Lu, Wenxiu Sun, Xiaogang Wang, Jifeng Dai, and Hongsheng Li. Fuseformer: Fusing fine-grained information in transformers for video inpainting. In ICCV, 2021. 2
21. Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030, 2021. 1
22. Ze Liu, Zheng Zhang, Yue Cao, Han Hu, and Xin Tong. Group-free 3d object detection via transformers. In ICCV, 2021. 1, 2, 4, 7, 8
23. Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015. 7
24. Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, and Chen Sun. Attention bottlenecks for multimodal fusion. arXiv preprint arXiv:2107.00135, 2021. 2
25. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019. 5
26. Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multimodal fusion transformer for end-to-end autonomous driving. In CVPR, 2021. 1, 2
27. Charles R Qi, Xinlei Chen, Or Litany, and Leonidas J Guibas. Imvotenet: Boosting 3d object detection in point clouds with image votes. In CVPR, 2020. 8
28. Charles R Qi, Or Litany, Kaiming He, and Leonidas J Guibas. Deep hough voting for 3d object detection in point clouds. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9277–9286, 2019. 7
29. Charles R Qi, Or Litany, Kaiming He, and Leonidas J Guibas. Deep hough voting for 3d object detection in point clouds. In ICCV, 2019. 8
30. Charles R Qi, Li Yi, Hao Su, and Leonidas J. Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In NIPS, 2017. 8
31. Xie Qian, Lai Yu-kun, Wu Jing, Wang Zhoutao, Zhang Yiming, Xu Kai, and Wang Jun. Mlcvnet: Multi-level context votenet for 3d object detection. In CVPR, 2020. 8
32. Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. In NeurIPS, 2021. 3
33. Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from RGBD images. In ECCV, 2012. 6
34. Shuran Song, Samuel P. Lichtenberg, and Jianxiong Xiao. SUN RGB-D: A RGB-D scene understanding benchmark suite. In CVPR, 2015. 6
35. Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In CVPR, 2015. 7
36. Sijie Song, Jiaying Liu, Yanghao Li, and Zongming Guo. Modality compensation network: Cross-modal adaptation for action recognition. In IEEE Trans. Image Process., 2020. 6
37. Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visuallinguistic representations. In ICLR, 2019. 1
38. Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016. 10
39. Abhinav Valada, Rohit Mohan, and Wolfram Burgard. Selfsupervised model adaptation for multimodal semantic segmentation. In IJCV, 2020. 6, 7
40. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017. 1, 2
41. Vibashan VS, Jeya Maria Jose Valanarasu, Poojan Oza, and Vishal M Patel. Image fusion transformer. arXiv preprint arXiv:2107.09011, 2021. 2
42. Yikai Wang, Wenbing Huang, Fuchun Sun, Tingyang Xu, Yu Rong, and Junzhou Huang. Deep multimodal fusion by channel exchanging. In NeurIPS, 2020. 5, 6, 7, 9
43. Yikai Wang, Fuchun Sun, Ming Lu, and Anbang Yao. Learning deep multimodal feature representation with asymmetric multi-layer fusion. In ACM MM, 2020. 7
44. Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. In NeurIPS, 2021. 1, 2, 5, 6
45. Amir Roshan Zamir, Alexander Sax, William B. Shen, Leonidas J. Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In CVPR, 2018. 5
46. Zaiwei Zhang, Bo Sun, Haitao Yang, and Qixing Huang. H3dnet: 3d object detection using hybrid geometric primitives. arXiv preprint arXiv:2006.05682, 2020. 8
47. Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip H.S. Torr, and Li Zhang. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In CVPR, 2021. 1, 2
48. Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020. 1, 2
