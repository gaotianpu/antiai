# A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP
网络结构PK：CNN、Transformer和MLP的实证研究 2021-8-30 原文：https://arxiv.org/abs/2108.13002

## Abstract
Convolutional neural networks (CNN) are the dominant deep neural network (DNN) architecture for computer vision. Recently, Transformer and multi-layer perceptron (MLP)-based models, such as Vision Transformer and MLP-Mixer, started to lead new trends as they showed promising results in the ImageNet classification task. In this paper, we conduct empirical studies on these DNN structures and try to understand their respective pros and cons. To ensure a fair comparison, we first develop a unified framework called SPACH which adopts separate modules for spatial and channel processing. Our experiments under the SPACH framework reveal that all structures can achieve competitive performance at a moderate scale. However, they demonstrate distinctive behaviors when the network size scales up. Based on our findings, we propose two hybrid models using convolution and Transformer modules. The resulting Hybrid-MS-S+ model achieves 83.9% top-1 accuracy with 63M parameters and 12.3G FLOPS. It is already on par with the SOTA models with sophisticated designs. The code and models are publicly available at https://github.com/microsoft/SPACH .

卷积神经网络(CNN)是计算机视觉的主要深度神经网络(DNN)架构。最近，基于Transformer和多层感知器(MLP)的模型，如Vit和MLP-Mixer，开始引领新的趋势，因为它们在ImageNet分类任务中显示出有希望的结果。在本文中，我们对这些DNN结构进行了实证研究，并试图了解它们各自的优缺点。为了确保公平的比较，我们首先开发了一个名为SPACH的统一框架，该框架采用独立的模块进行空间和信道处理。我们在SPACH框架下的实验表明，所有结构都可以在中等规模上实现竞争性能。然而，当网络规模扩大时，它们表现出独特的行为。基于我们的发现，我们提出了两种使用卷积和Transformer模块的复合模型。所得的Hybrid-MS-S+模型以63M参数和12.3G FLOPS实现了83.9%的顶级精度。它已经与设计复杂的SOTA模型不相上下。代码和模型可在https://github.com/microsoft/SPACH.

## 1. Introduction
Convolutional neural networks (CNNs) have been dominating the computer vision (CV) field since the renaissance of deep neural networks (DNNs). They have demonstrated effectiveness in numerous vision tasks from image classifi- cation [12], object detection [27], to pixel-based segmentation [11]. Remarkably, despite the huge success of Transformer structure [37] in natural language processing (NLP) [8], the CV society still focuses on the CNN structure for quite some time.

自深度神经网络(DNN)复兴以来，卷积神经网络(CNN)一直主导着计算机视觉(CV)领域。他们已经证明了从图像分类[12]、目标检测[27]到基于像素分割[11]的许多视觉任务的有效性。值得注意的是，尽管Transformer结构[37]在自然语言处理(NLP)[8]方面取得了巨大的成功，但CV学会在相当长的一段时间内仍然关注CNN结构。

The transformer structure finally made its grand debut in CV last year. Vision Transformer (ViT) [9] showed that a pure Transformer applied directly to a sequence of image patches can perform very well on image classification tasks, Interns at MSRA. if the training dataset is sufficiently large. DeiT [35] further demonstrated that Transformer can be successfully trained on typical-scale dataset, such as ImageNet-1K [7], with appropriate data augmentation and model regularization.

Transformer结构终于在去年的CV中首次亮相。MSRA的实习生表示，视觉Transformer(ViT)[9]表明，在训练数据集足够大的情况下，直接应用于图像块序列的纯Transformer可以很好地执行图像分类任务。DeiT[35]进一步证明，通过适当的数据增广和模型正则化，Transformer可以在典型规模的数据集(如ImageNet-1K[7])上成功训练。

Interestingly, before the heat of Transformer dissipated, the structure of multi-layer perceptrons (MLPs) was revived by Tolstikhin et al. in a work called MLP-Mixer [33]. MLPMixer is based exclusively on MLPs applied across spatial locations and feature channels. When trained on large datasets, MLP-Mixer attains competitive scores on image classification benchmarks. The success of MLP-Mixer suggests that neither convolution nor attention are necessary for good performance. It sparked further research on MLP as the authors wished [20, 26].

有趣的是，在Transformer的热量消散之前，Tolstikhinet al 在一项名为MLP-Mixer的工作中恢复了多层感知器(MLP)的结构[33]。MLPMixer仅基于跨空间位置和特征通道应用的MLP。当在大型数据集上进行训练时，MLP-Mixer在图像分类基准上获得了有竞争力的分数。MLP-Mixer的成功表明，卷积和注意力对于良好的性能都是不必要的。正如作者所愿，这引发了对MLP的进一步研究[20，26]。

However, as the reported accuracy on image classification benchmarks continues to increase by new network designs from various camps, no conclusion can be made as which structure among CNN, Transformer, and MLP performs the best or is most suitable for vision tasks. This is partly due to the pursuit of high scores that leads to multifarious tricks and exhaustive parameter tuning. As a result, network structures cannot be fairly compared in a systematic way. The work presented in this paper fills this blank by conducting a series of controlled experiments over CNN, Transformer, and MLP in a unified framework.

然而，随着来自不同阵营的新网络设计，图像分类基准的报告准确性不断提高，无法得出CNN、Transformer和MLP中哪种结构表现最好或最适合视觉任务的结论。这在一定程度上是由于追求高分导致了各种各样的技巧和详尽的调参。因此，无法以系统的方式公平地比较网络结构。本文提出的工作通过在统一的框架中对CNN、Transformer和MLP进行一系列受控实验来填补这一空白。

We first develop a unified framework called SPACH as shown in Fig. 1. It is mostly adopted from current Transformer and MLP frameworks, since convolution can also fit into this framework and is in general robust to optimization. The SPACH framework contains a plug-and-play module called mixing block which could be implemented as convolution layers, Transformer layers, or MLP layers. Aside from the mixing block, other components in the framework are kept the same when we explore different structures. This is in stark contrast to previous work which compares different network structures in different frameworks that vary greatly in layer cascade, normalization, and other non-trivial implementation details. As a matter of fact, we found that these structure-free components play an important role in the final performance of the model, and this is arXiv:2108.13002v2 [cs.CV] 25 Nov 2021 commonly neglected in the literature.

我们首先开发了一个名为SPACH的统一框架，如图1所示。它主要是从当前的Transformer和MLP框架中采用的，因为卷积也可以适用于该框架，并且通常对优化具有稳健性。SPACH框架包含一个称为混合块的即插即用模块，可以实现为卷积层、Transformer层或MLP层。除了混合块，当我们探索不同的结构时，框架中的其他组件保持不变。这与之前的工作形成鲜明对比，之前的工作比较了不同框架中的不同网络结构，这些框架在层级联、归一化和其他非琐碎的实现细节方面差异很大。事实上，我们发现这些无结构组件在模型的最终性能中起着重要作用，这是文献中通常忽略的arXiv:2108.13002v2[cs.CV]2021 11月25日。

With this unified framework, we design a series of controlled experiments to compare the three network structures. The results show that all three network structures could perform well on the image classification task when pre-trained on ImageNet-1K. In addition, each individual structure has its distinctive properties leading different behaviors when the network size scales up. We also find several common design choices which contribute a lot to the performance of our SPACH framework. The detailed findings are listed in the following.
*  Multi-stage design is standard in CNN models, but its effectiveness is largely overlooked in Transformerbased or MLP-based models. We find that the multistage framework consistently and notably outperforms the single-stage framework no matter which of the three network structures is chosen.
*  Local modeling is efficient and crucial. With only light-weight depth-wise convolutions, the convolution model can achieve similar performance as a Transformer model in our SPACH framework. By adding a local modeling bypass in both MLP and Transformer structures, a significant performance boost is obtained with negligible parameters and FLOPs increase.
*  MLP can achieve strong performance under small model sizes, but it suffers severely from over-fitting when the model size scales up. We believe that over- fitting is the main obstacle that prevents MLP from achieving SOTA performance.
*  Convolution and Transformer are complementary in the sense that convolution structure has the best generalization capability while Transformer structure has the largest model capacity among the three structures. This suggests that convolution is still the best choice in designing lightweight models but designing large models should take Transformer into account.

利用这个统一的框架，我们设计了一系列受控实验来比较三种网络结构。结果表明，当在ImageNet-1K上进行预训练时，所有三种网络结构都可以很好地执行图像分类任务。此外，当网络规模扩大时，每个单独的结构都有其独特的特性，导致不同的行为。我们还发现了几种常见的设计选择，它们对我们的SPACH框架的性能有很大贡献。详细调查结果如下所示。
* 多阶段设计在CNN模型中是标准的，但在基于Transformer或MLP的模型中，其有效性在很大程度上被忽视。我们发现，无论选择三种网络结构中的哪一种，多阶段框架都一致且显著地优于单阶段框架。
* 局部建模是高效和关键的。仅使用轻量级的深度卷积，卷积模型可以实现与SPACH框架中的Transformer模型类似的性能。通过在MLP和Transformer结构中添加局部建模旁路，可以在可忽略的参数和FLOP增加的情况下获得显著的性能提升。
* MLP可以在小模型尺寸下获得很强的性能，但当模型尺寸扩大时，它会严重地受到过拟合的影响。我们认为，过拟合是阻碍MLP实现SOTA性能的主要障碍。
* 卷积和Transformer是互补的，因为卷积结构具有最佳的泛化能力，而Transformer结构在三种结构中具有最大的模型容量。这表明卷积仍然是设计轻量级模型的最佳选择，但设计大型模型时应考虑Transformer。

Based on these findings, we propose two hybrid models of different scales which are built upon convolution and Transformer layers. Experimental results show that, when a sweet point between generalization capability and model capacity is reached, the performance of these straightforward hybrid models is already on par with SOTA models with sophisticated architecture designs.

基于这些发现，我们提出了两种不同尺度的复合模型，它们建立在卷积层和Transformer层上。实验结果表明，当达到泛化能力和模型能力之间的平衡点时，这些简单的复合模型的性能已经与具有复杂架构设计的SOTA模型不相上下。

## 2. Background
CNN and its variants have dominated the vision domain. During the evolution of CNN models, useful experience about the architecture design has been accumulated. Recently, two types of architectures, namely Transformer [9] and MLP [33], begin to emerge in the vision domain and have shown performance similar to the well-optimized CNNs. These results kindle a spark towards building better vision models beyond CNNs.

CNN及其变体已经主导了视觉领域。在CNN模型的发展过程中，积累了关于架构设计的有用经验。最近，两种类型的架构，即Transformer[9]和MLP[33]，开始出现在视觉领域，并显示出与优化良好的CNN类似的性能。这些结果点燃了在CNN之外构建更好视觉模型的火花。

Convolution-based vision models. Since the entrance of deep learning era pioneered by AlexNet [18], the computer vision community has devoted enormous efforts to designing better vision backbones. In the past decade, most work focused on improving the design of CNN, and a series of networks, including VGG [29], ResNet [12], SENet [15], Xception [2], MoblieNet [14,28], and EfficientNet [31,32], are designed. They achieve significant accuracy improvements in various vision tasks.

基于卷积的视觉模型。自AlexNet[18]开创的深度学习时代进入以来，计算机视觉社区为设计更好的视觉骨干付出了巨大努力。在过去的十年中，大多数工作都集中在改进CNN的设计上，设计了一系列网络，包括VGG[29]、ResNet[12]、SENet[15]、Xception[2]、MoblieNet[14，28]和EfficientNet[31，32]。它们在各种视觉任务中实现了显著的准确性改进。

A standard convolution layer learns filters in a 3D space, with two spatial dimensions and one channel dimension. Thus, the learning of spatial correlations and channel correlations are coupled inside a single convolution kernel. Differently, A depth-wise convolution layer only learns spatial correlations by moving the learning process of channel correlations to an additional 1x1 convolution. The fundamental hypothesis behind this design is that cross-channel correlations and spatial correlations are sufficiently decoupled that it is preferable not to map them jointly [2]. Recent work [31, 32] shows that depth-wise convolution can achieve both high accuracy and good efficiency, confirming this hypothesis to some extent. In addition, the idea of decoupling spatial and channel correlations is adopted in the vision Transformer. Therefore, this paper employs the spatial-channel decoupling idea in our framework design.

标准卷积层学习3D空间中的滤波器，具有两个空间维度和一个通道维度。因此，空间相关性和信道相关性的学习在单个卷积核内耦合。不同的是，深度卷积层仅通过将信道相关性的学习过程移动到附加的1x1卷积来学习空间相关性。该设计背后的基本假设是，交叉信道相关性和空间相关性充分解耦，因此最好不要联合映射它们[2]。最近的工作[31，32]表明，深度卷积可以实现高精度和高效率，在一定程度上证实了这一假设。此外，视觉Transformer采用了空间和信道相关性解耦的思想。因此，本文在框架设计中采用了空间信道解耦思想。

Transformer-based vision models. With the success of Transformer in natural language processing (NLP) [8, 37], many researchers start to explore the use of Transformer as a stand-alone architecture for vision tasks. They are facing two main challenges. First, Transformer operates over a group of tokens, but no natural tokens, similar to the words in natural language, exist in an image. Second, images have a strong local structure while the Transformer structure treats all tokens equally and ignores locality. The pioneering work ViT [9] solved the first challenge by simply dividing an image into non-overlapping patches and treat each patch as a visual token. ViT also reveals that Transformer models trained on large-scale datasets could attain SOTA image recognition performance. However, when the training data is insufficient, ViT does not perform well due to the lack of inductive biases. DeiT [35] mitigates the problem by introducing a regularization and augmentation pipeline on ImageNet-1K.

基于Transformer的视觉模型。随着Transformer在自然语言处理(NLP)方面的成功[8，37]，许多研究人员开始探索将Transformer用作视觉任务的独立架构。他们面临着两个主要挑战。首先，Transformer对一组token进行操作，但图像中不存在与自然语言中的单词类似的自然token。其次，图像具有强大的局部结构，而Transformer结构平等对待所有token，并忽略局部性。开创性的工作ViT[9]通过简单地将图像划分为不重叠的块，并将每个块视为视觉token，解决了第一个挑战。ViT还表明，在大规模数据集上训练的Transformer模型可以达到SOTA图像识别性能。然而，当训练数据不足时，由于缺乏归纳偏差，ViT表现不佳。DeiT[35]通过在ImageNet-1K上引入正则化和增强管道来缓解该问题。

Swin [21] and Twins [3] propose local ViT to address the second challenge. They adopt locally-grouped selfattention by computing the standard self-attention within non-overlapping windows. The local mechanism not only leads to performance improvement thanks to the reintroduction of locality, but also bring sufficient improvement on memory and computational efficiency. Thus, the pyramid structure becomes feasible again for vision Transformer.

Swin[21]和Twins[3]建议局部ViT解决第二个挑战。他们通过计算非重叠窗口内的标准自注意来采用局部分组的自注意。由于局部性的重新引入，局部机制不仅可以提高性能，而且可以充分提高内存和计算效率。因此，金字塔结构对于视觉Transformer再次变得可行。

Figure 1. Illustration of the proposed experimental framework named SPACH. 
图1.名为SPACH的拟议实验框架的说明。

There has been a blowout development in the design of Transformer-based vision models. Since this paper is not intended to review the progress of vision Transformer, we only briefly introduce some highly correlated Transformer models. CPVT [4] and CvT [39] introduce convolution into Transformer blocks, bringing the desired translationinvariance properties into ViT architecture. CaiT [36] introduces a LayerScale approach to empower effective training of deeper ViT network. It is also discovered that some class-attention layers built on top of ViT network offer more effective processing than the class embedding. LV-ViT [17] proposes a bag of training techniques to build a strong baseline for vision Transformer. LeViT [10] proposes a hybrid neural network for fast image classification inference.

基于Transformer的视觉模型设计有了井喷式的发展。由于本文不打算回顾视觉Transformer的进展，我们仅简要介绍了一些高度相关的Transformer模型。CPVT[4]和CvT[39]将卷积引入到Transformer块中，将所需的平移不变性特性引入到ViT架构中。CaiT[36]引入了LayerScale方法，以增强对更深层ViT网络的有效训练。还发现，一些构建在ViT网络之上的类关注层提供了比类嵌入更有效的处理。LV ViT[17]提出了一套训练技术，为视觉Transformer建立强大的基线。LeViT[10]提出了一种用于快速图像分类推理的混合神经网络。

MLP-based vision models. Although MLP is not a new concept for the computer vision community, the recent progress on MLP-based visual models surprisingly demonstrates, both conceptually and technically, that simple architecture can achieve competitive performance with CNN or Transformer [33]. The pioneering work MLPMixer proposed a Mixer architecture using channel-mixing MLPs and channel-mixing MLPs to communicate between different channels and spatial locations (tokens), respectively. It achieves promising results when trained on a largescale dataset (i.e., JFT [30]). ResMLP [34] built a similar MLP-based model with a deeper architecture. ResMLP does not need large-scale datasets and it achieves comparable accuracy/complexity trade-offs on ImageNet-1K with Transformer-based models. FF [23] showed that simply replacing the attention layer in ViT with an MLP layer applied over the patch dimension could achieve moderate performance on ImageNet classification. gMLP [20] proposed a gating mechanism on MLP and suggested that self-attention is not a necessary ingredient for scaling up machine learning models.

基于MLP的视觉模型。尽管MLP对于计算机视觉社区来说不是一个新概念，但基于MLP的视觉模型的最新进展令人惊讶地从概念上和技术上证明，简单的架构可以实现与CNN或Transformer竞争的性能[33]。开创性的工作MLPMixer提出了一种Mixer架构，使用信道混合MLP和信道混合MLPs分别在不同信道和空间位置(令牌)之间进行通信。当在大规模数据集(即JFT[30])上训练时，它获得了有希望的结果。ResMLP[34]构建了一个类似的基于MLP的模型，具有更深的架构。ResMLP不需要大规模数据集，它在基于Transformer的模型的ImageNet-1K上实现了相当的准确性/复杂性权衡。FF[23]表明，简单地将ViT中的关注层替换为应用于块维度上的MLP层可以在ImageNet分类中实现中等性能。gMLP[20]提出了一种关于MLP的门控机制，并指出自注意不是扩展机器学习模型的必要成分。

## 3. A Unified Experimental Framework 统一的实验框架
In order to fairly compare the three network structures, we are in need of a unified framework that excludes other performance-affecting factors. Since recent MLPbased networks have already shared a similar framework as Transformer-based networks, we build the unified experimental framework based on them and try to include CNNbased network in this framework as well.

为了公平地比较这三种网络结构，我们需要一个统一的框架，该框架排除了其他影响性能的因素。由于最近的基于MLP的网络已经与基于Transformer的网络共享了类似的框架，我们基于它们构建了统一的实验框架，并尝试将基于CNN的网络也包括在该框架中。

### 3.1. Overview of the SPACH Framework
We build our experimental framework with reference to ViT [9] and MLP-Mixer [33]. Fig. 1(a) shows the single-stage version of the SPACH framework, which is used for our empirical study. The architecture is very simple and consists mainly of a cascade of mixing blocks, plus some necessary auxiliary modules, such as patch embedding, global average pooling, and a linear classifier. Fig.1(b) shows the details of the mixing block. Note that the spatial mixing and channel mixing are performed in consecutive steps. The name SPACH for our framework is coined to emphasize the serial structure of SPAtial and CHannel processing.

我们参考ViT[9]和MLP-Mixer[33]构建了我们的实验框架。图1(a)显示了SPACH框架的单阶段版本，用于我们的实证研究。该架构非常简单，主要由一系列混合块组成，加上一些必要的辅助模块，如块嵌入、全局平均池和线性分类器。图1(b)显示了混合块的细节。注意，空间混合和通道混合是在连续的步骤中执行的。我们框架的名称SPACH是为了强调空间和通道处理的串行结构而创造的。

Table 1. SPACH and SPACH-MS model variants. C: feature dimension, R: expansion ratio of MLP in Fc, N: number of mixing blocks of SPACH, Ns: number of mixing blocks in the ith stage of SPACH-MS. 
表1.SPACH和SPACH-MS模型变体。C： 特征尺寸，R：MLP在Fc中的膨胀比，N：SPACH的混合块数，Ns：SPACH-MS第i阶段的混合块数目。

We also enable a multi-stage variation, referred to as SPACH-MS, as shown in Fig. 1(c). Multi-stage is an important mechanism in CNN-based networks to improve the performance. Unlike the single-stage SPACH, which processes the image in a low resolution by down-sampling the image by a large factor at the input, SPACH-MS is designed to keep a high-resolution in the initial stage of the framework and progressively perform down-sampling. Specifically, our SPACH-MS contains four stages with down-sample ratios of 4, 8, 16, and 32, respectively. Each stage contains Ns mixing blocks, where s is the stage index. Due to the extremely high computational cost of Transformer and MLP on high-resolution feature maps, we implement the mixing blocks in the first stage with convolutions only. The feature dimension within a stage remains constant, and will be multiplied with a factor of 2 after down-sampling. Let I ∈ R3×h×w denotes an input image, where 3 is the RGB channels and H × W is the spatial dimensions. Our SPACH framework first passes the input image through a Patch Embedding layer, which is the same as the one in ViT, to convert I into patch embeddings Xp ∈ RC× hp × wp . Here p denotes patch size, which is 16 in the single-stage implementation and 4 in the multi-stage implementation. After the cascades of mixing blocks, a classification head implemented by a linear layer is used for the supervised pretraining.

我们还启用了多阶段变化，称为SPACH-MS，如图所示。1(c)。多阶段是基于CNN的网络中提高性能的重要机制。与单阶段SPACH不同的是，SPACH-MS通过在输入端对图像进行大因子下采样来以低分辨率处理图像，SPACH-MS设计用于在框架的初始阶段保持高分辨率，并逐步执行下采样。具体而言，我们的SPACH-MS包含四个阶段，下采样率分别为4、8、16和32。每个阶段包含Ns个混合块，其中s是阶段索引。由于Transformer和MLP在高分辨率特征图上的计算成本极高，我们在第一阶段仅使用卷积来实现混合块。阶段内的特征尺寸保持不变，在下采样后将乘以2倍。让我∈ R3×h×w表示输入图像，其中3是RGB通道，h×w是空间维度。我们的SPACH框架首先将输入图像通过一个块嵌入层(Patch Embedding layer)，该层与ViT中的相同，以将I转换为块嵌入Xp∈ RC×hp×wp。这里p表示块大小，在单阶段实现中为16，在多阶段实现中为4。在混合块级联之后，由线性层实现的分类头用于监督预训练。

We list the hyper-parameters used in different model configurations in Table 1. Three model size for each variations of SPACH are designed, namely SPACH-XXS, SPACH-XS and SPACH-S, by controlling the number of blocks, the number of channels, and the expansion ratio of channel mixing MLP Fc. The model size, theoretical computational complexity (FLOPS), and empirical throughput are presented in Section 4. We measure the throughput using one P100 GPU.

我们在表1中列出了不同模型配置中使用的超参数。通过控制块数、信道数和信道混合MLP Fc的扩展比，为SPACH的每个变体设计了三种模型大小，即SPACH-XS、SPACH-SS和SPACH-S。第4节介绍了模型大小、理论计算复杂度(FLOPS)和经验吞吐量。我们使用一个P100 GPU测量吞吐量。

Figure 2. Three implementations of the spatial mixing module using convolution, Transformer, and MLP, respectively. P.E. denotes positional encoding, implemented by convolution in SPACH. 
图2.分别使用卷积、Transformer和MLP的空间混合模块的三种实现。P.E.表示位置编码，通过SPACH中的卷积实现。

### 3.2. Mixing Block Design
Mixing blocks are key components in the SPACH framework. As shown in Fig. 1(b), for an input feature X ∈ RC×H×W , where C and H ×W denote channel and spatial dimensions, it is first processed by a spatial mixing function Fs and then by a channel mixing function Fc. Fs focuses on aggregating context information from different spatial locations while Fc focuses on channel information fusion. Denoting the output as Y , we can formulate a mixing block as:

混合块是SPACH框架中的关键组件。如图1(b)所示，对于输入特征X∈ RC×H×W，其中C和H×W表示通道和空间维度，首先由空间混合函数Fs处理，然后由通道混合函数Fc处理。Fs专注于聚合来自不同空间位置的上下文信息，而Fc专注于信道信息融合。将输出表示为Y，我们可以将混合块表示为：

Y = Fs(Fc(X)). (1)

Following ViT [9], we use an MLP with appropriate normalization and residual connection to implement Fc. The MLP here can be also viewed as a 1x1 convolution (also known as point-wise convolution [2]) which is a special case of regular convolution. Note that Fc only performs channel fusion and does not explore any spatial context.

在ViT[9]之后，我们使用具有适当归一化和残差连接的MLP来实现Fc。这里的MLP也可以看作是1x1卷积(也称为逐点卷积[2])，这是正则卷积的特殊情况。请注意，Fc仅执行信道融合，而不探索任何空间上下文。

Table 2. Three desired properties in network design are seen in different network structures. 
表2.在不同的网络结构中可以看到网络设计中的三个期望特性。

The spatial mixing function Fs is the key to implement different architectures. As shown in Fig. 2, we implement three structures using convolution, self-attention, and MLP. The common components include normalization and residual connection. Specifically, the convolution structure is implemented by a 3x3 depth-wise convolution, as channel mixing will be handled separately in subsequent steps. For the Transformer structure, there is a positional embedding module in the original design. But recent research suggests that absolute positional embedding breaks translation variance, which is not suitable for images. In view of this and inspired by recent vision transformer design [4,39], we introduce a convolutional positional encoding (CPE) as a bypass in each spatial mixing module. The CPE module has negligible parameters and FLOPs. For MLP-based network, the pioneering work MLP-Mixer does not use any positional embedding, but we empirically find that adding the very lightweight CPE significantly improves the model performance, so we use the same treatment for MLP as for Transformer.

空间混合函数Fs是实现不同架构的关键。如图2所示，我们使用卷积、自关注和MLP实现了三种结构。常见的组件包括归一化和残余连接。具体而言，卷积结构由3x3深度卷积实现，因为信道混合将在后续步骤中单独处理。对于Transformer结构，原始设计中有一个位置嵌入模块。但最近的研究表明，绝对位置嵌入打破了平移方差，这不适用于图像。鉴于此，并受到最近视觉Transformer设计的启发[4，39]，我们在每个空间混合模块中引入卷积位置编码(CPE)作为旁路。CPE模块具有可忽略的参数和FLOP。对于基于MLP的网络，先驱工作MLP-Mixer没有使用任何位置嵌入，但我们根据经验发现，添加非常轻量级的CPE显著提高了模型性能，因此我们对MLP使用与Transformer相同的处理方法。

The three implementations of Fs have distinctive properties as listed in Table 2. First, the convolution structure only involves local connections so that it is computational efficient. Second, the self-attention structure uses dynamic weight for each input instance so that model capacity is increased. Moreover, it has a global receptive field, which enables information to flow freely across different positions [37]. Third, MLP structure has a global receptive field just as the self-attention structure, but it does not use dynamic weight. In summary, these three properties seen in different architectures are all desirable and may have positive influence on the model performance or efficiency. We can find convolution and self-attention have complementary properties thus there is potential to build hybrid model to combine all desirable properties. Besides, MLP structure seems to be inferior to self-attention in this analysis. 

如表2所示，Fs的三种实现具有不同的特性。首先，卷积结构只涉及局部连接，因此计算效率很高。其次，自我注意结构对每个输入实例使用动态权重，从而增加模型容量。此外，它有一个全局感受野，使信息能够在不同位置自由流动[37]。第三，MLP结构与自我注意结构一样具有全局感受野，但它不使用动态权重。总之，在不同架构中看到的这三个特性都是可取的，并可能对模型性能或效率产生积极影响。我们可以发现卷积和自我注意具有互补的特性，因此有可能建立复合模型来组合所有期望的特性。此外，在这一分析中，MLP结构似乎不如自我注意。

Figure 3. The multi-stage models (named with -MS suffix) always achieve a better performance than their single-stage counterparts.
图3.多阶段模型(以-MS后缀命名)总是比单阶段模型取得更好的性能。

## 4. Empirical Studies on Mixing Blocks 混合块的实证研究
In this section, we design a series of controlled experiments to compare the three network structures. We first introduce the experimental settings in Section 4.1, and then present our main findings in Section 4.2, 4.3, 4.4, and 4.5.

在本节中，我们设计了一系列受控实验来比较三种网络结构。我们首先在第4.1节介绍实验设置，然后在第4.2、4.3、4.4和4.5节介绍我们的主要发现。

### 4.1. Datasets and Training Pipelines
We conduct experiments on ImageNet-1K (IN-1K) [7] image classification which has 1k classes. The training set has 1.28M images while the validation set has 50k images. The Top-1 accuracy on a single crop is reported. Unless otherwise indicated, we use the input resolution of 224x224. Most of our training settings are inherited from DeiT [35]. We employ an AdamW [22] optimizer for 300 epochs with a cosine decay learning rate scheduler and 20 epochs of linear warm-up. The weight decay is 0.05, and the initial learning rate is 0.005 × batchsize 512 . 8 GPUs with minibatch 128 per GPU are used in training, resulting a total batch-size of 1024. We use exactly the same data augmentation and regularization configurations as DeiT, including Rand-Augment [5], random erasing [42], Mixup [41], CutMix [40], stochastic depth [16], and repeated augmentation [1, 13]. We use the same training pipeline for all comparing models. And the implementation is built upon PyTorch [24] and timm library [38].

我们对具有1K类的ImageNet-1K(IN-1K)[7]图像分类进行了实验。训练集有128M张图像，而验证集有50k张图像。报告了单次剪裁的Top-1精度。除非另有说明，我们使用224x224的输入分辨率。我们的大多数训练设置都继承自DeiT[35]。我们使用AdamW[22]优化器，用于300个周期，具有余弦衰减学习速率调度器和20个周期的线性预热。权重衰减为0.05，初始学习率为0.005×批大小512。训练中使用了8个GPU，每个GPU有128个小批，总批大小为1024。我们使用了与DeiT完全相同的数据增广和正则化配置，包括随机增广[5]、随机擦除[42]、混合[41]、CutMix[40]、随机深度[16]和重复增广[1，13]。我们对所有比较模型使用相同的训练管道。该实现基于PyTorch[24]和timm库[38]。

### 4.2. Multi-Stage is Superior to Single-Stage
Multi-stage design is standard in CNN models, but it is largely overlooked in Transformer-based or MLP-based models. Our first finding is that multi-stage design should always be adopted in vision models no matter which of the three network structures is chosen.

多阶段设计在CNN模型中是标准的，但在基于Transformer或基于MLP的模型中，它在很大程度上被忽略了。我们的第一个发现是，无论选择三种网络结构中的哪一种，视觉模型都应始终采用多阶段设计。

Table 3 compares the image classification performance between multi-stage framework and single-stage framework. For all three network scales and all three network structures, multi-stage framework consistently achieves better complexity-accuracy trade-off. For the sake of easy comparison, the changes of FLOPs and accuracy are highlighted in Table 3. Most of the multi-stage models are designed to have slightly fewer computational costs, but they manage to achieve a higher accuracy than the corresponding single-stage models. An accuracy loss of 2.6 points is observed for the Transformer model at the XXS scale, but it is understandable as the multi-stage model happens to have only half of the parameters and FLOPs of the corresponding single-stage model.

表3比较了多阶段框架和单阶段框架之间的图像分类性能。对于所有三个网络规模和所有三种网络结构，多阶段框架一致地实现了更好的复杂性和准确性权衡。为了便于比较，表3中突出显示了FLOP和精度的变化。大多数多阶段模型的计算成本略低，但它们能够实现比相应的单阶段模型更高的精度。在XXS尺度下，Transformer模型的精度损失为2.6点，但这是可以理解的，因为多阶段模型恰好只有相应单阶段模型的一半参数和FLOP。

Table 3. Model performance of SPACH and SPACH-MS at three network scales.
表3.SPACH和SPACH-MS在三种网络尺度下的模型性能。

Table 4. Both Transformer structure and MLP structure benefit from local modeling at a very small computational cost. The superscription - indicates without local modeling. 
表4.Transformer结构和MLP结构都以非常小的计算成本从局部建模中受益。subscription-表示没有本地建模。

In addition, Fig. 3 shows how the image classification accuracy changes with the size of model parameters and model throughput. Despite the different trends observed for different network structures, the multi-stage models always outperform their single-stage counterparts.

此外，图3显示了图像分类精度如何随着模型参数和模型吞吐量的大小而变化。尽管对于不同的网络结构观察到不同的趋势，但多阶段模型总是优于单阶段模型。

This finding is consistent with the results reported in recent work. Both Swin-Transformer [21] and TWins [3] adopt multi-stage framework and achieve a stronger performance than the single-stage framework DeiT [35]. Our empirical study suggests that the use of multi-stage framework can be an important reason.

这一发现与最近工作报告的结果一致。Swin Transformer[21]和TWins[3]均采用多阶段框架，并实现了比单阶段框架DeiT[35]更强的性能。我们的实证研究表明，使用多阶段框架可能是一个重要原因。

### 4.3. Local Modeling is Crucial 局部建模至关重要
Although it has been pointed out in many previous work [4,6,19,21,39] that local modeling is crucial for vision models, we will show in this subsection how amazingly efficient local modeling could be.

尽管在许多先前的工作[4,6,19,21,39]中已经指出，局部建模对于视觉模型至关重要，但我们将在本小节中展示局部建模的惊人效率。

In our empirical study, the spatial mixing block of the convolution structure is implemented by a 3 × 3 depth-wise convolution, which is a typical local modeling operation. It is so light-weight that it only contributes to 0.3% of the model parameter and 0.5% of the FLOPs. However, as Table 3 and Fig. 3 show, this structure can achieve competitive performance when compared with the Transformer structure in the XXS and XS configurations.

在我们的实证研究中，卷积结构的空间混合块由3×3深度卷积实现，这是一种典型的局部建模操作。它非常轻，只占模型参数的0.3%和FLOP的0.5%。然而，如表3和图3所示，与XXS和XS配置中的Transformer结构相比，该结构可以实现具有竞争力的性能。

It is due to the sheer efficiency of 3×3 depth-wise convolution that we propose to use it as a bypass in both MLP and Transformer structures. The increase of model parameters and inference FLOPs is almost negligible, but the locality of the models is greatly strengthened. In order to demonstrate how local modeling helps the performance of Transformer and MLP structures, we carry out an ablation study which removes this convolution bypass in the two structures.

由于3×3深度卷积的绝对效率，我们建议将其用作MLP和Transformer结构中的旁路。模型参数和推理FLOP的增加几乎可以忽略不计，但模型的局部性大大增强。为了证明局部建模如何帮助Transformer和MLP结构的性能，我们进行了消融研究，消除了这两种结构中的卷积旁路。

Table 4 shows the performance comparison between models with or without local modeling. The two models we pick are the top performers in Table 3 when multi-stage framework is used and network scale is S. We can clearly find that the convolution bypass only slightly decreases the throughput, but brings a notable accuracy increase to both models. Note that the convolution bypass is treated as convolutional positional embedding in Trans-MS-S, so we bring back the standard patch embedding as in ViT [9] in Trans-MS-S−. For MLP-MS-S−, we follow the practice in MLP-Mixer and do not use any positional embedding. This experiment confirms the importance of local modeling and suggests the use of 3×3 depth-wise convolution as a bypass for any designed network structures.

表4显示了具有或不具有局部建模的模型之间的性能比较。当使用多阶段框架且网络规模为S时，我们选择的两个模型在表3中表现最佳。我们可以清楚地发现，卷积旁路仅略微降低了吞吐量，但给两个模型带来了显著的精度提高。请注意，在Trans-MS-S中，卷积旁路被视为卷积位置嵌入，因此我们在Trans-MS中恢复了ViT[9]中的标准块嵌入. 对于MLP-MS-S−, 我们遵循MLP-Mixer中的实践，不使用任何位置嵌入。该实验证实了局部建模的重要性，并建议使用3×3深度卷积作为任何设计网络结构的旁路。

Table 5. The performance of MLP models are greatly boosted when weight sharing is adopted to alleviate over-fitting.
表5.当采用权重分担来缓解过拟合时，MLP模型的性能大大提高。

### 4.4. A Detailed Analysis of MLP
Due to the excessive number of parameters, MLP models suffer severely from over-fitting. We believe that over- fitting is the main obstacle for MLP to achieve SOTA performance. In this part, we discuss two mechanisms which can potentially alleviate this problem.

由于参数数量过多，MLP模型严重存在过拟合问题。我们认为，过拟合是MLP实现SOTA性能的主要障碍。在这一部分中，我们讨论了两种可能缓解这一问题的机制。

One is the use of multi-stage framework. We have already shown in Table 3 that multi-stage framework brings gain. Such gain is even more prominent for larger MLP models. In particular, the MLP-MS-S models achieves 2.6 accuracy gain over the single-stage model MLP-S. We believe this owes to the strong generalization capability of the multi-stage framework. Fig. 4 shows how the test accuracy increases with the decrease of training loss. Over- fitting can be observed when the test accuracy starts to flatten. These results also lead to a very promising baseline for MLP-based models. Without bells and whistles, MLP-MSS model achieves 82.1% ImageNet Top-1 accuracy, which is 5.7 points higher than the best results reported by MLPMixer [33] when ImageNet-1K is used as training data.

一个是使用多阶段框架。我们已经在表3中表明，多阶段框架带来了收益。对于更大的MLP模型，这种增益甚至更为显著。特别地，MLP-MS-S模型比单阶段模型MLP-S实现2.6精度增益。我们认为，这归功于多阶段框架的强大泛化能力。图4显示了测试精度如何随着训练损失的减少而增加。当测试精度开始变平时，可以观察到过拟合。这些结果也为基于MLP的模型提供了一个非常有前景的基线。无需鸣笛，MLP-MSS模型实现了82.1%的ImageNet Top-1精度，这比MLPMixer[33]报告的最佳结果高5.7点(当ImageNet-1K用作训练数据时)。

The other mechanism is parameter reduction through weight sharing. We apply weight-sharing on the spatial mixing function Fs. For the single-stage model, all N mixing blocks use the same Fs, while for the multi-stage model, each stage use the same same Fs for its Ns mixing blocks. We present the results of S models in Table 5. We can find that the shared-weight variants, denoted by ”+Shared”, achieve higher accuracy with almost the same model size and computation cost. Although they are still inferior to Transformer models, the performance is on par with or even better than convolution models. Fig. 4 confirms that using shared weights in the MLP-MS model further delays the appearance of over-fitting signs. Therefore, we conclude that MLP-based models remain competitive if they could solve or alleviate the over-fitting problem.

另一种机制是通过权重共享来减少参数。我们对空间混合函数Fs应用权重共享。对于单阶段模型，所有N个混合块使用相同的Fs，而对于多阶段模型，每个级对其Ns个混合块都使用相同的F。我们在表5中展示了S模型的结果。我们可以发现，用“+共享”表示的共享权重变量在几乎相同的模型大小和计算成本下实现了更高的精度。尽管它们仍然不如Transformer模型，但性能与卷积模型不相上下，甚至优于卷积模型。图4证实了在MLP-MS模型中使用共享权重进一步延迟了过拟合迹象的出现。因此，我们得出结论，如果基于MLP的模型能够解决或缓解过拟合问题，那么它们仍然具有竞争力。

### 4.5. Convolution and Transformer are Complementary 卷积和Transformer是互补的
We find that convolution and Transformer are complementary in the sense that convolution structure has the best generalization capability while Transformer structure has the largest model capacity among the three structures we investigated.

我们发现卷积和Transformer是互补的，因为卷积结构具有最佳的泛化能力，而Transformer结构在我们研究的三种结构中具有最大的模型容量。

Figure 4. Illustration of the over-fitting problem in MLP-based models. Both multi-stage framework and weight sharing alleviate the problem.
图4.基于MLP的模型中的过拟合问题说明。多阶段框架和权重共享都缓解了这个问题。

Figure 5. Conv-MS has a better generalization capability than Trans-MS as it achieves a higher test accuracy at the same training loss before the model saturates. 
图5.Conv MS比Trans MS具有更好的泛化能力，因为在模型饱和之前，它在相同的训练损失下实现了更高的测试精度。

Fig. 5 shows that, before the performance of Conv-MS saturates, it achieves a higher test accuracy than Trans-MS at the same training loss. This shows that convolution models generalize better than Transformer models. In particular, when the training loss is relatively large, the convolution models show great superiority against Transformer models. This suggests that convolution is still the best choice in designing lightweight vision models.

图5显示，在Conv MS的性能饱和之前，它在相同的训练损耗下实现了比Trans MS更高的测试精度。这表明，卷积模型比Transformer模型更易于推广。特别是，当训练损失相对较大时，卷积模型相对于Transformer模型显示出极大的优越性。这表明卷积仍然是设计轻量级视觉模型的最佳选择。

On the other hand, both Fig. 3 and Fig. 5 show that Transformer models achieve higher accuracy than the other two structures when we increase the model size and allow for higher computational cost. Recall that we have discussed three properties of network architectures in Section 3.2. It is now clear that the sparse connectivity helps to increase generalization capability, while dynamic weight and global receptive field help to increase model capacity.

另一方面，图3和图5都表明，当我们增加模型尺寸并允许更高的计算成本时，Transformer模型比其他两种结构实现更高的精度。回想一下，我们已经在第3.2节中讨论了网络架构的三个特性。现在很明显，稀疏连接有助于提高泛化能力，而动态权重和全局感受野有助于增加模型容量。

## 5. Hybrid Models 复合模型
As discussed in Section 3.2 and 4.4, convolution and Transformer structures have complementary characteristics and have potential to be used in a single model. Based on this observation, we construct hybrid models at the XS and S scales based on these two structures. The procedure we used to construct hybrid models is rather simple. We take a multi-stage convolution-based model as the base model, and replace some selected layers with Transformer layers. Considering the local modeling capability of convolutions and global modeling capability of Transformers, we tend to do such replacement in later stages of the model. The details of layer selection in the two hybrid models are listed as follows.
*  Hybrid-MS-XS: It is based on Conv-MS-XS. The last ten layers in Stage 3 and the last two layers in Stage 4 are replaced by Transformer layers. Stage 1 and 2 remain unchanged.
*  Hybrid-MS-S: It is based on Conv-MS-S. The last two layers in Stage 2, the last ten layers in Stage 3, and the last two layers in Stage 4 are replaced by Transformer layers. Stage 1 remains unchanged.

如第3.2节和第4.4节所述，卷积和Transformer结构具有互补的特性，有可能用于单个模型。基于这一观察，我们基于这两种结构构建了XS和S尺度的复合模型。我们用来构建复合模型的过程相当简单。我们采用基于多阶段卷积的模型作为基础模型，并用Transformer层替换某些选定层。考虑到卷积的局部建模能力和Transformer的全局建模能力，我们倾向于在模型的后期进行这种替换。下面列出了两种复合模型中图层选择的详情。
* Hybrid-MS-XS：基于Conv-MS-XS。阶段3中的最后十层和阶段4中的最后两层被Transformer层取代。阶段1和阶段2保持不变。
* Hybrid-MS-S：基于Conv-MS-S。阶段2中的最后两层、阶段3中的最后十层和阶段4中的最后两个层被Transformer层替换。阶段1保持不变。

In order to unleash the full potential of hybrid models, we further adopt the deep patch embedding layer (PEL) implementation as suggested in LV-ViT [17]. Different from default PEL which uses one large (16x16) convolution kernel, the deep PEL uses four convolution kernels with kernel size {7, 3, 3, 2}, stride {2, 1, 1, 2}, and channel number {64, 64, 64, C}. By using small kernel sizes and more convolution kernels, deep PEL helps a vision model to explore the locality inside single patch embedding vector. We mark models with deep PEL as ”Hybrid-MS-*+”.

为了释放复合模型的全部潜力，我们进一步采用LV ViT[17]中建议的深度块嵌入层(PEL,patch embedding layer)实现。与使用一个大(16x16)卷积内核的默认PEL不同，深度PEL使用四个卷积内核，内核大小为｛7，3，3，2｝，步幅为｛2，1，1，2｝，通道数为｛64，64，C｝。通过使用较小的核大小和更多的卷积核，深度PEL有助于视觉模型探索单个块嵌入向量内的局部性。我们将具有深度PEL的模型token为“混合MS-*+”。

Table 6 shows comparison between our hybrid models and some of the state-of-the-art models based on CNN, Transformer, or MLP. All listed models are trained on ImageNet-1K. Within the section of our models, we can find that hybrid models achieve better model size-performance trade-off than pure convolution models or Transformer models. The Hybrid-MS-XS achieves 82.4% top-1 accuracy with 28M parameters, which is higher than Conv-MSS with 44M parameters and only a little lower than TransMS-S with 40M parameters. In addition, the Hybrid-MS-S achieve 83.7% top-1 accuracy with 63M parameters, which has 0.8 point gain compared with Trans-MS-S.

表6显示了我们的复合模型与基于CNN、Transformer或MLP的一些最先进模型之间的比较。所有列出的模型都是在ImageNet-1K上训练的。在我们的模型部分中，我们可以发现复合模型比纯卷积模型或Transformer模型实现了更好的模型大小性能权衡。混合MS XS在28M参数下达到82.4%的顶级精度，高于Conv MSS(44M参数)，仅略低于TransMSS(40M参数)。此外，Hybrid-MS-S以63M参数达到83.7%的顶级精度，与Trans-MS-S相比，其增益为0.8点。

The Hybrid-MS-S+ model we proposed achieves 83.9% top-1 accuracy with 63M parameters. This number is higher than the accuracy achieved by SOTA models Swin-B and CaiT-S36, which have model size of 88M and 68.2M, respectively. The FLOPs of our model is also fewer than these two models. We believe Hybrid-MS-S can be serve as a strong yet simple baseline for future research on architecture design of vision models.

我们提出的Hybrid-MS-S+模型在63M个参数下达到了83.9%的顶级精度。这个数字高于SOTA模型Swin-B和CaiT-S36所达到的精度，它们的模型尺寸分别为88M和68.2M。我们模型的FLOP也比这两种模型少。我们相信，Hybrid-MS-S可以作为未来视觉模型架构设计研究的强大而简单的基线。

## 6. Conclusion
The objective of this work is to understand how the emerging Transformer and MLP structures compare with CNNs in the computer vision domain. We first built a simple and unified framework, called SPACH, that could use CNN, Transformer, or MLP as plug-and-play components. Under the SPACH framework, we discover with a little surprise that all three network structures are similarly competitive in terms of the accuracy-complexity tradeoff, although they show distinctive properties when the network scales up. In addition to the analysis of specific network structures, we also investigate two important design choices, namely multi-stage framework and local modeling, which are largely overlooked in previous work. Finally, inspired by the analysis, we propose two hybrid models which achieve SOTA performance on ImageNet-1k classification without bells and whistles.

这项工作的目的是了解新兴的Transformer和MLP结构如何与计算机视觉领域中的CNN进行比较。我们首先构建了一个简单统一的框架，称为SPACH，它可以使用CNN、Transformer或MLP作为即插即用组件。在SPACH框架下，我们有点惊讶地发现，所有三种网络结构在精度-复杂性权衡方面都具有相似的竞争力，尽管它们在网络扩展时表现出独特的特性。除了对特定网络结构的分析，我们还研究了两个重要的设计选择，即多阶段框架和局部建模，这在以前的工作中大多被忽略。最后，在分析的启发下，我们提出了两种复合模型，它们在ImageNet-1k分类中实现了SOTA性能，而不需要鸣笛。

Table 6. Comparison of different models on ImageNet-1K classification. Compared models are grouped according to network structures, and our models are listed in the last, Most models are pre-trained with 224x224 images, except ViT-B/16*, which uses 384x384 images. 

表6.ImageNet-1K分类的不同模型比较。根据网络结构对比较的模型进行分组，我们的模型在最后列出。除ViT-B/16*使用384x384图像外，大多数模型都使用224x224图像进行预训练。

Our work also raises several questions worth exploring. First, realizing the fact that the performance of MLP-based models is largely affected by over-fitting, is it possible to design a high-performing MLP model that is not subject to over-fitting? Second, current analyses suggest that neither convolution nor Transformer is the optimal structure across all model sizes. What is the best way to fuse these two structures? Last but not least, do better visual models exist beyond the known structures including CNN, Transformer, and MLP?

我们的工作也提出了几个值得探讨的问题。首先，认识到基于MLP的模型的性能在很大程度上受到过拟合的影响，是否有可能设计出不受过拟合影响的高性能MLP模型？其次，当前的分析表明，卷积和Transformer都不是所有模型尺寸的最佳结构。融合这两种结构的最佳方法是什么？最后但并非最不重要的是，除了已知的结构，包括CNN、Transformer和MLP，是否存在更好的视觉模型？

## References
1.  Maxim Berman, Herv´e J´egou, Andrea Vedaldi, Iasonas Kokkinos, and Matthijs Douze. Multigrain: a uni- fied image embedding for classes and instances. CoRR, abs/1902.05509, 2019. 5
2.  Franc¸ois Chollet. Xception: Deep learning with depthwise separable convolutions. In CVPR, pages 1800–1807. IEEE Computer Society, 2017. 2, 5
3.  Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, and Chunhua Shen. Twins: Revisiting spatial attention design in vision transformers. CoRR, abs/2104.13840, 2021. 2, 6
4.  Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, Xiaolin Wei, Huaxia Xia, and Chunhua Shen. Conditional positional encodings for vision transformers. arXiv preprint arXiv:2102.10882, 2021. 3, 5, 6
5.  Ekin Dogus Cubuk, Barret Zoph, Jon Shlens, and Quoc Le. Randaugment: Practical automated data augmentation with a reduced search space. In NeurIPS, 2020. 5
6.  St´ephane d’Ascoli, Hugo Touvron, Matthew L. Leavitt, Ari S. Morcos, Giulio Biroli, and Levent Sagun. Convit: Improving vision transformers with soft convolutional inductive biases. In ICML, volume 139 of Proceedings of Machine Learning Research, pages 2286–2296. PMLR, 2021. 6
7.  Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255. IEEE Computer Society, 2009. 1, 5
8.  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT (1), pages 4171–4186. Association for Computational Linguistics, 2019. 1, 2
9.  Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR. OpenReview.net, 2021. 1, 2, 3, 4, 6, 8
10.  Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Herv´e J´egou, and Matthijs Douze. Levit: a vision transformer in convnet’s clothing for faster inference. CoRR, abs/2104.01136, 2021. 3
11.  Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross B. Girshick. Mask R-CNN. In ICCV, pages 2980–2988. IEEE Computer Society, 2017. 1
12.  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778. IEEE Computer Society, 2016. 1, 2
13.  Elad Hoffer, Tal Ben-Nun, Itay Hubara, Niv Giladi, Torsten Hoefler, and Daniel Soudry. Augment your batch: Improving generalization through instance repetition. In CVPR, pages 8126–8135. IEEE, 2020. 5
14.  Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. CoRR, abs/1704.04861, 2017. 2
15.  Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In CVPR, pages 7132–7141. IEEE Computer Society, 2018. 2
16.  Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q. Weinberger. Deep networks with stochastic depth. In ECCV (4), volume 9908 of Lecture Notes in Computer Science, pages 646–661. Springer, 2016. 5
17.  Zihang Jiang, Qibin Hou, Li Yuan, Daquan Zhou, Yujun Shi, Xiaojie Jin, Anran Wang, and Jiashi Feng. All tokens matter: Token labeling for training better vision transformers. arXiv preprint arXiv:2104.10858, 2021. 3, 8
18.  Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, pages 1106–1114, 2012. 2
19.  Yawei Li, Kai Zhang, Jiezhang Cao, Radu Timofte, and Luc Van Gool. Localvit: Bringing locality to vision transformers. CoRR, abs/2104.05707, 2021. 6
20.  Hanxiao Liu, Zihang Dai, David R. So, and Quoc V. Le. Pay attention to mlps. CoRR, abs/2105.08050, 2021. 1, 3, 8
21.  Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. CoRR, abs/2103.14030, 2021. 2, 6, 8
22.  Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. CoRR, abs/1711.05101, 2017. 5
23.  Luke Melas-Kyriazi. Do you even need attention? A stack of feed-forward layers does surprisingly well on imagenet. CoRR, abs/2105.02723, 2021. 3, 8
24.  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas K¨opf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, pages 8024–8035, 2019. 5
25.  Ilija Radosavovic, Raj Prateek Kosaraju, Ross B. Girshick, Kaiming He, and Piotr Doll´ar. Designing network design spaces. In CVPR, pages 10425–10433. IEEE, 2020. 8
26.  Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. Global filter networks for image classification. CoRR, abs/2107.00645, 2021. 1
27.  Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. IEEE Trans. Pattern Anal. Mach. Intell., 39(6):1137–1149, 2017. 1
28.  Mark Sandler, Andrew G. Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In CVPR, pages 4510–4520. IEEE Computer Society, 2018. 2
29.  Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015. 2
30.  Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In ICCV, pages 843–852. IEEE Computer Society, 2017. 3
31.  Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In ICML, volume 97 of Proceedings of Machine Learning Research, pages 6105–6114. PMLR, 2019. 2
32.  Mingxing Tan and Quoc V. Le. Efficientnetv2: Smaller models and faster training. In ICML, volume 139 of Proceedings of Machine Learning Research, pages 10096–10106. PMLR, 2021. 2
33.  Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. Mlp-mixer: An all-mlp architecture for vision. CoRR, abs/2105.01601, 2021. 1, 2, 3, 7, 8
34.  Hugo Touvron, Piotr Bojanowski, Mathilde Caron, Matthieu Cord, Alaaeldin El-Nouby, Edouard Grave, Armand Joulin, Gabriel Synnaeve, Jakob Verbeek, and Herv´e J´egou. Resmlp: Feedforward networks for image classification with data-efficient training. CoRR, abs/2105.03404, 2021. 3, 8
35.  Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training data-efficient image transformers & distillation through attention. In ICML, volume 139 of Proceedings of Machine Learning Research, pages 10347–10357. PMLR, 2021. 1, 2, 5, 6, 8
36.  Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Herv´e J´egou. Going deeper with image transformers. CoRR, abs/2103.17239, 2021. 3, 8
37.  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pages 5998– 6008, 2017. 1, 2, 5
38.  Ross Wightman. Pytorch image models. https : / / github . com / rwightman / pytorch - image - models, 2019. 5
39.  Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions to vision transformers. CoRR, abs/2103.15808, 2021. 3, 5, 6, 8
40.  Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, Seong Joon Oh, Youngjoon Yoo, and Junsuk Choe. Cutmix: Regularization strategy to train strong classifiers with localizable features. In ICCV, pages 6022–6031. IEEE, 2019. 5
41.  Hongyi Zhang, Moustapha Ciss´e, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In ICLR (Poster). OpenReview.net, 2018. 5
42.  Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In AAAI, pages 13001–13008. AAAI Press, 2020. 5