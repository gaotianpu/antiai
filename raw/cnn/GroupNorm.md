# Group Normalization
2018.3.22 https://arxiv.org/abs/1803.08494

## Abstract
Batch Normalization (BN) is a milestone technique in the development of deep learning, enabling various networks to train. However, normalizing along the batch dimension introduces problems — BN’s error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation. This limits BN’s usage for training larger models and transferring features to computer vision tasks including detection, segmentation, and video, which require small batches constrained by memory consumption. In this paper, we present Group Normalization (GN) as a simple alternative to BN. GN divides the channels into groups and computes within each group the mean and variance for normalization. GN’s computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes. On ResNet-50 trained in ImageNet, GN has 10.6% lower error than its BN counterpart when using a batch size of 2; when using typical batch sizes, GN is comparably good with BN and outperforms other normalization variants. Moreover, GN can be naturally transferred from pre-training to fine-tuning. GN can outperform its BNbased counterparts for object detection and segmentation in COCO( https://github.com/facebookresearch/Detectron/blob/master/projects/GN ), and for video classification in Kinetics, showing that GN can effectively replace the powerful BN in a variety of tasks. GN can be easily implemented by a few lines of code in modern libraries.

批归一化(BN)是深度学习发展中的一项里程碑技术，使各种网络能够进行训练。然而，沿着批次维度进行归一化会带来问题 —— 由于批次统计估计不准确，当批次大小变小时，BN的误差会迅速增加。这限制了BN用于训练更大的模型和将特征迁移到计算机视觉任务，包括检测、分割和视频，这些任务需要小批量的内存消耗。在本文中，我们提出了组归一化(GN)作为BN的简单替代方案。GN将信道分成组，并在每组内计算平均值和方差以进行归一化。GN的计算与批量大小无关，其精度在大批量范围内是稳定的。在ImageNet中训练的ResNet-50上，当使用批量大小为2时，GN的误差比BN低10.6%; 当使用典型的批量大小时，GN与BN相当好，并且优于其他归一化变量。此外，GN可以自然地从预训练迁移到微调。GN在COCO中的目标检测和分割方面优于基于BN的同类算法( https://github.com/facebookresearch/Detectron/blob/master/projects/GN )，以及Kinetics中的视频分类，表明GN可以有效地在各种任务中取代强大的BN。GN可以通过现代库中的几行代码轻松实现。

## 1. Introduction
Batch Normalization (Batch Norm or BN) [26] has been established as a very effective component in deep learning, largely helping push the frontier in computer vision [59, 20] and beyond [54]. BN normalizes the features by the mean and variance computed within a (mini-)batch. This has been shown by many practices to ease optimization and enable very deep networks to converge. The stochastic uncertainty of the batch statistics also acts as a regularizer that can benefit generalization. BN has been a foundation of many stateof-the-art computer vision algorithms. 

批归一化(Batch Norm或BN)[26]已被确立为深度学习中非常有效的组成部分，在很大程度上有助于推动计算机视觉领域的前沿[59，20]和更高[54]。BN通过(迷你)批次内计算的平均值和方差对特征进行归一化。许多实践证明了这一点，以简化优化并使非常深层的网络能够收敛。批次统计的随机不确定性也可以作为正则化器，有利于泛化。BN是许多最先进计算机视觉算法的基础。

Despite its great success, BN exhibits drawbacks that are also caused by its distinct behavior of normalizing along the batch dimension. In particular, it is required for BN to work with a sufficiently large batch size (e.g., 32 per worker [26, 59, 20])(2In the context of this paper, we use “batch size” to refer to the number of samples per worker (e.g., GPU). BN’s statistics are computed for each worker, but not broadcast across workers, as is standard in many libraries. ). A small batch leads to inaccurate estimation of the batch statistics, and reducing BN’s batch size increases the model error dramatically (Figure 1). As a result, many recent models [59, 20, 57, 24, 63] are trained with non-trivial batch sizes that are memory-consuming. The heavy reliance on BN’s effectiveness to train models in turn prohibits people from exploring higher-capacity models that would be limited by memory.

尽管BN取得了巨大的成功，但它也表现出了一些缺点，这些缺点也是由其沿批次维度进行归一化的独特行为造成的。特别是，BN需要使用足够大的批量(例如每批次32个[26，59，20])(2在本文中，我们使用“批量”来指每个批次的样本数量(例如GPU)。BN的统计数据是为每个批次计算的，但不会像许多库的标准那样在批次之间传播。)。小批量会导致批量统计的不准确估计，而减少BN的批量大小会显著增加模型误差(图1)。因此，许多最近的模型[59，20，57，24，63]都是用非常规的批大小训练的，这些批大小消耗内存。严重依赖BN训练模型的有效性，反过来又阻碍了人们探索受内存限制的更高容量模型。

Figure 1. ImageNet classification error vs. batch sizes. This is a ResNet-50 model trained in the ImageNet training set using 8 workers (GPUs), evaluated in the validation set.
图1.ImageNet分类错误与批次大小。这是一个ResNet-50模型，在ImageNet训练集中使用8个批次(GPU)进行训练，并在验证集中进行评估。

The restriction on batch sizes is more demanding in computer vision tasks including detection [12, 47, 18], segmentation [38, 18], video recognition [60, 6], and other highlevel systems built on them. For example, the Fast/er and Mask R-CNN frameworks [12, 47, 18] use a batch size of 1 or 2 images because of higher resolution, where BN is “frozen” by transforming to a linear layer [20]; in video classification with 3D convolutions [60, 6], the presence of spatial-temporal features introduces a trade-off between the temporal length and batch size. The usage of BN often requires these systems to compromise between the model design and batch sizes. 

在计算机视觉任务中，对批量大小的限制更为苛刻，包括检测[12，47，18]、分割[38，18]，视频识别[60，6]和其他基于它们的高级系统。例如，Fast/er和Mask R-CNN框架[12，47，18]使用1或2个图像的批量大小，因为分辨率更高，其中BN通过转换为线性层而“冻结”[20]; 在使用3D卷积的视频分类[60，6]中，时空特征的存在引入了时间长度和批量大小之间的权衡。BN的使用通常要求这些系统在模型设计和批量大小之间做出妥协。

This paper presents Group Normalization (GN) as a simple alternative to BN. We notice that many classical features like SIFT [39] and HOG [9] are group-wise features and involve group-wise normalization. For example, a HOG vector is the outcome of several spatial cells where each cell is represented by a normalized orientation histogram. Analogously, we propose GN as a layer that divides channels into groups and normalizes the features within each group (Figure 2). GN does not exploit the batch dimension, and its computation is independent of batch sizes.

本文提出了组归一化(GN)作为BN的简单替代方案。我们注意到，许多经典特征，如SIFT[39]和HOG[9]都是逐组特征，并涉及逐组归一化。例如，HOG向量是几个空间单元的结果，其中每个单元由归一化方向直方图表示。类似地，我们建议GN作为一个层，将信道划分为多个组，并将每个组内的特征归一化(图2)。GN不利用批次维度，其计算与批次大小无关。

Figure 2. Normalization methods. Each subplot shows a feature map tensor, with N as the batch axis, C as the channel axis, and (H, W) as the spatial axes. The pixels in blue are normalized by the same mean and variance, computed by aggregating the values of these pixels. 
图2.归一化方法。每个子图显示一个特征图张量，其中N为批次轴，C为通道轴，(H，W)为空间轴。蓝色的像素由相同的平均值和方差进行归一化，通过聚集这些像素的值来计算。

GN behaves very stably over a wide range of batch sizes (Figure 1). With a batch size of 2 samples, GN has 10.6% lower error than its BN counterpart for ResNet-50 [20] in ImageNet [50]. With a regular batch size, GN is comparably good as BN (with a gap of ∼0.5%) and outperforms other normalization variants [3, 61, 51]. Moreover, although the batch size may change, GN can naturally transfer from pretraining to fine-tuning. GN shows improved results vs. its BN counterpart on Mask R-CNN for COCO object detection and segmentation [37], and on 3D convolutional networks for Kinetics video classification [30]. The effectiveness of GN in ImageNet, COCO, and Kinetics demonstrates that GN is a competitive alternative to BN that has been dominant in these tasks.

GN在广泛的批量大小范围内表现得非常稳定(图1)。当批量大小为2个样本时，GN在ImageNet[50]中的ResNet-50[20]的误差比BN的同类产品低10.6%。对于常规批量，GN与BN相当好(差距为∼0.5%)，并优于其他归一化变体[3，61，51]。此外，尽管批量大小可能会改变，但GN可以自然地从预训练迁移到微调。GN在用于COCO对象检测和分割的Mask R-CNN和用于动力学视频分类的3D卷积网络[30]上显示了与BN相对应的改进结果。GN在ImageNet、COCO和Kinetics中的有效性表明，GN是BN的竞争性替代品，在这些任务中占主导地位。

There have been existing methods, such as Layer Normalization (LN) [3] and Instance Normalization (IN) [61] (Figure 2), that also avoid normalizing along the batch dimension. These methods are effective for training sequential models (RNN/LSTM [49, 22]) or generative models (GANs [15, 27]). But as we will show by experiments, both LN and IN have limited success in visual recognition, for which GN presents better results. Conversely, GN could be used in place of LN and IN and thus is applicable for sequential or generative models. This is beyond the focus of this paper, but it is suggestive for future research.

已有一些方法，如层归一化(LN)[3]和实例归一化(IN)[61](图2)，也避免了沿批次维度进行归一化。这些方法对于训练序列模型(RNN/LSTM[49，22])或生成模型(GAN[15，27])是有效的。但正如我们将通过实验表明的，LN和IN在视觉识别方面的成功有限，而GN在视觉识别中表现出更好的效果。相反，GN可以代替LN和in，因此适用于顺序或生成模型。这超出了本文的重点，但对未来的研究具有启发性。

## 2. Related Work
### Normalization. 
It is well-known that normalizing the input data makes training faster [33]. To normalize hidden features, initialization methods [33, 14, 19] have been derived based on strong assumptions of feature distributions, which can become invalid when training evolves.

规一化. 众所周知，归一化输入数据可以使训练更快[33]。为了规一化隐藏特征，基于特征分布的强假设导出了初始化方法[33，14，19]，当训练发展时，这些方法可能会失效。

Normalization layers in deep networks had been widely used before the development of BN. Local Response Normalization (LRN) [40, 28, 32] was a component in AlexNet [32] and following models [64, 53, 58]. Unlike recent methods [26, 3, 61], LRN computes the statistics in a small neighborhood for each pixel.

在BN发展之前，深度网络中的归一化层已经被广泛使用。局部响应归一化(LRN)[40，28，32]是AlexNet[32]和后续模型[64，53，58]中的一个组成部分。与最近的方法[26，3，61]不同，LRN在每个像素的小邻域中计算统计数据。

Batch Normalization [26] performs more global normalization along the batch dimension (and as importantly, it suggests to do this for all layers). But the concept of “batch” is not always present, or it may change from time to time. For example, batch-wise normalization is not legitimate at inference time, so the mean and variance are pre-computed from the training set [26], often by running average; consequently, there is no normalization performed when testing. The pre-computed statistics may also change when the target data distribution changes [45]. These issues lead to inconsistency at training, transferring, and testing time. In addition, as aforementioned, reducing the batch size can have dramatic impact on the estimated batch statistics.

批归一化[26]沿批次维度执行更多的全局归一化(同样重要的是，它建议对所有层执行此操作)。但“批量”的概念并不总是存在，或者它可能会不时发生变化。例如，批归一化在推理时是不合法的，因此通常通过运行平均值从训练集[26]中预先计算平均值和方差; 因此，测试时没有执行归一化。当目标数据分布改变时，预计算的统计也可能改变[45]。这些问题导致训练、迁移和测试时间不一致。此外，如上所述，减少批次大小可能会对估计的批次统计数据产生重大影响。

Several normalization methods [3, 61, 51, 2, 46] have been proposed to avoid exploiting the batch dimension. Layer Normalization (LN) [3] operates along the channel dimension, and Instance Normalization (IN) [61] performs BN-like computation but only for each sample (Figure 2). Instead of operating on features, Weight Normalization (WN) [51] proposes to normalize the filter weights. These methods do not suffer from the issues caused by the batch dimension, but they have not been able to approach BN’s accuracy in many visual recognition tasks. We provide comparisons with these methods in context of the remaining sections.

已经提出了几种归一化方法[3，61，51，2，46]，以避免利用批次维度。层归一化(LN)[3]沿通道维度操作，实例归一化(IN)[61]执行BN类计算，但仅针对每个样本(图2)。权重归一化(WN)[51]建议对滤波器权重进行归一化，而不是对特征进行操作。这些方法不受批量维度引起的问题的影响，但在许多视觉识别任务中，它们无法达到BN的精度。我们在剩下的章节中对这些方法进行了比较。

### Addressing small batches. 
Ioffe [25] proposes Batch Renormalization (BR) that alleviates BN’s issue involving small batches. BR introduces two extra parameters that constrain the estimated mean and variance of BN within a certain range, reducing their drift when the batch size is small. BR has better accuracy than BN in the small-batch regime. But BR is also batch-dependent, and when the batch size decreases its accuracy still degrades [25].

解决小批量问题。Ioffe[25]提出了批量重整(BR)，以缓解BN涉及小批量的问题。BR引入了两个额外的参数，将BN的估计均值和方差限制在一定范围内，从而在批量较小时减少了它们的漂移。在小批量条件下，BR比BN具有更好的精度。但BR也依赖于批次，当批次大小减小时，其精度仍会下降[25]。

There are also attempts to avoid using small batches. The object detector in [43] performs synchronized BN whose mean and variance are computed across multiple GPUs. However, this method does not solve the problem of small batches; instead, it migrates the algorithm problem to engineering and hardware demands, using a number of GPUs proportional to BN’s requirements. Moreover, the synchronized BN computation prevents using asynchronous solvers (ASGD [10]), a practical solution to large-scale training widely used in industry. These issues can limit the scope of using synchronized BN.

也有人试图避免使用小批量。[43]中的对象检测器执行同步BN，其平均值和方差是跨多个GPU计算的。然而，这种方法并不能解决小批量的问题; 相反，它将算法问题迁移到工程和硬件需求，使用与BN需求成比例的多个GPU。此外，同步BN计算防止使用异步求解器(ASGD[10])，这是一种在工业中广泛使用的大规模训练的实用解决方案。这些问题可能会限制使用同步BN的范围。

Instead of addressing the batch statistics computation (e.g., [25, 43]), our normalization method inherently avoids this computation.

我们的归一化方法本质上避免了这种计算，而不是解决批量统计计算(例如[25，43])。

### Group-wise computation. 
Group convolutions have been presented by AlexNet [32] for distributing a model into two GPUs. The concept of groups as a dimension for model design has been more widely studied recently. The work of ResNeXt [63] investigates the trade-off between depth, width, and groups, and it suggests that a larger number of groups can improve accuracy under similar computational cost. MobileNet [23] and Xception [7] exploit channel-wise (also called “depth-wise”) convolutions, which are group convolutions with a group number equal to the channel number. ShuffleNet [65] proposes a channel shuffle operation that permutes the axes of grouped features. These methods all involve dividing the channel dimension into groups. Despite the relation to these methods, GN does not require group convolutions. GN is a generic layer, as we evaluate in standard ResNets [20].

分组计算。AlexNet[32]提出了组卷积，用于将模型分布到两个GPU中。组作为模型设计维度的概念最近得到了更广泛的研究。ResNeXt[63]的工作研究了深度、宽度和组之间的权衡，并表明在相似的计算成本下，更多的组可以提高精度。MobileNet[23]和Xception[7]利用了信道方向(也称为“深度方向”)卷积，这是一种组卷积，组编号等于信道编号。ShuffleNet[65]提出了一种信道混洗操作，该操作排列分组特征的轴。这些方法都涉及将通道维度分成组。尽管与这些方法有关，但GN不需要组卷积。正如我们在标准ResNets中评估的那样，GN是一个通用层[20]。

## 3. Group Normalization
The channels of visual representations are not entirely independent. Classical features of SIFT [39], HOG [9], and GIST [41] are group-wise representations by design, where each group of channels is constructed by some kind of histogram. These features are often processed by groupwise normalization over each histogram or each orientation. Higher-level features such as VLAD [29] and Fisher Vectors (FV) [44] are also group-wise features where a group can be thought of as the sub-vector computed with respect to a cluster.

视觉表现的信道并不完全独立。SIFT[39]、HOG[9]和GIST[41]的经典特征是设计上的分组表示，其中每组通道由某种直方图构成。这些特征通常通过对每个直方图或每个方向进行分组归一化来处理。诸如VLAD[29]和Fisher Vectors(FV)[44]等更高级别的特征也是分组特征，其中组可以被认为是相对于簇计算的子向量。

Analogously, it is not necessary to think of deep neural network features as unstructured vectors. For example, for conv1 (the first convolutional layer) of a network, it is reasonable to expect a filter and its horizontal flipping to exhibit similar distributions of filter responses on natural images. If conv1 happens to approximately learn this pair of filters, or if the horizontal flipping (or other transformations) is made into the architectures by design [11, 8], then the corresponding channels of these filters can be normalized together.

类似地，没有必要将深度神经网络特征视为非结构化向量。例如，对于网络的conv1(第一卷积层)，期望滤波器及其水平翻转在自然图像上呈现类似的滤波器响应分布是合理的。如果conv1恰好近似地学习了这对滤波器，或者如果通过设计将水平翻转(或其他变换)转换为架构[11，8]，那么这些滤波器的相应通道可以一起归一化。

The higher-level layers are more abstract and their behaviors are not as intuitive. However, in addition to orientations (SIFT [39], HOG [9], or [11, 8]), there are many factors that could lead to grouping, e.g., frequency, shapes, illumination, textures. Their coefficients can be interdependent. In fact, a well-accepted computational model in neuroscience is to normalize across the cell responses [21, 52, 55, 5], “with various receptive-field centers (covering the visual field) and with various spatiotemporal frequency tunings” (p183, [21]); this can happen not only in the primary visual cortex, but also “throughout the visual system” [5]. Motivated by these works, we propose new generic group-wise normalization for deep neural networks.

更高级的层更抽象，它们的行为也不那么直观。然而，除了取向(SIFT[39]、HOG[9]或[11，8])，还有许多因素可能导致分组，例如频率、形状、照明、纹理。它们的系数可以相互依赖。事实上，神经科学中一个广为接受的计算模型是对细胞反应进行归一化[21，52，55，5]，“用不同的感受野中心(覆盖视野)和不同的时空频率调谐”(p183，[21]); 这不仅可能发生在初级视觉皮层，也可能发生在“整个视觉系统”[5]。在这些工作的激励下，我们提出了一种新的深度神经网络的通用分组归一化。

### 3.1. Formulation
We first describe a general formulation of feature normalization, and then present GN in this formulation. A family of feature normalization methods, including BN, LN, IN, and GN, perform the following computation: 

xˆi = 1σi (xi − µi). (1)

Here x is the feature computed by a layer, and i is an index. In the case of 2D images, i = (iN , iC , iH, iW ) is a 4D vector indexing the features in (N, C, H, W) order, where N is the batch axis, C is the channel axis, and H and W are the spatial height and width axes. 

µ and σ in (1) are the mean and standard deviation (std) computed by: 

µi = 1m X k∈Si xk, σi = s 1m kX∈Si(xk − µi)2 + , (2) 

with  as a small constant. Si is the set of pixels in which the mean and std are computed, and m is the size of this set. Many types of feature normalization methods mainly differ in how the set Si is defined (Figure 2), discussed as follows.

In Batch Norm [26], the set Si is defined as:

Si = {k | kC = iC }, (3) 

where iC (and kC ) denotes the sub-index of i (and k) along the C axis. This means that the pixels sharing the same channel index are normalized together, i.e., for each channel, BN computes µ and σ along the (N, H, W) axes. In Layer Norm [3], the set is:

Si = {k | kN = iN }, (4) 

meaning that LN computes µ and σ along the (C, H, W) axes for each sample. In Instance Norm [61], the set is:

Si = {k | kN = iN , kC = iC }. (5) 

meaning that IN computes µ and σ along the (H, W) axes for each sample and each channel. The relations among BN, LN, and IN are in Figure 2.  


As in [26], all methods of BN, LN, and IN learn a perchannel linear transform to compensate for the possible lost of representational ability: 

yi = γxˆi + β, (6) 

where γ and β are trainable scale and shift (indexed by iC in all case, which we omit for simplifying notations).

Group Norm. Formally, a Group Norm layer computes µ and σ in a set Si defined as:

Si = {k | kN = iN , b kC C/Gc = b iC C/Gc}. (7)

Here G is the number of groups, which is a pre-defined hyper-parameter (G = 32 by default). C/G is the number of channels per group. b·c is the floor operation, and “b kC

C/G c = b iC

C/G c ” means that the indexes i and k are in the same group of channels, assuming each group of channels are stored in a sequential order along the C axis. GN computes µ and σ along the (H, W) axes and along a group of CG channels. The computation of GN is illustrated in Figure 2 (rightmost), which is a simple case of 2 groups (G = 2) each having 3 channels.

Given Si in Eqn.(7), a GN layer is defined by Eqn.(1), (2), and (6). Specifically, the pixels in the same group are normalized together by the same µ and σ. GN also learns the per-channel γ and β.

Relation to Prior Work. LN, IN, and GN all perform independent computations along the batch axis. The two extreme cases of GN are equivalent to LN and IN (Figure 2).

Relation to Layer Normalization [3]. GN becomes LN if we set the group number as G = 1. LN assumes all channels in a layer make “similar contributions” [3]. Unlike the case of fully-connected layers studied in [3], this assumption can be less valid with the presence of convolutions, as discussed in [3]. GN is less restricted than LN, because each group of channels (instead of all of them) are assumed to subject to the shared mean and variance; the model still has flexibility of learning a different distribution for each group. This leads to improved representational power of GN over LN, as shown by the lower training and validation error in experiments (Figure 4).

Relation to Instance Normalization [61]. GN becomes IN if we set the group number as G = C (i.e., one channel per group). But IN can only rely on the spatial dimension for computing the mean and variance and it misses the opportunity of exploiting the channel dependence.

### 3.2. Implementation
GN can be easily implemented by a few lines of code in PyTorch [42] and TensorFlow [1] where automatic differentiation is supported. Figure 3 shows the code based on TensorFlow. In fact, we only need to specify how the mean and variance (“moments”) are computed, along the appropriate axes as defined by the normalization method.

Figure 3. Python code of Group Norm based on TensorFlow.

## 4. Experiments
### 4.1. Image Classification in ImageNet
We experiment in the ImageNet classification dataset [50] with 1000 classes. We train on the ∼1.28M training images and evaluate on the 50,000 validation images, using the ResNet models [20].

#### Implementation details. 
As standard practice [20, 17], we use 8 GPUs to train all models, and the batch mean and variance of BN are computed within each GPU. We use the method of [19] to initialize all convolutions for all models. We use 1 to initialize all γ parameters, except for each residual block’s last normalization layer where we initialize γ by 0 following [16] (such that the initial state of a residual block is identity). We use a weight decay of 0.0001 for all weight layers, including γ and β (following [17] but unlike [20, 16]). We train 100 epochs for all models, and decrease the learning rate by 10× at 30, 60, and 90 epochs. During training, we adopt the data augmentation of [58] as implemented by [17]. We evaluate the top-1 classification error on the center crops of 224×224 pixels in the validation set. To reduce random variations, we report the median error rate of the final 5 epochs [16]. Other implementation details follow [17].

Our baseline is the ResNet trained with BN [20]. To compare with LN, IN, and GN, we replace BN with the specific variant. We use the same hyper-parameters for all models. We set G = 32 for GN by default.

#### Comparison of feature normalization methods. 
We first experiment with a regular batch size of 32 images (per GPU) [26, 20]. BN works successfully in this regime, so this is a strong baseline to compare with. Figure 4 shows the error curves, and Table 1 shows the final results.

Figure 4 shows that all of these normalization methods are able to converge. LN has a small degradation of 1.7% comparing with BN. This is an encouraging result, as it suggests that normalizing along all channels (as done by LN) of a convolutional network is reasonably good. IN also makes the model converge, but is 4.8% worse than BN.(3For completeness, we have also trained ResNet-50 with WN [51], which is filter (instead of feature) normalization. WN’s result is 28.2%. )<br/>
Figure 4. Comparison of error curves with a batch size of 32 images/GPU. We show the ImageNet training error (left) and validation error (right) vs. numbers of training epochs. The model is ResNet-50. 

Figure 5. Sensitivity to batch sizes: ResNet-50’s validation error of BN (left) and GN (right) trained with 32, 16, 8, 4, and 2 images/GPU.

Table 1. Comparison of error rates (%) of ResNet-50 in the ImageNet validation set, trained with a batch size of 32 images/GPU. The error curves are in Figure 4. 

In this regime where BN works well, GN is able to approach BN’s accuracy, with a decent degradation of 0.5% in the validation set. Actually, Figure 4 (left) shows that GN has lower training error than BN, indicating that GN is effective for easing optimization. The slightly higher validation error of GN implies that GN loses some regularization ability of BN. This is understandable, because BN’s mean and variance computation introduces uncertainty caused by the stochastic batch sampling, which helps regularization [26]. This uncertainty is missing in GN (and LN/IN). But it is possible that GN combined with a suitable regularizer will improve results. This can be a future research topic. 

Table 2. Sensitivity to batch sizes. We show ResNet-50’s validation error (%) in ImageNet. The last row shows the differences between BN and GN. The error curves are in Figure 5. This table is visualized in Figure 1.

#### Small batch sizes. 
Although BN benefits from the stochasticity under some situations, its error increases when the batch size becomes smaller and the uncertainty gets bigger. We show this in Figure 1, Figure 5, and Table 2. 

We evaluate batch sizes of 32, 16, 8, 4, 2 images per GPU. In all cases, the BN mean and variance are computed within each GPU and not synchronized. All models are trained in 8 GPUs. In this set of experiments, we adopt the linear learning rate scaling rule [31, 4, 16] to adapt to batch size changes — we use a learning rate of 0.1 [20] for the batch size of 32, and 0.1N/32 for a batch size of N. This linear scaling rule works well for BN if the total batch size changes (by changing the number of GPUs) but the perGPU batch size does not change [16]. We keep the same number of training epochs for all cases (Figure 5, x-axis). All other hyper-parameters are unchanged. 

Figure 6. Evolution of feature distributions of conv5 3’s output (before normalization and ReLU) from VGG-16, shown as the {1, 20, 80, 99} percentile of responses. The table on the right shows the ImageNet validation error (%). Models are trained with 32 images/GPU.

Table 3. Group division. We show ResNet-50’s validation error (%) in ImageNet, trained with 32 images/GPU. (Top): a given number of groups. (Bottom): a given number of channels per group. The last rows show the differences with the best number.

Figure 5 (left) shows that BN’s error becomes considerably higher with small batch sizes. GN’s behavior is more stable and insensitive to the batch size. Actually, Figure 5 (right) shows that GN has very similar curves (subject to random variations) across a wide range of batch sizes from 32 to 2. In the case of a batch size of 2, GN has 10.6% lower error rate than its BN counterpart (24.1% vs. 34.7%).

These results indicate that the batch mean and variance estimation can be overly stochastic and inaccurate, especially when they are computed over 4 or 2 images. However, this stochasticity disappears if the statistics are computed from 1 image, in which case BN becomes similar to IN at training time. We see that IN has a better result (28.4%) than BN with a batch size of 2 (34.7%).

The robust results of GN in Table 2 demonstrate GN’s strength. It allows to remove the batch size constraint imposed by BN, which can give considerably more memory (e.g., 16× or more). This will make it possible to train higher-capacity models that would be otherwise bottlenecked by memory limitation. We hope this will create new opportunities in architecture design.

#### Comparison with Batch Renorm (BR). 
BR [25] introduces two extra parameters (r and d in [25]) that constrain the estimated mean and variance of BN. Their values are controlled by rmax and dmax. To apply BR to ResNet-50, we have carefully chosen these hyper-parameters, and found that rmax = 1.5 and dmax = 0.5 work best for ResNet-50. With a batch size of 4, ResNet-50 trained with BR has an error rate of 26.3%. This is better than BN’s 27.3%, but still 2.1% higher than GN’s 24.2%.

#### Group division. 
Thus far all presented GN models are trained with a group number of G = 32. Next we evaluate different ways of dividing into groups. With a given fixed group number, GN performs reasonably well for all values of G we studied (Table 3, top panel). In the extreme case of G = 1, GN is equivalent to LN, and its error rate is higher than all cases of G > 1 studied.

We also evaluate fixing the number of channels per group (Table 3, bottom panel). Note that because the layers can have different channel numbers, the group number G can change across layers in this setting. In the extreme case of 1 channel per group, GN is equivalent to IN. Even if using as few as 2 channels per group, GN has substantially lower error than IN (25.6% vs. 28.4%). This result shows the effect of grouping channels when performing normalization.

#### Deeper models. 
We have also compared GN with BN on ResNet-101 [20]. With a batch size of 32, our BN baseline of ResNet-101 has 22.0% validation error, and the GN counterpart has 22.4%, slightly worse by 0.4%. With a batch size of 2, GN ResNet-101’s error is 23.0%. This is still a decently stable result considering the very small batch size, and it is 8.9% better than the BN counterpart’s 31.9%.

#### Results and analysis of VGG models. 
To study GN/BN compared to no normalization, we consider VGG-16 [56] that can be healthily trained without normalization layers. We apply BN or GN right after each convolutional layer. Figure 6 shows the evolution of the feature distributions of conv5 3 (the last convolutional layer). GN and BN behave qualitatively similar, while being substantially different with the variant that uses no normalization; this phenomenon is also observed for all other convolutional layers. This comparison suggests that performing normalization is essential for controlling the distribution of features.

For VGG-16, GN is better than BN by 0.4% (Figure 6, right). This possibly implies that VGG-16 benefits less from BN’s regularization effect, and GN (that leads to lower training error) is superior to BN in this case. 

### 4.2. Object Detection and Segmentation in COCO
Next we evaluate fine-tuning the models for transferring to object detection and segmentation. These computer vision tasks in general benefit from higher-resolution input, so the batch size tends to be small in common practice (1 or 2 images/GPU [12, 47, 18, 36]). As a result, BN is turned into a linear layer y = σγ (x − µ) + β where µ and σ are pre-computed from the pre-trained model and frozen [20]. We denote this as BN* , which in fact performs no normalization during fine-tuning. We have also tried a variant that fine-tunes BN (normalization is performed and not frozen) and found it works poorly (reducing ∼6 AP with a batch size of 2), so we ignore this variant.

We experiment on the Mask R-CNN baselines [18], implemented in the publicly available codebase of Detectron [13]. We use the end-to-end variant with the same hyperparameters as in [13]. We replace BN* with GN during finetuning, using the corresponding models pre-trained from ImageNet(4 Detectron [13] uses pre-trained models provided by the authors of [20]. For fair comparisons, we instead use the models pre-trained in this paper. The object detection and segmentation accuracy is statistically similar between these pre-trained models. ). During fine-tuning, we use a weight decay of 0 for the γ and β parameters, which is important for good detection results when γ and β are being tuned. We fine-tune with a batch size of 1 image/GPU and 8 GPUs.

The models are trained in the COCO train2017 set and evaluated in the COCO val2017 set (a.k.a minival). We report the standard COCO metrics of Average Precision (AP), AP50, and AP75, for bounding box detection (APbbox) and instance segmentation (APmask).

#### Results of C4 backbone. 
Table 4 shows the comparison of GN vs. BN* on Mask R-CNN using a conv4 backbone (“C4” [18]). This C4 variant uses ResNet’s layers of up to conv4 to extract feature maps, and ResNet’s conv5 layers as the Region-of-Interest (RoI) heads for classification and regression. As they are inherited from the pre-trained model, the backbone and head both involve normalization layers.

On this baseline, GN improves over BN* by 1.1 box AP and 0.8 mask AP. We note that the pre-trained GN model is slightly worse than BN in ImageNet (24.1% vs. 23.6%), but GN still outperforms BN* for fine-tuning. BN* creates inconsistency between pre-training and fine-tuning (frozen), which may explain the degradation.

We have also experimented with the LN variant, and found it is 1.9 box AP worse than GN and 0.8 worse than BN* . Although LN is also independent of batch sizes, its representational power is weaker than GN.

#### Results of FPN backbone. 
Next we compare GN and BN* on Mask R-CNN using a Feature Pyramid Network (FPN) backbone [35], the currently state-of-the-art framework in COCO. Unlike the C4 variant, FPN exploits all pre-trained layers to construct a pyramid, and appends randomly initialized layers as the head. In [35], the box head consists of two hidden fully-connected layers (2fc). We find that replacing the 2fc box head with 4conv1fc (similar to [48]) can better leverage GN. The resulting comparisons are in Table 5.


Table 5. Detection and segmentation ablation results in COCO, using Mask R-CNN with ResNet-50 FPN and a 4conv1fc bounding box head. BN* means BN is frozen.

Table 6. Detection and segmentation results in COCO using Mask R-CNN and FPN. Here BN* is the default Detectron baseline [13], and GN is applied to the backbone, box head, and mask head. “long” means training with more iterations. Code of these results are in https://github.com/facebookresearch/Detectron/blob/master/projects/GN. 


As a baseline, BN* has 38.6 box AP using the 4conv1fc head, on par with its 2fc counterpart using the same pretrained model (38.5 AP). By adding GN to all convolutional layers of the box head (but still using the BN* backbone), we increase the box AP by 0.9 to 39.5 (2nd row, Table 5). This ablation shows that a substantial portion of GN’s improvement for detection is from normalization in the head (which is also done by the C4 variant). On the contrary, applying BN to the box head (that has 512 RoIs per image) does not provide satisfactory result and is ∼9 AP worse — in detection, the batch of RoIs are sampled from the same image and their distribution is not i.i.d., and the non-i.i.d. distribution is also an issue that degrades BN’s batch statistics estimation [25]. GN does not suffer from this problem.

Next we replace the FPN backbone with the GN-based counterpart, i.e., the GN pre-trained model is used during fine-tuning (3rd row, Table 5). Applying GN to the backbone alone contributes a 0.5 AP gain (from 39.5 to 40.0), suggesting that GN helps when transferring features. 

Figure 7. Error curves in Kinetics with an input length of 32 frames. We show ResNet-50 I3D’s validation error of BN (left) and GN (right) using a batch size of 8 and 4 clips/GPU. The monitored validation error is the 1-clip error under the same data augmentation as the training set, while the final validation accuracy in Table 8 is 10-clip testing without data augmentation. 


Table 7. Detection and segmentation results trained from scratch in COCO using Mask R-CNN and FPN. Here the BN results are from [34], and BN is synced across GPUs [43] and is not frozen. Code of these results are in https://github.com/facebookresearch/Detectron/blob/master/projects/GN.

Table 6 shows the full results of GN (applied to the backbone, box head, and mask head), compared with the standard Detectron baseline [13] based on BN* . Using the same hyper-parameters as [13], GN increases over BN* by a healthy margin. Moreover, we found that GN is not fully trained with the default schedule in [13], so we also tried increasing the iterations from 180k to 270k (BN* does not benefit from longer training). Our final ResNet-50 GN model (“long”, Table 6) is 2.2 points box AP and 1.6 points mask AP better than its BN* variant.

#### Training Mask R-CNN from scratch. 
GN allows us to easily investigate training object detectors from scratch (without any pre-training). We show the results in Table 7, where the GN models are trained for 270k iterations(5 For models trained from scratch, we turn off the default StopGrad in Detectron that freezes the first few layers. ). To our knowledge, our numbers (41.0 box AP and 36.4 mask AP) are the best from-scratch results in COCO reported to date; they can even compete with the ImageNet-pretrained results in Table 6. As a reference, with synchronous BN [43], a concurrent work [34] achieves a from-scratch result of 34.5 box AP using R50 (Table 7), and 36.3 using a specialized backbone. 

Table 8. Video classification results in Kinetics: ResNet-50 I3D baseline’s top-1 / top-5 accuracy (%).

### 4.3. Video Classification in Kinetics
Lastly we evaluate video classification in the Kinetics dataset [30]. Many video classification models [60, 6] extend the features to 3D spatial-temporal dimensions. This is memory-demanding and imposes constraints on the batch sizes and model designs.

We experiment with Inflated 3D (I3D) convolutional networks [6]. We use the ResNet-50 I3D baseline as described in [62]. The models are pre-trained from ImageNet. For both BN and GN, we extend the normalization from over (H, W) to over (T, H, W), where T is the temporal axis. We train in the 400-class Kinetics training set and evaluate in the validation set. We report the top-1 and top-5 classifi- cation accuracy, using standard 10-clip testing that averages softmax scores from 10 clips regularly sampled.

We study two different temporal lengths: 32-frame and 64-frame input clips. The 32-frame clip is regularly sampled with a frame interval of 2 from the raw video, and the 64-frame clip is sampled continuously. The model is fully convolutional in spacetime, so the 64-frame variant consumes about 2× more memory. We study a batch size of 8 or 4 clips/GPU for the 32-frame variant, and 4 clips/GPU for the 64-frame variant due to memory limitation.

#### Results of 32-frame inputs. 
Table 8 (col. 1, 2) shows the video classification accuracy in Kinetics using 32-frame clips. For the batch size of 8, GN is slightly worse than BN by 0.3% top-1 accuracy and 0.1% top-5. This shows that GN is competitive with BN when BN works well. For the smaller batch size of 4, GN’s accuracy is kept similar (72.8 / 90.6 vs. 73.0 / 90.6), but is better than BN’s 72.1 / 90.0. BN’s accuracy is decreased by 1.2% when the batch size decreases from 8 to 4.

Figure 7 shows the error curves. BN’s error curves (left) have a noticeable gap when the batch size decreases from 8 to 4, while GN’s error curves (right) are very similar.

#### Results of 64-frame inputs. 
Table 8 (col. 3) shows the results of using 64-frame clips. In this case, BN has a result of 73.3 / 90.8. These appear to be acceptable numbers (vs. 73.3 / 90.7 of 32-frame, batch size 8), but the trade-off between the temporal length (64 vs. 32) and batch size (4 vs.
 8) could have been overlooked. Comparing col. 3 and col. 2 in Table 8, we find that the temporal length actually has positive impact (+1.2%), but it is veiled by BN’s negative effect of the smaller batch size.

GN does not suffer from this trade-off. The 64-frame variant of GN has 74.5 / 91.7 accuracy, showing healthy gains over its BN counterpart and all BN variants. GN helps the model benefit from temporal length, and the longer clip boosts the top-1 accuracy by 1.7% (top-5 1.1%) with the same batch size.

The improvement of GN on detection, segmentation, and video classification demonstrates that GN is a strong alternative to the powerful and currently dominant BN technique in these tasks.

## 5. Discussion and Future Work
We have presented GN as an effective normalization layer without exploiting the batch dimension. We have evaluated GN’s behaviors in a variety of applications. We note, however, that BN has been so influential that many state-ofthe-art systems and their hyper-parameters have been designed for it, which may not be optimal for GN-based models. It is possible that re-designing the systems or searching new hyper-parameters for GN will give better results.

In addition, we have shown that GN is related to LN and IN, two normalization methods that are particularly successful in training recurrent (RNN/LSTM) or generative (GAN) models. This suggests us to study GN in those areas in the future. We will also investigate GN’s performance on learning representations for reinforcement learning (RL) tasks, e.g., [54], where BN is playing an important role for training very deep models [20].

## Acknowledgement. 
We would like to thank Piotr Doll´ar and Ross Girshick for helpful discussions.

## References
1. M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard, et al. Tensor- flow: A system for large-scale machine learning. In Operating Systems Design and Implementation (OSDI), 2016.
2. D. Arpit, Y. Zhou, B. Kota, and V. Govindaraju. Normalization propagation: A parametric technique for removing internal covariate shift in deep networks. In ICML, 2016.
3. J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. arXiv:1607.06450, 2016.
4. L. Bottou, F. E. Curtis, and J. Nocedal. Optimization methods for large-scale machine learning. arXiv:1606.04838, 2016.
5. M. Carandini and D. J. Heeger. Normalization as a canonical neural computation. Nature Reviews Neuroscience, 2012.
6. J. Carreira and A. Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017.
7. F. Chollet. Xception: Deep learning with depthwise separable convolutions. In CVPR, 2017.
8. T. Cohen and M. Welling. Group equivariant convolutional networks. In ICML, 2016.
9. N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In CVPR, 2005.
10. J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao, A. Senior, P. Tucker, K. Yang, Q. V. Le, et al. Large scale distributed deep networks. In NIPS, 2012.
11. S. Dieleman, J. De Fauw, and K. Kavukcuoglu. Exploiting cyclic symmetry in convolutional neural networks. In ICML, 2016.
12. R. Girshick. Fast R-CNN. In ICCV, 2015.
13. R. Girshick, I. Radosavovic, G. Gkioxari, P. Doll´ar, and K. He. Detectron. https://github.com/ facebookresearch/detectron, 2018.
14. X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2010.
15. I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In NIPS, 2014.
16. P. Goyal, P. Doll´ar, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, and K. He. Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677, 2017.
17. S. Gross and M. Wilber. Training and investigating Residual Nets. https://github.com/facebook/fb. resnet.torch, 2016.
18. K. He, G. Gkioxari, P. Doll´ar, and R. Girshick. Mask RCNN. In ICCV, 2017.
19. K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015.
20. K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
21. D. J. Heeger. Normalization of cell responses in cat striate cortex. Visual neuroscience, 1992.
22. S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 1997.
23. A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam. MobileNets: Effi- cient convolutional neural networks for mobile vision applications. arXiv:1704.04861, 2017. 9
24. G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger. Densely connected convolutional networks. In CVPR, 2017.
25. S. Ioffe. Batch renormalization: Towards reducing minibatch dependence in batch-normalized models. In NIPS, 2017.
26. S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.
27. P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017.
28. K. Jarrett, K. Kavukcuoglu, Y. LeCun, et al. What is the best multi-stage architecture for object recognition? In ICCV, 2009.
29. H. Jegou, M. Douze, C. Schmid, and P. Perez. Aggregating local descriptors into a compact image representation. In CVPR, 2010.
30. W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev, et al. The Kinetics human action video dataset. arXiv:1705.06950, 2017.
31. A. Krizhevsky. One weird trick for parallelizing convolutional neural networks. arXiv:1404.5997, 2014.
32. A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.
33. Y. LeCun, L. Bottou, G. B. Orr, and K.-R. M¨uller. Efficient backprop. In Neural Networks: Tricks of the Trade. 1998.
34. Z. Li, C. Peng, G. Yu, X. Zhang, Y. Deng, and J. Sun. DetNet: A backbone network for object detection. arXiv:1804.06215, 2018.
35. T.-Y. Lin, P. Doll´ar, R. Girshick, K. He, B. Hariharan, and S. Belongie. Feature pyramid networks for object detection. In CVPR, 2017.
36. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Doll´ar. Focal loss for dense object detection. In ICCV, 2017.
37. T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick. Microsoft COCO: Common objects in context. In ECCV. 2014.
38. J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.
39. D. G. Lowe. Distinctive image features from scale-invariant keypoints. IJCV, 2004.
40. S. Lyu and E. P. Simoncelli. Nonlinear image representation using divisive normalization. In CVPR, 2008.
41. A. Oliva and A. Torralba. Modeling the shape of the scene: A holistic representation of the spatial envelope. IJCV, 2001.
42. A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in pytorch. 2017.
43. C. Peng, T. Xiao, Z. Li, Y. Jiang, X. Zhang, K. Jia, G. Yu, and J. Sun. MegDet: A large mini-batch object detector. In CVPR, 2018.
44. F. Perronnin and C. Dance. Fisher kernels on visual vocabularies for image categorization. In CVPR, 2007.
45. S.-A. Rebuffi, H. Bilen, and A. Vedaldi. Learning multiple visual domains with residual adapters. In NIPS, 2017.
46. M. Ren, R. Liao, R. Urtasun, F. H. Sinz, and R. S. Zemel. Normalizing the normalizers: Comparing and extending network normalization schemes. In ICLR, 2017.
47. S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.
48. S. Ren, K. He, R. Girshick, X. Zhang, and J. Sun. Object detection networks on convolutional feature maps. TPAMI, 2017.
49. D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 1986.
50. O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.
51. T. Salimans and D. P. Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In NIPS, 2016.
52. O. Schwartz and E. P. Simoncelli. Natural signal statistics and sensory gain control. Nature neuroscience, 2001.
53. P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.
54. D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, Y. Chen, T. Lillicrap, F. Hui, L. Sifre, G. van den Driessche, T. Graepel, and D. Hassabis. Mastering the game of go without human knowledge. Nature, 2017.
55. E. P. Simoncelli and B. A. Olshausen. Natural image statistics and neural representation. Annual review of neuroscience, 2001.
56. K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.
57. C. Szegedy, S. Ioffe, and V. Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. In ICLR Workshop, 2016.
58. C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.
59. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.
60. D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri. Learning spatiotemporal features with 3D convolutional networks. In ICCV, 2015.
61. D. Ulyanov, A. Vedaldi, and V. Lempitsky. Instance normalization: The missing ingredient for fast stylization. arXiv:1607.08022, 2016.
62. X. Wang, R. Girshick, A. Gupta, and K. He. Non-local neural networks. In CVPR, 2018.
63. S. Xie, R. Girshick, P. Doll´ar, Z. Tu, and K. He. Aggregated residual transformations for deep neural networks. In CVPR, 2017.
64. M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional neural networks. In ECCV, 2014.
65. X. Zhang, X. Zhou, M. Lin, and J. Sun. ShuffleNet: An extremely efficient convolutional neural network for mobile devices. In CVPR, 2018. 10
