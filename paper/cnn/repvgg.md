# RepVGG: Making VGG-style ConvNets Great Again
RepVGG：让VGG风格的卷积网络再次伟大 原文:https://arxiv.org/abs/2101.03697

## Abstract
We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inference time body composed of nothing but a stack of 3 × 3 convolution and ReLU, while the training-time model has a multi-branch topology. Such decoupling of the training-time and inference-time architecture is realized by a structural re-parameterization technique so that the model is named RepVGG. On ImageNet, RepVGG reaches over 80% top-1 accuracy, which is the first time for a plain model, to the best of our knowledge. On NVIDIA 1080Ti GPU, RepVGG models run 83% faster than ResNet-50 or 101% faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the stateof-the-art models like EfficientNet and RegNet. The code and trained models are available at https://github.com/megvii-model/RepVGG .

我们提出了一种简单但功能强大的卷积神经网络结构，它在推理时具有VGG风格的主体，仅由3×3卷积和ReLU的堆叠组成，而训练时模型具有多分支拓扑。这种训练时和推理时结构的解耦是通过结构重新参数化技术实现的，因此该模型被命名为RepVGG。据我们所知，在ImageNet上，RepVGG是首次使用普通模型在NVIDIA 1080Ti GPU上，达到了80%以上的top-1精度。RepVGG模型的运行速度比ResNet-50快83%，比ResNet-101快101%，具有更高的精度，与SOTA模型(如EfficientNet和RegNet)相比，显示出良好的精度-速度权衡。代码和训练过的模型可在https://github.com/megvii-model/RepVGG .

Figure 1: Top-1 accuracy on ImageNet vs. actual speed. Left: lightweight and middleweight RepVGG and baselines trained in 120 epochs. Right: heavyweight models trained in 200 epochs. The speed is tested on the same 1080Ti with a batch size of 128, full precision (fp32), single crop, and measured in examples/second. The input resolution is 300 for EfficientNet-B3 [35] and 224 for the others.

图1:ImageNet上的Top-1精度与实际速度的对比。左：轻量级和中量级RepVGG以及120轮训练的基线。右图：权重级模型经过200轮的训练。速度在相同的1080Ti上进行测试，批量大小为128，全精度(fp32)，单次裁剪，以实例/秒为单位测量。EfficientNet-B3[35]的输入分辨率为300，其他为224。

## 1. Introduction
A classic Convolutional Neural Network (ConvNet), VGG [31], achieved huge success in image recognition with a simple architecture composed of a stack of conv, ReLU, and pooling. With Inception [33, 34, 32, 19], ResNet [12] and DenseNet [17], a lot of research interests were shifted to well-designed architectures, making the models more and more complicated. Some recent architectures are based on automatic [44, 29, 23] or manual [28] architecture search, or a searched compound scaling strategy [35].

经典卷积神经网络(ConvNet)VGG[31]在图像识别方面取得了巨大成功，其简单的架构由一个由conv、ReLU和pooling组成的堆栈组成。随着Inception[33、34、32、19]、ResNet[12]和DenseNet[17]的推出，许多研究兴趣迁移到了设计良好的架构上，使得模型变得越来越复杂。最近的一些架构是基于自动[44、29、23]或手动[28]架构搜索，或搜索的复合缩放策略[35]。

Though many complicated ConvNets deliver higher accuracy than the simple ones, the drawbacks are significant. 1) The complicated multi-branch designs (e.g., residualaddition in ResNet and branch-concatenation in Inception) make the model difficult to implement and customize, slow down the inference and reduce the memory utilization. 2) Some components (e.g., depthwise conv in Xception [3] and MobileNets [16, 30] and channel shuffle in ShuffleNets [24, 41]) increase the memory access cost and lack supports of various devices. With so many factors affecting the inference speed, the amount of floating-point operations (FLOPs) does not precisely reflect the actual speed. Though some novel models have lower FLOPs than the oldfashioned ones like VGG and ResNet-18/34/50 [12], they may not run faster (Table. 4). Consequently, VGG and the original versions of ResNets are still heavily used for realworld applications in both academia and industry.

虽然许多复杂的ConvNets比简单的ConvNetworks具有更高的精度，但缺点是显著的:
1. 复杂的多分支设计(例如，ResNet中的残差加法和Inception中的分支串联)使模型难以实现和定制，减慢了推理速度并降低了内存利用率。
2. 一些组件(例如，Xception[3]和MobileNets[16，30]中的深度转换和ShuffleNets[24，41]中的通道切换)增加了内存访问成本，并且缺乏对各种设备的支持。

由于影响推理速度的因素很多，浮点运算(FLOP)的数量不能准确反映实际速度。尽管一些新模型的FLOP低于VGG和ResNet-18/34/50等老模型[12]，但它们可能不会运行得更快(表4)。因此，VGG和ResNets的原始版本仍然大量用于学术界和工业界的现实应用程序。


In this paper, we propose RepVGG, a VGG-style architecture which outperforms many complicated models (Fig. 1). RepVGG has the following advantages.
* The model has a VGG-like plain (a.k.a. feed-forward) topology 1 without any branches, which means every layer takes the output of its only preceding layer as input and feeds the output into its only following layer.
* The model’s body uses only 3 × 3 conv and ReLU.
* The concrete architecture (including the specific depth and layer widths) is instantiated with no automatic search [44], manual refinement [28], compound scaling [35], nor other heavy designs.

在本文中，我们提出了RepVGG，一种VGG风格的架构，其性能优于许多复杂的模型(图1)。RepVGG具有以下优点:
* 模型具有类似VGG的平面(即前馈)拓扑1，没有任何分支，这意味着每一层都将其前一层的输出作为输入，并将输出馈送到其后一层。
* 模型主体仅使用3×3 conv和ReLU。
* 具体结构(包括特定深度和层宽度)的实例化，不依赖自动搜索[44]、手动优化[28]、复合缩放[35]，也无需其他重型设计。

It is challenging for a plain model to reach a comparable level of performance as the multi-branch architectures. An explanation is that a multi-branch topology, e.g., ResNet, makes the model an implicit ensemble of numerous shallower models [36], so that training a multi-branch model avoids the gradient vanishing problem.

对于普通模型来说，要达到与多分支架构相当的性能水平是一个挑战。一种解释是，多分支拓扑(例如ResNet)使模型成为许多较浅模型的隐式集成[36]，因此训练多分支模型可以避免梯度消失问题。

Since the benefits of multi-branch architecture are all for training and the drawbacks are undesired for inference, we propose to decouple the training-time multibranch and inference-time plain architecture via structural re-parameterization, which means converting the architecture from one to another via transforming its parameters. To be specific, a network structure is coupled with a set of parameters, e.g., a conv layer is represented by a 4th-order kernel tensor. If the parameters of a certain structure can be converted into another set of parameters coupled by another structure, we can equivalently replace the former with the latter, so that the overall network architecture is changed.

由于多分支架构的优点只是为了训练，不用考虑推理陈本，因此我们建议通过结构重新参数化将训练时的多分支架构和推理时普通结构解耦，这意味着通过转换参数的方式将架构从一种结构转换为另外一种。具体来说，网络结构与一组参数耦合，例如，conv层由四阶核张量表示。如果某个结构的参数可以转换为另一个结构对应的另一组参数，我们可以用后者等效地替换前者，从而改变整个网络结构。

Specifically, we construct the training-time RepVGG using identity and 1×1 branches, which is inspired by ResNet but in a different way that the branches can be removed by structural re-parameterization (Fig. 2,4). After training, we perform the transformation with simple algebra, as an identity branch can be regarded as a degraded 1×1 conv, and the latter can be further regarded as a degraded 3 × 3 conv, so that we can construct a single 3 × 3 kernel with the trained parameters of the original 3 × 3 kernel, identity and 1 × 1 branches and batch normalization (BN) [19] layers. Consequently, the transformed model has a stack of 3 × 3 conv layers, which is saved for test and deployment.

具体来说，我们使用标识和1×1分支构造了训练时RepVGG，这是受ResNet启发的，但可以通过结构重新参数化以不同的方式删除分支(图2,4)。经过训练后，我们用简单代数进行变换，因为一个单位分支可以看作是一个退化的1×1 conv，而后者可以进一步看作是退化的3×3 conv，因此我们可以用原3×3核、单位分支和1×1分支以及批处理归一化(BN)[19]层的训练参数构造单个3×3内核。因此，转换后的模型有一个由3×3个conv层组成的堆栈，可以保存以供测试和部署。

Figure 2: Sketch of RepVGG architecture. RepVGG has 5 stages and conducts down-sampling via stride-2 convolution at the beginning of a stage. Here we only show the first 4 layers of a specific stage. As inspired by ResNet [12], we also use identity and 1 × 1 branches, but only for training.
RepVGG架构草图。RepVGG有5个阶段，在开始时通过2步长的卷积进行向下采样。这里我们只展示了特定阶段的前4层。受ResNet[12]启发，我们也使用标识和1×1分支，但仅用于训练。

Figure 4: Structural re-parameterization of a RepVGG block. For the ease of visualization, we assume C2 = C1 = 2, thus the 3 × 3 layer has four 3 × 3 matrices and the kernel of 1 × 1 layer is a 2 × 2 matrix.
RepVGG块的结构重新参数化。为了便于可视化，我们假设C2=C1=2，因此3×3层有四个3×3矩阵，1×1层的核心是一个2×2矩阵。

Notably, the body of an inference-time RepVGG only has one single type of operator: 3 × 3 conv followed by ReLU, which makes RepVGG fast on generic computing devices like GPUs. Even better, RepVGG allows for specialized hardware to achieve even higher speed because given the chip size and power consumption, the fewer types of operators we require, the more computing units we can integrate onto the chip. Consequently, an inference chip specialized for RepVGG can have an enormous number of 3×3-ReLU units and fewer memory units (because the plain topology is memory-economical, as shown in Fig. 3). Our contributions are summarized as follows.
* We propose RepVGG, a simple architecture with favorable speed-accuracy trade-off compared to the state-of-the-arts.
* We propose to use structural re-parameterization to decouple a training-time multi-branch topology with an inference-time plain architecture.
* We show the effectiveness of RepVGG in image classification and semantic segmentation, and the efficiency and ease of implementation.

值得注意的是，推理时的RepVGG主题只有一种类型的运算符：3×3卷积后跟ReLU，这使得RepVGGG在GPU等通用计算设备上运行速度更快。更好的是，RepVGG允许专用硬件加速，因为考虑到芯片尺寸和功耗，我们需要的操作员类型越少，我们可以集成到芯片上的计算单元就越多。因此，专门用于RepVGG的推理芯片可以有大量的3×3-ReLU单元和更少的内存单元(因为普通拓扑更节省内存，如图3所示)。我们的贡献总结如下:
* 我们建议使用RepVGG，这是一种简单的架构，与现有技术相比，具有良好的速度和精度权衡。
* 我们建议使用结构重新参数化，将训练时多分支拓扑与推理时普通架构解耦。
* 我们展示了RepVGG在图像分类和语义分割方面的有效性，以及实现的效率和易用性。

## 2. Related Work
### 2.1. From Single-path to Multi-branch 从单路径到多分支
After VGG [31] raised the top-1 accuracy of ImageNet classification to above 70%, there have been many innovations in making ConvNets complicated for high performance, e.g., the contemporary GoogLeNet [33] and later Inception models [34, 32, 19] adopted elaborately designed multi-branch architectures, ResNet [12] proposed a simplified two-branch architecture, and DenseNet [17] made the topology more complicated by connecting lower-level layers with numerous higher-level ones. Neural architecture search (NAS) [44, 29, 23, 35] and manual designing space design [28] can generate ConvNets with higher performance but at the costs of vast computing resources or manpower. Some large versions of NAS-generated models are even not trainable on ordinary GPUs, hence limiting the applications. Except for the inconvenience of implementation, the complicated models may reduce the degree of parallelism [24] hence slow down the inference.

在VGG[31]将ImageNet分类的前1精度提高到70%以上之后，让ConvNets复杂化以实现高性能方面有了许多创新，例如，当代的GoogLeNet[33]和后来的Inception模型[34、32、19]采用了精心设计的多分支架构，ResNet[12]提出了简化的两分支架构，而DenseNet[17]通过将低层与许多高层连接起来，使拓扑结构更加复杂。神经架构搜索(NAS)[44、29、23、35]和手动设计空间设计[28]可以生成性能更高的ConvNets，但需要耗费大量计算资源或人力。NAS生成模型的某些大型版本甚至无法在普通GPU上训练，因此限制了应用程序。除了实现的不便之外，复杂的模型可能会降低并行度[24]，从而减慢推理速度。

### 2.2. Effective Training of Single-path Models
There have been some attempts to train ConvNets without branches. However, the prior works mainly sought to make the very deep models converge with reasonable accuracy, but not achieve better performance than the complicated models. Consequently, the methods and resultant models were neither simple nor practical. An initialization method [37] was proposed to train extremely deep plain ConvNets. With a mean-field-theory-based scheme, 10,000-layer networks were trained over 99% accuracy on MNIST and 82% on CIFAR-10. Though the models were not practical (even LeNet-5 [21] can reach 99.3% accuracy on MNIST and VGG-16 can reach above 93% on CIFAR10), the theoretical contributions were insightful. A recent work [25] combined several techniques including Leaky ReLU, max-norm and careful initialization. On ImageNet, it showed that a plain ConvNet with 147M parameters could reach 74.6% top-1 accuracy, which was 2% lower than its reported baseline (ResNet-101, 76.6%, 45M parameters).

有人试图在没有分支结构的情况下训练ConvNets。然而，以往的工作主要是试图使非常深的模型以合理的精度收敛，但没有达到比复杂模型更好的性能。因此，这些方法和结果模型既不简单也不实用。一种初始化方法[37]被提出用来训练极深的普通ConvNets。使用基于平均场(mean-field-theory-based)理论的方案，在MNIST和CIFAR-10上分别对10000层网络进行了99%和82%以上的精度训练。尽管这些模型不实用(即使LeNet-5[21]在MNIST上也能达到99.3%的精度，而VGG-16在CIFAR10上也能达到93%以上)，但理论上的贡献很有见地。最近的一项工作[25]结合了多种技术，包括Leaky-ReLU、max-norm和仔细初始化。在ImageNet上，它显示具有147M参数的普通ConvNet可以达到74.6%的top-1精度，比其报告的基线(ResNet-101，76.6%，45M参数)低2%。

Notably, this paper is not merely a demonstration that plain models can converge reasonably well, and does not intend to train extremely deep ConvNets like ResNets. Rather, we aim to build a simple model with reasonable depth and favorable accuracy-speed trade-off, which can be simply implemented with the most common components (e.g., regular conv and BN) and simple algebra.

值得注意的是，本文不仅证明了普通模型可以很好地收敛，而且不打算训练像ResNets这样的非常深入的ConvNets。相反，我们的目标是建立一个具有合理深度和良好精度速度权衡的简单模型，它可以用最常见的组件(例如正则conv和BN)和简单的代数实现。

### 2.3. Model Re-parameterization
DiracNet [39] is a re-parameterization method related to ours. It builds deep plain models by encoding the kernel of a conv layer as Wˆ = diag(a)I + diag(b)Wnorm, where Wˆ is the eventual weight used for convolution (a 4thorder tensor viewed as a matrix), a and b are learned vectors, and Wnorm is the normalized learnable kernel. Compared to ResNets with comparable amount of parameters, the top-1 accuracy of DiracNet is 2.29% lower on CIFAR100 (78.46% vs. 80.75%) and 0.62% lower on ImageNet (72.21% of DiracNet-34 vs. 72.83% of ResNet-34). DiracNet differs from our method in two aspects. 1) The trainingtime behavior of RepVGG is implemented by the actual dataflow through a concrete structure which can be later converted into another, while DiracNet merely uses another mathematical expression of conv kernels for easier optimization. In other words, a training-time RepVGG is a real multi-branch model, but a DiracNet is not. 2) The performance of a DiracNet is higher than a normally parameterized plain model but lower than a comparable ResNet, while RepVGG models outperform ResNets by a large margin. Asym Conv Block (ACB) [10], DO-Conv [1] and ExpandNet [11] can also be viewed as structural reparameterization in the sense that they convert a block into a conv. Compared to our method, the difference is that they are designed for component-level improvements and used as a drop-in replacement for conv layers in any architecture, while our structural re-parameterization is critical for training plain ConvNets, as shown in Sect. 4.2.

DiracNet[39]是一种与我们相关的重新参数化方法。它通过将卷积层的内核编码为 Wˆ=diag(a)I+diag(b)Wnorm 来构建深度普通模型，其中Wˆ是用于卷积的最终权重(一个4th order张量被视为矩阵)，a和b是学习向量，Wnormal是归一化的可学习核。与具有可比参数量的ResNets相比，在CIFAR100上DiracNet的前1位精度低2.29%(78.46%比80.75%)，在ImageNet上低0.62%(DiracNet-34的72.21%比ResNet-34中的72.83%)。DiracNet与我们的方法在两个方面不同:
1. RepVGG的训练时是通过实际数据流通过一个具体的结构来实现的，该结构可以稍后转换为另一个结构，而DiracNet仅使用另一个conv内核的数学表达式来简化优化。换句话说，训练过程中RepVGG是一个真正的多分支模型，但DiracNet不是。
2. DiracNet的性能高于通常参数化的普通模型，但低于可比较的ResNet，而RepVGG模型的性能大大优于ResNets。

Asymly Conv Block(ACB)[10]、DO Conv[1]和ExpandNet[11]也可以被视为结构重新参数化，因为它们将块转换为Conv。与我们的方法相比，不同之处在于，它们是为组件级改进而设计的，并且在任何架构中用作Conv层的替代品，而我们的结构重新参数化对于训练普通卷积网络至关重要，如第4.2.节所示。

### 2.4. Winograd Convolution
RepVGG uses only 3 × 3 conv because it is highly optimized by some modern computing libraries like NVIDIA cuDNN [2] and Intel MKL [18] on GPU and CPU. Table. 1 shows the theoretical FLOPs, actual running time and computational density (measured in Tera FLoating-point Operations Per Second, TFLOPS) 2 tested with cuDNN 7.5.0 on a 1080Ti GPU. The theoretical computational density of 3 × 3 conv is around 4× as the others, suggesting the total theoretical FLOPs is not a comparable proxy for the actual speed among different architectures. Winograd [20] is a classic algorithm for accelerating 3 × 3 conv (only if the stride is 1), which has been well supported (and enabled by default) by libraries like cuDNN and MKL. For example, with the standard F (2 × 2, 3 × 3) Winograd, the amount of multiplications (MULs) of a 3 × 3 conv is reduced to 49 of the original. Since the multiplications are much more time-consuming than additions, we count the MULs to measure the computational costs with Winograd support (denoted by Wino MULs in Table. 4, 5). Note that the specific computing library and hardware determine whether to use Winograd for each operator because small-scale convolutions may not be accelerated due to the memory overhead. 3

RepVGG只使用3×3卷积，因为它由GPU和CPU上的一些现代计算库(如NVIDIA cuDNN[2]和Intel MKL[18])进行了高度优化。表1显示了在1080Ti GPU上使用cuDNN 7.5.0测试的理论FLOP、实际运行时间和计算密度(以每秒Tera FLoating point Operations Per Second，TFLOPS为单位测量)2。3×3 conv的理论计算密度约为其他的4倍，这表明理论上的总FLOP不能代表不同架构之间的实际速度。Winograd[20]是一种用于加速3×3 conv(仅当步幅为1时)的经典算法，cuDNN和MKL等库很好地支持(默认启用)该算法。例如，使用标准F(2×2，3×3)Winograd，3×3conv的乘法量(MUL)减少到原来的49。由于乘法比加法更耗时，我们计算MUL以测量Winograd支持下的计算成本(在表4、5中用Wino MUL表示)。请注意，特定的计算库和硬件决定是否对每个运算符使用Winograd，因为由于内存开销，小规模卷积可能不会加速。3.

## 3. Building RepVGG via Structural Re-param 
### 3.1. Simple is Fast, Memory-economical, Flexible
There are at least three reasons for using simple ConvNets: they are fast, memory-economical and Flexible.

使用简单ConvNets至少有三个原因：它们速度快、节省内存和灵活。

Fast, Many recent multi-branch architectures have lower theoretical FLOPs than VGG but may not run faster. For example, VGG-16 has 8.4× FLOPs as EfficientNet-B3 [35] but runs 1.8× faster on 1080Ti (Table. 4), which means the computational density of the former is 15× as the latter. Except for the acceleration brought by Winograd conv, the discrepancy between FLOPs and speed can be attributed to two important factors that have considerable affection on speed but are not taken into account by FLOPs: the memory access cost (MAC) and degree of parallelism [24]. For example, though the required computations of branch addition or concatenation are negligible, the MAC is significant. Moreover, MAC constitutes a large portion of time usage in groupwise convolution. On the other hand, a model with high degree of parallelism could be much faster than another one with low degree of parallelism, under the same FLOPs. As multi-branch topology is widely adopted in Inception and auto-generated architectures, multiple small operators are used instead of a few large ones. A prior work [24] reported that the number of fragmented operators (i.e. the number of individual conv or pooling operations in one building block) in NASNET-A [43] is 13, which is unfriendly to devices with strong parallel computing powers like GPU and introduces extra overheads such as kernel launching and synchronization. In contrast, this number is 2 or 3 in ResNets, and we make it 1: a single conv.

快速. 许多最近的多分支架构的理论FLOP低于VGG，但运行速度可能不会更快。例如，VGG-16有8.4×FLOP作为EfficientNet-B3[35]，但在1080Ti上运行速度快1.8×(表4)，这意味着前者的计算密度是后者的15×。除了Winograd conv带来的加速外，FLOP和速度之间的差异可归因于两个重要因素，这两个因素对速度有很大影响，但FLOP没有考虑到：内存访问成本(MAC)和并行度[24]。例如，尽管所需的分支加法或串联计算可以忽略不计，但MAC是重要的。此外，MAC在分组卷积的时间使用中占很大比例。另一方面，在相同的FLOP下，具有高并行度的模型可能比具有低并行度的另一个模型快得多。由于多分支拓扑在初始和自动生成架构中被广泛采用，因此使用了多个小操作符，而不是几个大操作符。先前的一项工作[24]报告称，NASNET-A[43]中分散的运算符的数量(即单个构造块中的单个conv或pooling操作的数量)为13，这对具有强大并行计算能力的设备(如GPU)不友好，并引入了额外的开销，如内核启动和同步。相比之下，ResNets中的这个数字是2或3，我们将其设置为1：单个conv。

Memory-economical, The multi-branch topology is memory-inefficient because the results of every branch need to be kept until the addition or concatenation, significantly raising the peak value of memory occupation. Fig. 3 shows that the input to a residual block need to be kept until the addition. Assuming the block maintains the feature map size, the peak value of extra memory occupation is 2× as the input. In contrast, a plain topology allows the memory occupied by the inputs to a specific layer to be immediately released when the operation is finished. When designing specialized hardware, a plain ConvNet allows deep memory optimizations and reduces the costs of memory units so that we can integrate more computing units onto the chip.

内存经济. 多分支拓扑是内存效率低下的，因为每个分支的结果都需要保留，直到add或concat操作，从而显著提高内存占用的峰值。图3显示，在add操作前，需要保留残差块的输入。假设块保持特征图大小，额外内存占用的峰值为2×作为输入。相比之下，普通拓扑允许在操作完成时立即释放特定层的输入所占用的内存。在设计专用硬件时，普通ConvNet允许深度内存优化，并降低内存单元的成本，以便我们可以将更多计算单元集成到芯片上。

Figure 3: Peak memory occupation in residual and plain model. If the residual block maintains the size of feature map, the peak value of extra memory occupied by feature maps will be 2× as the input. The memory occupied by the parameters is small compared to the features hence ignored.
剩余和普通模型中的峰值内存占用。如果残差块保持特征图的大小，则特征图占用的额外内存的峰值将为2×作为输入。与因此忽略的特征相比，参数占用的内存较小。

Flexible, The multi-branch topology imposes constraints on the architectural specification. For example, ResNet requires the conv layers to be organized as residual blocks, which limits the flexibility because the last conv layers of every residual block have to produce tensors of the same shape, or the shortcut addition will not make sense. Even worse, multi-branch topology limits the application of channel pruning [22, 14], which is a practical technique to remove some unimportant channels, and some methods can optimize the model structure by automatically discovering the appropriate width of each layer [8]. However, multibranch models make pruning tricky and result in significant performance degradation or low acceleration ratio [7, 22, 9]. In contrast, a plain architecture allows us to freely configure every conv layer according to our requirements and prune to obtain a better performance-efficiency trade-off.

灵活. 多分支拓扑对架构规范施加了约束。例如，ResNet要求将conv层组织为残差块，这限制了灵活性，因为每个残差块的最后conv层必须产生相同形状的张量，否则快捷方式的add就没有意义。更糟糕的是，多分支拓扑限制了通道修剪的应用[22，14]，这是一种删除一些不重要通道的实用技术，有些方法可以通过自动发现每个层的适当宽度来优化模型结构[8]。然而，多分支模型使修剪变得棘手，导致性能显著下降或加速比低[7,22,9]。相比之下，简单的架构允许我们根据需求自由配置每个conv层，并进行修剪，以获得更好的性能效率权衡。

### 3.2. Training-time Multi-branch Architecture
Plain ConvNets have many strengths but one fatal weakness: the poor performance. For example, with modern components like BN [19], a VGG-16 can reach over 72% top-1 accuracy on ImageNet, which seems outdated. Our structural re-parameterization method is inspired by ResNet, which explicitly constructs a shortcut branch to model the information flow as y = x + f(x) and uses a residual block to learn f. When the dimensions of x and f(x) do not match, it becomes y = g(x)+f (x), where g(x) is a convolutional shortcut implemented by a 1×1 conv. An explanation for the success of ResNets is that such a multibranch architecture makes the model an implicit ensemble of numerous shallower models [36]. Specifically, with n blocks, the model can be interpreted as an ensemble of 2n models, since every block branches the flow into two paths.

普通ConvNets有很多优点，但有一个致命的缺点：性能差。例如，使用BN[19]等现代组件，VGG-16在ImageNet上可以达到72%以上的前1精度，这似乎已经过时了。我们的结构重新参数化方法受到ResNet的启发，ResNet明确构造了一个快捷分支，以将信息流建模为y=x+f(x)，并使用一个残差块来学习f。当x和f(x)维度不匹配时，它变为 y=g(x)+f(x)，其中g(x)是由1×1 conv实现的卷积捷径。 ResNets成功的一个解释是，这种多分支架构使模型成为众多浅层模型的隐式集成[36]。具体来说，对于n个块，模型可以解释为2n个模型的集合，因为每个块将流分支为两条路径。

Since the multi-branch topology has drawbacks for inference but the branches seem beneficial to training [36], we use multiple branches to make an only-training-time ensemble of numerous models. To make most of the members shallower or simpler, we use ResNet-like identity (only if the dimensions match) and 1 × 1 branches so that the training-time information flow of a building block is y = x + g(x) + f(x). We simply stack several such blocks to construct the training-time model. From the same perspective as [36], the model becomes an ensemble of 3n members with n such blocks.

由于多分支拓扑在推理方面存在缺陷，但分支似乎有利于训练[36]，因此我们使用多个分支来制作众多模型的唯一训练时间集成。为了使大多数成员更浅或更简单，我们使用类ResNet标识(仅当维度匹配时)和1×1分支，以便构建块的训练时信息流为y=x+g(x)+f(x)。我们只需堆叠几个这样的块来构建训练时间模型。从与[36]相同的角度来看，该模型是由3n个成员和n个这样的块组成的集合。

### 3.3. Re-param for Plain Inference-time Model
In this subsection, we describe how to convert a trained block into a single 3 × 3 conv layer for inference. Note that we use BN in each branch before the addition (Fig. 4). Formally, we use W(3) ∈ RC2 ×C1 ×3×3 to denote the kernel of a 3 × 3 conv layer with C1 input channels and C2 output channels, and W(1) ∈ RC2×C1 for the kernel of 1 × 1 branch. We use μ(3),σ(3),γ(3),β(3) as the accumulated mean, standard deviation and learned scaling factor and bias of the BN layer following 3 × 3 conv, μ(1), σ(1), γ(1), β(1) for the BN following 1 × 1 conv, and μ(0), σ(0), γ(0), β(0) for the identity branch. Let M(1) ∈ RN ×C1 ×H1 ×W1 , M(2) ∈ RN ×C2 ×H2 ×W2 be the input and output, respectively, and ∗ be the convolution operator. If C1 = C2 , H1 = H2,W1 =W2,we have 

M(2) = bn(M(1) ∗ W(3), μ(3), σ(3), γ(3), β(3))
+ bn(M(1) ∗ W(1), μ(1), σ(1), γ(1), β(1)) (1)
+ bn(M(1), μ(0), σ(0), γ(0), β(0)) .

在本小节中，我们描述了如何将经过训练的块转换为单个3×3 conv层进行推理。注意，我们在添加之前在每个分支中使用BN(图4)。形式上，我们使用W(3)∈ RC2×C1×3×3表示带C1输入通道和C2输出通道的3×3 conv层的内核，以及W(1)∈ RC2×C1用于1×1分支的内核。我们使用μ(3)，σ(3)、γ(3)和β(3。设M(1)∈ RN×C1×H1×W1，M(2)∈ RN×C2×H2×W2分别为输入和输出∗ 是卷积算子。如果C1=C2，H1=H2，W1=W2，我们有 ...

Otherwise, we simply use no identity branch, hence the above equation only has the first two terms. Here bn is the inference-time BN function, formally, ∀1 ≤ i ≤ C2,

bn(M, μ, σ, γ, β):,i,:,: = (M:,i,:,: − μi) γi + βi . (2) σi

否则，我们只使用无恒等式分支，因此上述方程只有前两项。这里bn是推理时的bn函数，形式上，∀1.≤ 我≤ 指挥与控制，

We first convert every BN and its preceding conv layer into a conv with a bias vector. Let {W′,b′} be the kernel and bias converted from {W, μ, σ, γ, β}, we have

W′ = γiW , b′ =−μiγi +β . (3)

我们首先将每个BN及其前一个conv层转换为一个带有偏置矢量的conv。设{W′，b′}为核，由{W，μ，σ，γ，β}转换为偏压，我们得到

Then it is easy to verify that ∀1 ≤ i ≤ C2,

bn(M ∗ W, μ, σ, γ, β):,i,:,: = (M ∗ W′):,i,:,: + b′i .
(4)

This transformation also applies to the identity branch because an identity can be viewed as a 1 × 1 conv with an identity matrix as the kernel. After such transformations, we will have one 3 × 3 kernel, two 1 × 1 kernels, and three bias vectors. Then we obtain the final bias by adding up the three bias vectors, and the final 3 × 3 kernel by adding the 1×1 kernels onto the central point of 3×3 kernel, which can be easily implemented by first zero-padding the two 1 × 1 kernels to 3 × 3 and adding the three kernels up, as shown in Fig. 4. Note that the equivalence of such transformations requires the 3 × 3 and 1 × 1 layer to have the same stride, and the padding configuration of the latter shall be one pixel less than the former. For example, for a 3 × 3 layer that pads the input by one pixel, which is the most common case, the 1 × 1 layer should have padding = 0 

这种转换也适用于标识分支，因为标识可以被视为以标识矩阵为核心的1×1 conv。经过这样的变换，我们将有一个3×3核、两个1×1核和三个偏置向量。然后我们通过将三个偏差向量相加得到最终偏差，并通过将1×1内核添加到3×3内核的中心点来获得最终的3×3核，这可以很容易地通过先将两个1×1核填零为3×3并将三个内核相加来实现，如图4所示。注意，这种变换的等效性要求3×3和1×1层具有相同的步幅，后者的填充配置应比前者少一个像素。例如，对于将输入填充一个像素的3×3层(这是最常见的情况)，1×1层的填充应为0

### 3.4. Architectural Specification 架构说明
Table. 2 shows the specification of RepVGG including the depth and width. RepVGG is VGG-style in the sense that it adopts a plain topology and heavily uses 3 × 3 conv, but it does not use max pooling like VGG because we desire the body to have only one type of operator. We arrange the 3×3 layers into 5 stages, and the first layer of a stage downsamples with the stride = 2. For image classification, we use global average pooling followed by a fully-connected layer as the head. For other tasks, the task-specific heads can be used on the features produced by any layer.

表-2 显示了RepVGG的规格，包括深度和宽度。RepVGG是VGG风格的，因为它采用普通拓扑，大量使用3×3 conv，但它不像VGG那样使用最大池，因为我们希望主体只有一种类型的运算符。我们将3×3层分为5个阶段，阶段的第一层向下采样，步幅=2。对于图像分类，我们使用全局平均池，然后使用完全连接的层作为头部。对于其他任务，可以在任何图层生成的要素上使用特定于任务的标头。

Table 2: Architectural specification of RepVGG. Here 2 × 64a means stage2 has 2 layers each with 64a channels.

We decide the numbers of layers of each stage following three simple guidelines. 1) The first stage operates with large resolution, which is time-consuming, so we use only one layer for lower latency. 2) The last stage shall have more channels, so we use only one layer to save the parameters. 3) We put the most layers into the second last stage (with 14 × 14 output resolution on ImageNet), following ResNet and its recent variants [12, 28, 38] (e.g., ResNet-101 uses 69 layers in its 14 × 14-resolution stage). We let the five stages have 1, 2, 4, 14, 1 layers respectively to construct an instance named RepVGG-A. We also build a deeper RepVGG-B, which has 2 more layers in stage2, 3 and 4. We use RepVGG-A to compete against other lightweight and middleweight models including ResNet-18/34/50, and RepVGG-B against the high-performance ones.

我们根据三个简单的准则来决定每个阶段的层数。
1. 第一阶段以大分辨率运行，这很耗时，因此我们只使用一层来降低延迟。
2. 最后一个阶段应该有更多的通道，所以我们只使用一层来保存参数。
3. 我们将最多的层放入第二个最后阶段(ImageNet上的输出分辨率为14×14)，紧随其后的是ResNet及其最新变体[12、28、38](例如，ResNet-101在其14×14分辨率阶段使用了69层)。

我们让这五个阶段分别有1、2、4、14、1层来构造一个名为RepVGG-A的实例。我们还构建了一个更深的RepVGG-B，它在第2、3和4阶段又有2层。我们使用RepVGG-a与其他轻量级和中量级模型(包括ResNet-18/34/50)竞争，并与高性能模型竞争。

We determine the layer width by uniformly scaling the classic width setting of [64, 128, 256, 512] (e.g., VGG and ResNets). We use multiplier a to scale the first four stages and b for the last stage, and usually set b > a because we desire the last layer to have richer features for the classification or other down-stream tasks. Since RepVGG has only one layer in the last stage, a larger b does not significantly increase the latency nor the amount of parameters. Specifically, the width of stage2, 3, 4, 5 is [64a, 128a, 256a, 512b], respectively. To avoid large-scale conv on high-resolution feature maps, we scale down stage1 if a < 1 but do not scale it up, so that the width of stage1 is min(64, 64a).

我们通过均匀缩放经典宽度设置[64、128、256、512](例如VGG和ResNets)来确定层宽度。我们使用乘数a缩放前四个阶段，b缩放最后一个阶段，通常设置b>a，因为我们希望最后一层具有更丰富的特征，用于分类或其他下游任务。由于RepVGG在最后一个阶段只有一层，因此较大的b不会显著增加延迟或参数量。具体来说，阶段2、3、4、5的宽度分别为[64a、128a、256a、512b]。为了避免在高分辨率特征图上进行大规模转换，如果a<1，我们缩小阶段1，但不放大，因此阶段1的宽度为最小值(64，64a)。

To further reduce the parameters and computations, we may optionally interleave groupwise 3 × 3 conv layers with dense ones to trade accuracy for efficiency. Specifically, we set the number of groups g for the 3rd, 5th, 7th, ..., 21st layer of RepVGG-A and the additional 23rd, 25th and 27th layers of RepVGG-B. For the simplicity, we set g as 1, 2, or 4 globally for such layers without layer-wise tuning. We do not use adjacent groupwise conv layers because that would disable the inter-channel information exchange and bring a side effect [41]: outputs from a certain channel would be derived from only a small fraction of input channels. Note that the 1×1 branch shall have the same g as the 3×3 conv.

为了进一步减少参数和计算，我们可以选择将密集的3×3 conv层与成组的层数交错，以牺牲效率的准确性。具体来说，我们为RepVGG-A的第3、5、7、…、21层以及RepVGG-B的额外第23、25和27层设置了组g的数量。为了简单起见，我们将此类层的g全局设置为1、2或4，而不进行逐层调整。我们不使用相邻的逐组conv层，因为这将禁用通道间信息交换并带来副作用[41]：某个通道的输出将仅来自一小部分输入通道。注意，1×1支管的g应与3×3转换器的g相同。

## 4. Experiments
We compare RepVGG with the baselines on ImageNet, justify the significance of structural re-parameterization by a series of ablation studies and comparisons, and verify the generalization performance on semantic segmentation [42].

我们将RepVGG与ImageNet上的基线进行了比较，通过一系列消融研究和比较证明了结构重新参数化的重要性，并验证了语义分割的泛化性能[42]。

### 4.1. RepVGG for ImageNet Classification
We compare RepVGG with the classic and state-of-theart models including VGG-16 [31], ResNet [12], ResNeXt [38], EfficientNet [35], and RegNet [28] on ImageNet-1K [6], which comprises 1.28M images for training and 50K for validation. We use EfficientNet-B0/B3 and RegNet3.2GF/12GF as the representatives for middleweight and heavyweight state-of-the-art models, respectively. We vary the multipliers a and b to generate a series of RepVGG models to compare against the baselines (Table. 3).

我们将RepVGG与ImageNet-1K[6]上的经典和SOTA模型进行了比较，包括VGG-16[31]、ResNet[12]、ResNeXt[38]、EfficientNet[35]和RegNet[28]，其中包括1.28M个用于培训的图像和50K个用于验证的图像。我们分别使用EfficientNet-B0/B3和RegNet3.2GF/12GF作为中量级和权重级最先进车型的代表。我们改变乘数a和b，以生成一系列RepVGG模型，与基线进行比较(表3)。

We first compare RepVGG against ResNets [12], which are the most common benchmarks. We use RepVGGA0/A1/A2 for the comparisons with ResNet-18/34/50, respectively. To compare against the larger models, we construct the deeper RepVGG-B0/B1/B2/B3 with increasing width. For those RepVGG models with interleaved groupwise layers, we postfix g2/g4 to the model name.

我们首先比较RepVGG和ResNets[12]，后者是最常见的基准。我们使用RepVGGA0/A1/A2分别与ResNet-18/34/50进行比较。为了与较大的模型进行比较，我们构造了更深的RepVGG-B0/B1/B2/B3，宽度越大。对于那些具有交错组层的RepVGG模型，我们将g2/g4后缀到模型名称。

For training the lightweight and middleweight models, we only use the simple data augmentation pipeline including random cropping and left-right flipping, following the official PyTorch example [27]. We use a global batch size of 256 on 8 GPUs, a learning rate initialized as 0.1 and cosine annealing for 120 epochs, standard SGD with momentum coefficient of 0.9 and weight decay of 10−4 on the kernels of conv and fully-connected layers. For the heavyweight models including RegNetX-12GF, EfficientNet-B3 and RepVGG-B3, we use 5-epoch warmup, cosine learning rate annealing for 200 epochs, label smoothing [34] and mixup [40] (following [13]), and a data augmentation pipeline of Autoaugment [5], random cropping and flipping. RepVGG-B2 and its g2/g4 variants are trained in both settings. We test the speed of every model with a batch size of 128 on a 1080Ti GPU 4 by first feeding 50 batches to warm the hardware up, then 50 batches with time usage recorded. For the fair comparison, we test all the models on the same GPU, and all the conv-BN sequences of the baselines are also converted into a conv with bias (Eq. 3).

为了训练轻量级和中量级模型，我们只使用简单的数据增广管道，包括随机裁剪和左右翻转，遵循PyTorch的官方样本[27]。我们在8个GPU上使用256个全局批量大小，初始学习率为0.1，余弦退火120个周期，标准SGD动量系数为0.9，权重衰减为10−在conv和全连接层的核上。对于包括RegNetX-12GF、EfficientNet-B3和RepVGG-B3在内的权重级模型，我们使用了5周期预热、200周期余弦学习率退火、标签平滑[34]和混合[40](以下为[13])，以及自动增强[5]的数据增广管道、随机裁剪和翻转。RepVGG-B2及其g2/g4变体在两种设置中都经过训练。我们在1080Ti GPU 4上测试批次大小为128的每个模型的速度，方法是先喂食50批以预热硬件，然后喂食50批次并记录时间使用情况。为了进行公平比较，我们在同一个GPU上测试所有模型，基线的所有conv BN序列也都转换为带偏差的conv(公式3)。

Table. 4 shows the favorable accuracy-speed tradeoff of RepVGG: RepVGG-A0 is 1.25% and 33% better than ResNet-18 in terms of accuracy and speed, RepVGGA1 is 0.29%/64% better than ResNet-34, RepVGG-A2 is 0.17%/83% better than ResNet-50. With interleaved groupwise layers (g2/g4), the RepVGG models are further accelerated with reasonable accuracy decrease: RepVGG-B1g4 is 0.37%/101% better than ResNet-101, and RepVGGB1g2 is impressively 2.66× as fast as ResNet-152 with the same accuracy. Though the number of parameters is not our primary concern, all the above RepVGG models are more parameter-efficient than ResNets. Compared to the classic VGG-16, RepVGG-B2 has only 58% parameters, runs 10% faster and shows 6.57% higher accuracy. Compared to the highest-accuracy (74.5%) VGG to the best of our knowledge trained with RePr [26] (a pruning-based training method), RepVGG-B2 outperforms by 4.28% in accuracy.

表-4显示了RepVGG良好的精度-速度权衡：RepVGG-A0在精度和速度方面比ResNet-18高1.25%和33%，RepVGGA1比ResNet-34高0.29%/64%，RepVGG-A2比ResNet-50高0.17%/83%。通过交错分组层(g2/g4)，RepVG模型进一步加速，精度降低合理：RepVGG-B1g4比ResNet-101高0.37%/101%，RepVGGB1g2的速度是ResNet-152的2.66倍，准确度相当。虽然参数的数量不是我们主要关心的问题，但上述所有RepVGG模型的参数效率都高于ResNets。与经典的VGG-16相比，RepVGG-B2只有58%的参数，运行速度提高了10%，精度提高了6.57%。据我们所知，与使用RePr[26](一种基于修剪的训练方法)训练的最高准确度(74.5%)VGG相比，RepVGG-B2的准确度优于4.28%。

Compared with the state-of-the-art baselines, RepVGG also shows favorable performance, considering its simplicity: RepVGG-A2 is 1.37%/59% better than EfficientNetB0, RepVGG-B1 performs 0.39% better than RegNetX3.2GF and runs slightly faster. Notably, RepVGG models reach above 80% accuracy with 200 epochs (Table. 5), which is the first time for plain models to catch up with the state-of-the-arts, to the best of our knowledge. Compared to RegNetX-12GF, RepVGG-B3 runs 31% faster, which is impressive considering that RepVGG does not require a lot of manpower to refine the design space like RegNet [28], and the architectural hyper-parameters are set casually.

与SOTA基线相比，考虑到其简单性，RepVGG也显示出良好的性能：RepVGG-A2比EfficientNetB0好1.37%/59%，RepVVGG-B1比RegNetX3.2GF好0.39%，运行速度稍快。值得注意的是，RepVGG模型在200个时期内达到了80%以上的精度(表5)，据我们所知，这是普通模型第一次赶上SOTA水平。与RegNetX-12GF相比，RepVGG-B3的运行速度快31%，这是令人印象深刻的，因为RepVGG不需要像RegNet[28]那样耗费大量人力来优化设计空间，而且架构超参数是随意设置的。

As two proxies of computational complexity, we count the theoretical FLOPs and Wino MULs as described in Sect. 2.4. For example, we found out that none of the conv in EfficientNet-B0/B3 is accelerated by Winograd algorithm. Table. 4 shows Wino MULs is a better proxy on GPU, e.g., ResNet-152 runs slower than VGG-16 with lower theoretical FLOPs but higher Wino MULs. Of course, the actual speed should always be the golden standard.

作为计算复杂性的两个代表，我们计算理论FLOP和Wino MUL，如第2.4.节所述。例如，我们发现EfficientNet-B0/B3中的conv都没有被Winograd算法加速。表4显示Wino MUL在GPU上是一个更好的智能体，例如，ResNet-152的运行速度比VGG-16慢，理论FLOP更低，但Wino MULs更高。当然，实际速度应该始终是黄金标准。

### 4.2. Structural Re-parameterization is the Key
In this subsection, we verify the significance of our structural re-parameterization technique (Table. 6). All the models are trained from scratch for 120 epochs with the same simple training settings described above. First, we conduct ablation studies by removing the identity and/or 1 × 1 branch from every block of RepVGG-B0. With both branches removed, the training-time model degrades into an ordinary plain model and only achieves 72.39% accuracy. The accuracy is lifted to 73.15% with 1 × 1 or 74.79% with identity. The accuracy of the full featured RepVGGB0 is 75.14%, which is 2.75% higher than the ordinary plain model. Seen from the inference speed of the training-time (i.e., not yet converted) models, removing the identity and 1 × 1 branches via structural re-parameterization brings significant speedup.

在本小节中，我们验证了结构重新参数化技术的重要性(表6)。所有模型都是用上述相同的简单训练设置从头开始训练120个时代的。首先，我们通过从RepVGG-B0的每个块中删除身份和/或1×1分支来进行消融研究。当两个分支都被删除时，训练时间模型退化为普通的平面模型，仅达到72.39%的精度。1×1的精度提高到73.15%，同一性提高到74.79%。全功能RepVGGB0的精度为75.14%，比普通普通模型高2.75%。从训练时间(即尚未转换的)模型的推理速度来看，通过结构重新参数化去除恒等式和1×1分支带来了显著的加速。

Then we construct a series of variants and baselines for comparison on RepVGG-B0 (Table. 7). Again, all the models are trained from scratch in 120 epochs.
* Identity w/o BN removes the BN in identity branch.
* Post-addition BN removes the BN layers in the three branches and appends a BN layer after the addition. In other words, the position of BN is changed from preaddition to post-addition.
* +ReLU in branches inserts ReLU into each branch (after BN and before addition). Since such a block cannot be converted into a single conv layer, it is of no practical use, and we merely desire to see whether more nonlinearity will bring higher performance.
* DiracNet [39] adopts a well-designed reparameterization of conv kernels, as introduced in Sect. 2.2. We use its official PyTorch code to build the layers to replace the original 3 × 3 conv.
* Trivial Re-param is a simpler re-parameterization of conv kernels by directly adding an identity kernel to the 3 × 3 kernel, which can be viewed a degraded versionofDiracNet(Wˆ =I+W[39]).
* Asymmetric Conv Block (ACB) [10] can be viewed as another form of structural re-parameterization. We compare with ACB to see whether the improvement of our structural re-parameterization is due to the component-level over-parameterization (i.e., the extra parameters making every 3 × 3 conv stronger).
* Residual Reorg builds each stage by re-organizing it in a ResNet-like manner (2 layers per block). Specifically, the resultant model has one 3×3 layer in the first and last stages and 2, 3, 8 residual blocks in stage2, 3, 4, and uses shortcuts just like ResNet-18/34.

然后，我们构建了一系列变体和基线，以便在RepVGG-B0上进行比较(表7)。同样，所有的模特都是在120个时代里从头开始训练的。
* 标识w/o BN删除标识分支中的BN。
* 添加后BN去除三个分支中的BN层，并在添加后附加一个BN层。换言之，BN的位置从前加变为后加。
* +分支中的ReLU将ReLU插入每个分支(BN之后和添加之前)。由于这样的块不能转换为单个conv层，因此没有实际用途，我们只想看看更多的非线性是否会带来更高的性能。
* DiracNet[39]采用了第。2.2.我们使用其官方PyTorch代码构建层来替换原来的3×3 conv。
* Trivial Re-param是通过直接向3×3内核添加标识内核，对conv内核进行更简单的重新参数化，可以将其视为DiracNet的降级版本(Wˆ=I+W[39])。
* 非对称转换块(ACB)[10]可被视为另一种形式的结构重新参数化。我们与ACB进行比较，以了解我们的结构重新参数化的改进是否是由于组件级别的过度参数化(即，额外的参数使每3×3转换更强)。
* 剩余重组通过以类似ResNet的方式重新组织每个阶段(每个区块2层)来构建每个阶段。具体来说，合成模型在第一阶段和最后阶段有一个3×3层，在第二阶段、第三阶段、第四阶段有2、第三、第八个残差块，并且使用的快捷方式与ResNet-18/34类似。

We reckon the superiority of structural re-param over DiractNet and Trivial Re-param lies in the fact that the former relies on the actual dataflow through a concrete structure with nonlinear behavior (BN), while the latter merely uses another mathematical expression of conv kernels. The former “re-param” means “using the params of a structure to parameterize another structure”, but the latter means “computing the params first with another set of params, then using them for other computations”. With nonlinear components like a training-time BN, the former cannot be approximated by the latter. As evidences, the accuracy is decreased by removing the BN and improved by adding ReLU. In other words, though a RepVGG block can be equivalently converted into a single conv for inference, the inferencetime equivalence does not imply the training-time equivalence, as we cannot construct a conv layer to have the same training-time behavior as a RepVGG block.

我们认为，与DiractNet和Trivial re-param相比，结构化re-param的优势在于，前者依赖于通过具有非线性行为(BN)的具体结构的实际数据流，而后者仅使用另一个conv核的数学表达式。前者的“re-param”意思是“使用一个结构的参数来参数化另一个结构”，而后者的意思是“首先使用另一组参数计算参数，然后将它们用于其他计算”。对于非线性分量，如训练时间BN，前者不能用后者近似。作为证据，去除BN会降低准确度，添加ReLU会提高准确度。换句话说，尽管RepVGG块可以等效地转换为单个conv进行推理，但推理时间等效并不意味着训练时间等效，因为我们不能构造一个conv层来具有与RepVGG块相同的训练时间行为。

The comparison with ACB suggests the success of RepVGG should not be simply attributed to the effect of over-parameterization of every component, since ACB uses more parameters but yields inferior performance. As a double check, we replace every 3 × 3 conv of ResNet-50 with a RepVGG block and train from scratch for 120 epochs. The accuracy is 76.34%, which is merely 0.03% higher than the ResNet-50 baseline, suggesting that RepVGGstyle structural re-parameterization is not a generic overparameterization technique, but a methodology critical for training powerful plain ConvNets. Compared to Residual Reorg, a real residual network with the same number of 3 × 3 conv and additional shortcuts for both training and inference, RepVGG outperforms by 0.58%, which is not surprising since RepVGG has far more branches. For example, the branches make stage4 of RepVGG an ensemble of 2 × 315 = 2.8 × 107 models [36], while the number for Residual Reorg is 28 = 256.

与ACB的比较表明，RepVGG的成功不应仅仅归因于每个组件的过度参数化的影响，因为ACB使用了更多的参数，但性能较差。作为双重检查，我们将ResNet-50的每3×3 conv替换为一个RepVGG块，并从头开始训练120个小时。精度为76.34%，仅比ResNet-50基线高0.03%，这表明RepVGG style结构重新参数化不是一种通用的过参数化技术，而是训练强大的普通ConvNets的关键方法。与具有相同数量的3×3 conv和用于训练和推理的附加捷径的真实残差网络Restival Reorg相比，RepVGG的表现优于0.58%，这并不奇怪，因为RepVGG有更多分支。例如，分支机构使RepVGG的阶段4成为2×315=2.8×107模型的集合[36]，而剩余重组的数量为28=256。

### 4.3. Semantic Segmentation 语义分割
We verify the generalization performance of ImageNet pretrained RepVGG for semantic segmentation on Cityscapes [4] (Table. 8). We use the PSPNet [42] framework, a poly learning rate policy with base of 0.01 and power of 0.9, weight decay of 10−4 and a global batch size of 16 on 8 GPUs for 40 epochs. For the fair comparison, we only change the ResNet-50/101 backbone to RepVGG-B1g2/B2 and keep other settings identical. Following the official PSPNet-50/101 [42] which uses dilated conv in the last two stages of ResNet-50/101, we also make all the 3 × 3 conv layers in the last two stages of RepVGG-B1g2/B2 dilated. However, the current inefficient implementation of 3 × 3 dilated conv (though the FLOPs is the same as 3 × 3 regular conv) slows down the inference. For the ease of comparison, we build another two PSPNets (denoted by fast) with dilation only in the last 5 layers (i.e., the last 4 layers of stage4 and the only layer of stage5), so that the PSPNets run slightly faster than the ResNet-50/101-backbone counterparts. RepVGG backbones outperform ResNet-50 and ResNet-101 by 1.71% and 1.01% respectively in mean IoU with higher speed, and RepVGG-B1g2-fast outperforms the ResNet-101 backbone by 0.37 in mIoU and runs 62% faster. Interestingly, dilation seems more effective for larger models, as using more dilated conv layers does not improve the performance compared to RepVGG-B1g2-fast, but raises the mIoU of RepVGG-B2 by 1.05% with reasonable slowdown.

我们验证了ImageNet预训练RepVGG在城市景观语义分割方面的泛化性能[4](表8)。我们使用PSPNet[42]框架，一种基于0.01和0.9次幂的多元学习率策略，权重衰减为10−4个，全局批量大小为16个，在8个GPU上运行40个周期。为了进行公平比较，我们只将ResNet-50/101主干更改为RepVGG-B1g2/B2，并保持其他设置相同。继官方PSPNet-50/101[42]在ResNet-50/101的最后两个阶段使用了扩展conv之后，我们还对RepVGG-B1g2/B2最后两个步骤中的所有3×3 conv层进行了扩展。然而，当前3×3扩展conv的低效率实现(尽管FLOP与3×3常规conv相同)减慢了推理速度。为了便于比较，我们构建了另外两个PSPNet(用fast表示)，仅在最后5层(即阶段4的最后4层和阶段5的唯一一层)进行扩展，因此PSPNet的运行速度略快于ResNet-50/101骨干网。RepVGG主干在平均IoU上以更高的速度分别比ResNet-50和ResNet-101快1.71%和1.01%，RepVGG-B1g2-fast在mIoU方面比ResNet-102主干快0.37，运行速度快62%。有趣的是，对于较大的模型而言，扩展似乎更有效，因为与RepVGG-B1g2相比，使用更多的扩展conv层并不能快速提高性能，但会在合理的减速下将RepVGG-B2的mIoU提高1.05%。

### 4.4. Limitations 局限
RepVGG models are fast, simple and practical ConvNets designed for the maximum speed on GPU and specialized hardware, less concerning the number of parameters. They are more parameter-efficient than ResNets but may be less favored than the mobile-regime models like MobileNets [16, 30, 15] and ShuffleNets [41, 24] for low-power devices.

RepVGG模型是一种快速、简单、实用的ConvNets，专为GPU和专用硬件上的最高速度而设计，较少涉及参数的数量。它们比ResNets的参数效率更高，但可能不如MobileNets[16,30,15]和ShuffleNets[41,24]等低功耗设备的移动机制模型受欢迎。

## 5. Conclusion 结论
We proposed RepVGG, a simple architecture with a stack of 3 × 3 conv and ReLU, which is especially suitable for GPU and specialized inference chips. With our structural re-parameterization method, it reaches over 80% top-1 accuracy on ImageNet and shows favorable speed-accuracy trade-off compared to the state-of-the-art models.

我们提出了RepVGG，这是一种简单的结构，具有3×3 conv和ReLU的堆栈，特别适合于GPU和专用推理芯片。通过我们的结构重新参数化方法，它在ImageNet上达到了80%以上的top-1精度，与SOTA模型相比，显示了良好的速度精度权衡。

## References
1. Jinming Cao, Yangyan Li, Mingchao Sun, Ying Chen, Dani Lischinski, Daniel Cohen-Or, Baoquan Chen, and Changhe Tu. Do-conv: Depthwise over-parameterized convolutional layer. arXiv preprint arXiv:2006.12030, 2020. 3
2. Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer. cudnn: Efficient primitives for deep learning. arXiv preprint arXiv:1410.0759, 2014. 3
3. Franc ̧ois Chollet. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE con- ference on computer vision and pattern recognition, pages 1251–1258, 2017. 1
4. Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In 2016 IEEE Conference on Computer Vision and Pattern Recog- nition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 3213–3223. IEEE Computer Society, 2016. 8
5. EkinDCubuk,BarretZoph,DandelionMane,VijayVasude- van, and Quoc V Le. Autoaugment: Learning augmentation strategies from data. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 113–123, 2019. 6, 7
6. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical im- age database. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pages 248–255. IEEE, 2009. 6
7. Xiaohan Ding, Guiguang Ding, Yuchen Guo, and Jungong Han. Centripetal sgd for pruning very deep convolutional networks with complicated structure. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recogni- tion, pages 4943–4953, 2019. 4
8. Xiaohan Ding, Guiguang Ding, Yuchen Guo, Jungong Han, and Chenggang Yan. Approximated oracle filter pruning for destructive cnn width optimization. In International Confer- ence on Machine Learning, pages 1607–1616, 2019. 4
9. Xiaohan Ding, Guiguang Ding, Jungong Han, and Sheng Tang. Auto-balanced filter pruning for efficient convolu- tional neural networks. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018. 4
10. Xiaohan Ding, Yuchen Guo, Guiguang Ding, and Jungong Han. Acnet: Strengthening the kernel skeletons for power- ful cnn via asymmetric convolution blocks. In Proceedings of the IEEE International Conference on Computer Vision, pages 1911–1920, 2019. 3, 7, 8
11. Shuxuan Guo, Jose M Alvarez, and Mathieu Salzmann. Expandnets: Linear over-parameterization to train compact convolutional networks. Advances in Neural Information Processing Systems, 33, 2020. 3
12. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed- ings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 1, 2, 5, 6
13. Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Jun- yuan Xie, and Mu Li. Bag of tricks for image classification with convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recogni- tion, pages 558–567, 2019. 6
14. Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In International Conference on Computer Vision (ICCV), volume 2, page 6, 2017. 4
15. Andrew Howard, Ruoming Pang, Hartwig Adam, Quoc V. Le, Mark Sandler, Bo Chen, Weijun Wang, Liang-Chieh Chen, Mingxing Tan, Grace Chu, Vijay Vasudevan, and Yukun Zhu. Searching for mobilenetv3. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pages 1314–1324. IEEE, 2019. 8
16. Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco An- dreetto, and Hartwig Adam. Mobilenets: Efficient convolu- tional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017. 1, 8
17. Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kil- ian Q. Weinberger. Densely connected convolutional net- works. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 2261–2269. IEEE Computer Society, 2017. 1, 2
18. Intel. Intel mkl. https://software.intel.com/ content/www/us/en/develop/tools/math- kernel-library.html, 2020. 3
19. Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal co- variate shift. In International Conference on Machine Learn- ing, pages 448–456, 2015. 1, 2, 4
20. Andrew Lavin and Scott Gray. Fast algorithms for convo- lutional neural networks. In Proceedings of the IEEE Con- ference on Computer Vision and Pattern Recognition, pages 4013–4021, 2016. 3
21. Yann LeCun, Le ́on Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recog- nition. Proceedings of the IEEE, 86(11):2278–2324, 1998. 3
22. Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. arXiv preprint arXiv:1608.08710, 2016. 4
23. Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, and Kevin Murphy. Progressive neural architecture search. In Proceedings of the European Conference on Com- puter Vision (ECCV), pages 19–34, 2018. 1, 2
24. NingningMa,XiangyuZhang,Hai-TaoZheng,andJianSun. Shufflenet v2: Practical guidelines for efficient cnn architec- ture design. In Proceedings of the European conference on computer vision (ECCV), pages 116–131, 2018. 1, 3, 4, 8
25. Oyebade K Oyedotun, Djamila Aouada, Bjo ̈rn Ottersten, et al. Going deeper with neural networks without skip con- nections. In 2020 IEEE International Conference on Image Processing (ICIP), pages 1756–1760. IEEE, 2020. 3
26. Aaditya Prakash, James A. Storer, Dinei A. F. Floreˆncio, and Cha Zhang. Repr: Improved training of convolutional fil- ters. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16- 20, 2019, pages 10666–10675. Computer Vision Foundation / IEEE, 2019. 6
27. PyTorch. Pytorch example. https://github.com/ pytorch / examples / blob / master / imagenet / main.py, 2019. 6
28. Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dolla ́r. Designing network design spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10428– 10436, 2020. 1, 2, 5, 6, 7
29. Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V Le. Regularized evolution for image classifier architecture search. In Proceedings of the aaai conference on artificial intelligence, volume 33, pages 4780–4789, 2019. 1, 2
30. Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zh- moginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recogni- tion, pages 4510–4520, 2018. 1, 8
31. Karen Simonyan and Andrew Zisserman. Very deep convo- lutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 1, 2, 6
32. Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander A Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. In Thirty-first AAAI conference on artificial intelligence, 2017. 1, 2
33. Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1–9, 2015. 1, 2
34. Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception archi- tecture for computer vision. In Proceedings of the IEEE con- ference on computer vision and pattern recognition, pages 2818–2826, 2016. 1, 2, 6
35. Mingxing Tan and Quoc V Le. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946, 2019. 1, 2, 3, 6
36. Andreas Veit, Michael J Wilber, and Serge Belongie. Resid- ual networks behave like ensembles of relatively shallow net- works. In Advances in neural information processing sys- tems, pages 550–558, 2016. 2, 4, 8
37. Lechao Xiao, Yasaman Bahri, Jascha Sohl-Dickstein, Samuel Schoenholz, and Jeffrey Pennington. Dynamical isometry and a mean field theory of cnns: How to train 10,000-layer vanilla convolutional neural networks. In In- ternational Conference on Machine Learning, pages 5393– 5402, 2018. 3
38. Saining Xie, Ross Girshick, Piotr Dolla ́r, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1492–1500,
2017. 5, 6
39. Sergey Zagoruyko and Nikos Komodakis. Diracnets: Train-
ing very deep neural networks without skip-connections.
arXiv preprint arXiv:1706.00388, 2017. 3, 7, 8
40. Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimiza-
tion. arXiv preprint arXiv:1710.09412, 2017. 6
41. Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural net- work for mobile devices. In Proceedings of the IEEE con- ference on computer vision and pattern recognition, pages
6848–6856, 2018. 1, 6, 8
42. Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang
Wang, and Jiaya Jia. Pyramid scene parsing network. In 2017 IEEE Conference on Computer Vision and Pattern Recog- nition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 6230–6239. IEEE Computer Society, 2017. 6, 8
43. Barret Zoph and Quoc V Le. Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578, 2016. 4
44. Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V Le. Learning transferable architectures for scalable image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8697–8710, 2018. 1, 2

