# ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
ShuffleNet：一种用于移动设备的高效卷积神经网络 2017-7-4 https://arxiv.org/abs/1707.01083

## 阅读笔记
* pointwise group convolution  
* channel shuffle

## Abstract
We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs). The new architecture utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy. Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1 error (absolute 7.8%) than recent MobileNet [12] on ImageNet classification task, under the computation budget of 40 MFLOPs. On an ARM-based mobile device, ShuffleNet achieves ∼13× actual speedup over AlexNet while maintaining comparable accuracy.

我们介绍了一种计算效率极高的CNN架构ShuffleNet，该架构专为计算能力非常有限的移动设备(例如，10-150个MFLOP)设计。新的架构利用了两种新的操作，逐点组卷积和信道shuffle，在保持精度的同时大大降低了计算成本。ImageNet分类和MS COCO对象检测的实验表明，ShuffleNet优于其他结构，例如，在40个MFLOP的计算预算下，在ImageNet[12]分类任务中，比最近的MobileNet[12]更低的top-1错误(绝对7.8%)。在基于ARM的移动设备上，ShuffleNet实现了∼13倍于AlexNet的实际速度，同时保持相当的准确性。

## 1. Introduction
Building deeper and larger convolutional neural networks (CNNs) is a primary trend for solving major visual recognition tasks [21, 9, 33, 5, 28, 24]. The most accurate CNNs usually have hundreds of layers and thousands of channels [9, 34, 32, 40], thus requiring computation at billions of FLOPs. This report examines the opposite extreme: pursuing the best accuracy in very limited computational budgets at tens or hundreds of MFLOPs, focusing on common mobile platforms such as drones, robots, and smartphones. Note that many existing works [16, 22, 43, 42, 38, 27] focus on pruning, compressing, or low-bit representing a “basic” network architecture. Here we aim to explore a highly efficient basic architecture specially designed for our desired computing ranges.

构建更深更大的卷积神经网络(CNN)是解决主要视觉识别任务的主要趋势[21，9，33，5，28，24]。最精确的CNN通常有数百层和数千个信道[9，34，32，40]，因此需要数十亿FLOP的计算。本报告考察了相反的极端：在数十或数百个MFLOP的有限计算预算中追求最佳精度，重点关注无人机、机器人和智能手机等常见移动平台。请注意，许多现有的工作[16，22，43，42，38，27]集中于修剪、压缩或低位表示“基本”网络架构。在这里，我们的目标是探索一种高效的基础架构，专门为我们所需的计算范围设计。

We notice that state-of-the-art basic architectures such as Xception [3] and ResNeXt [40] become less efficient in extremely small networks because of the costly dense 1 × 1 convolutions. We propose using pointwise group convolutions to reduce computation complexity of 1 × 1 convolutions. To overcome the side effects brought by group convolutions, we come up with a novel channel shuffle operation to help the information flowing across feature channels. Based on the two techniques, we build a highly efficient architecture called ShuffleNet. Compared with popular structures like [30, 9, 40], for a given computation complexity budget, our ShuffleNet allows more feature map channels, which helps to encode more information and is especially critical to the performance of very small networks.

我们注意到，最先进的基本架构，如Xception[3]和ResNeXt[40]，由于成本高昂的密集1×1卷积，在非常小的网络中变得效率较低。我们建议使用逐点组卷积来降低1×1卷积的计算复杂度。为了克服组卷积带来的副作用，我们提出了一种新的信道混洗操作，以帮助信息在特征信道上流动。基于这两种技术，我们构建了一个名为ShuffleNet的高效架构。与像[30，9，40]这样的流行结构相比，对于给定的计算复杂度预算，我们的ShuffleNet允许更多的特征映射通道，这有助于编码更多的信息，并且对非常小的网络的性能尤为关键。

We evaluate our models on the challenging ImageNet classification [4, 29] and MS COCO object detection [23] tasks. A series of controlled experiments shows the effectiveness of our design principles and the better performance over other structures. Compared with the state-of-the-art architecture MobileNet [12], ShuffleNet achieves superior performance by a significant margin, e.g. absolute 7.8% lower ImageNet top-1 error at level of 40 MFLOPs.

我们在具有挑战性的ImageNet分类[4，29]和MS COCO对象检测[23]任务上评估了我们的模型。一系列受控实验表明，我们的设计原理是有效的，并且比其他结构具有更好的性能。与最先进的架构MobileNet[12]相比，ShuffleNet以显著的优势实现了优异的性能，例如，在40个MFLOP级别下，ImageNet top 1错误绝对降低了7.8%。

We also examine the speedup on real hardware, i.e. an off-the-shelf ARM-based computing core. The ShuffleNet model achieves ∼13× actual speedup (theoretical speedup is 18×) over AlexNet [21] while maintaining comparable accuracy.

我们还研究了实际硬件上的加速，即现成的基于ARM的计算核心。ShuffleNet模型实现了∼13倍实际加速(理论加速为18倍)，同时保持可比精度。

## 2. Related Work
Efficient Model Designs The last few years have seen the success of deep neural networks in computer vision tasks [21, 36, 28], in which model designs play an important role. The increasing needs of running high quality deep neural networks on embedded devices encourage the study on efficient model designs [8]. For example, GoogLeNet [33] increases the depth of networks with much lower complexity compared to simply stacking convolution layers. SqueezeNet [14] reduces parameters and computation significantly while maintaining accuracy. ResNet [9, 10] utilizes the efficient bottleneck structure to achieve impressive performance. SENet [13] introduces an architectural unit that boosts performance at slight computation cost. Concurrent with us, a very recent work [46] employs reinforcement learning and model search to explore efficient model designs. The proposed mobile NASNet model achieves comparable performance with our counterpart ShuffleNet model (26.0% @ 564 MFLOPs vs. 26.3% @ 524 MFLOPs for ImageNet classification error). But [46] do not report results on extremely tiny models (e.g. complexity less than 150 MFLOPs), nor evaluate the actual inference time on mobile devices.

高效的模型设计过去几年，深度神经网络在计算机视觉任务中取得了成功[21，36，28]，其中模型设计发挥了重要作用。在嵌入式设备上运行高质量深度神经网络的需求不断增加，这鼓励了对高效模型设计的研究[8]。例如，与简单堆叠卷积层相比，GoogleLeNet[33]以更低的复杂度增加了网络的深度。SqueezeNet[14]在保持精度的同时显著减少了参数和计算。ResNet[9，10]利用高效的瓶颈结构实现令人印象深刻的性能。SENet[13]引入了一种架构单元，它可以以很低的计算成本提高性能。与我们同时，最近的一项工作[46]采用强化学习和模型搜索来探索有效的模型设计。所提出的移动NASNet模型实现了与我们的对应ShuffleNet模型相当的性能(对于ImageNet分类错误，在564个MFLOP时为26.0%，在524个MFLoP时为263%)。但[46]没有报告极小模型的结果(例如复杂度小于150 MFLOP)，也没有评估移动设备上的实际推断时间。

Figure 1. Channel shuffle with two stacked group convolutions. GConv stands for group convolution. a) two stacked convolution layers with the same number of groups. Each output channel only relates to the input channels within the group. No cross talk; b) input and output channels are fully related when GConv2 takes data from different groups after GConv1; c) an equivalent implementation to b) using channel shuffle. 
图1.具有两个堆叠组卷积的信道混洗。GConv代表组卷积。a) 具有相同组数的两个堆叠卷积层。每个输出通道仅与组内的输入通道相关。没有相声; b) 当GConv2从GConv1之后的不同组获取数据时，输入和输出通道完全相关; c) 与b)使用信道混洗的等效实现。

Group Convolution. The concept of group convolution, which was first introduced in AlexNet [21] for distributing the model over two GPUs, has been well demonstrated its effectiveness in ResNeXt [40]. Depthwise separable convolution proposed in Xception [3] generalizes the ideas of separable convolutions in Inception series [34, 32]. Recently, MobileNet [12] utilizes the depthwise separable convolutions and gains state-of-the-art results among lightweight models. Our work generalizes group convolution and depthwise separable convolution in a novel form.

组卷积。组卷积的概念首次在AlexNet[21]中引入，用于在两个GPU上分布模型，已在ResNeXt[40]中充分证明了其有效性。Xception[3]中提出的深度可分离卷积推广了Inception系列[34，32]中可分离卷积的思想。最近，MobileNet[12]利用深度可分离卷积，在轻量级模型中获得了最先进的结果。我们的工作以一种新的形式推广了组卷积和深度可分离卷积。

Channel Shuffle Operation. To the best of our knowledge, the idea of channel shuffle operation is rarely mentioned in previous work on efficient model design, although CNN library cuda-convnet [20] supports “random sparse convolution” layer, which is equivalent to random channel shuffle followed by a group convolutional layer. Such “random shuffle” operation has different purpose and been seldom exploited later. Very recently, another concurrent work [41] also adopt this idea for a two-stage convolution. However, [41] did not specially investigate the effectiveness of channel shuffle itself and its usage in tiny model design.

信道混洗操作。据我们所知，尽管CNN库cuda convnet[20]支持“随机稀疏卷积”层，这等同于随机信道混洗，然后是组卷积层，但在之前关于有效模型设计的工作中很少提到信道混洗操作的想法。这种“随机洗牌”操作有不同的目的，后来很少被利用。最近，另一项同时进行的工作[41]也采用了这种两阶段卷积的思想。然而，[41]没有专门研究信道混洗本身的有效性及其在微小模型设计中的应用。

Model Acceleration. This direction aims to accelerate inference while preserving accuracy of a pre-trained model. Pruning network connections [6, 7] or channels [38] reduces redundant connections in a pre-trained model while maintaining performance. Quantization [31, 27, 39, 45, 44] and factorization [22, 16, 18, 37] are proposed in literature to reduce redundancy in calculations to speed up inference. Without modifying the parameters, optimized convolution algorithms implemented by FFT [25, 35] and other methods [2] decrease time consumption in practice. Distilling [11] transfers knowledge from large models into small ones, which makes training small models easier.

模型加速。该方向旨在加速推理，同时保持预训练模型的准确性。修剪网络连接[6，7]或信道[38]减少了预训练模型中的冗余连接，同时保持了性能。文献中提出了量化[31，27，39，45，44]和因子分解[22，16，18，37]，以减少计算中的冗余，从而加快推断。在不修改参数的情况下，由FFT[25，35]和其他方法[2]实现的优化卷积算法在实践中减少了时间消耗。提取[11]将知识从大模型迁移到小模型，这使得训练小模型更容易。

## 3. Approach
### 3.1. Channel Shuffle for Group Convolutions 用于组卷积的信道混洗
Modern convolutional neural networks [30, 33, 34, 32, 9, 10] usually consist of repeated building blocks with the same structure. Among them, state-of-the-art networks such as Xception [3] and ResNeXt [40] introduce efficient depthwise separable convolutions or group convolutions into the building blocks to strike an excellent trade-off between representation capability and computational cost. However, we notice that both designs do not fully take the 1 × 1 convolutions (also called pointwise convolutions in [12]) into account, which require considerable complexity. For example, in ResNeXt [40] only 3 × 3 layers are equipped with group convolutions. As a result, for each residual unit in ResNeXt the pointwise convolutions occupy 93.4% multiplication-adds (cardinality = 32 as suggested in [40]). In tiny networks, expensive pointwise convolutions result in limited number of channels to meet the complexity constraint, which might significantly damage the accuracy.

现代卷积神经网络[30，33，34，32，9，10]通常由具有相同结构的重复构建块组成。其中，最先进的网络如Xception[3]和ResNeXt[40]将高效的深度可分离卷积或组卷积引入构建块中，以在表示能力和计算成本之间实现出色的权衡。然而，我们注意到这两种设计都没有完全考虑到1×1卷积(在[12]中也称为逐点卷积)，这需要相当大的复杂性。例如，在ResNeXt[40]中，只有3×3层配备了组卷积。因此，对于ResNeXt中的每个残差单元，逐点卷积占据93.4%的乘法相加(基数=32，如[40]所示)。在微小的网络中，昂贵的逐点卷积会导致信道数量有限，无法满足复杂度约束，这可能会严重损害精度。

To address the issue, a straightforward solution is to apply channel sparse connections, for example group convolutions, also on 1 × 1 layers. By ensuring that each convolution operates only on the corresponding input channel group, group convolution significantly reduces computation cost. However, if multiple group convolutions stack together, there is one side effect: outputs from a certain channel are only derived from a small fraction of input channels. Fig 1 (a) illustrates a situation of two stacked group convolution layers. It is clear that outputs from a certain group only relate to the inputs within the group. This property blocks information flow between channel groups and weakens representation.

为了解决这个问题，一个简单的解决方案是在1×1层上应用信道稀疏连接，例如组卷积。通过确保每个卷积仅在相应的输入信道组上操作，组卷积显著降低了计算成本。然而，如果多组卷积叠加在一起，则有一个副作用：来自某个通道的输出仅来自输入通道的一小部分。图1(a)说明了两个堆叠组卷积层的情况。很明显，某一组的输出仅与该组内的输入有关。此属性阻止信道组之间的信息流并削弱表示。

Figure 2. ShuffleNet Units. a) bottleneck unit [9] with depthwise convolution (DWConv) [3, 12]; b) ShuffleNet unit with pointwise group convolution (GConv) and channel shuffle; c) ShuffleNet unit with stride = 2. 
图2.ShuffleNet单元。a) 具有深度卷积(DWConv)的瓶颈单元[9][3，12]; b) 具有逐点组卷积(GConv)和信道混洗的ShuffleNet单元; c) 步幅为2的ShuffleNet单元。

If we allow group convolution to obtain input data from different groups (as shown in Fig 1 (b)), the input and output channels will be fully related. Specifically, for the feature map generated from the previous group layer, we can first divide the channels in each group into several subgroups, then feed each group in the next layer with different subgroups. This can be efficiently and elegantly implemented by a channel shuffle operation (Fig 1 (c)): suppose a convolutional layer with g groups whose output has g × n channels; we first reshape the output channel dimension into (g, n), transposing and then flattening it back as the input of next layer. Note that the operation still takes effect even if the two convolutions have different numbers of groups. Moreover, channel shuffle is also differentiable, which means it can be embedded into network structures for end-to-end training.

如果我们允许组卷积从不同组获得输入数据(如图1(b)所示)，输入和输出通道将完全相关。具体而言，对于从上一组层生成的特征图，我们可以首先将每组中的通道划分为几个子组，然后将下一层中的每个组与不同的子组进行馈送。这可以通过信道混洗操作高效而优雅地实现(图1(c))：假设具有g个组的卷积层，其输出具有g×n个信道; 我们首先将输出通道维度重塑为(g，n)，进行转置，然后将其平坦化，作为下一层的输入。请注意，即使两个卷积具有不同的组数，该操作仍然有效。此外，信道混洗也是可区分的，这意味着它可以嵌入到网络结构中进行端到端训练。

Channel shuffle operation makes it possible to build more powerful structures with multiple group convolutional layers. In the next subsection we will introduce an efficient network unit with channel shuffle and group convolution.

信道混洗操作使构建具有多组卷积层的更强大的结构成为可能。在下一小节中，我们将介绍一种具有信道混洗和组卷积的高效网络单元。

### 3.2. ShuffleNet Unit
Taking advantage of the channel shuffle operation, we propose a novel ShuffleNet unit specially designed for small networks. We start from the design principle of bottleneck unit [9] in Fig 2 (a). It is a residual block. In its residual branch, for the 3 × 3 layer, we apply a computational economical 3 × 3 depthwise convolution [3] on the bottleneck feature map. Then, we replace the first 1 × 1 layer with pointwise group convolution followed by a channel shuffle operation, to form a ShuffleNet unit, as shown in Fig 2 (b). The purpose of the second pointwise group convolution is to recover the channel dimension to match the shortcut path. For simplicity, we do not apply an extra channel shuffle operation after the second pointwise layer as it results in comparable scores. The usage of batch normalization (BN) [15] and nonlinearity is similar to [9, 40], except that we do not use ReLU after depthwise convolution as suggested by [3]. As for the case where ShuffleNet is applied with stride, we simply make two modifications (see Fig 2 (c)): (i) add a 3 × 3 average pooling on the shortcut path; (ii) replace the element-wise addition with channel concatenation, which makes it easy to enlarge channel dimension with little extra computation cost.

利用信道混洗操作，我们提出了一种专门为小型网络设计的新型ShuffleNet单元。我们从图2(a)中瓶颈单元[9]的设计原理开始。这是一个残留块。在其残余分支中，对于3×3层，我们在瓶颈特征图上应用了计算经济的3×3深度卷积[3]。然后，我们将第一个1×1层替换为逐点组卷积，然后进行信道混洗操作，以形成ShuffleNet单元，如图2(b)所示。第二个逐点组卷积的目的是恢复通道维度以匹配快捷路径。为了简单起见，我们不在第二个逐点层之后应用额外的通道混洗操作，因为它会导致可比较的分数。批次归一化(BN)[15]和非线性的使用类似于[9，40]，除了我们没有如[3]所建议的那样在深度卷积后使用ReLU。对于ShuffleNet使用步幅的情况，我们只做了两个修改(见图2(c))：(i)在快捷路径上添加3×3平均池; (ii)用信道级联代替逐元素加法，这使得在几乎没有额外计算成本的情况下很容易扩大信道维度。

Thanks to pointwise group convolution with channel shuffle, all components in ShuffleNet unit can be computed efficiently. Compared with ResNet [9] (bottleneck design) and ResNeXt [40], our structure has less complexity under the same settings. For example, given the input size c × h × w and the bottleneck channels m, ResNet unit requires hw(2cm + 9m2) FLOPs and ResNeXt has hw(2cm + 9m2/g) FLOPs, while our ShuffleNet unit requires only hw(2cm/g + 9m) FLOPs, where g means the number of groups for convolutions. In other words, given a computational budget, ShuffleNet can use wider feature maps. We find this is critical for small networks, as tiny networks usually have an insufficient number of channels to process the information.
 
由于逐点组卷积和信道混洗，ShuffleNet单元中的所有组件都可以有效地计算。与ResNet[9](瓶颈设计)和ResNeXt[40]相比，我们的结构在相同的设置下具有更少的复杂性。例如，给定输入大小c×h×w和瓶颈信道m，ResNet单元需要hw(2cm+9m2)FLOP，ResNeXt具有hw(2 cm+9m2/g)FLOP; 而我们的ShuffleNet单元只需要hw(2 cm/g+9m)FLOP。其中g表示卷积的组数。换句话说，给定计算预算，ShuffleNet可以使用更广泛的特征图。我们发现这对于小型网络至关重要，因为小型网络通常没有足够的通道来处理信息。

Table 1. ShuffleNet architecture. The complexity is evaluated with FLOPs, i.e. the number of floating-point multiplication-adds. Note that for Stage 2, we do not apply group convolution on the first pointwise layer because the number of input channels is relatively small.
表1.ShuffleNet架构。使用FLOP评估复杂性，即浮点乘法相加的数量。请注意，对于阶段2，我们不在第一逐点层上应用组卷积，因为输入通道的数量相对较小。

Table 2. Classification error vs. number of groups g (smaller number represents better performance) 
表2.分类误差与组数g(数量越小表示性能越好)

In addition, in ShuffleNet depthwise convolution only performs on bottleneck feature maps. Even though depthwise convolution usually has very low theoretical complexity, we find it difficult to efficiently implement on lowpower mobile devices, which may result from a worse computation/memory access ratio compared with other dense operations. Such drawback is also referred in [3], which has a runtime library based on TensorFlow [1]. In ShuffleNet units, we intentionally use depthwise convolution only on bottleneck in order to prevent overhead as much as possible.

此外，在ShuffleNet中，深度卷积仅在瓶颈特征图上执行。尽管深度卷积通常具有非常低的理论复杂度，但我们发现很难在低功率移动设备上有效地实现，这可能是由于与其他密集运算相比，计算/内存访问比更差。[3]中也提到了这种缺陷，它有一个基于TensorFlow[1]的运行时库。在ShuffleNet单元中，我们有意只在瓶颈上使用深度卷积，以尽可能避免开销。

### 3.3. Network Architecture
Built on ShuffleNet units, we present the overall ShuffleNet architecture in Table 1. The proposed network is mainly composed of a stack of ShuffleNet units grouped into three stages. The first building block in each stage is applied with stride = 2. Other hyper-parameters within a stage stay the same, and for the next stage the output channels are doubled. Similar to [9], we set the number of bottleneck channels to 1/4 of the output channels for each ShuffleNet unit. Our intent is to provide a reference design as simple as possible, although we find that further hyper-parameter tunning might generate better results.

基于ShuffleNet单元，我们在表1中展示了整个ShuffleNet架构。所提出的网络主要由ShuffleNet单元的堆栈组成，分为三个阶段。每个阶段中的第一个构建块应用步幅=2。一个阶段中的其他超参数保持不变，下一个阶段的输出通道加倍。与[9]类似，我们将每个ShuffleNet单元的瓶颈信道数设置为输出信道的1/4。我们的目的是提供尽可能简单的参考设计，尽管我们发现进一步的超参数调整可能会产生更好的结果。

In ShuffleNet units, group number g controls the connection sparsity of pointwise convolutions. Table 1 explores different group numbers and we adapt the output channels to ensure overall computation cost roughly unchanged (∼140 MFLOPs). Obviously, larger group numbers result in more output channels (thus more convolutional filters) for a given complexity constraint, which helps to encode more information, though it might also lead to degradation for an individual convolutional filter due to limited corresponding input channels. In Sec 4.1.1 we will study the impact of this number subject to different computational constrains.

在ShuffleNet单元中，组号g控制逐点卷积的连接稀疏性。表1探讨了不同的组号，我们调整了输出通道，以确保总体计算成本大致不变(∼140 MFLOP)。显然，对于给定的复杂度约束，较大的组号会导致更多的输出信道(从而导致更多的卷积滤波器)，这有助于编码更多的信息，尽管由于相应的输入信道有限，这也可能导致单个卷积滤波器的性能下降。在第4.1.1节中，我们将研究该数字在不同计算约束下的影响。

To customize the network to a desired complexity, we can simply apply a scale factor s on the number of channels. For example, we denote the networks in Table 1 as ”ShuffleNet 1×”, then ”ShuffleNet s×” means scaling the number of filters in ShuffleNet 1× by s times thus overall complexity will be roughly s2 times of ShuffleNet 1×.

为了将网络定制为所需的复杂度，我们可以简单地在信道数量上应用比例因子s。例如，我们将表1中的网络表示为“ShuffleNet 1×”，那么“Shuff Net s×”意味着将Shuffle Net中的过滤器数量缩放1×s倍，因此总体复杂度将大约为Shuffley Net 1 x的2倍。

## 4. Experiments
We mainly evaluate our models on the ImageNet 2012 classification dataset [29, 4]. We follow most of the training settings and hyper-parameters used in [40], with two exceptions: (i) we set the weight decay to 4e-5 instead of 1e-4 and use linear-decay learning rate policy (decreased from 0.5 to 0); (ii) we use slightly less aggressive scale augmentation for data preprocessing. Similar modifications are also referenced in [12] because such small networks usually suffer from underfitting rather than overfitting. It takes 1 or 2 days to train a model for 3×105 iterations on 4 GPUs, whose batch size is set to 1024. To benchmark, we compare single crop top-1 performance on ImageNet validation set, i.e. cropping 224×224 center view from 256× input image and evaluating classification accuracy. We use exactly the same settings for all models to ensure fair comparisons.

我们主要在ImageNet 2012分类数据集上评估我们的模型[29，4]。我们遵循了[40]中使用的大多数训练设置和超参数，但有两个例外：(i)我们将权重衰减设置为4e-5而不是1e-4，并使用线性衰减学习率策略(从0.5降至0); (ii)我们在数据预处理中使用稍微不太激进的尺度增强。[12]中也引用了类似的修改，因为此类小型网络通常会出现不足拟合而不是过拟合。在4个GPU(批量大小设置为1024)上训练3×105次迭代的模型需要1或2天。为了进行基准测试，我们比较了ImageNet验证集上的单次裁剪top 1性能，即从256×输入图像中裁剪224×224中心视图，并评估分类准确性。我们对所有模型使用完全相同的设置，以确保公平比较。

Table 3. ShuffleNet with/without channel shuffle (smaller number represents better performance) 
表3.带/不带频道混洗的ShuffleNet(数字越小表示性能越好)

### 4.1. Ablation Study
The core idea of ShuffleNet lies in pointwise group convolution and channel shuffle operation. In this subsection we evaluate them respectively.

ShuffleNet的核心思想在于逐点组卷积和信道混洗操作。在本小节中，我们分别对它们进行评估。

### 4.1.1 Pointwise Group Convolutions 逐点组卷积
To evaluate the importance of pointwise group convolutions, we compare ShuffleNet models of the same complexity whose numbers of groups range from 1 to 8. If the group number equals 1, no pointwise group convolution is involved and then the ShuffleNet unit becomes an ”Xception-like” [3] structure. For better understanding, we also scale the width of the networks to 3 different complexities and compare their classification performance respectively. Results are shown in Table 2.

为了评估逐点组卷积的重要性，我们比较了相同复杂度的ShuffleNet模型，其群数范围为1到8。如果群数等于1，则不涉及逐点组卷积，然后Shuffle Net单元变成“类Xception”[3]结构。为了更好地理解，我们还将网络的宽度缩放为3种不同的复杂性，并分别比较它们的分类性能。结果如表2所示。

From the results, we see that models with group convolutions (g > 1) consistently perform better than the counterparts without pointwise group convolutions (g = 1). Smaller models tend to benefit more from groups. For example, for ShuffleNet 1× the best entry (g = 8) is 1.2% better than the counterpart, while for ShuffleNet 0.5× and 0.25× the gaps become 3.5% and 4.4% respectively. Note that group convolution allows more feature map channels for a given complexity constraint, so we hypothesize that the performance gain comes from wider feature maps which help to encode more information. In addition, a smaller network involves thinner feature maps, meaning it benefits more from enlarged feature maps.

从结果中，我们看到，具有组卷积(g>1)的模型始终比没有逐点组卷积的模型(g=1)表现得更好。较小的模型往往从群体中受益更多。例如，对于ShuffleNet 1×，最佳条目(g=8)比对应条目好1.2%，而对于ShuffeNet 0.5×和0.25×，差距分别为3.5%和4.4%。请注意，对于给定的复杂度约束，组卷积允许更多的特征图通道，因此我们假设性能增益来自更宽的特征图，这有助于编码更多信息。此外，较小的网络包含较薄的特征图，这意味着它从放大的特征图中受益更多。

Table 2 also shows that for some models (e.g. ShuffleNet 0.5×) when group numbers become relatively large (e.g. g = 8), the classification score saturates or even drops. With an increase in group number (thus wider feature maps), input channels for each convolutional filter become fewer, which may harm representation capability. Interestingly, we also notice that for smaller models such as ShuffleNet 0.25× larger group numbers tend to better results consistently, which suggests wider feature maps bring more benefits for smaller models.

表2还显示，对于某些模型(例如ShuffleNet 0.5×)，当组号变得相对较大(例如g=8)时，分类分数饱和甚至下降。随着组数的增加(因此特征图更宽)，每个卷积滤波器的输入通道变得更少，这可能会损害表示能力。有趣的是，我们还注意到，对于较小的模型，如ShuffleNet，0.25×更大的组号往往一致地获得更好的结果，这表明更宽的特征图为较小的模型带来更多的好处。

#### 4.1.2 Channel Shuffle vs. No Shuffle
The purpose of shuffle operation is to enable cross-group information flow for multiple group convolution layers. Table 3 compares the performance of ShuffleNet structures (group number is set to 3 or 8 for instance) with/without channel shuffle. The evaluations are performed under three different scales of complexity. It is clear that channel shuffle consistently boosts classification scores for different settings. Especially, when group number is relatively large (e.g. g = 8), models with channel shuffle outperform the counterparts by a significant margin, which shows the importance of cross-group information interchange.

混洗操作的目的是为多个组卷积层启用跨组信息流。表3比较了有/无信道混洗的ShuffleNet结构(例如，组号设置为3或8)的性能。评估是在三种不同的复杂程度下进行的。很明显，频道洗牌会不断提高不同设置的分类分数。特别是，当组数相对较大(例如g=8)时，具有信道混洗的模型显著优于对应模型，这表明了跨组信息交换的重要性。

### 4.2. Comparison with Other Structure Units 与其他结构单元的比较
Recent leading convolutional units in VGG [30], ResNet [9], GoogleNet [33], ResNeXt [40] and Xception [3] have pursued state-of-the-art results with large models (e.g. ≥ 1GFLOPs), but do not fully explore lowcomplexity conditions. In this section we survey a variety of building blocks and make comparisons with ShuffleNet under the same complexity constraint.

VGG[30]、ResNet[9]、GoogleNet[33]、ResNeXt[40]和Xception[3]中最新的领先卷积单元已利用大型模型(例如。≥ 1GFLOP)，但不完全探索低复杂度条件。在本节中，我们调查了各种构建块，并在相同的复杂性约束下与ShuffleNet进行了比较。

For fair comparison, we use the overall network architecture as shown in Table 1. We replace the ShuffleNet units in Stage 2-4 with other structures, then adapt the number of channels to ensure the complexity remains unchanged. The structures we explored include:
* VGG-like. Following the design principle of VGG net [30], we use a two-layer 3×3 convolutions as the basic building block. Different from [30], we add a Batch Normalization layer [15] after each of the convolutions to make end-to-end training easier.
* ResNet. We adopt the ”bottleneck” design in our experiment, which has been demonstrated more efficient in [9] . Same as [9], the bottleneck ratio(1 In the bottleneck-like units (like ResNet, ResNeXt or ShuffleNet) bottleneck ratio implies the ratio of bottleneck channels to output channels.For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times the width of the bottleneck feature map. ) is also 1 : 4. 
* Xception-like. The original structure proposed in [3] involves fancy designs or hyper-parameters for different stages, which we find difficult for fair comparison on small models. Instead, we remove the pointwise group convolutions and channel shuffle operation from ShuffleNet (also equivalent to ShuffleNet with g = 1). The derived structure shares the same idea of “depthwise separable convolution” as in [3], which is called an Xception-like structure here.
* ResNeXt. We use the settings of cardinality = 16 and bottleneck ratio = 1 : 2 as suggested in [40]. We also explore other settings, e.g. bottleneck ratio = 1 : 4, and get similar results.

为了公平比较，我们使用了如表1所示的整体网络架构。我们将第2-4阶段中的ShuffleNet单元替换为其他结构，然后调整信道数以确保复杂性保持不变。我们探索的结构包括：
* 类似VGG。遵循VGG网络的设计原理[30]，我们使用两层3×3卷积作为基本构建块。与[30]不同，我们在每个卷积之后添加了一个批次归一化层[15]，以使端到端训练更容易。
* ResNet。我们在实验中采用了“瓶颈”设计，这在[9]中得到了证明。与[9]相同，瓶颈比(1在类瓶颈单元(如ResNet、ResNeXt或ShuffleNet)中，瓶颈比意味着瓶颈通道与输出通道的比率。例如，瓶颈比=1:4意味着输出特征图是瓶颈特征图宽度的4倍。)也是1:4。
* 类似Xception。[3]中提出的原始结构涉及不同阶段的花哨设计或超参数，我们发现很难在小型模型上进行公平比较。相反，我们从ShuffleNet(也等同于g＝1的shuffle Net)中移除逐点组卷积和信道混洗操作。派生结构与[3]中的“深度可分离卷积”概念相同，这里称为类Xception结构。
* ResNeXt。我们使用了[40]中建议的基数=16和瓶颈比=1:2的设置。我们还探索了其他设置，例如瓶颈比=1:4，并得到了类似的结果。

Table 4. Classification error vs. various structures (%, smaller number represents better performance). We do not report VGG-like structure on smaller networks because the accuracy is significantly worse.
表4.分类误差与各种结构(%，数字越小表示性能越好)。我们没有在较小的网络上报告类似VGG的结构，因为其准确性明显较差。

Table 5. ShuffleNet vs. MobileNet [12] on ImageNet Classification
表5.关于ImageNet分类的ShuffleNet与MobileNet[12]

We use exactly the same settings to train these models. Results are shown in Table 4. Our ShuffleNet models outperform most others by a significant margin under different complexities. Interestingly, we find an empirical relationship between feature map channels and classification accuracy. For example, under the complexity of 38 MFLOPs, output channels of Stage 4 (see Table 1) for VGG-like, ResNet, ResNeXt, Xception-like, ShuffleNet models are 50, 192, 192, 288, 576 respectively, which is consistent with the increase of accuracy. Since the efficient design of ShuffleNet, we can use more channels for a given computation budget, thus usually resulting in better performance.

我们使用完全相同的设置来训练这些模型。结果如表4所示。在不同的复杂性下，我们的ShuffleNet模型以显著的优势优于大多数其他模型。有趣的是，我们发现了特征图通道和分类精度之间的经验关系。例如，在38个MFLOP的复杂度下，VGG-like、ResNet、ResNeXt、Xception-like、ShuffleNet模型的第4阶段(见表1)的输出通道分别为50、192、192和288、576，这与精度的提高一致。由于ShuffleNet的高效设计，我们可以在给定的计算预算中使用更多的信道，因此通常会产生更好的性能。

Note that the above comparisons do not include GoogleNet or Inception series [33, 34, 32]. We find it nontrivial to generate such Inception structures to small networks because the original design of Inception module involves too many hyper-parameters. As a reference, the first GoogleNet version [33] has 31.3% top-1 error at the cost of 1.5 GFLOPs (See Table 6). More sophisticated Inception versions [34, 32] are more accurate, however, involve significantly increased complexity. Recently, Kim et al. propose a lightweight network structure named PVANET [19] which adopts Inception units. Our reimplemented PVANET (with 224×224 input size) has 29.7% classification error with a computation complexity of 557 MFLOPs, while our ShuffleNet 2x model (g = 3) gets 26.3% with 524 MFLOPs (see Table 6).

请注意，上述比较不包括GoogleNet或Inception系列[33，34，32]。我们发现，为小型网络生成这样的Inception结构并不简单，因为Inceptation模块的原始设计涉及太多超参数。作为参考，第一个GoogleNet版本[33]有31.3%的top-1错误，代价是1.5 GFLOP(见表6)。然而，更复杂的Inception版本[34，32]更为准确，涉及到显著增加的复杂性。最近，Kimet al 提出了一种名为PVANET[19]的轻量级网络结构，它采用了Inception单元。我们重新实现的PVANET(输入大小为224×224)有29.7%的分类错误，计算复杂度为557个MFLOP，而我们的ShuffleNet 2x模型(g=3)有524个MFLOPs，分类错误率为26.3%(见表6)。

### 4.3. Comparison with MobileNets and Other Frameworks
Recently Howard et al. have proposed MobileNets [12] which mainly focus on efficient network architecture for mobile devices. MobileNet takes the idea of depthwise separable convolution from [3] and achieves state-of-the-art results on small models.

最近Howardet al 提出了MobileNets[12]，其主要关注移动设备的高效网络架构。MobileNet采用了[3]中深度可分离卷积的思想，并在小模型上获得了最先进的结果。

Table 5 compares classification scores under a variety of complexity levels. It is clear that our ShuffleNet models are superior to MobileNet for all the complexities. Though our ShuffleNet network is specially designed for small models (< 150 MFLOPs), we find it is still better than MobileNet for higher computation cost, e.g. 3.1% more accurate than MobileNet 1× at the cost of 500 MFLOPs. For smaller networks (∼40 MFLOPs) ShuffleNet surpasses MobileNet by 7.8%. Note that our ShuffleNet architecture contains 50 layers while MobileNet only has 28 layers. For better understanding, we also try ShuffleNet on a 26-layer architecture by removing half of the blocks in Stage 2-4 (see ”ShuffleNet 0.5× shallow (g = 3)” in Table 5). Results show that the shallower model is still significantly better than the corresponding MobileNet, which implies that the effectiveness of ShuffleNet mainly results from its efficient structure, not the depth.
 
表5比较了各种复杂程度下的分类得分。很明显，我们的ShuffleNet模型在所有复杂性方面都优于MobileNet。尽管我们的ShuffleNet网络是专门为小型模型(<150 MFLOP)设计的，但我们发现它在更高的计算成本方面仍优于MobileNet，例如，在500 MFLOP的成本下，比MobileNet1×精确3.1%。对于较小的网络(∼40个MFLOP)ShuffleNet超过MobileNet 7.8%。请注意，我们的ShuffleNet架构包含50层，而MobileNet只有28层。为了更好地理解，我们还尝试在26层架构上使用ShuffleNet，方法是在阶段2-4中删除一半的块(参见表5中的“Shuffle Net 0.5×shall(g=3)”)。结果表明，较浅的模型仍然明显优于相应的MobileNet，这意味着ShuffleNet的有效性主要来自其有效的结构，而不是深度。

Table 6. Complexity comparison. *Implemented by BVLC (https://github.com/BVLC/caffe/tree/master/models/bvlc googlenet)<br/>
Table 7. Object detection results on MS COCO (larger numbers represents better performance). For MobileNets we compare two results: 1) COCO detection scores reported by [12]; 2) finetuning from our reimplemented MobileNets, whose training and finetuning settings are exactly the same as that for ShuffleNets.
表7.MS COCO上的目标检测结果(数字越大表示性能越好)。对于MobileNets，我们比较了两个结果：1)[12]报告的COCO检测得分; 2) 从我们重新实现的MobileNets进行微调，其训练和微调设置与ShuffleNets完全相同。

Table 8. Actual inference time on mobile device (smaller number represents better performance). The platform is based on a single Qualcomm Snapdragon 820 processor. All results are evaluated with single thread. 
表8.移动设备上的实际推理时间(数字越小表示性能越好)。该平台基于单个高通骁龙820处理器。所有结果都使用单线程进行评估。

Table 6 compares our ShuffleNet with a few popular models. Results show that with similar accuracy ShuffleNet is much more efficient than others. For example, ShuffleNet 0.5× is theoretically 18× faster than AlexNet [21] with comparable classification score. We will evaluate the actual running time in Sec 4.5.
表6将我们的ShuffleNet与一些流行的模型进行了比较。结果表明，在相似的精度下，ShuffleNet比其他方法更有效。例如，ShuffleNet 0.5×理论上比AlexNet[21]快18倍，分类得分相当。我们将在第4.5节中评估实际运行时间。

It is also worth noting that the simple architecture design makes it easy to equip ShuffeNets with the latest advances such as [13, 26]. For example, in [13] the authors propose Squeeze-and-Excitation (SE) blocks which achieve state-of-the-art results on large ImageNet models. We find SE modules also take effect in combination with the backbone ShuffleNets, for instance, boosting the top-1 error of ShuffleNet 2× to 24.7% (shown in Table 5). Interestingly, though negligible increase of theoretical complexity, we find ShuffleNets with SE modules are usually 25 ∼ 40% slower than the “raw” ShuffleNets on mobile devices, which implies that actual speedup evaluation is critical on low-cost architecture design. In Sec 4.5 we will make further discussion.

同样值得注意的是，简单的架构设计使ShuffeNets很容易配备最新的技术，如[13，26]。例如，在[13]中，作者提出了挤压和激发(SE)块，在大型ImageNet模型上实现了最先进的结果。例如，我们发现SE模块与主干ShuffleNets结合也会起作用，将ShuffleNet 2×的top 1错误提高到24.7%(如表5所示)。有趣的是，尽管理论复杂性的增加可以忽略不计，但我们发现带有SE模块的ShuffleNets通常为25∼ 在移动设备上比“原始”ShuffleNets慢40%，这意味着实际的加速评估对低成本架构设计至关重要。在第4.5节中，我们将进一步讨论。

### 4.4. Generalization Ability 泛化能力
To evaluate the generalization ability for transfer learning, we test our ShuffleNet model on the task of MS COCO object detection [23]. We adopt Faster-RCNN [28] as the detection framework and use the publicly released Caffe code [28, 17] for training with default settings. Similar to [12], the models are trained on the COCO train+val dataset excluding 5000 minival images and we conduct testing on the minival set. Table 7 shows the comparison of results trained and evaluated on two input resolutions. Comparing ShuffleNet 2× with MobileNet whose complexity are comparable (524 vs. 569 MFLOPs), our ShuffleNet 2× surpasses MobileNet by a significant margin on both resolutions; our ShuffleNet 1× also achieves comparable results with MobileNet on 600× resolution, but has ∼4× complexity reduction. We conjecture that this significant gain is partly due to ShuffleNet’s simple design of architecture without bells and whistles.

为了评估迁移学习的泛化能力，我们在MS COCO对象检测任务上测试了我们的ShuffleNet模型[23]。我们采用Faster RCNN[28]作为检测框架，并使用公开发布的Caffe代码[28，17]进行默认设置的训练。与[12]相似，模型在COCO训练+val数据集上训练，不包括5000个最小值图像，我们在最小值集上进行测试。表7显示了在两种输入分辨率下训练和评估的结果的比较。将ShuffleNet 2×与复杂度相当的MobileNet(524对569 MFLOP)进行比较，我们的Shuffle Net 2 x在两种分辨率上都大大超过了MobileNet; 我们的ShuffleNet 1×也在600×分辨率上取得了与MobileNet相当的结果，但∼复杂度降低4倍。我们推测，这一显著的增长部分归功于ShuffleNet简单的架构设计，没有铃铛和哨声。

### 4.5. Actual Speedup Evaluation 实际加速评估
Finally, we evaluate the actual inference speed of ShuffleNet models on a mobile device with an ARM platform. Though ShuffleNets with larger group numbers (e.g. g = 4 or g = 8) usually have better performance, we find it less efficient in our current implementation. Empirically g = 3 usually has a proper trade-off between accuracy and actual inference time. As shown in Table 8, three input resolutions are exploited for the test. Due to memory access and other overheads, we find every 4× theoretical complexity reduction usually results in ∼2.6× actual speedup in our implementation. Nevertheless, compared with AlexNet [21] our ShuffleNet 0.5× model still achieves ∼13× actual speedup under comparable classification accuracy (the theoretical speedup is 18×), which is much faster than previous AlexNet-level models or speedup approaches such as [14, 16, 22, 42, 43, 38].

最后，我们评估了在具有ARM平台的移动设备上ShuffleNet模型的实际推理速度。尽管具有较大组号(例如g=4或g=8)的ShuffleNets通常具有更好的性能，但我们发现它在当前的实现中效率较低。经验上，g=3通常在准确度和实际推断时间之间有适当的权衡。如表8所示，测试采用了三种输入分辨率。由于内存访问和其他开销，我们发现每减少4倍理论复杂度通常会导致∼2.6倍的实际速度。尽管如此，与AlexNet[21]相比，我们的ShuffleNet 0.5×模型仍然实现了∼在可比分类精度下的13倍实际加速(理论加速为18倍)，比以前的AlexNet级别模型或加速方法(如[14，16，22，42，43，38])快得多。

## References
1. M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467, 2016. 4
2. H. Bagherinezhad, M. Rastegari, and A. Farhadi. Lcnn: Lookup-based convolutional neural network. arXiv preprint arXiv:1611.06473, 2016. 2
3. F. Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016. 1, 2, 3, 4, 5, 6
4. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. FeiFei. Imagenet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pages 248–255. IEEE, 2009. 1, 4
5. R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 580–587,2014. 1
6. S. Han, H. Mao, and W. J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149, 2015. 2
7. S. Han, J. Pool, J. Tran, and W. Dally. Learning both weights and connections for efficient neural network. In Advances in Neural Information Processing Systems, pages 1135–1143,2015. 2
8. K. He and J. Sun. Convolutional neural networks at constrained time cost. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5353– 5360, 2015. 1
9. K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016. 1, 2, 3, 4, 5, 6
10. K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In European Conference on Computer Vision, pages 630–645. Springer, 2016. 1, 2
11. G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 2
12. A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017. 1, 2, 3, 5, 6, 7
13. J. Hu, L. Shen, and G. Sun. Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 2017. 1, 6, 7
14. F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally, and K. Keutzer. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and¡ 0.5 mb model size. arXiv preprint arXiv:1602.07360, 2016. 1, 7, 8
15. S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015. 3, 5
16. M. Jaderberg, A. Vedaldi, and A. Zisserman. Speeding up convolutional neural networks with low rank expansions. arXiv preprint arXiv:1405.3866, 2014. 1, 2, 8
17. Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. In Proceedings of the 22nd ACM international conference on Multimedia, pages 675–678. ACM, 2014. 7
18. J. Jin, A. Dundar, and E. Culurciello. Flattened convolutional neural networks for feedforward acceleration. arXiv preprint arXiv:1412.5474, 2014. 2
19. K.-H. Kim, S. Hong, B. Roh, Y. Cheon, and M. Park. Pvanet: Deep but lightweight neural networks for real-time object detection. arXiv preprint arXiv:1608.08021, 2016. 6
20. A. Krizhevsky. cuda-convnet: High-performance c++/cuda implementation of convolutional neural networks, 2012. 2
21. A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012. 1, 2, 7, 8
22. V. Lebedev, Y. Ganin, M. Rakhuba, I. Oseledets, and V. Lempitsky. Speeding-up convolutional neural networks using fine-tuned cp-decomposition. arXiv preprint arXiv:1412.6553, 2014. 1, 2, 8
23. T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, pages 740–755. Springer, 2014. 1, 7
24. J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3431–3440, 2015. 1
25. M. Mathieu, M. Henaff, and Y. LeCun. Fast training of convolutional networks through ffts. arXiv preprint arXiv:1312.5851, 2013. 2
26. P. Ramachandran, B. Zoph, and Q. V. Le. Swish: a self-gated activation function. arXiv preprint arXiv:1710.05941, 2017. 7
27. M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi. Xnornet: Imagenet classification using binary convolutional neural networks. In European Conference on Computer Vision, pages 525–542. Springer, 2016. 1, 2
28. S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems, pages 91–99, 2015. 1, 7
29. O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211–252,2015. 1, 4
30. K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 1, 2, 5, 7
31. D. Soudry, I. Hubara, and R. Meir. Expectation backpropagation: Parameter-free training of multilayer neural networks with continuous or discrete weights. In Advances in Neural Information Processing Systems, pages 963–971, 2014. 2
32. C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi. Inceptionv4, inception-resnet and the impact of residual connections on learning. arXiv preprint arXiv:1602.07261, 2016. 1, 2, 6
33. C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–9, 2015. 1, 2, 5, 6, 7
34. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2818–2826, 2016. 1, 2, 6
35. N. Vasilache, J. Johnson, M. Mathieu, S. Chintala, S. Piantino, and Y. LeCun. Fast convolutional nets with fbfft: A gpu performance evaluation. arXiv preprint arXiv:1412.7580, 2014. 2
36. O. Vinyals, A. Toshev, S. Bengio, and D. Erhan. Show and tell: A neural image caption generator. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3156–3164, 2015. 1
37. M. Wang, B. Liu, and H. Foroosh. Design of efficient convolutional layers using single intra-channel convolution, topological subdivisioning and spatial ”bottleneck” structure. arXiv preprint arXiv:1608.04337, 2016. 2
38. W. Wen, C. Wu, Y. Wang, Y. Chen, and H. Li. Learning structured sparsity in deep neural networks. In Advances in Neural Information Processing Systems, pages 2074–2082,2016. 1, 2, 8
39. J. Wu, C. Leng, Y. Wang, Q. Hu, and J. Cheng. Quantized convolutional neural networks for mobile devices. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4820–4828, 2016. 2
40. S. Xie, R. Girshick, P. Doll´ar, Z. Tu, and K. He. Aggregated residual transformations for deep neural networks. arXiv preprint arXiv:1611.05431, 2016. 1, 2, 3, 4, 5, 6
41. T. Zhang, G.-J. Qi, B. Xiao, and J. Wang. Interleaved group convolutions for deep neural networks. In International Conference on Computer Vision, 2017. 2
42. X. Zhang, J. Zou, K. He, and J. Sun. Accelerating very deep convolutional networks for classification and detection. IEEE transactions on pattern analysis and machine intelligence, 38(10):1943–1955, 2016. 1, 8
43. X. Zhang, J. Zou, X. Ming, K. He, and J. Sun. Efficient and accurate approximations of nonlinear convolutional networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1984–1992, 2015. 1, 8
44. A. Zhou, A. Yao, Y. Guo, L. Xu, and Y. Chen. Incremental network quantization: Towards lossless cnns with lowprecision weights. arXiv preprint arXiv:1702.03044, 2017. 2
45. S. Zhou, Y. Wu, Z. Ni, X. Zhou, H. Wen, and Y. Zou. Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv preprint arXiv:1606.06160, 2016. 2
46. B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le. Learning transferable architectures for scalable image recognition. arXiv preprint arXiv:1707.07012, 2017. 1, 2
