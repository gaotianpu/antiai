# Densely Connected Convolutional Networks
密集连接卷积网络 2016-8-15 https://arxiv.org/abs/1608.06993

## 阅读笔记
* [pytorch实现](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)

## Abstract
Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections—one between each layer and its subsequent layer—our network has L(L 2 +1) direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less computation to achieve high performance. Code and pre-trained models are available at https://github.com/liuzhuang13/DenseNet.

最近的研究表明，如果卷积网络在靠近输入的层和靠近输出的层之间包含较短的连接，则卷积网络可以更深、更准确、更有效地进行训练。在本文中，我们接受这一观察，并介绍了密集卷积网络(Dense Convolutional Network，简称DenseNet)，它以前馈方式将每一层连接到每一层。而具有L层的传统卷积网络具有L个连接，每个层与其后续层之间有一个连接。对于每个图层，所有先前图层的要素图都用作输入，其自身的要素图用作所有后续图层的输入。DenseNets有几个令人信服的优点：它们缓解了消失梯度问题，加强了特征传播，鼓励了特征重用，并大大减少了参数的数量。我们在四个竞争激烈的对象识别基准任务(CIFAR-10、CIFAR-100、SVHN和ImageNet)上评估了我们提出的架构。DenseNets在大多数方面都比最先进的技术有了显著的改进，同时需要较少的计算来实现高性能。代码和预训练的模型可在https://github.com/liuzhuang13/DenseNet.

## 1. Introduction
Convolutional neural networks (CNNs) have become the dominant machine learning approach for visual object recognition. Although they were originally introduced over 20 years ago [18], improvements in computer hardware and network structure have enabled the training of truly deep CNNs only recently. The original LeNet5 [19] consisted of 5 layers, VGG featured 19 [29], and only last year Highway Networks [34] and Residual Networks (ResNets) [11] have surpassed the 100-layer barrier.

卷积神经网络(CNN)已成为视觉对象识别的主要机器学习方法。尽管它们最初是在20多年前引入的[18]，但计算机硬件和网络结构的改进直到最近才使真正深度CNN的训练成为可能。最初的LeNet5[19]由5层组成，VGG有19层[29]，去年的Highway网[34]和残差网络(ResNets)[11]才突破了100层的障碍。

As CNNs become increasingly deep, a new research problem emerges: as information about the input or gradient passes through many layers, it can vanish and “wash out” by the time it reaches the end (or beginning) of the network. Many recent publications address this or related problems. ResNets [11] and Highway Networks [34] bypass signal from one layer to the next via identity connections. Stochastic depth [13] shortens ResNets by randomly dropping layers during training to allow better information and gradient flow. FractalNets [17] repeatedly combine several parallel layer sequences with different number of convolutional blocks to obtain a large nominal depth, while maintaining many short paths in the network. Although these different approaches vary in network topology and training procedure, they all share a key characteristic: they create short paths from early layers to later layers. 

随着CNN变得越来越深，一个新的研究问题出现了：当关于输入或梯度的信息通过许多层时，当它到达网络的末端(或开始)时，它可能会消失并“洗掉”。最近的许多出版物都讨论了这一问题或相关问题。ResNets[11]和Highway Networks[34]通过身份连接将信号从一层绕过到下一层。随机深度[13]通过在训练期间随机丢弃层来缩短ResNets，以允许更好的信息和梯度流。FractalNets[17]反复组合具有不同数量卷积块的多个并行层序列，以获得较大的标称深度，同时保持网络中的许多短路径。尽管这些不同的方法在网络拓扑和训练过程中有所不同，但它们都有一个关键特征：它们创建了从早期层到后期层的短路径。

In this paper, we propose an architecture that distills this insight into a simple connectivity pattern: to ensure maximum information flow between layers in the network, we connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers. Figure 1 illustrates this layout schematically. Crucially, in contrast to ResNets, we never combine features through summation before they are passed into a layer; instead, we combine features by concatenating them. Hence, the  $l^{th}$ layer has  inputs, consisting of the feature-maps of all preceding convolutional blocks. Its own feature-maps are passed on to all L−l subsequent layers. This introduces L(L+1)/2 connections in an L-layer network, instead of just L, as in traditional architectures. Because of its dense connectivity pattern, we refer to our approach as Dense Convolutional Network (DenseNet).

在本文中，我们提出了一种架构，将这种洞察力提炼为一种简单的连接模式：为了确保网络中各层之间的最大信息流，我们将所有层(具有匹配的特征图大小)直接相互连接。为了保持前馈特性，每个层从所有先前层获得额外输入，并将其自身的特征映射传递给所有后续层。图1示意性地说明了这种布局。关键的是，与ResNets相反，我们从不在将特性传递到一个层之前通过求和来组合它们; 相反，我们通过级连功能来组合它们。因此，$l^｛th｝$层具有输入，由所有先前卷积块的特征图组成。它自己的特征地图被传递给所有L−l后续层。这在L层网络中引入了L(L+1)/2连接，而不是传统架构中的L。由于其稠密的连接模式，我们将我们的方法称为稠密卷积网络(DenseNet)。

Figure 1: A 5-layer dense block with a growth rate of k = 4. Each layer takes all preceding feature-maps as input.
图1：一个5层密集块，其增长率为k=4。每个层都将前面的所有特征图作为输入。

A possibly counter-intuitive effect of this dense connectivity pattern is that it requires fewer parameters than traditional convolutional networks, as there is no need to relearn redundant feature-maps. Traditional feed-forward architectures can be viewed as algorithms with a state, which is passed on from layer to layer. Each layer reads the state from its preceding layer and writes to the subsequent layer. It changes the state but also passes on information that needs to be preserved. ResNets [11] make this information preservation explicit through additive identity transformations. Recent variations of ResNets [13] show that many layers contribute very little and can in fact be randomly dropped during training. This makes the state of ResNets similar to (unrolled) recurrent neural networks [21], but the number of parameters of ResNets is substantially larger because each layer has its own weights. Our proposed DenseNet architecture explicitly differentiates between information that is added to the network and information that is preserved. DenseNet layers are very narrow (e.g., 12 filters per layer), adding only a small set of feature-maps to the “collective knowledge” of the network and keep the remaining featuremaps unchanged—and the final classifier makes a decision based on all feature-maps in the network.

这种密集连接模式的一个可能与直觉相反的效果是，它需要比传统卷积网络更少的参数，因为不需要重新学习冗余特征图。传统的前馈架构可以被看作是一种具有状态的算法，这种状态可以从一层传递到另一层。每一层从其前一层读取状态，并写入下一层。它改变了状态，但也传递了需要保留的信息。ResNets[11]通过加性身份转换使这种信息保存显式。ResNets的最新变化[13]表明，许多层的贡献很小，实际上可以在训练期间随机丢弃。这使得ResNets的状态类似于(展开的)递归神经网络[21]，但ResNets参数的数量大大增加，因为每个层都有自己的权重。我们提出的DenseNet架构明确区分了添加到网络中的信息和保留的信息。DenseNet层非常窄(例如，每层12个过滤器)，仅向网络的“集体知识”添加一小组特征图，并保持其余特征图不变，最终分类器根据网络中的所有特征图做出决策。

Besides better parameter efficiency, one big advantage of DenseNets is their improved flow of information and gradients throughout the network, which makes them easy to train. Each layer has direct access to the gradients from the loss function and the original input signal, leading to an implicit deep supervision [20]. This helps training of deeper network architectures. Further, we also observe that dense connections have a regularizing effect, which reduces over- fitting on tasks with smaller training set sizes.

除了更好的参数效率外，DenseNets的一大优势是其在整个网络中改进的信息流和梯度，这使其易于训练。每一层都可以直接从损失函数和原始输入信号获得梯度，从而实现隐式深度监控[20]。这有助于训练更深层次的网络架构。此外，我们还观察到，密集连接具有正则化效果，这减少了对较小训练集大小的任务的过度拟合。

We evaluate DenseNets on four highly competitive benchmark datasets (CIFAR-10, CIFAR-100, SVHN, and ImageNet). Our models tend to require much fewer parameters than existing algorithms with comparable accuracy. Further, we significantly outperform the current state-ofthe-art results on most of the benchmark tasks.

我们在四个极具竞争力的基准数据集(CIFAR-10、CIFAR-100、SVHN和ImageNet)上评估了DenseNets。我们的模型往往需要比现有算法少得多的参数，并且具有相当的精度。此外，在大多数基准任务中，我们的表现明显优于当前最先进的结果。

## 2. Related Work
The exploration of network architectures has been a part of neural network research since their initial discovery. The recent resurgence in popularity of neural networks has also revived this research domain. The increasing number of layers in modern networks amplifies the differences between architectures and motivates the exploration of different connectivity patterns and the revisiting of old research ideas.

自从神经网络最初被发现以来，对网络结构的探索一直是神经网络研究的一部分。最近神经网络的流行也使这一研究领域重新活跃起来。现代网络中层数的增加扩大了架构之间的差异，并促使人们探索不同的连接模式和重新审视旧的研究思路。

A cascade structure similar to our proposed dense network layout has already been studied in the neural networks literature in the 1980s [3]. Their pioneering work focuses on fully connected multi-layer perceptrons trained in a layerby-layer fashion. More recently, fully connected cascade networks to be trained with batch gradient descent were proposed [40]. Although effective on small datasets, this approach only scales to networks with a few hundred parameters. In [9, 23, 31, 41], utilizing multi-level features in CNNs through skip-connnections has been found to be effective for various vision tasks. Parallel to our work, [1] derived a purely theoretical framework for networks with cross-layer connections similar to ours.

20世纪80年代神经网络文献中已经研究了类似于我们提出的密集网络布局的级联结构[3]。他们的开创性工作专注于以分层方式训练的完全连接的多层感知器。最近，有人提出了用批梯度下降训练完全连接的级联网络[40]。尽管这种方法对小数据集有效，但它只适用于具有数百个参数的网络。在[9，23，31，41]中，通过跳跃连接利用CNN中的多级特征已被发现对各种视觉任务有效。与我们的工作平行，[1]导出了一个与我们类似的跨层连接网络的纯理论框架。

Highway Networks [34] were amongst the first architectures that provided a means to effectively train end-to-end networks with more than 100 layers. Using bypassing paths along with gating units, Highway Networks with hundreds of layers can be optimized without difficulty. The bypassing paths are presumed to be the key factor that eases the training of these very deep networks. This point is further supported by ResNets [11], in which pure identity mappings are used as bypassing paths. ResNets have achieved impressive, record-breaking performance on many challenging image recognition, localization, and detection tasks, such as ImageNet and COCO object detection [11]. Recently, stochastic depth was proposed as a way to successfully train a 1202-layer ResNet [13]. Stochastic depth improves the training of deep residual networks by dropping layers randomly during training. This shows that not all layers may be needed and highlights that there is a great amount of redundancy in deep (residual) networks. Our paper was partly inspired by that observation. ResNets with pre-activation also facilitate the training of state-of-the-art networks with > 1000 layers [12].

Highway Networks[34]是首批提供有效训练具有100层以上的端到端网络的架构之一。使用旁路路径和门单元，可以毫无困难地优化具有数百层的Highway网络。旁路路径被认为是简化这些深度网络训练的关键因素。ResNets[11]进一步支持这一点，其中使用纯身份映射作为旁路路径。ResNets在许多具有挑战性的图像识别、定位和检测任务(如ImageNet和COCO对象检测)中取得了令人印象深刻的、破纪录的性能[11]。最近，随机深度被提出作为成功训练1202层ResNet的一种方法[13]。随机深度通过在训练期间随机丢弃层来改进深度残差网络的训练。这表明并非所有层都需要，并强调了深度(剩余)网络中存在大量冗余。我们的论文部分受到了这一观察的启发。具有预激活功能的ResNets也有助于训练具有超过1000层的最先进网络[12]。

An orthogonal approach to making networks deeper (e.g., with the help of skip connections) is to increase the network width. The GoogLeNet [36, 37] uses an “Inception module” which concatenates feature-maps produced by filters of different sizes. In [38], a variant of ResNets with wide generalized residual blocks was proposed. In fact, simply increasing the number of filters in each layer of ResNets can improve its performance provided the depth is sufficient [42]. FractalNets also achieve competitive results on several datasets using a wide network structure [17].

使网络更深的正交方法(例如，借助跳过连接)是增加网络宽度。GoogleLeNet[36,37]使用了一个“Inception模块”，它将不同大小的过滤器生成的特征图连接起来。在[38]中，提出了一种具有广义残差块的ResNets变体。事实上，只要深度足够，只需增加ResNets每层中的过滤器数量就可以提高其性能[42]。FractalNets还使用广泛的网络结构在多个数据集上取得了竞争性结果[17]。

Figure 2: A deep DenseNet with three dense blocks. The layers between two adjacent blocks are referred to as transition layers and change feature-map sizes via convolution and pooling.
图2：具有三个致密块的深层致密网。两个相邻块之间的层被称为过渡层，并通过卷积和池来改变特征图大小。

Instead of drawing representational power from extremely deep or wide architectures, DenseNets exploit the potential of the network through feature reuse, yielding condensed models that are easy to train and highly parameterefficient. Concatenating feature-maps learned by different layers increases variation in the input of subsequent layers and improves efficiency. This constitutes a major difference between DenseNets and ResNets. Compared to Inception networks [36, 37], which also concatenate features from different layers, DenseNets are simpler and more efficient.

DenseNets没有从极深或极广的架构中汲取代表性力量，而是通过功能重用来挖掘网络的潜力，生成易于训练且参数化程度高的精简模型。将由不同层学习的特征图连接起来，增加了后续层输入的变化并提高了效率。这构成了DenseNets和ResNets之间的主要区别。与Inception网络[36，37]相比，DenseNets更简单、更高效，Inceptions网络也将不同层的功能连接起来。

There are other notable network architecture innovations which have yielded competitive results. The Network in Network (NIN) [22] structure includes micro multi-layer perceptrons into the filters of convolutional layers to extract more complicated features. In Deeply Supervised Network (DSN) [20], internal layers are directly supervised by auxiliary classifiers, which can strengthen the gradients received by earlier layers. Ladder Networks [27, 25] introduce lateral connections into autoencoders, producing impressive accuracies on semi-supervised learning tasks. In [39], Deeply-Fused Nets (DFNs) were proposed to improve information flow by combining intermediate layers of different base networks. The augmentation of networks with pathways that minimize reconstruction losses was also shown to improve image classification models [43].

还有其他值得注意的网络架构创新，已经产生了具有竞争力的结果。网络中的网络(NIN)[22]结构在卷积层的滤波器中包括微多层感知器，以提取更复杂的特征。在深度监督网络(DSN)[20]中，内部层由辅助分类器直接监督，这可以增强早期层接收到的梯度。梯形网络[27，25]将横向连接引入自动编码器，在半监督学习任务中产生了令人印象深刻的精确度。在[39]中，提出了深度融合网(DFN)，以通过组合不同基础网络的中间层来改善信息流。增加具有最小化重建损失的路径的网络也被证明可以改善图像分类模型[43]。

## 3. DenseNets
Consider a single image x0 that is passed through a convolutional network. The network comprises L layers, each of which implements a non-linear transformation $H_l (·)$, where indexes the layer. $H_l (·)$ can be a composite function of operations such as Batch Normalization (BN) [14], rectified linear units (ReLU) [6], Pooling [19], or Convolution (Conv). We denote the output of the $l^{th}$ layer as $x_l$ .

考虑通过卷积网络的单个图像x0。该网络包括L个层，每个层实现非线性变换$H_L(·)$，其中对层进行索引$H_ l(·)$可以是诸如批处理归一化(BN)[14]、校正线性单元(ReLU)[6]、混合[19]或卷积(Conv)等操作的复合函数。我们将$l^{th}$层的输出表示为$x_l$。

ResNets. Traditional convolutional feed-forward networks connect the output of the  th layer as input to the $(l + 1)^{th}$ layer [16], which gives rise to the following layer transition: $x_l = H_l(x_l −1)$. ResNets [11] add a skip-connection that bypasses the non-linear transformations with an identity function: 

ResNets。传统的卷积前馈网络将第th层的输出作为输入连接到$(l+1)^｛th｝$层[16]，这导致以下层转换：$x_l＝H_l(x_l−1)$. ResNets[11]添加了一个跳过连接，该连接通过恒等函数绕过非线性变换：

$x_l = H_l(x_{l −1}) + x_{l −1}$. 

An advantage of ResNets is that the gradient can flow directly through the identity function from later layers to the earlier layers. However, the identity function and the output of H` are combined by summation, which may impede the information flow in the network.

ResNets的一个优点是，梯度可以直接通过恒等函数从后面的层流到前面的层。然而，恒等函数和H`的输出通过求和结合在一起，这可能会阻碍网络中的信息流。

Dense connectivity. To further improve the information flow between layers we propose a different connectivity pattern: we introduce direct connections from any layer to all subsequent layers. Figure 1 illustrates the layout of the resulting DenseNet schematically. Consequently, the  th layer receives the feature-maps of all preceding layers, x0, . . . , xl −1, as input: 

密集连接。为了进一步改善层之间的信息流，我们提出了一种不同的连接模式：我们引入了从任何层到所有后续层的直接连接。图1示意性地显示了生成的DenseNet的布局。因此，第th层接收所有先前层的特征图，x0，…，xl−1，作为输入：

$x_l = H_l ([x_0, x_1, . . . , x_{l −1}]) $ , (2) 

where [$x_0, x_1, . . . , x_{l−1}$] refers to the concatenation of the feature-maps produced in layers 0, . . . , l−1. Because of its dense connectivity we refer to this network architecture as Dense Convolutional Network (DenseNet). For ease of implementation, we concatenate the multiple inputs of H` (·) in eq. (2) into a single tensor.

其中[$x_0，x_1，…，x_{l−1} $]是指在层0、…、l中生成的特征图的级联−1.由于其密集连通性，我们将该网络架构称为密集卷积网络(dense Convolutional network。为了便于实现，我们将等式(2)中的H`(·)的多个输入连接到一个张量中。

Composite function. Motivated by [12], we define H` (·) as a composite function of three consecutive operations: batch normalization (BN) [14], followed by a rectified linear unit (ReLU) [6] and a 3 × 3 convolution (Conv).

复合函数。在[12]的启发下，我们将H`(·)定义为三个连续操作的复合函数：批归一化(BN)[14]，然后是校正线性单元(ReLU)[6]和3×3卷积(Conv)。

Pooling layers. The concatenation operation used in Eq. (2) is not viable when the size of feature-maps changes. However, an essential part of convolutional networks is down-sampling layers that change the size of feature-maps. To facilitate down-sampling in our architecture we divide the network into multiple densely connected dense blocks; see Figure 2. We refer to layers between blocks as transition layers, which do convolution and pooling. The transition layers used in our experiments consist of a batch normalization layer and an 1×1 convolutional layer followed by a 2×2 average pooling layer.

汇集层。当特征图的大小改变时，等式(2)中使用的连接操作是不可行的。然而，卷积网络的一个重要部分是对改变特征图大小的层进行下采样。为了便于架构中的下采样，我们将网络划分为多个密集连接的密集块; 参见图2。我们将块之间的层称为转换层，它执行卷积和池。我们实验中使用的过渡层包括一个批量归一化层和一个1×。

Growth rate. If each function H produces k featuremaps, it follows that the  th layer has k0 +k ×(` −1) input feature-maps, where k0 is the number of channels in the input layer. An important difference between DenseNet and existing network architectures is that DenseNet can have very narrow layers, e.g., k = 12. We refer to the hyperparameter k as the growth rate of the network. We show in Section 4 that a relatively small growth rate is sufficient to obtain state-of-the-art results on the datasets that we tested on. One explanation for this is that each layer has access to all the preceding feature-maps in its block and, therefore, to the network’s “collective knowledge”. One can view the feature-maps as the global state of the network. Each layer adds k feature-maps of its own to this state. The growth rate regulates how much new information each layer contributes to the global state. The global state, once written, can be accessed from everywhere within the network and, unlike in traditional network architectures, there is no need to replicate it from layer to layer.

增长率。如果每个函数H产生k个特征映射，则第H层具有k0+k×(`−1) 输入特征图，其中k0是输入层中的通道数。DenseNet和现有网络架构之间的一个重要区别是，DenseNetwork可以具有非常窄的层，例如，k＝12。我们将超参数k称为网络的增长率。我们在第4节中表明，相对较小的增长率足以获得我们测试的数据集的最新结果。对此的一个解释是，每个层都可以访问其块中的所有先前特征图，因此也可以访问网络的“集体知识”。可以将特征图视为网络的全局状态。每个图层都会将自己的k个要素地图添加到此状态。增长率决定了每一层对全局状态贡献的新信息量。全局状态一旦写入，就可以从网络中的任何地方访问，与传统的网络架构不同，不需要从一层复制到另一层。

Table 1: DenseNet architectures for ImageNet. The growth rate for all the networks is k = 32. Note that each “conv” layer shown in the table corresponds the sequence BN-ReLU-Conv. 
表1:ImageNet的DenseNet架构。所有网络的增长率为k=32。请注意，表中显示的每个“conv”层对应于序列BN-ReLU-conv。

Bottleneck layers. Although each layer only produces k output feature-maps, it typically has many more inputs. It has been noted in [37, 11] that a 1×1 convolution can be introduced as bottleneck layer before each 3×3 convolution to reduce the number of input feature-maps, and thus to improve computational efficiency. We find this design especially effective for DenseNet and we refer to our network with such a bottleneck layer, i.e., to the BN-ReLU-Conv(1× 1)-BN-ReLU-Conv(3×3) version of H` , as DenseNet-B. In our experiments, we let each 1×1 convolution produce 4k feature-maps.

瓶颈层。虽然每个层仅生成k个输出特征图，但它通常有更多的输入。在[37，11]中已经指出，可以在每个3×3卷积之前引入1×1卷积作为瓶颈层，以减少输入特征图的数量，从而提高计算效率。我们发现这种设计对DenseNet特别有效，我们将我们的网络称为具有这种瓶颈层的网络，即H‘的BN-ReLU-Conv(1×。在我们的实验中，我们让每个1×1卷积产生4k个特征图。

Compression. To further improve model compactness, we can reduce the number of feature-maps at transition layers. If a dense block contains m feature-maps, we let the following transition layer generate b θmc output featuremaps, where 0 <θ ≤1 is referred to as the compression factor. When θ = 1, the number of feature-maps across transition layers remains unchanged. We refer the DenseNet with θ <1 as DenseNet-C, and we set θ = 0.5 in our experiment. When both the bottleneck and transition layers with θ < 1 are used, we refer to our model as DenseNet-BC.

压缩。为了进一步提高模型的紧凑性，我们可以减少过渡层的特征图数量。如果稠密块包含m个特征映射，我们让下面的过渡层生成bθmc输出特征映射，其中0<θ≤1被称为压缩因子。θ=1时，过渡层上的特征图数量保持不变。我们将θ<1的DenseNet称为DenseNetwork-C，并在我们的实验中设置θ=0.5。当使用θ<1的瓶颈层和过渡层时，我们将我们的模型称为DenseNet BC。

Implementation Details. On all datasets except ImageNet, the DenseNet used in our experiments has three dense blocks that each has an equal number of layers. Before entering the first dense block, a convolution with 16 (or twice the growth rate for DenseNet-BC) output channels is performed on the input images. For convolutional layers with kernel size 3×3, each side of the inputs is zero-padded by one pixel to keep the feature-map size fixed. We use 1×1 convolution followed by 2×2 average pooling as transition layers between two contiguous dense blocks. At the end of the last dense block, a global average pooling is performed and then a softmax classifier is attached. The feature-map sizes in the three dense blocks are 32× 32, 16×16, and 8×8, respectively. We experiment with the basic DenseNet structure with configurations {L = 40, k = 12}, {L = 100, k = 12} and {L = 100, k = 24}. For DenseNetBC, the networks with configurations {L = 100, k = 12}, {L= 250, k= 24} and {L= 190, k= 40} are evaluated.

实施细节。在除ImageNet之外的所有数据集上，我们实验中使用的DenseNet有三个密集块，每个块都有相等数量的层。在进入第一个密集块之前，对输入图像执行16个(或DenseNet BC的两倍增长率)输出通道的卷积。对于内核大小为3×3的卷积层，输入的每一侧都被零填充一个像素，以保持特征图大小不变。我们使用1×1卷积和2×2平均池作为两个相邻致密块之间的过渡层。在最后一个密集块的末尾，执行全局平均池，然后附加softmax分类器。三个密集区块的特征图大小分别为32×32、16×16和8×8。我们对基本的DenseNet结构进行了实验，其配置为{L=40，k=12}，{L=100，k=12]和{L=10，k=24}。对于DenseNetBC，评估配置为{L=100，k=12}，{L=250，k=24}和{L=190，k=40}的网络。

In our experiments on ImageNet, we use a DenseNet-BC structure with 4 dense blocks on 224×224 input images. The initial convolution layer comprises 2k convolutions of size 7×7 with stride 2; the number of feature-maps in all other layers also follow from setting k. The exact network configurations we used on ImageNet are shown in Table 1.

在我们对ImageNet的实验中，我们在224×224个输入图像上使用了具有4个密集块的DenseNet BC结构。初始卷积层包括2k个大小为7×; 所有其他层中的特征图数量也从设置k开始。我们在ImageNet上使用的确切网络配置如表1所示。

## 4. Experiments
We empirically demonstrate DenseNet’s effectiveness on several benchmark datasets and compare with state-of-theart architectures, especially with ResNet and its variants.

我们通过经验证明了DenseNet在多个基准数据集上的有效性，并将其与最先进的架构(尤其是ResNet及其变体)进行了比较。

Table 2: Error rates (%) on CIFAR and SVHN datasets. k denotes network’s growth rate. Results that surpass all competing methods are bold and the overall best results are blue. “+” indicates standard data augmentation (translation and/or mirroring). ∗ indicates results run by ourselves. All the results of DenseNets without data augmentation (C10, C100, SVHN) are obtained using Dropout. DenseNets achieve lower error rates while using fewer parameters than ResNet. Without data augmentation, DenseNet performs better by a large margin.

表2:CIFAR和SVHN数据集的错误率(%)。k表示网络的增长率。超过所有竞争方法的结果都是大胆的，总体最佳结果是蓝色的。“+”表示标准数据增广(转换和/或镜像)。∗ 表示我们自己运行的结果。使用Dropout可获得未进行数据增广(C10、C100、SVHN)的DenseNets的所有结果。DenseNets在使用比ResNet更少的参数的同时实现更低的错误率。如果没有数据增广，DenseNet的性能将大大提高。

### 4.1. Datasets
CIFAR. The two CIFAR datasets [15] consist of colored natural images with 32×32 pixels. CIFAR-10 (C10) consists of images drawn from 10 and CIFAR-100 (C100) from 100 classes. The training and test sets contain 50,000 and 10,000 images respectively, and we hold out 5,000 training images as a validation set. We adopt a standard data augmentation scheme (mirroring/shifting) that is widely used for these two datasets [11, 13, 17, 22, 28, 20, 32, 34]. We denote this data augmentation scheme by a “+” mark at the end of the dataset name (e.g., C10+). For preprocessing, we normalize the data using the channel means and standard deviations. For the final run we use all 50,000 training images and report the final test error at the end of training.

SVHN. The Street View House Numbers (SVHN) dataset [24] contains 32×32 colored digit images. There are 73,257 images in the training set, 26,032 images in the test set, and 531,131 images for additional training. Following common practice [7, 13, 20, 22, 30] we use all the training data without any data augmentation, and a validation set with 6,000 images is split from the training set. We select the model with the lowest validation error during training and report the test error. We follow [42] and divide the pixel values by 255 so they are in the [0, 1] range.

ImageNet. The ILSVRC 2012 classification dataset [2] consists 1.2 million images for training, and 50,000 for validation, from 1, 000 classes. We adopt the same data augmentation scheme for training images as in [8, 11, 12], and apply a single-crop or 10-crop with size 224×224 at test time. Following [11, 12, 13], we report classification errors on the validation set.

### 4.2. Training
All the networks are trained using stochastic gradient descent (SGD). On CIFAR and SVHN we train using batch size 64 for 300 and 40 epochs, respectively. The initial learning rate is set to 0.1, and is divided by 10 at 50% and 75% of the total number of training epochs. On ImageNet, we train models for 90 epochs with a batch size of 256. The learning rate is set to 0.1 initially, and is lowered by 10 times at epoch 30 and 60. Note that a naive implementation of DenseNet may contain memory inefficiencies. To reduce the memory consumption on GPUs, please refer to our technical report on the memory-efficient implementation of DenseNets [26].

Following [8], we use a weight decay of 10−4 and a Nesterov momentum [35] of 0.9 without dampening. We adopt the weight initialization introduced by [10]. For the three datasets without data augmentation, i.e., C10, C100 and SVHN, we add a dropout layer [33] after each convolutional layer (except the first one) and set the dropout rate to 0.2. The test errors were only evaluated once for each task and model setting.

Table 3: The top-1 and top-5 error rates on the ImageNet validation set, with single-crop / 10- crop testing. 

Figure 3: Comparison of the DenseNets and ResNets top-1 error rates (single-crop testing) on the ImageNet validation dataset as a function of learned parameters (left) and FLOPs during test-time (right). 

### 4.3. Classification Results on CIFAR and SVHN
We train DenseNets with different depths, L, and growth rates, k. The main results on CIFAR and SVHN are shown in Table 2. To highlight general trends, we mark all results that outperform the existing state-of-the-art in boldface and the overall best result in blue.

Accuracy. Possibly the most noticeable trend may originate from the bottom row of Table 2, which shows that DenseNet-BC with L = 190 and k = 40 outperforms the existing state-of-the-art consistently on all the CIFAR datasets. Its error rates of 3.46% on C10+ and 17.18% on C100+ are significantly lower than the error rates achieved by wide ResNet architecture [42]. Our best results on C10 and C100 (without data augmentation) are even more encouraging: both are close to 30% lower than FractalNet with drop-path regularization [17]. On SVHN, with dropout, the DenseNet with L = 100 and k = 24 also surpasses the current best result achieved by wide ResNet. However, the 250-layer DenseNet-BC doesn’t further improve the performance over its shorter counterpart. This may be explained by that SVHN is a relatively easy task, and extremely deep models may overfit to the training set.

Capacity. Without compression or bottleneck layers, there is a general trend that DenseNets perform better as L and k increase. We attribute this primarily to the corresponding growth in model capacity. This is best demonstrated by the column of C10+ and C100+. On C10+, the error drops from 5.24% to 4.10% and finally to 3.74% as the number of parameters increases from 1.0M, over 7.0M to 27.2M. On C100+, we observe a similar trend. This suggests that DenseNets can utilize the increased representational power of bigger and deeper models. It also indicates that they do not suffer from overfitting or the optimization difficulties of residual networks [11].

Parameter Efficiency. The results in Table 2 indicate that DenseNets utilize parameters more efficiently than alternative architectures (in particular, ResNets). The DenseNetBC with bottleneck structure and dimension reduction at transition layers is particularly parameter-efficient. For example, our 250-layer model only has 15.3M parameters, but it consistently outperforms other models such as FractalNet and Wide ResNets that have more than 30M parameters. We also highlight that DenseNet-BC with L = 100 and k = 12 achieves comparable performance (e.g., 4.51% vs 4.62% error on C10+, 22.27% vs 22.71% error on C100+) as the 1001-layer pre-activation ResNet using 90% fewer parameters. Figure 4 (right panel) shows the training loss and test errors of these two networks on C10+. The 1001-layer deep ResNet converges to a lower training loss value but a similar test error. We analyze this effect in more detail below.

Overfitting. One positive side-effect of the more efficient use of parameters is a tendency of DenseNets to be less prone to overfitting. We observe that on the datasets without data augmentation, the improvements of DenseNet architectures over prior work are particularly pronounced. On C10, the improvement denotes a 29% relative reduction in error from 7.33% to 5.19%. On C100, the reduction is about 30% from 28.20% to 19.64%. In our experiments, we observed potential overfitting in a single setting: on C10, a 4× growth of parameters produced by increasing k = 12 to k = 24 lead to a modest increase in error from 5.77% to 5.83%. The DenseNet-BC bottleneck and compression layers appear to be an effective way to counter this trend.

### 4.4. Classification Results on ImageNet
We evaluate DenseNet-BC with different depths and growth rates on the ImageNet classification task, and compare it with state-of-the-art ResNet architectures. To ensure a fair comparison between the two architectures, we eliminate all other factors such as differences in data preprocessing and optimization settings by adopting the publicly available Torch implementation for ResNet by [8] (https://github.com/facebook/fb.resnet.torch). We simply replace the ResNet model with the DenseNetBC network, and keep all the experiment settings exactly the same as those used for ResNet.

Figure 4: Left: Comparison of the parameter efficiency on C10+ between DenseNet variations. Middle: Comparison of the parameter efficiency between DenseNet-BC and (pre-activation) ResNets. DenseNet-BC requires about 1/3 of the parameters as ResNet to achieve comparable accuracy. Right: Training and testing curves of the 1001-layer pre-activation ResNet [12] with more than 10M parameters and a 100-layer DenseNet with only 0.8M parameters.

We report the single-crop and 10-crop validation errors of DenseNets on ImageNet in Table 3. Figure 3 shows the single-crop top-1 validation errors of DenseNets and ResNets as a function of the number of parameters (left) and FLOPs (right). The results presented in the figure reveal that DenseNets perform on par with the state-of-the-art ResNets, whilst requiring significantly fewer parameters and computation to achieve comparable performance. For example, a DenseNet-201 with 20M parameters model yields similar validation error as a 101-layer ResNet with more than 40M parameters. Similar trends can be observed from the right panel, which plots the validation error as a function of the number of FLOPs: a DenseNet that requires as much computation as a ResNet-50 performs on par with a ResNet-101, which requires twice as much computation.

It is worth noting that our experimental setup implies that we use hyperparameter settings that are optimized for ResNets but not for DenseNets. It is conceivable that more extensive hyper-parameter searches may further improve the performance of DenseNet on ImageNet.

## 5. Discussion
Superficially, DenseNets are quite similar to ResNets: Eq. (2) differs from Eq. (1) only in that the inputs to H` (·) are concatenated instead of summed. However, the implications of this seemingly small modification lead to substantially different behaviors of the two network architectures.

从表面上看，DenseNets与ResNets非常相似：等式(2)与等式(1)的不同之处仅在于H`(·)的输入是级联的，而不是求和的。然而，这一看似微小的修改的含义导致了两种网络架构的不同行为。

Model compactness. As a direct consequence of the input concatenation, the feature-maps learned by any of the DenseNet layers can be accessed by all subsequent layers. This encourages feature reuse throughout the network, and leads to more compact models.

模型紧凑性。作为输入连接的直接结果，任何DenseNet层学习的特征地图都可以被所有后续层访问。这鼓励了整个网络中的特性重用，并导致了更紧凑的模型。

The left two plots in Figure 4 show the result of an experiment that aims to compare the parameter efficiency of all variants of DenseNets (left) and also a comparable ResNet architecture (middle). We train multiple small networks with varying depths on C10+ and plot their test accuracies as a function of network parameters. In comparison with other popular network architectures, such as AlexNet [16] or VGG-net [29], ResNets with pre-activation use fewer parameters while typically achieving better results [12]. Hence, we compare DenseNet (k = 12) against this architecture. The training setting for DenseNet is kept the same as in the previous section.

图4中左边的两个图显示了一个实验的结果，该实验旨在比较DenseNets(左)和ResNet架构(中)的所有变体的参数效率。我们在C10+上训练具有不同深度的多个小网络，并将其测试精度绘制为网络参数的函数。与其他流行的网络架构(如AlexNet[16]或VGG-net[29])相比，具有预激活功能的ResNets使用更少的参数，同时通常会获得更好的结果[12]。因此，我们将DenseNet(k=12)与此架构进行比较。DenseNet的训练设置与上一节中的相同。

The graph shows that DenseNet-BC is consistently the most parameter efficient variant of DenseNet. Further, to achieve the same level of accuracy, DenseNet-BC only requires around 1/3 of the parameters of ResNets (middle plot). This result is in line with the results on ImageNet we presented in Figure 3. The right plot in Figure 4 shows that a DenseNet-BC with only 0.8M trainable parameters is able to achieve comparable accuracy as the 1001-layer (pre-activation) ResNet [12] with 10.2M parameters.

该图显示，DenseNet BC始终是DenseNetwork的参数效率最高的变体。此外，为了达到相同的精度水平，DenseNet BC只需要ResNets(中间图)参数的1/3左右。该结果与我们在图3中所示的ImageNet上的结果一致。图4中的右图显示，仅具有0.8M可训练参数的DenseNet BC能够实现与具有10.2M参数的1001层(预激活)ResNet[12]相当的精度。

Implicit Deep Supervision. One explanation for the improved accuracy of dense convolutional networks may be that individual layers receive additional supervision from the loss function through the shorter connections. One can interpret DenseNets to perform a kind of “deep supervision”. The benefits of deep supervision have previously been shown in deeply-supervised nets (DSN; [20]), which have classifiers attached to every hidden layer, enforcing the intermediate layers to learn discriminative features.

隐性深度监管。稠密卷积网络精度提高的一个解释可能是，各个层通过较短的连接从损耗函数接收额外的监督。人们可以将DenseNets解释为执行某种“深度监督”。深度监督的好处以前已经在深度监督网络(DSN; [20])中显示出来，该网络将分类器连接到每个隐藏层，强制中间层学习区别特征。

DenseNets perform a similar deep supervision in an implicit fashion: a single classifier on top of the network provides direct supervision to all layers through at most two or three transition layers. However, the loss function and gradient of DenseNets are substantially less complicated, as the same loss function is shared between all layers.

DenseNets以隐式方式执行类似的深度监控：网络顶部的单个分类器通过最多两个或三个过渡层向所有层提供直接监控。然而，DenseNets的损失函数和梯度基本上不那么复杂，因为所有层之间共享相同的损失函数。

Stochastic vs. deterministic connection. There is an interesting connection between dense convolutional networks and stochastic depth regularization of residual networks [13]. In stochastic depth, layers in residual networks are randomly dropped, which creates direct connections betest error (%) test error (%) test error (%) training loss tween the surrounding layers. As the pooling layers are never dropped, the network results in a similar connectivity pattern as DenseNet: there is a small probability for any two layers, between the same pooling layers, to be directly connected—if all intermediate layers are randomly dropped. Although the methods are ultimately quite different, the DenseNet interpretation of stochastic depth may provide insights into the success of this regularizer.

随机与确定性连接。稠密卷积网络和残差网络的随机深度正则化之间存在有趣的联系[13]。在随机深度中，残差网络中的层被随机丢弃，这在周围层之间的测试误差(%)、测试错误(%)和训练损失(%)之间产生了直接联系。由于池层从未被丢弃，因此网络会产生与DenseNet相似的连接模式：如果所有中间层都被随机丢弃，那么在相同池层之间的任何两个层直接连接的可能性很小。尽管这些方法最终是完全不同的，但对随机深度的DenseNet解释可能会为该正则化器的成功提供见解。

Feature Reuse. By design, DenseNets allow layers access to feature-maps from all of its preceding layers (although sometimes through transition layers). We conduct an experiment to investigate if a trained network takes advantage of this opportunity. We first train a DenseNet on C10+ with L = 40 and k = 12. For each convolutional layer  within a block, we compute the average (absolute) weight assigned to connections with layer s. Figure 5 shows a heat-map for all three dense blocks. The average absolute weight serves as a surrogate for the dependency of a convolutional layer on its preceding layers. A red dot in position (`, s) indicates that the layer  makes, on average, strong use of feature-maps produced s-layers before. Several observations can be made from the plot:
1. All layers spread their weights over many inputs within the same block. This indicates that features extracted by very early layers are, indeed, directly used by deep layers throughout the same dense block.
2. The weights of the transition layers also spread their weight across all layers within the preceding dense block, indicating information flow from the first to the last layers of the DenseNet through few indirections.
3. The layers within the second and third dense block consistently assign the least weight to the outputs of the transition layer (the top row of the triangles), indicating that the transition layer outputs many redundant features (with low weight on average). This is in keeping with the strong results of DenseNet-BC where exactly these outputs are compressed.
4. Although the final classification layer, shown on the very right, also uses weights across the entire dense block, there seems to be a concentration towards final feature-maps, suggesting that there may be some more high-level features produced late in the network.

功能重用。根据设计，DenseNets允许图层从其前面的所有图层访问要素地图(尽管有时通过过渡图层)。我们进行了一项实验，以调查受过训练的网络是否利用了这个机会。我们首先在C10+上训练L=40和k=12的DenseNet。对于块内的每个卷积层，我们计算分配给层s连接的平均(绝对)权重。图5显示了所有三个密集块的热图。平均绝对权重用作卷积层对其先前层的依赖性的替代。位置(`，s)中的一个红点表示该层平均强烈地使用了之前生成的s层特征图。从图中可以观察到以下几点：
1. 所有层将其权重分布在同一块内的多个输入上。这表明，早期地层提取的特征实际上被同一致密区块的深层直接使用。
2. 过渡层的权重也将其权重分布在前一个密集块内的所有层上，这表明信息从密集网的第一层到最后一层通过几个间接方向流动。
3. 第二和第三密集块内的层始终将最小权重分配给过渡层的输出(三角形的顶行)，这表明过渡层输出了许多冗余特征(平均权重较低)。这与DenseNet BC的强大结果是一致的，其中这些输出正是压缩的。
4. 虽然最后的分类层(如右图所示)也使用了整个密集块的权重，但似乎集中在最终的特征图上，这表明网络后期可能会产生一些更高级的特征。

Figure 5: The average absolute filter weights of convolutional layers in a trained DenseNet. The color of pixel (s, ) encodes the average L1 norm (normalized by number of input feature-maps) of the weights connecting convolutional layer s to ` within a dense block. Three columns highlighted by black rectangles correspond to two transition layers and the classification layer. The first row encodes weights connected to the input layer of the dense block.

图5：经过训练的DenseNet中卷积层的平均绝对滤波器权重。像素的颜色编码将卷积层连接到稠密块内的权重的平均L1范数(由输入特征映射的数量归一化)。由黑色矩形突出显示的三列对应于两个过渡层和分类层。第一行编码连接到密集块的输入层的权重。

## 6. Conclusion
We proposed a new convolutional network architecture, which we refer to as Dense Convolutional Network (DenseNet). It introduces direct connections between any two layers with the same feature-map size. We showed that DenseNets scale naturally to hundreds of layers, while exhibiting no optimization difficulties. In our experiments, DenseNets tend to yield consistent improvement in accuracy with growing number of parameters, without any signs of performance degradation or overfitting. Under multiple settings, it achieved state-of-the-art results across several highly competitive datasets. Moreover, DenseNets require substantially fewer parameters and less computation to achieve state-of-the-art performances. Because we adopted hyperparameter settings optimized for residual networks in our study, we believe that further gains in accuracy of DenseNets may be obtained by more detailed tuning of hyperparameters and learning rate schedules.

我们提出了一种新的卷积网络结构，我们称之为稠密卷积网络(Dense convolutional network，DenseNet)。它引入了具有相同特征图大小的任意两个层之间的直接连接。我们表明，DenseNets可以自然地扩展到数百层，而没有表现出优化困难。在我们的实验中，随着参数数量的增加，DenseNets倾向于在精度方面不断提高，而没有任何性能下降或过度拟合的迹象。在多种设置下，它在多个极具竞争力的数据集中实现了最先进的结果。此外，DenseNets需要更少的参数和更少的计算来实现最先进的性能。由于我们在研究中采用了针对残差网络优化的超参数设置，我们相信，通过更详细地调整超参数和学习速率计划，可以进一步提高DenseNets的准确性。

Whilst following a simple connectivity rule, DenseNets naturally integrate the properties of identity mappings, deep supervision, and diversified depth. They allow feature reuse throughout the networks and can consequently learn more compact and, according to our experiments, more accurate models. Because of their compact internal representations and reduced feature redundancy, DenseNets may be good feature extractors for various computer vision tasks that build on convolutional features, e.g., [4, 5]. We plan to study such feature transfer with DenseNets in future work.

在遵循简单的连通性规则的同时，DenseNets自然地集成了身份映射、深度监控和多样化深度的属性。根据我们的实验，它们允许在整个网络中重用特征，从而可以学习更紧凑、更准确的模型。由于其紧凑的内部表示和减少的特征冗余，DenseNets可能是基于卷积特征的各种计算机视觉任务的良好特征提取器，例如[4,5]。我们计划在未来的工作中使用DenseNets研究这种功能迁移。

## Acknowledgements. 
The authors are supported in part by the NSF III-1618134, III-1526012, IIS-1149882, the Of- fice of Naval Research Grant N00014-17-1-2175 and the Bill and Melinda Gates foundation. GH is supported by the International Postdoctoral Exchange Fellowship Program of China Postdoctoral Council (No.20150015). ZL is supported by the National Basic Research Program of China Grants 2011CBA00300, 2011CBA00301, the NSFC 61361136003. We also thank Daniel Sedra, Geoff Pleiss and Yu Sun for many insightful discussions.

## Reference
1. C. Cortes, X. Gonzalvo, V. Kuznetsov, M. Mohri, and S. Yang. Adanet: Adaptive structural learning of artificial neural networks. arXiv preprint arXiv:1607.01097, 2016. 2 Source layer (s)
2. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. FeiFei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 5
3. S. E. Fahlman and C. Lebiere. The cascade-correlation learning architecture. In NIPS, 1989. 2
4. J. R. Gardner, M. J. Kusner, Y. Li, P. Upchurch, K. Q. Weinberger, and J. E. Hopcroft. Deep manifold traversal: Changing labels with convolutional features. arXiv preprint arXiv:1511.06421, 2015. 8
5. L. Gatys, A. Ecker, and M. Bethge. A neural algorithm of artistic style. Nature Communications, 2015. 8
6. X. Glorot, A. Bordes, and Y. Bengio. Deep sparse rectifier neural networks. In AISTATS, 2011. 3
7. I. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. In ICML, 2013. 5
8. S. Gross and M. Wilber. Training and investigating residual nets, 2016. 5, 7
9. B. Hariharan, P. Arbeláez, R. Girshick, and J. Malik. Hypercolumns for object segmentation and fine-grained localization. In CVPR, 2015. 2
10. K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015. 5
11. K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016. 1, 2, 3, 4, 5, 6
12. K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016. 2, 3, 5, 7
13. G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger. Deep networks with stochastic depth. In ECCV, 2016. 1, 2, 5, 8
14. S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015. 3
15. A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Tech Report, 2009. 5
16. A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012. 3, 7
17. G. Larsson, M. Maire, and G. Shakhnarovich. Fractalnet: Ultra-deep neural networks without residuals. arXiv preprint arXiv:1605.07648, 2016. 1, 3, 5, 6
18. Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1(4):541–551, 1989. 1
19. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. 1, 3
20. C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeplysupervised nets. In AISTATS, 2015. 2, 3, 5, 7
21. Q. Liao and T. Poggio. Bridging the gaps between residual learning, recurrent neural networks and visual cortex. arXiv preprint arXiv:1604.03640, 2016. 2
22. M. Lin, Q. Chen, and S. Yan. Network in network. In ICLR, 2014. 3, 5
23. J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015. 2
24. Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. Reading digits in natural images with unsupervised feature learning, 2011. In NIPS Workshop, 2011. 5
25. M. Pezeshki, L. Fan, P. Brakel, A. Courville, and Y. Bengio. Deconstructing the ladder network architecture. In ICML, 2016. 3
26. G. Pleiss, D. Chen, G. Huang, T. Li, L. van der Maaten, and K. Q. Weinberger. Memory-efficient implementation of densenets. arXiv preprint arXiv:1707.06990, 2017. 5
27. A. Rasmus, M. Berglund, M. Honkala, H. Valpola, and T. Raiko. Semi-supervised learning with ladder networks. In NIPS, 2015. 3
28. A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio. Fitnets: Hints for thin deep nets. In ICLR, 2015. 5
29. O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. IJCV. 1, 7
30. P. Sermanet, S. Chintala, and Y. LeCun. Convolutional neural networks applied to house numbers digit classification. In ICPR, pages 3288–3291. IEEE, 2012. 5
31. P. Sermanet, K. Kavukcuoglu, S. Chintala, and Y. LeCun. Pedestrian detection with unsupervised multi-stage feature learning. In CVPR, 2013. 2
32. J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014. 5
33. N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. JMLR, 2014. 6
34. R. K. Srivastava, K. Greff, and J. Schmidhuber. Training very deep networks. In NIPS, 2015. 1, 2, 5
35. I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in deep learning. In ICML, 2013. 5
36. C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015. 2, 3
37. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016. 2, 3, 4
38. S. Targ, D. Almeida, and K. Lyman. Resnet in resnet: Generalizing residual architectures. arXiv preprint arXiv:1603.08029, 2016. 2
39. J. Wang, Z. Wei, T. Zhang, and W. Zeng. Deeply-fused nets. arXiv preprint arXiv:1605.07716, 2016. 3
40. B. M. Wilamowski and H. Yu. Neural network learning without backpropagation. IEEE Transactions on Neural Networks, 21(11):1793–1803, 2010. 2
41. S. Yang and D. Ramanan. Multi-scale recognition with dagcnns. In ICCV, 2015. 2
42. S. Zagoruyko and N. Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016. 3, 5, 6
43. Y. Zhang, K. Lee, and H. Lee. Augmenting supervised neural networks with unsupervised objectives for large-scale image classification. In ICML, 2016. 3