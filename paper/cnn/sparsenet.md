# SparseNet: A Sparse DenseNet for Image Classification
稀疏网络：一种用于图像分类的稀疏密集网络 2018-4-15 https://arxiv.org/abs/1804.05340

## Abstract. 
Deep neural networks have made remarkable progresses on various computer vision tasks. Recent works have shown that depth, width and shortcut connections of networks are all vital to their performances. In this paper, we introduce a method to sparsify DenseNet which can reduce connections of a L-layer DenseNet from O(L2 ) to O(L), and thus we can simultaneously increase depth, width and connections of neural networks in a more parameter-efficient and computation-efficient way. Moreover, an attention module is introduced to further boost our network’s performance. We denote our network as SparseNet. We evaluate SparseNet on datasets of CIFAR(including CIFAR10 and CIFAR100) and SVHN. Experiments show that SparseNet can obtain improvements over the state-of-the-art on CIFAR10 and SVHN. Furthermore, while achieving comparable performances as DenseNet on these datasets, SparseNet is ×2.6 smaller and ×3.7 faster than the original DenseNet. Keywords: neural networks DenseNet SparseNet 

深度神经网络在各种计算机视觉任务上取得了显著进展。最近的研究表明，网络的深度、宽度和快捷连接对其性能都至关重要。在本文中，我们介绍了一种稀疏化DenseNet的方法，该方法可以将L层DenseNetwork的连接从O(L2)减少到O(L)，从而可以以更高效的参数和计算方式同时增加神经网络的深度、宽度和连接。此外，还引入了关注模块，以进一步提高网络性能。我们将网络称为SparseNet。我们在CIFAR(包括CIFAR10和CIFAR100)和SVHN的数据集上评估SparseNet。实验表明，SparseNet可以在CIFAR10和SVHN上获得优于最新技术的改进。此外，尽管在这些数据集上实现了与DenseNet相当的性能，但SparseNet比原始DenseNetwork小×2.6，快×3.7。关键词：神经网络密集网稀疏网.

## 1 Introduction
Deep convolutional neural networks have achieved great successes on many computer vision tasks, such as object classification, detection and segmentation [1] [2] [3]. ‘Depth’ played a significant role while neural networks are achieving their successes. From AlexNet[1] to VGGNet[4] and GoogLeNet[5], their performances on various computer vision tasks are boosting as network’s depth is increasing.

深度卷积神经网络在许多计算机视觉任务中取得了巨大的成功，例如对象分类、检测和分割[1][2][3]当神经网络取得成功时，“深度”发挥了重要作用。从AlexNet[1]到VGGNet[4]和GoogleLeNet[5]，随着网络深度的增加，他们在各种计算机视觉任务上的表现都在提高。

Experiments[6] have shown if we simply stack layers without changing network’s structure, its performance would get worse otherwise. Because gradients of network’s parameters will vanish as depth is increasing. To settle this problem, He[6] proposed ResNet, which introduced a residual learning framework by adding identity-mapping shortcuts. ResNet extended its depth up to over 100 layers and achieved state-of-art performances in many computer vision tasks. However, when ResNet is getting deeper(e.g. over 1000), it will suffer from the overfitting problem.

实验[6]表明，如果我们简单地堆叠层而不改变网络的结构，其性能会变得更差。因为随着深度的增加，网络参数的梯度将消失。为了解决这个问题，他[6]提出了ResNet，它通过添加身份映射快捷方式引入了一个剩余学习框架。ResNet将其深度扩展到100多层，并在许多计算机视觉任务中实现了最先进的性能。然而，当ResNet越来越深(例如超过1000)时，它将受到过度拟合问题的影响。

Huang[7] proposed a new training procedure, named stochastic depth, solved this problem. Take ResNet for example, Huang[7] trained shallower subnetworks by randomly dropping residual modules(while retaining shortcut connections). The vanishing-gradient problem has been alleviated since only shallower networks are trained in the training phase. This training procedure can extend depth of networks to over 1000 layers(e.g. 1202 layers) and the performance on image classification has been further improved.

黄[7]提出了一种新的训练程序，称为随机深度，解决了这个问题。以ResNet为例，Huang[7]通过随机丢弃剩余模块(同时保留快捷连接)来训练较浅的子网络。由于在训练阶段只训练较浅的网络，所以消失梯度问题得到了缓解。此训练程序可将网络的深度扩展到1000层以上(例如1202层)，图像分类性能得到进一步提高。

Zagoruyko[8] improved ResNet from another aspect. He introduced a wider(more channels in convolution layers) and shallower ResNet variant. The performance of wide ResNet with only 16 layers exceeds that of original ResNet with over 1000 layers. Another benefit brought by wide ResNet is the training is very fast since it can take advantage of the parallel of GPU computing.

Zagoruyko[8]从另一个方面改进了ResNet。他引入了更宽(卷积层中的通道更多)和更浅的ResNet变体。只有16层的宽ResNet的性能超过了超过1000层的原始ResNet。宽ResNet带来的另一个好处是训练速度非常快，因为它可以利用GPU计算的并行性。

By gradually increasing width of neural networks, Han[9] presented deep pyramidal residual Networks. For the original ResNet, width only doubled after downsampling happened. For example, there are 4 modules in original ResNet[6]: Conv2 x, Conv3 x, Conv4 x and Conv5 x. Width for each module are 64, 128, 256 and 512. Within every module, dimensions are all the same. For pyramidal residual networks, width for each residual unit are always increasing no matter they are in the same module or not. Experiments shown pyramidal residual networks had superior generalization ability compared to the original residual networks. So except increasing the depth, properly increasing width is also another way to boost network’s performance.

通过逐渐增加神经网络的宽度，Han[9]提出了深度金字塔残差网络。对于原始ResNet，宽度仅在下采样发生后增加一倍。例如，原始ResNet[6]中有4个模块：Conv2 x、Conv3 x、Conv 4 x和Conv5 x。每个模块的宽度分别为64、128、256和512。每个模块内的尺寸都相同。对于金字塔残差网络，每个残差单元的宽度总是在增加，无论它们是否在同一模块中。实验表明，与原始残差网络相比，金字塔残差网络具有更好的泛化能力。因此，除了增加深度之外，适当增加宽度也是提高网络性能的另一种方式。

Besides increasing depth or width, increasing number of shortcut connections is another effective way of improving network’s performance. It can gain network’s performance from two aspects. 
1. It shortens the distance between input and output and thus alleviates the vanishing-gradient problem with shorter forward flows. Highway networks[10] and ResNet[6] proposed different ways of shortcut connections, both of which made training easier. 
2. Shortcut connections can take advantage of multi-scale feature maps, which can improve performances on various computer vision tasks[11][3][12][13].

除了增加深度或宽度之外，增加快捷连接的数量也是提高网络性能的另一种有效方法。它可以从两个方面获得网络性能。
1. 它缩短了输入和输出之间的距离，从而缓解了具有更短前向流的消失梯度问题。公路网[10]和ResNet[6]提出了不同的快捷连接方式，这两种方式都使培训变得更容易。
2. 快捷连接可以利用多尺度特征图，这可以提高各种计算机视觉任务的性能[11][3][12][13]。

Huang[14]takes this idea to the extreme. He proposed DenseNet, for the l th layer of which, it takes all previous (l − 1) layers as its input(connections of this layer is O(l)). By this kind of network structure design, it not only alleviates vanishing-gradient problem, but also achieves better feature reuse. DenseNet achieves superior performance on datasets of CIFAR-10,CIFAR-100 and SVHN. However, it has its own disadvantages. There are total L(L − 1) 2 connections for a L-layer DenseNet. The excessive connections not only decrease networks’ computation-efficiency and parameter-efficiency, but also make networks more prone to overfitting. As we can see from upper of Fig.1, When I modify connections of a 40-layer DenseNet, the test error rates on CIFAR10 first decrease and then increase as connection is increasing. When the connections is 22, the error rate reaches the lowest, 5.11%. However, as we can see from bottom of Fig.1, error rates on the training datasets is decreasing as connections is increasing.

黄[14]将这种想法推向了极端。他提出了DenseNet，对于它的第1层，它采用了所有先前的(1)− 1) 层作为其输入(该层的连接是O(l))。通过这种网络结构设计，不仅缓解了消失梯度问题，而且实现了更好的特征重用。DenseNet在CIFAR-10、CIFAR-100和SVHN数据集上实现了优异的性能。然而，它也有自己的缺点。总共有L(L− 1) 用于L层DenseNet的2个连接。过度连接不仅降低了网络的计算效率和参数效率，而且使网络更容易过度拟合。从图1的上部可以看到，当我修改40层DenseNet的连接时，CIFAR10上的测试错误率会随着连接的增加先降低，然后增加。当连接数为22时，错误率最低，为5.11%。然而，正如我们从图1底部看到的，随着连接的增加，训练数据集的错误率正在下降。

To settle this problem, we proposed a method to sparsify DenseNet. Zeiler[15] found out that for a deep neural network, shallower layers can learn concrete features, whereas deeper layers can learn abstract features. Based on this observation, we can drop connections from middle layers and reduce connections for each layer from O(n) to O(1). So total connections of the sparsified DenseNet is O(n). As we can see in Fig. 2, left is a small part of DenseNet, right is a small part of SparseNet. the dotted line are dropped connections. So our idea for sparsifing is simply dropping connections from middle layers and only retaining the nearest and farthest connections. And then we can extend the network to deeper or wider, which would result in better performance. As we can see in Fig. 2, while keeping the overall parameters unchanged, by dropping some connections and then extend network’s width or depth, the performance of networks are getting better.

为了解决这个问题，我们提出了一种稀疏DenseNet的方法。Zeiler[15]发现，对于深度神经网络，较浅层可以学习具体特征，而较深层可以学习抽象特征。根据这一观察，我们可以从中间层删除连接，并将每个层的连接从O(n)减少到O(1)。所以稀疏密集网的总连接数是O(n)。如图2所示，左边是DenseNet的一小部分，右边是SparseNet的小部分。虚线表示断开的连接。因此，我们的稀疏化思想是简单地从中间层删除连接，只保留最近和最远的连接。然后我们可以将网络扩展到更深或更广的地方，这将带来更好的性能。如图2所示，在保持总体参数不变的同时，通过删除一些连接，然后扩展网络的宽度或深度，网络的性能正在变得更好。

Fig. 1: test/train error rate on CIFAR10 of different paths(connections) in DenseNet 
图1:DenseNet中不同路径(连接)的CIFAR10上的测试/列车错误率

Fig. 2: Left is DenseNet, input to layers are from all previous layers; right is SparseNet, dotted lines are dropped connections. input to layers are from at most two previous layers.
图2：左边是DenseNet，层的输入来自所有先前层; 右边是SparseNet，虚线是丢弃的连接。层的输入来自至多两个先前层。

Beside changing networks’ depth, width or shortcut connections to boost model’s performance, we can also borrow our knowledge about human visual processing mechanism. The most significant feature of human visual system lies in its attention mechanism. When we skim images, we can automatically focus on important regions, and then devote more attentional resources to those regions. Recently some researchers on computer vision are enlightened by attention mechanism of human visual system. They designed mechanisms which can firstly select most significant regions in an image(e.g. foreground regions for object segmentation), and then pay more attention to those regions. Attention mechanism has made progresses on various computer vision tasks, such as image classification[16],image segmentation[17], human pose estimation[18] and so on. Recently, Hu[19] took advantage of attention mechanism from another perspective, he put different amounts of ‘attentional resources’ to different channels of feature maps. To be specific, he increases weights on channels which have informative features and decreases weights on channels which have less useful features. He proposed SE module, which can calibrate feature responses adaptively for different channels in cost of slightly more computation and parameters.

除了改变网络的深度、宽度或快捷连接以提高模型的性能之外，我们还可以借用我们对人类视觉处理机制的知识。人类视觉系统最显著的特征在于它的注意机制。当我们浏览图像时，我们可以自动聚焦于重要区域，然后将更多的注意力投入到这些区域。近年来，一些计算机视觉研究者受到了人类视觉系统注意机制的启发。他们设计了一种机制，可以首先选择图像中最重要的区域(例如用于对象分割的前景区域)，然后更加关注这些区域。注意机制在各种计算机视觉任务上取得了进展，如图像分类[16]、图像分割[17]、人体姿势估计[18]等。最近，胡[19]从另一个角度利用了注意机制，他将不同数量的“注意资源”放在了特征图的不同通道上。具体来说，他增加了具有信息特征的通道的权重，并减少了具有较少有用特征的频道。他提出了SE模块，该模块可以针对不同的通道自适应地校准特征响应，只需稍微增加计算量和参数。

Fig. 3: Wider Sparse DenseNet and Deeper Sparse DenseNet are networks extended to wider or deeper after drop some middle connections. setup of DenseNet is k(growth rate)=12, layer=40; setup of Wider Sparse DenseNet is k=16, layer=40, path(total connections)=12; setup of Deeper Sparse DenseNet is k=12, path=12, layer=64.
图3：Wider Sparse DenseNet和Deeper Sparse DenseNet是在删除一些中间连接后扩展到更宽或更深的网络。DenseNet的设置为k(增长率)=12，层=40; 宽稀疏密集网的设置为k=16，层=40，路径(总连接)=12; 深度稀疏密集网的设置为k=12，路径=12，层=64。

The SE module has been proved to be effective for ResNet[6],Inception[5] and Inception ResNet[20]. However, the improvement is ignorable when it applied to our SparseNet. To settle this problem, we present a new attention mechanism. Its structure is shown in Fig 4. It consists of one global average pooling layer and two convolution modules(includes convolution layer, ReLU layer and batch normalization layer). Borrowing idea of shortcut connections, outputs of both global average pooling layer and the first convolution module are taken as input to the second convolution module. And then outputs of the second convolution module are used to calibrate the original network’s output.

SE模块已被证明对ResNet[6]、Inception[5]和Inception ResNet[20]有效。然而，当它应用于我们的SparseNet时，这种改进是不可忽视的。为了解决这个问题，我们提出了一种新的注意机制。它的结构如图4所示。它由一个全局平均池层和两个卷积模块(包括卷积层、ReLU层和批归一化层)组成。借用快捷连接的思想，将全局平均池层和第一卷积模块的输出作为第二卷积模块输入。然后使用第二卷积模块的输出来校准原始网络的输出。

There are two contributions in our paper: 
1. We present an effective way to sparsify DenseNet, which can improve network’s performance by simultaneously increasing depth, width and shortcut connections of networks. Besides, 
2. we also proposed an attention mechanism, which can further boost network’s performance. 

本文有两个贡献：
1. 我们提出了一种有效的稀疏化DenseNet的方法，它可以通过同时增加网络的深度、宽度和快捷连接来提高网络性能。此外
2. 我们还提出了一种注意机制，可以进一步提高网络的性能。

## 2 Related work
### 2.1 Convolutional neural networks
Since 2012, neural networks, as a new way of constructing models, have made big steps in various computer vision regions. AlexNet[1], which consists of 8 layers, won the image classification championship of ILSVRC-2012. It reduced error rate on ImageNet dataset from 25.8%(best performance in 2011) to 16.4%. In 2014, when VGGNet[4] and Inception Net[21] were introduced, depth of networks had been easily extended to 20 layers and the accuracy of image classification also improved a lot. As network goes deeper, simply stacking layers would degrade its performance. To solve the problem, He[6] introduced ResNet, which learns Residual function H(x) − x,instead of target function H(x) directly. ResNet can be extended to over 100 layers and the performance can be further improved.

自2012年以来，神经网络作为一种构建模型的新方法，在各种计算机视觉领域取得了重大进展。由8层组成的AlexNet[1]赢得了ILSVRC-2012图像分类冠军。它将ImageNet数据集的错误率从25.8%(2011年最佳性能)降至16.4%。2014年，当引入VGGNet[4]和Inception Net[21]时，网络的深度很容易扩展到20层，图像分类的准确性也有了很大提高。随着网络的深入，简单地堆叠层会降低其性能。为了解决这个问题，He[6]引入了ResNet，它学习残差函数H(x)− x、 而不是直接目标函数H(x)。ResNet可以扩展到100层以上，性能可以进一步提高。

Fig. 4: SparseNet with attention module 
图4：带注意力模块的SparseNet

Many researches have been made on ResNet variants. He[22] changed the conventional ”post-activation” of the weight layers to ”pre-activation”. To be specific, he put BN layer and ReLU layer before Conv layer. As the result turned out, this identity-mapping change made training easier and thus the performance of networks better. Han[9] introduced Deep Pyramidal Residual Networks, which increase width gradually layer by layer and rearrange the convolution module. Experiments showed their network architecture has superior generalization ability compared to original ResNet.

已经对ResNet变体进行了许多研究。他将权重层的传统“后激活”改为“预激活”。具体来说，他把BN层和ReLU层放在Conv层之前。结果表明，这种身份映射的改变使训练变得更容易，从而提高了网络的性能。Han[9]引入了深度金字塔残差网络，它逐层逐渐增加宽度，并重新排列卷积模块。实验表明，与原始ResNet相比，其网络结构具有更好的泛化能力。

Targ[23] proposed ResNet in ResNet(RiR), which changed convolution module to a small deep dual-stream architecture. RiR makes network generalize between residual stream which is similar to a residual block and transient stream which is a standard convolutional layer. Huang[7] constructs a very deep ResNet. By randomly dropping some residual modules with probability p, they can train different shallower subnetworks in the training phase. In the testing phase, they use the whole deep network, whereas recalibrated every residual module with the survival probability (1 − p). In this way, ResNet can be expended to over 1200 layers. Zagoruyko[8] introduced a ResNet variant with wider width and shallower depth, named WRN(Wide Residual Networks), which can improve ResNet’s performance further. Huang[14] presented DenseNet with layers connected to its all previous layers. With this kind of network design, it not only accomplishes feature reuse, but also alleviates the vanishing-gradient problem.

Targ[23]提出了ResNet in ResNet(RiR)，将卷积模块改为小型深双流架构。RiR使网络在类似于残差块的残差流和作为标准卷积层的瞬态流之间进行泛化。黄[7]构建了一个非常深的ResNet。通过随机丢弃一些概率为p的剩余模块，它们可以在训练阶段训练不同的较浅子网络。在测试阶段，他们使用整个深度网络，而使用生存概率(1− p)。这样，ResNet可以扩展到1200多个层。Zagoruyko[8]引入了一种宽度更宽、深度更浅的ResNet变体，称为WRN(宽残差网络)，它可以进一步提高ResNet的性能。Huang[14]介绍了DenseNet，其中的层与之前的所有层相连。通过这种网络设计，不仅实现了特征重用，而且缓解了消失梯度问题。

### 2.2 Attention mechanism
Attention mechanism has achieved many progresses in areas such as machine translation[24]. Recently attention mechanism is playing a significant role in various computer vision tasks. Harley[17]learned weights of pixels in multiple scales using attention mechanism, and calculating the weighted average value as the final result of segmentation. Chu[18] improve human pose estimation using multi-context attention module. They use holistic attention model to get global consistency information of human body; while using body part attention module to get detailed information for each human part. Wang[16] proposed a residual attention network for image classification, which achieved state-of-art performance on CIFAR dataset. By attention residual learning, they can easily extend their networks to hundreds of layers. Hu[19] proposed SENet(Squeeze-Excitation networks), which calibrate weights for different channels by explicitly modeling channel interdependencies. SENet won ILSVRC-2017 image classification championship. 

注意机制在机器翻译等领域取得了许多进展[24]。近年来，注意机制在各种计算机视觉任务中发挥着重要作用。Harley[17]使用注意力机制在多个尺度上学习像素的权重，并计算加权平均值作为分割的最终结果。Chu[18]使用多上下文关注模块改进了人体姿势估计。他们使用整体注意模型来获得人体的全局一致性信息; 同时使用身体部位关注模块来获取每个人体部位的详情。Wang[16]提出了一种用于图像分类的残余注意力网络，该网络在CIFAR数据集上实现了最先进的性能。通过注意力剩余学习，他们可以轻松地将网络扩展到数百层。Hu[19]提出了SENet(挤压激励网络)，它通过显式建模信道相互依赖性来校准不同信道的权重。SENet获得ILSVRC-2017图像分类冠军。

## 3 SparseNet
## 3.1 DenseNet
We represent the input image as x0,output of the i th layer as xi and each convolutional module as function H. Since input to the i th layer is outputs of all previous layers. The formula is presented as follows: 

我们将输入图像表示为x0，第i层的输出表示为xi，每个卷积模块表示为函数H。因为第i层输入是之前所有层的输出。公式如下：

xi = H([x0, x1, , xi−1]), 

where [x0, x1, , xi−1] is the concatenation of outputs of all previous layers. DenseNet is composed of several dense blocks connected by transition layer. Normally, size of feature map decreased by 14 for each block. For example, size of feature map for the first block is h × w, then h2 × w2 for the second block, h4 × w4 for the third block. In DenseNet, number of output feature-maps for each convolution module are always the same, which is denoted by k. Thus the output number of the i th layer is (k0 + (i − 1) × k), where k0 is the number input to the first dense block. k was referenced as growth rate.

其中[x0，x1，xi−1] 是所有先前层的输出的级联。DenseNet由几个由过渡层连接的致密块组成。通常情况下，每个块的特征图大小减少了14。例如，第一个块的特征图大小为h×w，第二个块为h2×w2，第三个块为h4×w4。在DenseNet中，每个卷积模块的输出特征图的数量总是相同的，用k表示。因此，第i层的输出数量为(k0+(i− 1) ×k)，其中k0是输入到第一稠密块的数。k作为生长速率。

As DenseNet goes deeper, number of input feature-maps would become excess very soon. To settle this problem, the author put 1 × 1 convolution module(as bottleneck layer) before the 3 × 3 convolution module. Thus, the convolutional module has changed from BN+ReLU+3 × 3Conv to BN+ReLU+1 × 1Conv+BN+ReLU+3 × 3Conv.(And the new convolution module is counted as two layers instead of one). The normal setup for output feature-maps of bottleneck layer(1 × 1Conv) is 4k. Thus, inputs to every 3 × 3 Conv layer is fixed to 4k. To further improve model compactness, number of feature-maps can also be reduced in transition layer. The normal setup is number of feature-maps is reduced by 12 . This kind of DenseNet is called DenseNet-BC.

随着DenseNet的深入，输入特征图的数量将很快变得过剩。为了解决这个问题，作者将1×1卷积模块(作为瓶颈层)置于3×3卷积模块之前。因此，卷积模已从BN+ReLU+3×3Conv变为BN+ReLU+1×1Conv+BN+ReLU+3×3Conv。(新的卷积模块被计算为两层而不是一层)。瓶颈层(1×1Conv)输出特征图的正常设置为4k。因此，每3×3 Conv层的输入固定至4k。为了进一步提高模型的紧凑性，还可以减少过渡层中的特征图数量。通常的设置是特征图的数量减少12。这种DenseNet称为DenseNetwork BC。

### 3.2 SparseNet
We introduce a method to sparsify DenseNet. The basic idea is dropping connections from middle layers and preserving only the farthest and nearest connections. The formula is as followings: 

我们介绍了一种稀疏DenseNet的方法。基本思想是删除中间层的连接，只保留最远和最近的连接。公式如下：

xi = H([x0, x1, ...xn/2, xi−n/2, ..., xi−1]), 

where n denotes number of connections we will preserve(We call it ‘path’). As DenseNet does, we also use bottleneck layer and compress the model in the transition layer, the hyperparameters are set the same as that of DenseNet.

其中n表示我们将保留的连接数(我们称之为“路径”)。与DenseNet一样，我们也使用瓶颈层并在过渡层中压缩模型，超参数设置与Dense Net相同。

Moreover, we also make a structure optimization. In DenseNet, layer number are the same for all dense blocks. However, in our SparseNet, the layer number in each block is increasing. We will talk about the advantages of this arrangement in section 4.6.

此外，我们还进行了结构优化。在DenseNet中，所有密集块的层数相同。然而，在我们的SparseNet中，每个块中的层数都在增加。我们将在第4.6节中讨论这种安排的优点。

### 3.3 Attention mechanism
We proposed an attention mechanism to further boost network’s performance. Structure is shown in Figure 3. Suppose the input is x, the left part is a convolution module, we denoted the function as H. The right part is the attention mechanism module, and denote it as F. It consists of one global Pooling layer and two 1 × 1 convolution modules. The input to the second convolution module is the concatenation of outputs of both global pooling layer and the first convolution module. Then the final result is calculated as H(x) + H(x) × F(x).

我们提出了一种注意机制，以进一步提高网络性能。结构如图3所示。假设输入是x，左侧部分是卷积模块，我们将该函数表示为H。右侧部分是注意机制模块，并将其表示为F。它由一个全局池层和两个1×1卷积模块组成。第二卷积模块的输入是全局池层和第一卷积模块两者的输出的级联。然后将最终结果计算为H(x)+H(x)×F(x)。

### 3.4 Framework
To summarize, as it is shown in Fig.5, We proposed three networks. 
(a) is the original DenseNet; 
(b) is the basic SparseNet(path = 2, since connections to every layer is at most 2); 
(c) is SparseNet-bc, by adding bottleneck layers and reducing number of feature-maps in transition layer; 
(d) is SparseNet-abc, by adding attention mechanism on SparseNet-bc. 
The whole framework is shown in Figure 6.

总之，如图5所示，我们提出了三个网络。
* (a) 是原始的DenseNet; 
* (b) 是基本的SparseNet(路径＝2，因为到每个层的连接最多为2); 
* (c) 是SparseNet bc，通过增加瓶颈层和减少过渡层中的特征图数量; 
* (d) 是SparseNet abc，通过在SparseNetwork bc上添加注意机制。
整个框架如图6所示。

### 3.5 Implementation details
All our models include three sparse blocks. The layers within each sparse block are increasing. Besides bottleneck layer, all convolutional kernels are 3×3. blocks are connected with transition layer, which reduced feature map size by 14 and feature map number by 12 (feature map number will remain the same for the basic SparseNet). After the last block, a global pooling layer and a softmax classifier is attached. For each network(SparseNet, SparseNet-bc, SparseNet-abc), we construct three different sizes of parameters. denoting by V1, V2, V3 and V4. For V1, the layer number for three blocks are 8,12,16; 12,18,24 for V2; 16, 24,32 for V3 and 20,30,40 for V4. other parameters are listed in table 1.

我们所有的模型都包括三个稀疏块。每个稀疏块中的层都在增加。除瓶颈层外，所有卷积核均为3×3。块与过渡层连接，从而将特征图大小减少了14，特征图数量减少了12(对于基本稀疏网络，特征图编号将保持不变)。在最后一个块之后，附加了全局池层和softmax分类器。对于每个网络(SpareNet、SpareNetbc、SparseNetabc)，我们构造了三个不同大小的参数。用V1、V2、V3和V4表示。对于V1，三个块的层数为8、12、16; V2为12,18,24; V3为16、24、32，V4为20、30、40。其他参数见表1。

Fig. 5: a is DenseNet; b is SparseNet(path=2); c is SparseNet-bc; d is SparseNetabc.
图5:a为DenseNet; b是SparseNet(路径=2); c是SparseNet bc; d是SparseNetabc。

Fig. 6: the framework of SparseNet for image classification. Between Sparse blocks are transition layer. 
图6：用于图像分类的SparseNet框架。稀疏块之间是过渡层。

Table 1: setups of networks. 
表1：网络设置。

## 4 Experiments
### 4.1 Datasets
CIFAR. CIFAR[25] are colored images with three channels. Their sizes are 32 × 32. CIFAR10 consists of 10 classes and CIFAR100 consists of 100 classes. Both are composed of 50,000 training images and 10,000 test images.

CIFAR。CIFAR[25]是具有三个通道的彩色图像。它们的大小为32×32。CIFAR10由10个类组成，CIFAR100由100个类组成。两者都由50000张训练图像和10000张测试图像组成。

SVHN. The Street View House Numbers(SVHN)[26] are also colored images with three channels. Their sizes are 32 × 32. SVHN includes 73,257 training images, 531,131 additional training images and 26,032 test images. We training our model using all the training images.

SVHN.街景房屋编号(SVHN)[26]也是带有三个通道的彩色图像。SVHN包括73257幅训练图像、531131幅附加训练图像和26032幅测试图像。我们使用所有训练图像训练我们的模型。

### 4.2 Training
All networks are trained using stochastic gradient descent. The weight decay is 0.0001, Nesterov momentum is 0.9 without dampening. We initialize parameters as He[27] does. All datasets are augmented with method introduced in Huang[14]. For CIFAR, the training epoch is 280. the initial learning rate is 0.1, and decreasing learning rate to 0.01,0.001,0.0002 at epoch 150, 200 and 250. For SVHN, the total epoch is 40, and decreasing to 0.01 and 0.001 at epoch 20 and 30. the batch size of both datasets are 64.

### 4.3 Classification Results on CIFAR and SVHN
Results on datasets of CIFAR and SVHN are shown in table 2. Compared to DenseNet, SparseNet achieves superior performances on all datasets. On CIFAR10, SparseNet decreases error rate from 3.46% to 3.24%. On CIFAR100, SparseNet achieves error rate of 16.98%, while DenseNet achieved 17.18%. On SVHN, SparseNet also achieves lower error rate(1.69% v.s. 1.74% ). Furthermore, SpareNet outperforms the existing state-of-art on CIFAR10 and SVHN. Its error rates are lower than PyramidNet on CIFAR10, which achieved 3.31% and DenseNet on SVHN, which achieved 1.74%.

使用随机梯度下降训练所有网络。权重衰减为0.0001，Nesterov动量为0.9，无阻尼。我们像He[27]一样初始化参数。所有数据集都用Huang[14]中介绍的方法进行了扩充。对于CIFAR，训练周期为280。初始学习率为0.1，在第150、200和250周期时，学习率降至0.01,0.001,0.0002。对于SVHN，总周期为40，在第20和30周期时降至0.01和0.001。两个数据集的批大小均为64。

Table 2: Error rates on datasets of CIFAR and SVHN.
表2:CIFAR和SVHN数据集上的错误率。

### 4.4 Attention mechanism
As we can see from table 2, attention mechanism can boost networks’ performance for most model sizes(V1, V2 and V3) with only 2% increasing in parameters and 1% increasing in inference time. We also compared our attention mechanism to SE module[19] on SparseNet-V1. Results are shown in Fig. 7. In the whole training phase, our attention mechanism is always superior to SE module.Besides that, the effect of SE module on SparseNet-V1 is nearly neglectable.

如表2所示，注意机制可以提高大多数模型大小(V1、V2和V3)的网络性能，而参数增加2%，推理时间增加1%。我们还将注意力机制与SparseNet-V1上的SE模块[19]进行了比较。结果如图7所示。在整个训练阶段，我们的注意力机制始终优于SE模块。此外，SE模块对SparseNet-V1的影响几乎可以忽略。

Fig. 7: original is the SparseNet; attention is SparseNet+attention module; SE is the SparseNet+SE module(the epoch and learning rates are set as DenseNet)
图7：原始是SparseNet; 注意力是SparseNet+注意力模块; SE是SparseNet+SE模块(周期和学习速率设置为DenseNet)

### 4.5 Parameter Efficiency and Computation Efficiency of SparseNet
The results in Fig.8 indicate that SparseNet utilizes parameters more efficiently than alternative models. SparseNet-abc-v1 achieves lower test error on CIFAR10 than pre-activation ResNet of 10001 layers, while latter has 10 times more parameters than the former one. For DenseNet-BC, the best model achieves 3.46%, while SparseNet achives lower test errror(3.40%) with ×2.6 less parameters. For the recent CondenseNet[30], which designed for mobile devices, Our SparseNet is still more parameter-efficient.

图8中的结果表明，SparseNet比其他模型更有效地利用参数。SparseNet-abc-v1在CIFAR10上的测试误差低于10001层的预激活ResNet，而后者的参数是前者的10倍。对于DenseNet BC，最佳模型实现了3.46%，而SparseNet实现了更低的测试误差(3.40%)，参数减少了2.6倍。对于最近为移动设备设计的CondenseNet[30]，我们的SparseNet仍然具有更高的参数效率。

To analyze SparseNet’s computation, we compared FLOPs1 (floating-point operations) of pre-activation ResNet, DenseNet and SparseNet. Results are shown in Fig. 8. It shows SparseNet is more computation-efficient than the other two models. Compared to the best DenseNet Model, SparseNet is ×3.7 faster than DenseNet. 

为了分析SparseNet的计算，我们比较了预激活ResNet、DenseNet和SparseNetwork的FLOPs1(浮点运算)。结果如图8所示。它表明SparseNet比其他两种模型更高效。与最佳的DenseNet模型相比，SparseNet比DenseNetwork快3.7倍。

Fig. 8: Comparison parameter-efficiency on CIFAR10 of different models
图8：不同模型CIFAR10的参数效率比较

Fig. 9: Comparison of SparseNet-abc and DenseNet error rate on CIFAR10 as a function of FLOPs.
图8：不同模型CIFAR10的参数效率比较

### 4.6 Structure optimization
We also analyzed the effectiveness of our layer arrangement for each sparse block. We compared two kinds of block arrangements. One is the increasing arrangement: 8-12-16; the other is equal arrangement: 12-12-12. Results are listed in table 3. It shows that our increasing arrangement is superior not only in computation but also in accuracy.

我们还分析了每个稀疏块的层排列的有效性。我们比较了两种街区安排。一种是递增排列：8-12-16; 另一种是等距排列：12-12-12。结果列于表3。这表明我们的递增排列不仅在计算上而且在精度上都是优越的。

Table 3: Results of two block arrangements on CIFAR10. 
表3:CIFAR10的两个区块安排的结果。

## 5 Discussion
### 5.1 Where to drop connections
In this section, we experimented different methods of reducing connections. Take SparseNet-V1(path=14) for example, we tried 5 different ways of dropping connections: 
1. 14-0: only preserving the farthest 14 connections; 
2. 10-4: preserving the farthest 10 connections and nearest 4 connections; 
3. 7-7(ours): preserving the farthest 7 connections and nearest 7 connections; 
4. 4-10: preserving the farthest 4 connections and nearest 10 connections; 
5. 0-14: only preserving the nearest 14 connections.

在本节中，我们尝试了减少连接的不同方法。以SparseNet-V1(路径=14)为例，我们尝试了5种不同的断开连接方式：
1. 14-0：仅保留最远的14个连接; 
2. 10-4：保留最远10个连接和最近4个连接; 
3. 7-7(我们的)：保留最远7个连接和最近7个连接; 
4. 4-10：保留最远的4个连接和最近的10个连接; 
5. 0-14：仅保留最近的14个连接。

As we can see from Fig. 10, different dropping connection methods resulted in different error rates. And our dropping connections method(7-7) achieves best performance. Besides 7-7, 0-14 also achieved comparable performance to our method. One possible explanation is that the method of preserving the nearest 14 connections contains as much information as method of preserving the farthest 7 connections and nearest 7 connections for SparseNet-V1.

如图10所示，不同的分线连接方法导致不同的错误率。我们的丢弃连接方法(7-7)实现了最佳性能。除了7-7，0-14也取得了与我们的方法相当的性能。一种可能的解释是，保存最近的14个连接的方法包含的信息与保存SparseNet-V1的最远7个连接和最近7个连接的相同。

### 5.2 How layers, growth rate and path influence network’s performance
We also analyzed how networks’ layer, width and shortcut connections influence network’s performance. In our experiments, we set up three layers: 28(8 layers per block), 52 layers(16 layers per block) and 76 layers(24 layers per block). We set the range of growth rate(k) to be [6,26]. The parameters of all models are around 1M. So when we set the layer number and the growth rate, the number of connections(path) is also determined. The results are showed in Fig. 11. We can see that for each layer setup, all test errors are experiencing decreasing first and then increasing, resulting the optimal test error are always in the somewhere middle. For different layer setups, the lowest test error is within layers of 52, which is between 28 and 76. The results showed that none of the three factors shouldn’t be set to be extreme. Only by increasing layers, growth rate and path synchronously, can SparseNet achieve better performance. 

我们还分析了网络的层、宽度和快捷连接对网络性能的影响。在我们的实验中，我们设置了三层：28层(每个区块8层)、52层(每个块16层)和76层(每个街区24层)。我们将增长率(k)的范围设置为[6,26]。所有模型的参数都在1M左右。因此，当我们设置层数和增长率时，连接数(路径)也会被确定。结果如图11所示。我们可以看到，对于每个层设置，所有测试误差都经历了先减小后增大的过程，因此最佳测试误差总是在中间的某个位置。对于不同的层设置，最低的测试误差在52层之间，即28到76层之间。结果表明，这三个因素都不应设置为极端。只有同步增加层、增长率和路径，SparseNet才能获得更好的性能。

Fig. 10: different methods of reducing connections in SparseNet-V1. 
图10：减少SparseNet-V1中连接的不同方法。

## 6 Conclusion
In this work, we proposed a method to sparsify DenseNet. After reducing shortcut connections, we can expend the network to deeper and wider. Moreover, we also introduced an attention model, which can boost networks’ performance further. Experiments showed that compared to DenseNet, our model achieved comparable performance on datasets of CIFAR and SVHN with much less parameters and much lower computation. Besides, we analyzed several ways of reducing connections and how layers, growth rate and shortcut connections in- fluence networks’ performance. In future work, we will apply our models to other computer vision tasks, for example object detection, object segmentation, human pose estimation and so on.

在这项工作中，我们提出了一种稀疏DenseNet的方法。在减少快捷连接之后，我们可以将网络扩展到更深更广的地方。此外，我们还引入了一种注意力模型，可以进一步提高网络的性能。实验表明，与DenseNet相比，我们的模型在CIFAR和SVHN数据集上的性能相当，参数更少，计算量更低。此外，我们还分析了几种减少连接的方法，以及层、增长率和快捷连接对网络性能的影响。在未来的工作中，我们将把我们的模型应用于其他计算机视觉任务，例如对象检测、对象分割、人体姿态估计等。

## References
1. Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. In: Advances in neural information processing systems.
(2012) 1097–1105
2. Girshick, R.: Fast r-cnn. arXiv preprint arXiv:1504.08083 (2015)
16 Wenqi Liu Kun Zeng
Fig. 11: Error rates of different setups of layers, growth rate and path on CIFAR10.
3. Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic
segmentation. In: Proceedings of the IEEE conference on computer vision and
pattern recognition. (2015) 3431–3440
4. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale
image recognition. arXiv preprint arXiv:1409.1556 (2014)
5. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D.,
Vanhoucke, V., Rabinovich, A., et al.: Going deeper with convolutions, Cvpr (2015)
6. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition.
In: Proceedings of the IEEE conference on computer vision and pattern recognition.
(2016) 770–778
7. Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q.: Deep networks with
stochastic depth. In: European Conference on Computer Vision, Springer (2016)
646–661
8. Zagoruyko, S., Komodakis, N.: Wide residual networks. arXiv preprint
arXiv:1605.07146 (2016)
9. Han, D., Kim, J., Kim, J.: Deep pyramidal residual networks. arXiv preprint
arXiv:1610.02915 (2016)
10. Srivastava, R.K., Greff, K., Schmidhuber, J.: Training very deep networks. arXiv
preprint arXiv:1507.06228 (2015)
11. Hariharan, B., Arbel´aez, P., Girshick, R., Malik, J.: Hypercolumns for object
segmentation and fine-grained localization. In: Proceedings of the IEEE conference
on computer vision and pattern recognition. (2015) 447–456
12. Sermanet, P., Kavukcuoglu, K., Chintala, S., LeCun, Y.: Pedestrian detection
with unsupervised multi-stage feature learning. In: Computer Vision and Pattern
Recognition (CVPR), 2013 IEEE Conference on, IEEE (2013) 3626–3633
13. Yang, S., Ramanan, D.: Multi-scale recognition with dag-cnns. In: Computer
Vision (ICCV), 2015 IEEE International Conference on, IEEE (2015) 1215–1223
SparseNet: A Sparse DenseNet for Image Classification 17
14. Huang, G., Liu, Z., Weinberger, K.Q., van der Maaten, L.: Densely connected
convolutional networks. In: Proceedings of the IEEE conference on computer vision
and pattern recognition. Volume 1. (2017) 3
15. Zeiler, M.D., Fergus, R.: Visualizing and understanding convolutional networks.
In: European conference on computer vision, Springer (2014) 818–833
16. Wang, F., Jiang, M., Qian, C., Yang, S., Li, C., Zhang, H., Wang, X.,
Tang, X.: Residual attention network for image classification. arXiv preprint
arXiv:1704.06904 (2017)
17. Harley, A.W., Derpanis, K.G., Kokkinos, I.: Segmentation-aware convolutional
networks using local attention masks. In: IEEE International Conference on Computer Vision (ICCV). Volume 2. (2017) 7
18. Chu, X., Yang, W., Ouyang, W., Ma, C., Yuille, A.L., Wang, X.: Multi-context
attention for human pose estimation. arXiv preprint arXiv:1702.07432 1(2) (2017)
19. Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. arXiv preprint
arXiv:1709.01507 (2017)
20. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception architecture for computer vision. In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition. (2016) 2818–2826
21. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D.,
Vanhoucke, V., Rabinovich, A.: Going deeper with convolutions. In: The IEEE
Conference on Computer Vision and Pattern Recognition (CVPR). (June 2015)
22. He, K., Zhang, X., Ren, S., Sun, J.: Identity mappings in deep residual networks.
In: European Conference on Computer Vision, Springer (2016) 630–645
23. Targ, S., Almeida, D., Lyman, K.: Resnet in resnet: generalizing residual architectures. arXiv preprint arXiv:1603.08029 (2016)
24. Bahdanau, D., Cho, K., Bengio, Y.: Neural machine translation by jointly learning
to align and translate. arXiv preprint arXiv:1409.0473 (2014)
25. Krizhevsky, A., Hinton, G.: Learning multiple layers of features from tiny images.
(2009)
26. Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., Ng, A.Y.: Reading digits
in natural images with unsupervised feature learning. In: NIPS workshop on deep
learning and unsupervised feature learning. Volume 2011. (2011) 5
27. He, K., Zhang, X., Ren, S., Sun, J.: Delving deep into rectifiers: Surpassing humanlevel performance on imagenet classification. In: Proceedings of the IEEE international conference on computer vision. (2015) 1026–1034
28. Xie, S., Girshick, R., Doll´ar, P., Tu, Z., He, K.: Aggregated residual transformations
for deep neural networks. In: Computer Vision and Pattern Recognition (CVPR),
2017 IEEE Conference on, IEEE (2017) 5987–5995
29. Larsson, G., Maire, M., Shakhnarovich, G.: Fractalnet: Ultra-deep neural networks
without residuals. In: ICLR. (2017)
30. Huang, G., Liu, S., van der Maaten, L., Weinberger, K.Q.: Condensenet: An effi-
cient densenet using learned group convolutions. arXiv preprint arXiv:1711.09224
(2017)
