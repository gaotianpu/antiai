# Aggregated Residual Transformations for Deep Neural Networks
深度神经网络的聚合残差变换 2016-11-16 原文：https://arxiv.org/abs/1611.05431

# 阅读笔记
* 基数, cardinality
* Inception: 拆分-转换-合并(split-transform-merge),计算复杂度低
* 论文中对前人作品的总结很有启发

## Abstract
We present a simple, highly modularized network architecture for image classification. Our network is constructed by repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new dimension, which we call “cardinality” (the size of the set of transformations), as an essential factor in addition to the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover, increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its ResNet counterpart. The code and models are publicly available online(https://github.com/facebookresearch/ResNeXt).

我们提出了一种简单、高度模块化的图像分类网络架构。我们的网络是通过重复一个构建块来构建的，该构建块聚合了一组具有相同拓扑的转换。我们的简单设计产生了一个同质、多分支的架构，只需设置几个超参数。这个策略公开了一个新维度，我们称之为“基数”(转换集的大小)，它是深度(卷积层数量)和宽度(卷积核数量)维度之外的一个重要因素。在ImageNet-1K数据集上，我们的经验表明，即使在保持复杂度的限制条件下，增加基数也可以提高分类精度。此外，当我们增加容量时，增加基数比增加深度和宽度更有效。我们的名为ResNeXt的模型是我们加入ILSVRC的基础2016年的分类任务，我们获得了第二名。我们进一步研究了ImageNet-5K集合和COCO检测集合上的ResNeXt，也显示出比ResNet对应物更好的结果。代码和模型可在线公开获取(https://github.com/facebookresearch/ResNeXt)。

## 1. Introduction
Research on visual recognition is undergoing a transition from “feature engineering” to “network engineering” [25, 24, 44, 34, 36, 38, 14]. In contrast to traditional handdesigned features (e.g., SIFT [29] and HOG [5]), features learned by neural networks from large-scale data [33] require minimal human involvement during training, and can be transferred to a variety of recognition tasks [7, 10, 28]. Nevertheless, human effort has been shifted to designing better network architectures for learning representations.

视觉识别研究正经历着从“特征工程”到“网络工程”的过渡[25，24，44，34，36，38，14]。与传统的手工设计特征(如SIFT[29]和HOG[5])不同，神经网络从大规模数据中学习的特征[33]在训练期间需要最少的人工参与，并且可以迁移到各种识别任务[7、10、28]。然而，人类的努力已经迁移到为学习表示设计更好的网络架构。

Designing architectures becomes increasingly difficult with the growing number of hyper-parameters (width[Width refers to the number of channels in a layer.], filter sizes, strides, etc.), especially when there are many layers. The VGG-nets [36] exhibit a simple yet effective strategy of constructing very deep networks: stacking building blocks of the same shape. This strategy is inherited by ResNets [14] which stack modules of the same topology. This simple rule reduces the free choices of hyper-parameters, and depth is exposed as an essential dimension in neural networks. Moreover, we argue that the simplicity of this rule may reduce the risk of over-adapting the hyperparameters to a specific dataset. The robustness of VGGnets and ResNets has been proven by various visual recognition tasks [7, 10, 9, 28, 31, 14] and by non-visual tasks involving speech [42, 30] and language [4, 41, 20].

随着超参数(宽度-层中通道数，卷积核尺寸、步幅等)数量的增加，设计架构变得越来越困难，尤其是当有许多层时。VGG网络[36]展示了构建非常深网络的简单而有效的策略：堆叠形状相同的构建块。该策略由ResNets[14]继承，后者将相同拓扑的模块堆叠在一起。这个简单的规则减少了超参数的自由选择，而深度是神经网络中的一个基本维度。此外，我们认为该规则的简单性可以降低超参数过度适应特定数据集的风险。VGGnets和ResNets的健壮性已被各种视觉识别任务[7、10、9、28、31、14]以及涉及语音[42、30]和语言[4、41、20]的非视觉任务所证明。

Unlike VGG-nets, the family of Inception models [38, 17, 39, 37] have demonstrated that carefully designed topologies are able to achieve compelling accuracy with low theoretical complexity. The Inception models have evolved over time [38, 39], but an important common property is a split-transform-merge strategy. In an Inception module, the input is split into a few lower-dimensional embeddings (by 1×1 convolutions), transformed by a set of specialized filters (3×3, 5×5, etc.), and merged by concatenation. It can be shown that the solution space of this architecture is a strict subspace of the solution space of a single large layer (e.g., 5×5) operating on a high-dimensional embedding. The split-transform-merge behavior of Inception modules is expected to approach the representational power of large and dense layers, but at a considerably lower computational complexity.

与VGG网络不同，Inception模型家族[38、17、39、37]已经证明，精心设计的拓扑结构能够以较低的理论复杂度实现令人信服的准确性。Inception模型已经随着时间的推移而发展[38,39]，但一个重要的共同属性是拆分-转换-合并(split-transform-merge)策略。在初始模块中，输入被分割成几个低维嵌入(通过1×1卷积)，通过一组专用卷积核(3×3、5×5等)进行变换，然后通过级联(concatenation)进行合并。可以看出，该架构的解空间是操作高维嵌入的单个大层(如5×5)解空间的严格子空间。Inception模块的拆分-转换-合并行为有望接近大型密集层的表示能力，但计算复杂度要低得多。

Despite good accuracy, the realization of Inception models has been accompanied with a series of complicating factors — the filter numbers and sizes are tailored for each individual transformation, and the modules are customized stage-by-stage. Although careful combinations of these components yield excellent neural network recipes, it is in general unclear how to adapt the Inception architectures to new datasets/tasks, especially when there are many factors and hyper-parameters to be designed.

尽管精度很高，但Inception模型的实现伴随着一系列复杂因素 —— 卷积核的数量和大小是为每个单独的转换定制的，模块是逐步定制的。尽管这些组件的精心组合产生了出色的神经网络配方，但通常不清楚如何使Inception架构适应新的数据集/任务，尤其是当有许多因素和超参数需要设计时。

In this paper, we present a simple architecture which adopts VGG/ResNets’ strategy of repeating layers, while exploiting the split-transform-merge strategy in an easy, extensible way. A module in our network performs a set of transformations, each on a low-dimensional embedding, whose outputs are aggregated by summation. We pursuit a simple realization of this idea — the transformations to be aggregated are all of the same topology (e.g., Fig. 1 (right)). This design allows us to extend to any large number of transformations without specialized designs.

在本文中，我们提出了一种简单的架构，它采用VGG/ResNets的重复层策略，同时以一种简单、可扩展的方式利用拆分-转换-合并策略。我们网络中的一个模块执行一组转换，每个转换都基于低维嵌入，其输出通过求和进行聚合。我们追求这个想法的简单实现 —— 要聚合的转换都是相同的拓扑结构(例如，图1(右))。这种设计允许我们在没有专门设计的情况下扩展到任何大量转换。

Figure 1. Left: A block of ResNet [14]. Right: A block of ResNeXt with cardinality = 32, with roughly the same complexity. A layer is shown as (# in channels, filter size, # out channels).
图1.左：ResNet的一个块[14]。右：一个基数为32的ResNeXt块，其复杂度大致相同。层显示为(#输入通道数，卷积核大小，#输出通道数)。

Interestingly, under this simplified situation we show that our model has two other equivalent forms (Fig. 3). The reformulation in Fig. 3(b) appears similar to the InceptionResNet module [37] in that it concatenates multiple paths; but our module differs from all existing Inception modules in that all our paths share the same topology and thus the number of paths can be easily isolated as a factor to be investigated. In a more succinct reformulation, our module can be reshaped by Krizhevsky et al.’s grouped convolutions [24] (Fig. 3(c)), which, however, had been developed as an engineering compromise.

有趣的是，在这种简化的情况下，我们表明我们的模型有两种其他等效形式(图3)。图3(b)中的重制与InceptionResNet模块[37]相似，因为它连接多条路径; 但我们的模块不同于所有现有的Inception模块，因为我们的所有路径都共享相同的拓扑，因此可以很容易地将路径的数量作为一个待调查的因素进行隔离。在更简洁的重新表述中，我们的模块可以通过Krizhevskyet al 的分组卷积[24](图3(c))进行重塑，但这是作为工程折衷方案开发的。

Figure 3. Equivalent building blocks of ResNeXt. (a): Aggregated residual transformations, the same as Fig. 1 right. (b): A block equivalent to (a), implemented as early concatenation. (c): A block equivalent to (a,b), implemented as grouped convolutions [24]. Notations in bold text highlight the reformulation changes. A layer is denoted as (# input channels, filter size, # output channels).
图3.ResNeXt的等效构建块。(a) ：聚合残差变换，如右图1所示。(b)：相当于(A)的块，实现为早期串联。(c) ：相当于(A，b)的块，实现为分组卷积[24]。粗体文本中的注释突出显示了重制的更改。层表示为(输入通道数、卷积核大小、输出通道数)。

We empirically demonstrate that our aggregated transformations outperform the original ResNet module, even under the restricted condition of maintaining computational complexity and model size — e.g., Fig. 1(right) is designed to keep the FLOPs complexity and number of parameters of Fig. 1(left). We emphasize that while it is relatively easy to increase accuracy by increasing capacity (going deeper or wider), methods that increase accuracy while maintaining (or reducing) complexity are rare in the literature.

我们从经验上证明，即使在保持计算复杂度和模型大小的限制条件下，我们的聚合转换仍优于原始ResNet模块 —— 例如，图1(右)旨在保持图1(左)的FLOP复杂度和参数数量。我们强调，虽然通过增加容量(更深或更广)来提高精度相对容易，但在文献中，提高精度同时保持(或降低)复杂度的方法很少。

Our method indicates that cardinality (the size of the set of transformations) is a concrete, measurable dimension that is of central importance, in addition to the dimensions of width and depth. Experiments demonstrate that increasing cardinality is a more effective way of gaining accuracy than going deeper or wider, especially when depth and width starts to give diminishing returns for existing models.

我们的方法表明，基数(变换集的大小)是一个具体的、可测量的维度，除了宽度和深度维度之外，它也是一个重要的维度。实验表明，增加基数比增加深度和宽度能更有效地提高精度，尤其是当深度和宽度开始为现有模型带来递减回报时。

Our neural networks, named ResNeXt (suggesting the next dimension), outperform ResNet-101/152 [14], ResNet200 [15], Inception-v3 [39], and Inception-ResNet-v2 [37] on the ImageNet classification dataset. In particular, a 101-layer ResNeXt is able to achieve better accuracy than ResNet-200 [15] but has only 50% complexity. Moreover, ResNeXt exhibits considerably simpler designs than all Inception models. ResNeXt was the foundation of our submission to the ILSVRC 2016 classification task, in which we secured second place. This paper further evaluates ResNeXt on a larger ImageNet-5K set and the COCO object detection dataset [27], showing consistently better accuracy than its ResNet counterparts. We expect that ResNeXt will also generalize well to other visual (and non-visual) recognition tasks.

在ImageNet分类数据集上，我们的名为ResNeXt的神经网络(暗示下一维度)的表现优于ResNet-101/152[14]、ResNet200[15]、Inception-v3[39]和Incepton-ResNet-v2[37]。特别是，101层ResNeXt能够实现比ResNet-200更好的精度[15]，但复杂度只有50%。此外，ResNeXt的设计比所有Inception模型都要简单得多。ResNeXt是我们提交ILSVRC 2016分类任务的基础，我们获得了第二名。本文在更大的ImageNet-5K集合和COCO目标检测数据集[27]上进一步评估了ResNeXt，显示出比ResNet对应物更好的准确性。我们期望ResNeXt也能很好地推广到其他视觉(和非视觉)识别任务。

## 2. Related Work
### Multi-branch convolutional networks. 
The Inception models [38, 17, 39, 37] are successful multi-branch architectures where each branch is carefully customized. ResNets [14] can be thought of as two-branch networks where one branch is the identity mapping. Deep neural decision forests [22] are tree-patterned multi-branch networks with learned splitting functions.

多分支卷积网络。Inception模型[38、17、39、37]是成功的多分支架构，其中每个分支都经过精心定制。ResNets[14]可以看作是两个分支网络，其中一个分支是恒等映射。深层神经决策森林[22]是具有学习分裂函数的树型多分支网络。

### Grouped convolutions. 
The use of grouped convolutions dates back to the AlexNet paper [24], if not earlier. The motivation given by Krizhevsky et al. [24] is for distributing the model over two GPUs. Grouped convolutions are supported by Caffe [19], Torch [3], and other libraries, mainly for compatibility of AlexNet. To the best of our knowledge, there has been little evidence on exploiting grouped convolutions to improve accuracy. A special case of grouped convolutions is channel-wise convolutions in which the number of groups is equal to the number of channels. Channel-wise convolutions are part of the separable convolutions in [35].

分组卷积。分组卷积的使用可以追溯到AlexNet论文[24]，如果不是更早的话。Krizhevskyet al [24]给出的动机是将模型分布在两个GPU上。Caffe[19]、Torch[3]和其他库支持分组卷积，主要是为了与AlexNet兼容。据我们所知，利用分组卷积来提高精度的证据很少。分组卷积的一个特殊情况是通道卷积，其中组数等于通道数。通道卷积是[35]中可分离卷积的一部分。

### Compressing convolutional networks. 
Decomposition (at spatial [6, 18] and/or channel [6, 21, 16] level) is a widely adopted technique to reduce redundancy of deep convolutional networks and accelerate/compress them. Ioannou et al. [16] present a “root”-patterned network for reducing computation, and branches in the root are realized by grouped convolutions. These methods [6, 18, 21, 16] have shown elegant compromise of accuracy with lower complexity and smaller model sizes. Instead of compression, our method is an architecture that empirically shows stronger representational power.

压缩卷积网络。分解(在空间[6，18]and/or通道[6，21，16]级别)是一种广泛采用的技术，用于减少深度卷积网络的冗余并加速/压缩它们。Ioannouet al [16]提出了一种“根”模式网络，用于减少计算，根中的分支通过分组卷积实现。这些方法[6、18、21、16]显示了精确性与较低复杂度和较小模型尺寸之间的完美折衷。我们的方法不是压缩，而是一种在经验上表现出更强表现力的架构。

### Ensembling. 
Averaging a set of independently trained networks is an effective solution to improving accuracy [24], widely adopted in recognition competitions [33]. Veit et al. [40] interpret a single ResNet as an ensemble of shallower networks, which results from ResNet’s additive behaviors [15]. Our method harnesses additions to aggregate a set of transformations. But we argue that it is imprecise to view our method as ensembling, because the members to be aggregated are trained jointly, not independently.

集成。平均一组独立训练的网络是提高准确性的有效解决方案[24]，在识别比赛中广泛采用[33]。Veitet al [40]将单个ResNet解释为浅层网络的集成，这是ResNet的加性行为的结果[15]。我们的方法利用加法来聚合一组转换。但我们认为，将我们的方法视为集成是不精确的，因为要集成的成员是联合训练的，而不是独立训练的。

## 3. Method 
### 3.1. Template 模板
We adopt a highly modularized design following VGG/ResNets. Our network consists of a stack of residual blocks. These blocks have the same topology, and are subject to two simple rules inspired by VGG/ResNets: (i) if producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes), and (ii) each time when the spatial map is downsampled by a factor of 2, the width of the blocks is multiplied by a factor of 2. The second rule ensures that the computational complexity, in terms of FLOPs (floating-point operations, in # of multiply-adds), is roughly the same for all blocks.

我们采用了VGG/ResNets之后的高度模块化设计。我们的网络由一堆残差块组成。这些块具有相同的拓扑结构，并受VGG/ResNets启发的两个简单规则的约束：
(i)如果生成相同大小的空间图，则块共享相同的超参数(宽度和卷积核大小)，
以及(ii)每次空间图按系数2进行下采样时，块的宽度乘以系数2。第二条规则确保所有块的计算复杂度大致相同，即FLOP(浮点运算，以乘法加法的#表示)。

With these two rules, we only need to design a template module, and all modules in a network can be determined accordingly. So these two rules greatly narrow down the design space and allow us to focus on a few key factors. The networks constructed by these rules are in Table 1.

有了这两个规则，我们只需要设计一个模板模块，就可以相应地确定网络中的所有模块。因此，这两条规则大大缩小了设计空间，使我们能够专注于几个关键因素。由这些规则构建的网络如表1所示。

Table 1. (Left) ResNet-50. (Right) ResNeXt-50 with a 32×4d template (using the reformulation in Fig. 3(c)). Inside the brackets are the shape of a residual block, and outside the brackets is the number of stacked blocks on a stage. “C=32” suggests grouped convolutions [24] with 32 groups. The numbers of parameters and FLOPs are similar between these two models.
表1.(左)ResNet-50。(右)ResNeXt-50，带有32×4d模板(使用图3(c)中的重制)。括号内是残差块的形状，括号外是一个阶段上堆叠块的数量。“C=32”表示32组的分组卷积[24]。这两种模型的参数和FLOP数量相似。

### 3.2. Revisiting Simple Neurons 重新审视简单神经元
The simplest neurons in artificial neural networks perform inner product (weighted sum), which is the elementary transformation done by fully-connected and convolutional layers. Inner product can be thought of as a form of aggregating transformation:

人工神经网络中最简单的神经元执行内积(加权和)，这是由完全连接和卷积层完成的基本变换。内积可以看作是一种聚合转化的形式：

$\sum_{i=1}^{D}w_ix_i$ , (1)

where x = [$x_1,x_2,...,x_D$] is a D-channel input vector to the neuron and $w_i$ is a filter’s weight for the i-th channel. This operation (usually including some output nonlinearity) is referred to as a “neuron”. See Fig. 2 

其中x=[$x_1,x_2,...,x_D$]是神经元的D通道输入向量，$w_i$是第i通道的卷积核权重。这种操作(通常包括一些输出非线性)称为“神经元”。见图2

Figure 2. A simple neuron that performs inner product.
图2.执行内积的简单神经元。

The above operation can be recast as a combination of splitting, transforming, and aggregating. (i) Splitting: the vector x is sliced as a low-dimensional embedding, and in the above, it is a single-dimension subspace $x_i$. (ii) Transforming: the low-dimensional representation is transformed, and in the above, it is simply scaled: $w_ix_i$ . (iii) Aggregating: the transformations in all embeddings are aggregated by $\sum_{i=1}^D$

上述操作可以重新转换为拆分-转换-聚合的组合。
1. 拆分：向量x被切片为低维嵌入，在上面，它是一个一维子空间$x_i$。
2. 转换：低维表示被转换，在上面，它被简单地缩放：$w_ix_i$。
3. 聚合：所有嵌入中的转换通过以下方式聚合 $\sum_{i=1}^D$ 。

### 3.3. Aggregated Transformations 聚合转换
Given the above analysis of a simple neuron, we consider replacing the elementary transformation ($w_ix_i$) with a more generic function, which in itself can also be a network. In contrast to “Network-in-Network” [26] that turns out to increase the dimension of depth, we show that our “Network-in-Neuron” expands along a new dimension.

鉴于上述对简单神经元的分析，我们考虑用更通用的函数替换初等变换($w_ix_i$)，该函数本身也可以是网络。与增加深度维度的“网络中的网络”[26]相反，我们展示了我们的“神经元网络”沿着一个新维度扩展。

Formally, we present aggregated transformations as:

形式上，我们将聚合转换表示为：

F(x) = $\sum_{i=1}^CT_i(x)$ , (2)

where $T_i(x)$ can be an arbitrary function. Analogous to a simple neuron, $T_i$ should project x into an (optionally lowdimensional) embedding and then transform it.

其中，$T_i(x)$可以是任意函数。类似于一个简单的神经元，$T_i$应该将x投射到一个(可选的低维)嵌入物中，然后对其进行变换。

In Eqn.(2), C is the size of the set of transformations to be aggregated. We refer to C as cardinality [2]. In Eqn.(2) C is in a position similar to D in Eqn.(1), but C need not equal D and can be an arbitrary number. While the dimension of width is related to the number of simple transformations (inner product), we argue that the dimension of cardinality controls the number of more complex transformations. We show by experiments that cardinality is an essential dimension and can be more effective than the dimensions of width and depth.

在公式(2)中 ，C是要聚合的转换集的大小。我们称C为基数[2]。在公式(2) C的位置类似于等式(1)中的D，但C不必等于D，可以是任意数。虽然宽度的维度与简单转换(内积)的数量有关，但我们认为基数的维度控制着更复杂的转换数量。我们通过实验表明，基数是一个基本维度，比宽度和深度维度更有效。

In this paper, we consider a simple way of designing the transformation functions: all $T_i$’s have the same topology. This extends the VGG-style strategy of repeating layers of the same shape, which is helpful for isolating a few factors and extending to any large number of transformations. We set the individual transformation $T_i$ to be the bottleneckshaped architecture [14], as illustrated in Fig. 1 (right). In this case, the first 1×1 layer in each $T_i$ produces the lowdimensional embedding.

在本文中，我们考虑一种设计变换函数的简单方法：所有$T_i$都具有相同的拓扑。这扩展了VGG风格的重复相同形状层的策略，这有助于隔离一些因素并扩展到任何大量变换。我们将单个变换$T_i$设置为瓶颈状结构[14]，如图1(右)所示。在这种情况下，每个$T_i$中的第一个1×1层产生低维嵌入。

The aggregated transformation in Eqn.(2) serves as the residual function [14] (Fig. 1 right):

方程式(2)中的聚合变换。 作为残差函数[14](图1右)：

$y = x + \sum_{i=1}^CTi(x)$ , (3)

where y is the output.

其中y是输出。

#### Relation to Inception-ResNet. 
Some tensor manipulations show that the module in Fig. 1(right) (also shown in Fig. 3(a)) is equivalent to Fig. 3(b).3 Fig. 3(b) appears similar to the Inception-ResNet [37] block in that it involves branching and concatenating in the residual function. But unlike all Inception or Inception-ResNet modules, we share the same topology among the multiple paths. Our module requires minimal extra effort designing each path.

与Inception-ResNet的关系。一些张量操作表明，图1(右)中的模块(也如图3(a)所示)等同于图3(b).3。 图3(b)类似于Inception ResNet[37]块，因为它涉及残差函数中的分支和串联。但与所有Inception或Inception-ResNet模块不同，我们在多条路径之间共享相同的拓扑。我们的模块需要最少的额外努力来设计每条路径。

#### Relation to Grouped Convolutions. 
The above module becomes more succinct using the notation of grouped convolutions [24].$^4$ This reformulation is illustrated in Fig. 3(c). All the low-dimensional embeddings (the first 1×1 layers) can be replaced by a single, wider layer (e.g., 1×1, 128-d in Fig 3(c)). Splitting is essentially done by the grouped convolutional layer when it divides its input channels into groups. The grouped convolutional layer in Fig. 3(c) performs 32 groups of convolutions whose input and output channels are 4-dimensional. The grouped convolutional layer concatenates them as the outputs of the layer. The block in Fig. 3(c) looks like the original bottleneck residual block in Fig. 1(left), except that Fig. 3(c) is a wider but sparsely connected module.

与分组卷积的关系。使用分组卷积的符号，上述模块变得更加简洁[24]$^4$。此重制如图3(c)所示。所有低维嵌入件(前1×1层)可替换为单个更宽的层(如图3(c)中的1×1，128-d)。当分组卷积层将其输入通道划分为组时，基本上由分组卷积层来完成分割。图3(c)中的分组卷积层执行32组卷积，其输入和输出通道为4维。分组卷积层将它们连接为层的输出。图3(c)中的模块看起来像图1(左)中的原始瓶颈残留模块，但图3(c)是一个较宽但连接稀疏的模块。

Figure 4. (Left): Aggregating transformations of depth = 2. (Right): An equivalent block, which is trivially wider.
图4.(左)：深度=2的聚合转换。(右)：一个等价的块，它通常更宽。

We note that the reformulations produce nontrivial topologies only when the block has depth ≥3. If the block has depth = 2 (e.g., the basic block in [14]), the reformulations lead to trivially a wide, dense module. See the illustration in Fig. 4.

我们注意到，只有当块的有深度≥3，重制才会产生非平凡的拓扑. 如果块的深度为2(例如，[14]中的基本块)，则重制通常会生成一个宽而密的模块。参见图4中的插图。

#### Discussion. 
We note that although we present reformulations that exhibit concatenation (Fig. 3(b)) or grouped convolutions (Fig. 3(c)), such reformulations are not always applicable for the general form of Eqn.(3), e.g., if the transformation Ti takes arbitrary forms and are heterogenous. We choose to use homogenous forms in this paper because they are simpler and extensible. Under this simplified case, grouped convolutions in the form of Fig. 3(c) are helpful for easing implementation.

讨论。我们注意到，尽管我们提出了显示串联(图3(b))或分组卷积(图3)的重制，但这种重制并不总是适用于一般形式的等式。(3) 例如，如果转换$T_i$具有任意形式并且是异质的。在本文中，我们选择使用同质形式，因为它们更简单且可扩展。在这种简化的情况下，图3(c)形式的分组卷积有助于简化实现。

### 3.4. Model Capacity 模型容量
Our experiments in the next section will show that our models improve accuracy when maintaining the model complexity and number of parameters. This is not only interesting in practice, but more importantly, the complexity and number of parameters represent inherent capacity of models and thus are often investigated as fundamental properties of deep networks [8].

我们在下一节的实验将表明，在保持模型复杂度和参数数量的同时，我们的模型可以提高精度。这不仅在实践中很有趣，而且更重要的是，参数的复杂度和数量代表了模型的固有能力，因此通常作为深层网络的基本属性进行研究[8]。

When we evaluate different cardinalities C while preserving complexity, we want to minimize the modification of other hyper-parameters. We choose to adjust the width of the bottleneck (e.g., 4-d in Fig 1(right)), because it can be isolated from the input and output of the block. This strategy introduces no change to other hyper-parameters (depth or input/output width of blocks), so is helpful for us to focus on the impact of cardinality.

当我们在保持复杂度的同时评估不同的基数C时，我们希望最小化对其他超参数的修改。我们选择调整瓶颈的宽度(如图1(右)中的4-d)，因为它可以与块的输入和输出隔离。此策略不会改变其他超参数(块的深度或输入/输出宽度)，因此有助于我们关注基数的影响。

In Fig. 1(left), the original ResNet bottleneck block [14] has 256·64+3·3·64·64+64·256 ≈ 70k parameters and proportional FLOPs (on the same feature map size). With bottleneck width d, our template in Fig. 1(right) has:
在图1(左)中，原始ResNet瓶颈块[14]有256·64+3·3·64·64+64·256≈ 70k参数和比例FLOP(在相同的特征图大小上)。对于瓶颈宽度d，图1(右)中的模板有：

C ·(256·d+3·3·d·d+d·256) (4)

parameters and proportional FLOPs. When C = 32 and d = 4, Eqn.(4) ≈ 70k. Table 2 shows the relationship between cardinality C and bottleneck width d.

参数和比例FLOP。当C=32和d=4时，公式(4) ≈ 70k。表2显示了基数C和瓶颈宽度d之间的关系。

Table 2. Relations between cardinality and width (for the template of conv2), with roughly preserved complexity on a residual block. The number of parameters is ∼70k for the template of conv2. The number of FLOPs is ∼0.22 billion (# params×56×56 for conv2).
表2.基数和宽度之间的关系(对于conv2的模板)，残差块的复杂度大致保持不变。conv2的模板的参数数量为~70k。FLOP的数量为∼2.2亿(conv2的#params×56×56)。

Because we adopt the two rules in Sec. 3.1, the above approximate equality is valid between a ResNet bottleneck block and our ResNeXt on all stages (except for the subsampling layers where the feature maps size changes). Table 1 compares the original ResNet-50 and our ResNeXt-50 that is of similar capacity.$^5$ We note that the complexity can only be preserved approximately, but the difference of the complexity is minor and does not bias our results.

因为我们采用了第3.1节中的两条规则，所以上述近似等式在所有阶段的ResNet瓶颈块和我们的ResNeXt之间都是有效的(除了特征图大小发生变化的子采样层)。表1比较了原始的ResNet-50和容量相似的ResNeXt-50 $^5$。我们注意到，复杂度只能大致保持不变，但复杂度的差异很小，不会影响我们的结果。

## 4. Implementation details
Our implementation follows [14] and the publicly available code of fb.resnet.torch [11]. On the ImageNet dataset, the input image is 224×224 randomly cropped from a resized image using the scale and aspect ratio augmentation of [38] implemented by [11]. The shortcuts are identity connections except for those increasing dimensions which are projections (type B in [14]). Downsampling of conv3, 4, and 5 is done by stride-2 convolutions in the 3×3 layer of the first block in each stage, as suggested in [11]. We use SGD with a mini-batch size of 256 on 8 GPUs (32 per GPU). The weight decay is 0.0001 and the momentum is 0.9. We start from a learning rate of 0.1, and divide it by 10 for three times using the schedule in [11]. We adopt the weight initialization of [13]. In all ablation comparisons, we evaluate the error on the single 224×224 center crop from an image whose shorter side is 256.

我们的实现遵循[14]和公开代码fb.resnet.torch[11]。在ImageNet数据集上，输入图像是224×224，使用[11]实现的[38]的比例和纵横比增大从调整大小的图像中随机裁剪而来。快捷连接是恒等连接，除了那些增加尺寸的投影([14]中的B类型)。conv3、4和5的下采样是通过每个阶段第一个块的3×3层中的跨2步卷积完成的，如[11]所示。我们在8个GPU(每个GPU 32个)上使用最小批量大小为256的SGD。权重衰减为0.0001，动量为0.9。我们从0.1的学习速率开始，用[11]中的时间表将其除以10三次。我们采用[13]的权重初始化。在所有消融比较中，我们从较短边为256的图像中评估单个224×224中心裁剪的误差。

Our models are realized by the form of Fig. 3(c). We perform batch normalization (BN) [17] right after the convolutions in Fig. 3(c).$^6$ ReLU is performed right after each BN, expect for the output of the block where ReLU is performed after the adding to the shortcut, following [14].

我们的模型是通过图3(c)的形式实现的。我们在图3(c).$^6$ 中的卷积之后立即执行批归一化(BN)[17] 6。 ReLU在每个BN之后执行，除了在添加到快捷连接之后执行ReLU的块的输出外，[14]。

We note that the three forms in Fig. 3 are strictly equivalent, when BN and ReLU are appropriately addressed as mentioned above. We have trained all three forms and obtained the same results. We choose to implement by Fig. 3(c) because it is more succinct and faster than the other two forms.

我们注意到，当BN和ReLU如上文所述得到适当处理时，图3中的三种形式是严格等效的。我们已经训练了所有三种形式，并获得了相同的结果。我们选择通过图3(c)实现，因为它比其他两种形式更简洁、更快。

## 5. Experiments
### 5.1. Experiments on ImageNet-1K
We conduct ablation experiments on the 1000-class ImageNet classification task [33]. We follow [14] to construct 50-layer and 101-layer residual networks. We simply replace all blocks in ResNet-50/101 with our blocks.

我们对1000-分类的ImageNet分类任务进行了消融实验[33]。我们按照[14]构建了50层和101层残差网络。我们只需将ResNet-50/101中的所有块替换为我们的块。

#### Notations. 
Because we adopt the two rules in Sec. 3.1, it is sufficient for us to refer to an architecture by the template. For example, Table 1 shows a ResNeXt-50 constructed by a template with cardinality = 32 and bottleneck width = 4d (Fig. 3). This network is denoted as ResNeXt-50 (32×4d) for simplicity. We note that the input/output width of the template is fixed as 256-d (Fig. 3), and all widths are doubled each time when the feature map is subsampled (see Table 1).

说明。因为我们采用了第3.1节中的两条规则，所以我们可以通过模板引用架构。例如，表1显示了由基数=32、瓶颈宽度=4d的模板构造的ResNeXt-50(图3)。为了简单起见，该网络表示为ResNeXt-50(32×4d)。我们注意到，模板的输入/输出宽度固定为256-d(图3)，并且每次对特征图进行子采样时，所有宽度都会加倍(见表1)。

#### Cardinality vs. Width. 
We first evaluate the trade-off between cardinality C and bottleneck width, under preserved complexity as listed in Table 2. Table 3 shows the results and Fig. 5 shows the curves of error vs. epochs. Comparing with ResNet-50 (Table 3 top and Fig. 5 left), the 32×4d ResNeXt-50 has a validation error of 22.2%, which is 1.7% lower than the ResNet baseline’s 23.9%. With cardinality C increasing from 1 to 32 while keeping complexity, the error rate keeps reducing. Furthermore, the 32×4d ResNeXt also has a much lower training error than the ResNet counterpart, suggesting that the gains are not from regularization but from stronger representations.

基数vs.宽度。我们首先评估基数C和瓶颈宽度之间的权衡，如表2所示，在保持复杂度的情况下。表3显示了结果，图5显示了错误与时间的曲线。与ResNet-50(表3顶部和图5左侧)相比，32×4d ResNeXt-50的验证误差为22.2%，比ResNet基线的23.9%低1.7%。随着基数C从1增加到32，同时保持复杂度，错误率不断降低。此外，32×4d ResNeXt的训练误差也远低于ResNet对应物，这表明收益不是来自正则化，而是来自更强的表示。

Figure 5. Training curves on ImageNet-1K. (Left): ResNet/ResNeXt-50 with preserved complexity (∼4.1 billion FLOPs, ∼25 million parameters); (Right): ResNet/ResNeXt-101 with preserved complexity (∼7.8 billion FLOPs, ∼44 million parameters).
图5.ImageNet-1K上的训练曲线。(左)：保留复杂度的ResNet/ResNeXt-50(∼41亿FLOP，∼2500万个参数); (右)：保留复杂度的ResNet/ResNeXt-101(∼78亿FLOP，∼4400万个参数)。

Table 3. Ablation experiments on ImageNet-1K. (Top): ResNet50 with preserved complexity (∼4.1 billion FLOPs); (Bottom): ResNet-101 with preserved complexity (∼7.8 billion FLOPs). The error rate is evaluated on the single crop of 224×224 pixels.
表3.ImageNet-1K上的消融实验。(顶部)：ResNet-50，保持复杂度(∼41亿FLOP); (底部)：ResNet-101保持了复杂度(∼78亿FLOP)。在224×224像素的单个裁剪上评估错误率。

Similar trends are observed in the case of ResNet-101 (Fig. 5 right, Table 3 bottom), where the 32×4d ResNeXt101 outperforms the ResNet-101 counterpart by 0.8%. Although this improvement of validation error is smaller than that of the 50-layer case, the improvement of training error is still big (20% for ResNet-101 and 16% for 32×4d ResNeXt-101, Fig. 5 right). In fact, more training data will enlarge the gap of validation error, as we show on an ImageNet-5K set in the next subsection.

在ResNet-101的情况下也观察到了类似的趋势(图5右侧，表3底部)，其中32×4d ResNeXt101的表现优于ResNet-10l对应产品0.8%。尽管验证误差的改进小于50层情况，但训练误差的改进仍然很大(ResNet-101为20%，32×4d ResNeXt-101为16%，右图5)。事实上，更多的训练数据将扩大验证错误的差距，正如我们在下一小节的ImageNet-5K集合中所示。

Table 3 also suggests that with complexity preserved, increasing cardinality at the price of reducing width starts to show saturating accuracy when the bottleneck width is small. We argue that it is not worthwhile to keep reducing width in such a trade-off. So we adopt a bottleneck width no smaller than 4d in the following.

表3还表明，在保持复杂度的情况下，当瓶颈宽度较小时，以减小宽度为代价增加基数开始显示饱和精度。我们认为，在这种权衡中，不值得继续减少宽度。因此，我们在下面采用不小于4d的瓶颈宽度。

#### Increasing Cardinality vs. Deeper/Wider. 
Next we investigate increasing complexity by increasing cardinality C or increasing depth or width. The following comparison can also be viewed as with reference to 2× FLOPs of the ResNet-101 baseline. We compare the following variants that have ∼15 billion FLOPs. (i) Going deeper to 200 layers. We adopt the ResNet-200 [15] implemented in [11]. (ii) Going wider by increasing the bottleneck width. (iii) Increasing cardinality by doubling C.

基数增加 vs 加深/加宽。接下来，我们研究通过增加基数C或增加深度或宽度来增加复杂度。以下比较也可以视为参考ResNet-101基线的2×FLOP。我们比较了以下具有∼150亿FLOP。
(i) 加深到200层。我们采用了[11]中实施的ResNet-200[15]。
(ii)通过增加瓶颈宽度加宽。
(iii)通过加倍C来增加基数。

Table 4. Comparisons on ImageNet-1K when the number of FLOPs is increased to 2× of ResNet-101’s. The error rate is evaluated on the single crop of 224×224 pixels. The highlighted factors are the factors that increase complexity.
表4.当FLOP数量增加到ResNet-101的2倍时，ImageNet-1K上的比较。在224×224像素的单个裁剪上评估错误率。突出显示的因素是增加复杂度的因素。

Table 4 shows that increasing complexity by 2× consistently reduces error vs. the ResNet-101 baseline (22.0%). But the improvement is small when going deeper (ResNet200, by 0.3%) or wider (wider ResNet-101, by 0.7%).

表4显示，与ResNet-101基线(22.0%)相比，复杂度增加2倍始终可以减少错误。但是，当加深(ResNet200，0.3%)或加宽(ResNet-101，0.7%)时，改善很小。

On the contrary, increasing cardinality C shows much better results than going deeper or wider. The 2×64d ResNeXt-101 (i.e., doubling C on 1×64d ResNet-101 baseline and keeping the width) reduces the top-1 error by 1.3% to 20.7%. The 64×4d ResNeXt-101 (i.e., doubling C on 32×4d ResNeXt-101 and keeping the width) reduces the top-1 error to 20.4%.

相反，增加基数C比加深或加宽显示出更好的结果。2×64d ResNeXt-101(即在1×64d ResNet-101基线上加倍C并保持宽度)将前1个错误减少1.3%至20.7%。64×4d ResNeXt-101(即，在32×4d ReNeXt-101C上加倍并保持宽度)将top-1错误降低到20.4%。

We also note that 32×4d ResNet-101 (21.2%) performs better than the deeper ResNet-200 and the wider ResNet101, even though it has only ∼50% complexity. This again shows that cardinality is a more effective dimension than the dimensions of depth and width.

我们还注意到，32×4d ResNet-101(21.2%)的性能优于较深的ResNet-200和较宽的ResNet101，尽管它只有∼50%的复杂度。这再次表明，基数比深度和宽度维度更有效。

#### Residual connections. 
The following table shows the effects of the residual (shortcut) connections:

残差连接。下表显示了残差(快捷方式)连接的效果：

| | setting | w/ residual | w/o residual 
--- | --- | --- | --- 
ResNet-50 | 1 × 64d | 23.9 | 31.2 
ResNeXt-50 | 32 × 4d | 22.2 |  26.1 

Removing shortcuts from the ResNeXt-50 increases the error by 3.9 points to 26.1%. Removing shortcuts from its ResNet-50 counterpart is much worse (31.2%). These comparisons suggest that the residual connections are helpful for optimization, whereas aggregated transformations are stronger representations, as shown by the fact that they perform consistently better than their counterparts with or without residual connections.

从ResNeXt-50中删除快捷方式会使错误增加3.9点，达到26.1%。从ResNet-50的对应版本中删除快捷方式要糟糕得多(31.2%)。这些比较表明，残差连接有助于优化，而聚合转换是更强的表示，事实表明，它们的性能始终优于有或无残差连接的对应项。

#### Performance. 
For simplicity we use Torch’s built-in grouped convolution implementation, without special optimization. We note that this implementation was brute-force and not parallelization-friendly. On 8 GPUs of NVIDIA M40, training 32×4d ResNeXt-101 in Table 3 takes 0.95s per mini-batch, vs. 0.70s of ResNet-101 baseline that has similar FLOPs. We argue that this is a reasonable overhead. We expect carefully engineered lower-level implementation (e.g., in CUDA) will reduce this overhead. We also expect that the inference time on CPUs will present less overhead. Training the 2×complexity model (64×4d ResNeXt-101) takes 1.7s per mini-batch and 10 days total on 8 GPUs.

性能. 为了简单起见，我们使用Torch的内置分组卷积实现，无需特殊优化。我们注意到，这个实现是暴力的，不利于并行化。在NVIDIA M40的8个GPU上，表3中的32×4d ResNeXt-101训练每小批次需要0.95s，而具有类似FLOP的ResNet-101基线需要0.70s。我们认为这是一个合理的开销。我们希望精心设计的低级别实现(例如在CUDA中)将减少此开销。我们还希望CPU上的推理时间会带来更少的开销。在8个GPU上训练2×复杂度模型(64×4d ResNeXt-101)每小批次需要1.7s，总共需要10天。

Table 5. State-of-the-art models on the ImageNet-1K validation set (single-crop testing). The test size of ResNet/ResNeXt is 224×224 and 320×320 as in [15] and of the Inception models is 299×299.

表5.ImageNet-1K验证集的SOTA(单次裁剪测试)。ResNet/ResNeXt的测试集的尺寸为224×224和320×320，如[15]所示，而Inception模型是299×299。

#### Comparisons with state-of-the-art results. 
Table 5 shows more results of single-crop testing on the ImageNet validation set. In addition to testing a 224×224 crop, we also evaluate a 320×320 crop following [15]. Our results compare favorably with ResNet, Inception-v3/v4, and Inception-ResNet-v2, achieving a single-crop top-5 error rate of 4.4%. In addition, our architecture design is much simpler than all Inception models, and requires considerably fewer hyper-parameters to be set by hand.

与SOTA比较的结果。表5显示了ImageNet验证集上的更多单次裁剪测试结果。除了测试224×224裁剪外，我们还评估了以下320×320裁剪[15]。我们的结果与ResNet、Inception-v3/v4和Incepton-ResNet-v2的结果相比较，前者的错误率为4.4%。此外，我们的架构设计比所有的Inception模型都简单得多，并且需要手动设置的超参数要少得多。

ResNeXt is the foundation of our entries to the ILSVRC 2016 classification task, in which we achieved 2nd place. We note that many models (including ours) start to get saturated on this dataset after using multi-scale and/or multicrop testing. We had a single-model top-1/top-5 error rates of 17.7%/3.7% using the multi-scale dense testing in [14], on par with Inception-ResNet-v2’s single-model results of 17.8%/3.7% that adopts multi-scale, multi-crop testing. We had an ensemble result of 3.03% top-5 error on the test set, on par with the winner’s 2.99% and Inception-v4/InceptionResNet-v2’s 3.08% [37]. 

ResNeXt是我们加入ILSVRC 2016分类任务的基础，我们在其中获得了第二名。我们注意到，在使用多尺度and/or多点测试后，许多模型(包括我们的模型)在该数据集上开始饱和。使用[14]中的多尺度密集测试，我们得到了17.7%/3.7%的单模型top-1/top-5错误率，与Inception-ResNet-v2采用多尺度多裁剪测试得出的17.8%/3.7%单模型结果不相上下。在测试集上，我们的前五名错误率为3.03%，与获胜者的2.99%和Inception-v4/InceptonResNet-v2的3.08%持平[37]。

### 5.2. Experiments on ImageNet-5K
The performance on ImageNet-1K appears to saturate. But we argue that this is not because of the capability of the models but because of the complexity of the dataset. Next we evaluate our models on a larger ImageNet subset that has 5000 categories.

ImageNet-1K上的性能似乎已饱和。但我们认为，这并不是因为模型的能力，而是因为数据集的复杂度。接下来，我们在一个更大的ImageNet子集上评估我们的模型，该子集有5000个类别。

Our 5K dataset is a subset of the full ImageNet-22K set [33]. The 5000 categories consist of the original ImageNet1K categories and additional 4000 categories that have the largest number of images in the full ImageNet set. The 5K set has 6.8 million images, about 5× of the 1K set. There is no official train/val split available, so we opt to evaluate on the original ImageNet-1K validation set. On this 1K-class val set, the models can be evaluated as a 5K-way classification task (all labels predicted to be the other 4K classes are automatically erroneous) or as a 1K-way classification task (softmax is applied only on the 1K classes) at test time.

我们的5K数据集是完整ImageNet-22K集合的子集[33]。5000个类别由原始ImageNet1K类别和其他4000个类别组成，这些类别在整个ImageNet集合中拥有最多的图像。5K集有680万张图像，约为1K集的5倍。没有可用的官方train/val拆分，因此我们选择在原始ImageNet-1K验证集上进行评估。在这个1K类值集上，模型可以在测试时评估为5K方式分类任务(所有预测为其他4K类的标签都自动错误)或1K方式分类(softmax仅适用于1K类)。

Figure 6. ImageNet-5K experiments. Models are trained on the 5K set and evaluated on the original 1K validation set, plotted as a 1K-way classification task. ResNeXt and its ResNet counterpart have similar complexity.
图6.ImageNet-5K实验。模型在5K集合上进行训练，并在原始的1K验证集合上进行评估，绘制为1K方式分类任务。ResNeXt及其对应的ResNet具有类似的复杂度。

Table 6. Error (%) on ImageNet-5K. The models are trained on ImageNet-5K and tested on the ImageNet-1K val set, treated as a 5K-way classification task or a 1K-way classification task at test time. ResNeXt and its ResNet counterpart have similar complexity. The error is evaluated on the single crop of 224×224 pixels.
表6.ImageNet-5K上的错误(%)。模型在ImageNet-5K上进行训练，并在ImageNet-1K值集上进行测试，在测试时被视为5K方式分类任务或1K方式分类。ResNeXt及其对应的ResNet具有类似的复杂度。在224×224像素的单个裁剪上评估错误。

The implementation details are the same as in Sec. 4. The 5K-training models are all trained from scratch, and are trained for the same number of mini-batches as the 1Ktraining models (so 1/5× epochs). Table 6 and Fig. 6 show the comparisons under preserved complexity. ResNeXt-50 reduces the 5K-way top-1 error by 3.2% comparing with ResNet-50, and ResNetXt-101 reduces the 5K-way top-1 error by 2.3% comparing with ResNet-101. Similar gaps are observed on the 1K-way error. These demonstrate the stronger representational power of ResNeXt.

实施细节与第4节相同。5K训练模型都是从头开始训练的，并且训练的小批量数量与1K训练模型相同(因此1/5×epoch)。表6和图6显示了在保持复杂度下的比较。与ResNet-50相比，ResNeXt-50减少了3.2%的5K向top-1误差，与ResNet-101相比，ResNetXt-101减少了2.3%的5K向top-1错误。在1K向误差上也观察到了类似的差距。这表明ResNeXt具有更强的代表性。

Moreover, we find that the models trained on the 5K set (with 1K-way error 22.2%/5.7% in Table 6) perform competitively comparing with those trained on the 1K set (21.2%/5.6% in Table 3), evaluated on the same 1K-way classification task on the validation set. This result is achieved without increasing the training time (due to the same number of mini-batches) and without fine-tuning. We argue that this is a promising result, given that the training task of classifying 5K categories is a more challenging one.

此外，我们发现，在5K集上训练的模型(表6中1K向误差为22.2%/5.7%)与在验证集上相同的1K向分类任务上训练的1K集模型(表3中为21.2%/5.6%)相比具有竞争力。这一结果是在不增加训练时间(由于相同的小批量)和不进行微调的情况下实现的。我们认为这是一个很有希望的结果，因为对5K类别进行分类的训练任务更具挑战性。

### 5.3. Experiments on CIFAR
We conduct more experiments on CIFAR-10 and 100 datasets [23]. We use the architectures as in [14] and replace the basic residual block by the bottleneck template [[1×1, 64] ,[3×3, 64],[1×1, 256]]. Our networks start with a single 3×3 conv 1×1, 256 layer, followed by 3 stages each having 3 residual blocks, and end with average pooling and a fully-connected classifier (total 29-layer deep), following [14]. We adopt the same translation and flipping data augmentation as [14]. Implementation details are in the appendix.

我们在CIFAR-10和100数据集上进行了更多实验[23]。我们使用[14]中的架构，并用瓶颈模板替换基本残差块[[1×1, 64] ,[3×3, 64],[1×1, 256]]. 我们的网络从单个3×3 conv 1×1256层开始，然后是3个阶段，每个阶段有3个残差块，最后是平均池化和一个完全连接的分类器(共29层深)，最后我们采用与[14]相同的平移和翻转数据增广。实施细节见附录。

Table 7. Test error (%) and model size on CIFAR. Our results are the average of 10 runs.
表7.CIFAR上的测试错误(%)和模型大小。我们的成绩是平均10分。

We compare two cases of increasing complexity based on the above baseline: (i) increase cardinality and fix all widths, or (ii) increase width of the bottleneck and fix cardinality = 1. We train and evaluate a series of networks under these changes. Fig. 7 shows the comparisons of test error rates vs. model sizes. We find that increasing cardinality is more effective than increasing width, consistent to what we have observed on ImageNet-1K. Table 7 shows the results and model sizes, comparing with the Wide ResNet [43] which is the best published record. Our model with a similar model size (34.4M) shows results better than Wide ResNet. Our larger method achieves 3.58% test error (average of 10 runs) on CIFAR-10 and 17.31% on CIFAR-100. To the best of our knowledge, these are the state-of-the-art results (with similar data augmentation) in the literature including unpublished technical reports.

基于上述基线，我们比较了两种复杂度增加的情况：(i)增加基数并修复所有宽度，或(ii)增加瓶颈宽度并修复基数=1。我们在这些变化下对一系列网络进行了训练和评估。图7显示了测试错误率与模型尺寸的比较。我们发现增加基数比增加宽度更有效，这与我们在ImageNet-1K上观察到的一致。表7显示了结果和模型大小，与最佳出版记录Wide ResNet[43]进行了比较。我们的模型具有相似的模型大小(34.4M)，其结果优于Wide ResNet。我们更大的方法在CIFAR-10上达到3.58%的测试误差(平均10次)，在CIFAR-100上达到17.31%。据我们所知，这些是文献中最先进的结果(具有类似的数据增广)，包括未发表的技术报告。

Figure 7. Test error vs. model size on CIFAR-10. The results are computed with 10 runs, shown with standard error bars. The labels show the settings of the templates.
图7.CIFAR-10上的测试错误与模型大小。结果通过10次运行进行计算，显示标准错误条。标签显示模板的设置。

### 5.4. Experiments on COCO object detection
Next we evaluate the generalizability on the COCO object detection set [27]. We train the models on the 80k training set plus a 35k val subset and evaluate on a 5k val subset (called minival), following [1]. We evaluate the COCOstyle Average Precision (AP) as well as AP@IoU=0.5 [27]. We adopt the basic Faster R-CNN [32] and follow [14] to plug ResNet/ResNeXt into it. The models are pre-trained on ImageNet-1K and fine-tuned on the detection set. Implementation details are in the appendix.

接下来，我们评估COCO目标检测集的泛化性[27]。我们在80k训练集和35k val子集上训练模型，并在5k val子集(称为minival)上进行评估，如下[1]。我们评估了COCOstyle平均精度(AP)以及AP@IoU=0.5 [27]. 我们采用基本的Faster R-CNN[32]，并按照[14]将ResNet/ResNeXt插入其中。这些模型在ImageNet-1K上进行了预先训练，并在检测集上进行了微调。实施细节见附录。

Table 8. Object detection results on the COCO minival set. ResNeXt and its ResNet counterpart have similar complexity.
表8.COCO minival集合上的目标检测结果。ResNeXt及其对应的ResNet具有类似的复杂度。

Table 8 shows the comparisons. On the 50-layer baseline, ResNeXt improves AP@0.5 by 2.1% and AP by 1.0%, without increasing complexity. ResNeXt shows smaller improvements on the 101-layer baseline. We conjecture that more training data will lead to a larger gap, as observed on the ImageNet-5K set.
表8显示了这些比较。在50层基线上，ResNeXt提高了AP@0.5在不增加复杂度的情况下，降低2.1%，降低1.0%。ResNeXt在101层基线上显示出较小的改进。我们推测，更多的训练数据将导致更大的差距，正如在ImageNet-5K集合上观察到的那样。

It is also worth noting that recently ResNeXt has been adopted in Mask R-CNN [12] that achieves state-of-the-art results on COCO instance segmentation and object detection tasks.

还值得注意的是，最近ResNeXt已在Mask R-CNN[12]中采用，在COCO实例分割和目标检测任务上取得了最先进的结果。

## Acknowledgment
S.X. and Z.T.’s research was partly supported by NSF IIS-1618477. The authors would like to thank Tsung-Yi Lin and Priya Goyal for valuable discussions.

S.X.和Z.T.的研究部分得到了NSF IIS-1618477的支持。作者感谢Tsung Yi Lin和Priya Goyal的宝贵讨论。

## A. Implementation Details: CIFAR 实施细节
We train the models on the 50k training set and evaluate on the 10k test set. The input image is 32×32 randomly cropped from a zero-padded 40×40 image or its flipping, following [14]. No other data augmentation is used. The first layer is 3×3 conv with 64 filters. There are 3 stages each having 3 residual blocks, and the output map size is 32, 16, and 8 for each stage [14]. The network ends with a global average pooling and a fully-connected layer. Width is increased by 2× when the stage changes (downsampling), as in Sec. 3.1. The models are trained on 8 GPUs with a mini-batch size of 128, with a weight decay of 0.0005 and a momentum of 0.9. We start with a learning rate of 0.1 and train the models for 300 epochs, reducing the learning rate at the 150-th and 225-th epoch. Other implementation details are as in [11].

我们在50k训练集上训练模型，并在10k测试集上进行评估。输入图像为32×32，是从一个填充了零的40×40图像或其翻转图像中随机裁剪的，如下[14]。未使用其他数据增广。第一层是带64个卷积核的3×3 conv。有3个阶段，每个阶段有3个残差块，每个阶段的输出映射大小为32、16和8[14]。网络以全局平均池化和全连接层结束。如第3.1节所述，当阶段变化(下采样)时，宽度增加2倍。模型在8个GPU上进行训练，最小批量大小为128，权重衰减为0.0005，动量为0.9。我们从0.1的学习率开始，对模型进行300个周期的训练，降低了第150和225个周期的学习率。其他实现细节如[11]所示。

## B. Implementation Details: Object Detection
We adopt the Faster R-CNN system [32]. For simplicity we do not share the features between RPN and Fast R-CNN. In the RPN step, we train on 8 GPUs with each GPU holding 2 images per mini-batch and 256 anchors per image. We train the RPN step for 120k mini-batches at a learning rate of 0.02 and next 60k at 0.002. In the Fast R-CNN step, we train on 8 GPUs with each GPU holding 1 image and 64 regions per mini-batch. We train the Fast R-CNN step for 120k mini-batches at a learning rate of 0.005 and next 60k at 0.0005, We use a weight decay of 0.0001 and a momentum of 0.9. Other implementation details are as in https://github.com/rbgirshick/py-faster-rcnn .

我们采用更快的R-CNN系统[32]。为了简单起见，我们不共享RPN和Fast R-CNN之间的功能。在RPN步骤中，我们在8个GPU上进行训练，每个GPU每个小批次包含2个图像，每个图像包含256个锚。我们以0.02的学习率对120k个小批次的RPN步骤进行训练，然后以0.002的学习率训练60k个小批量的RPN。在Fast R-CNN步骤中，我们在8个GPU上进行训练，每个GPU每个小批次包含1个图像和64个区域。我们以0.005的学习率对120k个小批次的快速R-CNN步骤进行训练，然后以0.0005的学习率训练60k个小批量。我们使用0.0001的权重衰减和0.9的动量。其他实施细节如所示https://github.com/rbgirshick/py-faster-rcnn .

## References
1. S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Insideoutside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016.
2. G. Cantor. ¨Uber unendliche, lineare punktmannichfaltigkeiten, arbeiten zur mengenlehre aus den jahren 1872-1884. 1884.
3. R. Collobert, S. Bengio, and J. Mari´ethoz. Torch: a modular machine learning software library. Technical report, Idiap, 2002.
4. A. Conneau, H. Schwenk, L. Barrault, and Y. Lecun. Very deep convolutional networks for natural language processing. arXiv:1606.01781, 2016.
5. N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In CVPR, 2005.
6. E. Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus. Exploiting linear structure within convolutional networks for efficient evaluation. In NIPS, 2014.
7. J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In ICML, 2014.
8. D. Eigen, J. Rolfe, R. Fergus, and Y. LeCun. Understanding deep architectures using a recursive convolutional network. arXiv:1312.1847, 2013.
9. R. Girshick. Fast R-CNN. In ICCV, 2015.
10. R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.
11. S. Gross and M. Wilber. Training and investigating Residual Nets. https://github.com/ facebook/fb.resnet.torch, 2016.
12. K. He, G. Gkioxari, P. Doll´ar, and R. Girshick. Mask R-CNN. arXiv:1703.06870, 2017.
13. K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015.
14. K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
15. K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
16. Y. Ioannou, D. Robertson, R. Cipolla, and A. Criminisi. Deep roots: Improving cnn efficiency with hierarchical filter groups. arXiv:1605.06489, 2016.
17. S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.
18. M. Jaderberg, A. Vedaldi, and A. Zisserman. Speeding up convolutional neural networks with low rank expansions. In BMVC, 2014.
19. Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. arXiv:1408.5093, 2014.
20. N. Kalchbrenner, L. Espeholt, K. Simonyan, A. v. d. Oord, A. Graves, and K. Kavukcuoglu. Neural machine translation in linear time. arXiv:1610.10099, 2016.
21. Y.-D. Kim, E. Park, S. Yoo, T. Choi, L. Yang, and D. Shin. Compression of deep convolutional neural networks for fast and low power mobile applications. In ICLR, 2016.
22. P. Kontschieder, M. Fiterau, A. Criminisi, and S. R. Bul`o. Deep convolutional neural decision forests. In ICCV, 2015.
23. A. Krizhevsky. Learning multiple layers of features from tiny images. Tech Report, 2009.
24. A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.
25. Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.
26. M. Lin, Q. Chen, and S. Yan. Network in network. In ICLR, 2014.
27. T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick. Microsoft COCO: Common objects in context. In ECCV. 2014.
28. J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.
29. D. G. Lowe. Distinctive image features from scaleinvariant keypoints. IJCV, 2004.
30. A. Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu. Wavenet: A generative model for raw audio. arXiv:1609.03499, 2016.
31. P. O. Pinheiro, R. Collobert, and P. Dollar. Learning to segment object candidates. In NIPS, 2015.
32. S. Ren, K. He, R. Girshick, and J. Sun. Faster RCNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.
33. O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.
34. P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.
35. L. Sifre and S. Mallat. Rigid-motion scattering for texture classification. arXiv:1403.1687, 2014.
36. K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.
37. C. Szegedy, S. Ioffe, and V. Vanhoucke. Inceptionv4, inception-resnet and the impact of residual connections on learning. In ICLR Workshop, 2016.
38. C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.
39. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.
40. A. Veit, M. Wilber, and S. Belongie. Residual networks behave like ensembles of relatively shallow network. In NIPS, 2016.
41. Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv:1609.08144, 2016.
42. W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, and G. Zweig. The Microsoft 2016 Conversational Speech Recognition System. arXiv:1609.03528, 2016.
43. S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.
44. M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional neural networks. In ECCV, 2014.