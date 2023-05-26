# Squeeze-and-Excitation Networks
挤压和激励网络 2017-9-5  https://arxiv.org/abs/1709.01507

## Abstract
The central building block of convolutional neural networks (CNNs) is the convolution operator, which enables networks to construct informative features by fusing both spatial and channel-wise information within local receptive fields at each layer. A broad range of prior research has investigated the spatial component of this relationship, seeking to strengthen the representational power of a CNN by enhancing the quality of spatial encodings throughout its feature hierarchy. In this work, we focus instead on the channel relationship and propose a novel architectural unit, which we term the “Squeeze-and-Excitation” (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels. We show that these blocks can be stacked together to form SENet architectures that generalise extremely effectively across different datasets. We further demonstrate that SE blocks bring significant improvements in performance for existing state-of-the-art CNNs at slight additional computational cost. Squeeze-and-Excitation Networks formed the foundation of our ILSVRC 2017 classification submission which won first place and reduced the top-5 error to 2.251%, surpassing the winning entry of 2016 by a relative improvement of ∼25%. Models and code are available at https://github.com/hujie-frank/SENet.

卷积神经网络(CNN)的核心构建块是卷积算子，它使网络能够通过融合每个层的局部感受野内的空间和信道信息来构建信息特征。之前的大量研究已经调查了这种关系的空间成分，试图通过提高整个特征层次的空间编码质量来增强CNN的表现力。在这项工作中，我们将重点放在信道关系上，并提出了一种新的架构单元，我们称之为“挤压和激励”(SE)块，它通过显式建模信道之间的相互依赖性自适应地重新校准信道特征响应。我们展示了这些块可以堆叠在一起以形成SENet架构，该架构在不同的数据集中非常有效地通用。我们进一步证明，SE块在略微增加计算成本的情况下，为现有最先进的CNN带来了显著的性能改进。挤压和激励网络为我们的ILSVRC 2017分类提交奠定了基础，该分类获得了第一名，并将前五名的误差降至2.251%，以相对提高的∼25%. 模型和代码可在https://github.com/hujie-frank/SENet.

Index Terms—Squeeze-and-Excitation, Image representations, Attention, Convolutional Neural Networks.  

索引术语 挤压和激励，图像表示，注意力，卷积神经网络。

## 1 INTRODUCTION
CONVOLUTIONAL neural networks (CNNs) have proven to be useful models for tackling a wide range of visual tasks [1], [2], [3], [4]. At each convolutional layer in the network, a collection of filters expresses neighbourhood spatial connectivity patterns along input channels—fusing spatial and channel-wise information together within local receptive fields. By interleaving a series of convolutional layers with non-linear activation functions and downsampling operators, CNNs are able to produce image representations that capture hierarchical patterns and attain global theoretical receptive fields. A central theme of computer vision research is the search for more powerful representations that capture only those properties of an image that are most salient for a given task, enabling improved performance.

卷积神经网络(CNN)已被证明是处理各种视觉任务的有用模型[1]，[2]，[3]，[4]。在网络中的每个卷积层，一组滤波器表示沿输入信道的邻域空间连通性模式，将空间和信道信息融合在局部接收场中。通过将一系列卷积层与非线性激活函数和下采样算子交织，CNN能够生成捕获分层模式并获得全局理论接受域的图像表示。计算机视觉研究的一个中心主题是寻找更强大的表示，只捕捉图像中对给定任务最显著的属性，从而提高性能。

As a widely-used family of models for vision tasks, the development of new neural network architecture designs now represents a key frontier in this search. Recent research has shown that the representations produced by CNNs can be strengthened by integrating learning mechanisms into the network that help capture spatial correlations between features. One such approach, popularised by the Inception family of architectures [5], [6], incorporates multi-scale processes into network modules to achieve improved performance. Further work has sought to better model spatial dependencies [7], [8] and incorporate spatial attention into the structure of the network [9].

作为视觉任务的一个广泛使用的模型家族，新的神经网络架构设计的发展现在代表了这一研究的一个关键前沿。最近的研究表明，通过将学习机制集成到网络中，有助于捕捉特征之间的空间相关性，CNN生成的表示可以得到加强。Inception系列架构[5]、[6]推广了一种这样的方法，它将多尺度过程合并到网络模块中，以提高性能。进一步的工作试图更好地建模空间相关性[7]、[8]，并将空间注意力纳入网络结构[9]。

In this paper, we investigate a different aspect of network design - the relationship between channels. We introduce a new architectural unit, which we term the Squeeze-andExcitation (SE) block, with the goal of improving the quality of representations produced by a network by explicitly modelling the interdependencies between the channels of its convolutional features. To this end, we propose a mechanism that allows the network to perform feature recalibration, through which it can learn to use global information to selectively emphasise informative features and suppress less useful ones.

在本文中，我们研究了网络设计的另一个方面 —— 信道之间的关系。我们引入了一个新的架构单元，我们称之为挤压和激励(SE)块，其目的是通过显式建模其卷积特征的通道之间的相互依赖关系来提高网络生成的表示质量。为此，我们提出了一种机制，允许网络执行特征重新校准，通过该机制，网络可以学习使用全局信息来选择性地强调信息性特征并抑制不太有用的特征。

The structure of the SE building block is depicted in Fig. 1. For any given transformation Ftr mapping the input X to the feature maps U where U ∈ R H×W×C , e.g. a convolution, we can construct a corresponding SE block to perform feature recalibration. The features U are first passed through a squeeze operation, which produces a channel descriptor by aggregating feature maps across their spatial dimensions (H × W). The function of this descriptor is to produce an embedding of the global distribution of channel-wise feature responses, allowing information from the global receptive field of the network to be used by all its layers. The aggregation is followed by an excitation operation, which takes the form of a simple self-gating mechanism that takes the embedding as input and produces a collection of per-channel modulation weights. These weights are applied to the feature maps U to generate the output of the SE block which can be fed directly into subsequent layers of the network.

SE构建块的结构如图1所示。对于任何给定的变换，Ftr将输入X映射到特征映射U，其中U∈ ×C，例如卷积，我们可以构造相应的SE块来执行特征重新校准。特征U首先通过挤压操作，该操作通过在其空间维度(H×W)上聚集特征图来生成通道描述符。该描述符的功能是生成信道特征响应的全局分布的嵌入，允许网络的所有层使用来自网络的全局接收场的信息。聚合之后是激励操作，该操作采用简单的自选通机制的形式，该机制将嵌入作为输入，并产生每个信道调制权重的集合。这些权重被应用于特征图U以生成SE块的输出，该输出可以直接馈送到网络的后续层中。

Fig. 1. A Squeeze-and-Excitation block. 

It is possible to construct an SE network (SENet) by simply stacking a collection of SE blocks. Moreover, these SE blocks can also be used as a drop-in replacement for the original block at a range of depths in the network architecture (Section 6.4). While the template for the building block is generic, the role it performs at different depths differs throughout the network. In earlier layers, it excites informative features in a class-agnostic manner, strengthening the shared low-level representations. In later layers, the SE blocks become increasingly specialised, and respond to different inputs in a highly class-specific manner (Section 7.2). As a consequence, the benefits of the feature recalibration performed by SE blocks can be accumulated through the network.

可以通过简单地堆叠SE块的集合来构建SE网络(SENet)。此外，这些SE块也可以在网络架构中的某个深度范围内用作原始块的替换(第6.4节)。虽然构建块的模板是通用的，但在整个网络中，它在不同深度执行的角色不同。在早期的层中，它以一种与类无关的方式激发信息特征，加强了共享的低级表示。在后面的层中，SE块变得越来越专业化，并以高度特定的方式响应不同的输入(第7.2节)。因此，SE块执行的特征重新校准的好处可以通过网络积累。

The design and development of new CNN architectures is a difficult engineering task, typically requiring the selection of many new hyperparameters and layer configurations. By contrast, the structure of the SE block is simple and can be used directly in existing state-of-the-art architectures by replacing components with their SE counterparts, where the performance can be effectively enhanced. SE blocks are also computationally lightweight and impose only a slight increase in model complexity and computational burden.

设计和开发新的CNN架构是一项艰巨的工程任务，通常需要选择许多新的超参数和层配置。相比之下，SE块的结构很简单，可以直接在现有的最先进的架构中使用，方法是将组件替换为SE对应的组件，从而有效地提高性能。SE块在计算上也是轻量级的，并且只略微增加了模型复杂性和计算负担。

To provide evidence for these claims, we develop several SENets and conduct an extensive evaluation on the ImageNet dataset [10]. We also present results beyond ImageNet that indicate that the benefits of our approach are not restricted to a specific dataset or task. By making use of SENets, we ranked first in the ILSVRC 2017 classification competition. Our best model ensemble achieves a 2.251% top-5 error on the test set1 . This represents roughly a 25% relative improvement when compared to the winner entry of the previous year (top-5 error of 2.991%). 

为了为这些主张提供证据，我们开发了几个SENet，并对ImageNet数据集进行了广泛评估[10]。我们还展示了ImageNet之外的结果，这些结果表明我们的方法的好处并不局限于特定的数据集或任务。通过使用SENets，我们在ILSVRC 2017分类比赛中排名第一。我们的最佳模型集合在测试集1上获得了2.251%的前5名误差。与前一年的获胜者相比，这大约代表了25%的相对改进(前5名的误差为2.991%)。

## 2 RELATED WORK
Deeper architectures. VGGNets [11] and Inception models [5] showed that increasing the depth of a network could significantly increase the quality of representations that it was capable of learning. By regulating the distribution of the inputs to each layer, Batch Normalization (BN) [6] added stability to the learning process in deep networks and produced smoother optimisation surfaces [12]. Building on these works, ResNets demonstrated that it was possible to learn considerably deeper and stronger networks through the use of identity-based skip connections [13], [14]. Highway networks [15] introduced a gating mechanism to regulate the flow of information along shortcut connections. Following these works, there have been further reformulations of the connections between network layers [16], [17], which show promising improvements to the learning and representational properties of deep networks.

更深层次的架构。VGGNets[11]和Inception模型[5]表明，增加网络的深度可以显著提高其能够学习的表示质量。通过调节每个层的输入分布，批量规范化(BN)[6]为深度网络中的学习过程增加了稳定性，并产生了更平滑的优化曲面[12]。在这些工作的基础上，ResNets证明，通过使用基于身份的跳过连接，可以学习更深、更强的网络[13]、[14]。高速公路网络[15]引入了一种门控机制，以调节沿快捷连接的信息流。在这些工作之后，对网络层之间的连接进行了进一步的重新表述[16]、[17]，这显示了对深度网络的学习和表征特性的有希望的改进。

An alternative, but closely related line of research has focused on methods to improve the functional form of the computational elements contained within a network. Grouped convolutions have proven to be a popular approach for increasing the cardinality of learned transformations [18], [19]. More flexible compositions of operators can be achieved with multi-branch convolutions [5], [6], [20], [21], which can be viewed as a natural extension of the grouping operator. In prior work, cross-channel correlations are typically mapped as new combinations of features, either independently of spatial structure [22], [23] or jointly by using standard convolutional filters [24] with 1 × 1 convolutions. Much of this research has concentrated on the objective of reducing model and computational complexity, reflecting an assumption that channel relationships can be formulated as a composition of instance-agnostic functions with local receptive fields. In contrast, we claim that providing the unit with a mechanism to explicitly model dynamic, non-linear dependencies between channels using global information can ease the learning process, and significantly enhance the representational power of the network.

另一个可供选择但密切相关的研究方向集中于改进网络中计算元素的功能形式的方法。分组卷积已被证明是提高学习变换基数的一种流行方法[18]，[19]。使用多分支卷积[5]、[6]、[20]、[21]可以实现更灵活的运算符组合，这可视为分组运算符的自然扩展。在先前的工作中，交叉信道相关性通常被映射为新的特征组合，或者独立于空间结构[22]、[23]，或者通过使用具有1×1卷积的标准卷积滤波器[24]联合映射。这项研究的大部分集中在降低模型和计算复杂度的目标上，反映了一种假设，即信道关系可以表述为具有局部接受域的实例未知函数的组合。相反，我们声称，为该单元提供一种机制，使用全局信息显式地建模信道之间的动态、非线性依赖关系，可以简化学习过程，并显著增强网络的代表能力。

Algorithmic Architecture Search. Alongside the works described above, there is also a rich history of research that aims to forgo manual architecture design and instead seeks to learn the structure of the network automatically. Much of the early work in this domain was conducted in the neuro-evolution community, which established methods for searching across network topologies with evolutionary methods [25], [26]. While often computationally demanding, evolutionary search has had notable successes which include finding good memory cells for sequence models [27], [28] and learning sophisticated architectures for largescale image classification [29], [30], [31]. With the goal of reducing the computational burden of these methods, efficient alternatives to this approach have been proposed based on Lamarckian inheritance [32] and differentiable architecture search [33].

算法架构搜索。除了上述工作之外，还有一段丰富的研究历史，旨在放弃手动架构设计，而寻求自动学习网络结构。该领域的大部分早期工作是在神经进化社区进行的，该社区建立了用进化方法搜索网络拓扑的方法[25]，[26]。虽然进化搜索通常需要计算，但它已经取得了显著的成功，包括为序列模型[27]、[28]找到了良好的存储单元，并学习了大规模图像分类的复杂架构[29]、[30]、[31]。为了减少这些方法的计算负担，基于拉马克遗传[32]和可微结构搜索[33]，提出了该方法的有效替代方案。

By formulating architecture search as hyperparameter optimisation, random search [34] and other more sophisticated model-based optimisation techniques [35], [36] can also be used to tackle the problem. Topology selection as a path through a fabric of possible designs [37] and direct architecture prediction [38], [39] have been proposed as additional viable architecture search tools. Particularly strong results have been achieved with techniques from reinforcement learning [40], [41], [42], [43], [44]. SE blocks 3 can be used as atomic building blocks for these search algorithms, and were demonstrated to be highly effective in this capacity in concurrent work [45].

通过将架构搜索表述为超参数优化、随机搜索[34]和其他更复杂的基于模型的优化技术[35]，[36]也可以用来解决这个问题。拓扑选择作为通过可能设计结构的路径[37]和直接架构预测[38]、[39]已被提议作为额外的可行架构搜索工具。强化学习[40]、[41]、[42]、[43]、[44]的技术已经取得了特别显著的成果。SE块3可以用作这些搜索算法的原子构建块，并被证明在并发工作中在这一能力方面非常有效[45]。

Attention and gating mechanisms. Attention can be interpreted as a means of biasing the allocation of available computational resources towards the most informative components of a signal [46], [47], [48], [49], [50], [51]. Attention mechanisms have demonstrated their utility across many tasks including sequence learning [52], [53], localisation and understanding in images [9], [54], image captioning [55], [56] and lip reading [57]. In these applications, it can be incorporated as an operator following one or more layers representing higher-level abstractions for adaptation between modalities. Some works provide interesting studies into the combined use of spatial and channel attention [58], [59]. Wang et al. [58] introduced a powerful trunk-and-mask attention mechanism based on hourglass modules [8] that is inserted between the intermediate stages of deep residual networks. By contrast, our proposed SE block comprises a lightweight gating mechanism which focuses on enhancing the representational power of the network by modelling channel-wise relationships in a computationally efficient manner. 

注意力和门控机制。注意力可以被解释为将可用计算资源的分配偏向于信号中信息量最大的分量的一种手段[46]、[47]、[48]、[49]、[50]、[51]。注意力机制已在许多任务中证明了其效用，包括序列学习[52]、[53]、图像定位和理解[9]、[54]、图像字幕[55]、[56]和唇读[57]。在这些应用中，它可以作为一个操作符，跟随一个或多个表示模态之间适应的高级抽象的层。一些作品对空间注意力和渠道注意力的组合使用进行了有趣的研究[58]，[59]。Wanget al [58]介绍了一种基于沙漏模块[8]的强大的躯干和面罩注意机制，该机制插入深度残差网络的中间阶段之间。相比之下，我们提出的SE块包括一个轻量级门控机制，该机制通过以计算高效的方式对信道关系进行建模，重点增强网络的表示能力。

## 3 SQUEEZE-AND-EXCITATION BLOCKS
A Squeeze-and-Excitation block is a computational unit which can be built upon a transformation Ftr mapping an input X ∈ R H0×W0×C 0 to feature maps U ∈ R H×W×C. In the notation that follows we take Ftr to be a convolutional operator and use V = [v1, v2, . . . , vC ] to denote the learned set of filter kernels, where vc refers to the parameters of the c-th filter. We can then write the outputs as U = [u1, u2, . . . , uC ], where 

挤压和激励块是一种计算单元，可基于映射输入X的变换Ftr构建∈ R H0×W0×C 0到要素图U∈ 长×宽×长。在下面的符号中，我们将Ftr作为卷积运算符，并使用V＝[v1，v2，…，vC]表示学习的滤波器核集合，其中vC表示第c个滤波器的参数。然后我们可以将输出写为U＝[u1，u2，…，uC]，其中

uc = vc ∗ X = C X0 s=1 v s c ∗ x s . (1)

Here ∗ denotes convolution, vc = [v 1 c , v 2 c , . . . , vC 0 c ], X = [x 1 , x 2 , . . . , xC 0 ] and uc ∈ R H×W . v s c is a 2D spatial kernel representing a single channel of vc that acts on the corresponding channel of X. To simplify the notation, bias terms are omitted. Since the output is produced by a summation through all channels, channel dependencies are implicitly embedded in vc, but are entangled with the local spatial correlation captured by the filters. The channel relationships modelled by convolution are inherently implicit and local (except the ones at top-most layers). We expect the learning of convolutional features to be enhanced by explicitly modelling channel interdependencies, so that the network is able to increase its sensitivity to informative features which can be exploited by subsequent transformations. Consequently, we would like to provide it with access to global information and recalibrate filter responses in two steps, squeeze and excitation, before they are fed into the next transformation. A diagram illustrating the structure of an SE block is shown in Fig. 1.

在这里∗ 表示卷积，vc=[v1c，v2c，…，vC0c]，X=[x1，x2，…，xC0]和uc∈ ×W.vsc是一个2D空间核，表示作用于X的对应通道上的单个vc通道。为了简化符号，省略了偏置项。由于输出是通过所有信道的求和产生的，因此信道依赖性隐式嵌入在vc中，但与滤波器捕获的局部空间相关性纠缠。由卷积建模的信道关系本质上是隐式的和局部的(最顶层的除外)。我们期望通过显式建模信道相互依赖性来增强卷积特征的学习，从而使网络能够提高其对信息特征的敏感度，这些信息特征可通过后续转换加以利用。因此，我们希望为其提供获取全局信息的途径，并在将滤波器响应输入下一次转换之前，分两步(挤压和激励)重新校准滤波器响应。SE块的结构示意图如图1所示。

### 3.1 Squeeze: Global Information Embedding
In order to tackle the issue of exploiting channel dependencies, we first consider the signal to each channel in the output features. Each of the learned filters operates with a local receptive field and consequently each unit of the transformation output U is unable to exploit contextual information outside of this region.

为了解决利用信道依赖性的问题，我们首先考虑输出特性中每个信道的信号。每个学习的滤波器都与局部接受场一起工作，因此变换输出U的每个单元都不能利用该区域之外的上下文信息。

To mitigate this problem, we propose to squeeze global spatial information into a channel descriptor. This is achieved by using global average pooling to generate channel-wise statistics. Formally, a statistic z ∈ R C is generated by shrinking U through its spatial dimensions H × W, such that the c-th element of z is calculated by: 

为了缓解这个问题，我们建议将全局空间信息压缩到信道描述符中。这是通过使用全局平均池来生成信道统计来实现的。形式上，统计z∈ R C是通过将U缩小到其空间尺寸H×W而生成的，因此z的第C个元素通过以下公式计算：

zc = Fsq(uc) = 1 H × WX H i=1 X W j=1 uc(i, j). (2)

Discussion. The output of the transformation U can be interpreted as a collection of the local descriptors whose statistics are expressive for the whole image. Exploiting such information is prevalent in prior feature engineering work [60], [61], [62]. We opt for the simplest aggregation technique, global average pooling, noting that more sophisticated strategies could be employed here as well.

讨论. 变换U的输出可以被解释为局部描述符的集合，其统计量表示整个图像。利用此类信息在先前的特征工程工作[60]、[61]、[62]中很普遍。我们选择了最简单的聚合技术，即全局平均池，注意到这里也可以采用更复杂的策略。

### 3.2 Excitation: Adaptive Recalibration
To make use of the information aggregated in the squeeze operation, we follow it with a second operation which aims to fully capture channel-wise dependencies. To fulfil this objective, the function must meet two criteria: first, it must be flexible (in particular, it must be capable of learning a nonlinear interaction between channels) and second, it must learn a non-mutually-exclusive relationship since we would like to ensure that multiple channels are allowed to be emphasised (rather than enforcing a one-hot activation). To meet these criteria, we opt to employ a simple gating mechanism with a sigmoid activation: 

为了利用挤压操作中聚集的信息，我们在第二个操作之后进行，该操作旨在完全捕获通道相关性。为了实现这一目标，该功能必须满足两个标准：第一，它必须灵活(特别是，它必须能够学习渠道之间的非线性互动); 第二，它必须学习非互斥关系，因为我们希望确保允许强调多个渠道(而不是强制执行一次激活)。为了满足这些标准，我们选择使用简单的门控机制和sigmoid激活：

s = Fex(z,W) = σ(g(z,W)) = σ(W2δ(W1z)), (3) 

where δ refers to the ReLU [63] function, W1 ∈ RC r ×C and W2 ∈ R C× C r . To limit model complexity and aid generalisation, we parameterise the gating mechanism by forming a bottleneck with two fully-connected (FC) layers around the non-linearity, i.e. a dimensionality-reduction layer with reduction ratio r (this parameter choice is discussed in Section 6.1), a ReLU and then a dimensionality-increasing layer returning to the channel dimension of the transformation output U. The final output of the block is obtained by rescaling U with the activations s: xec = Fscale(uc, sc) = sc uc, (4) where Xe = [xe1, xe2, . . . , xeC ] and Fscale(uc, sc) refers to channel-wise multiplication between the scalar sc and the feature map uc ∈ R H×W .

其中δ表示ReLU[63]函数，W1∈ 钢筋混凝土r×C和W2∈ R C×C R。为了限制模型的复杂性和帮助推广，我们通过在非线性周围形成具有两个完全连接(FC)层的瓶颈来参数化选通机制，即具有缩减比R的降维层(该参数选择在第6.1节中讨论)，一个ReLU，然后一个维度增加层返回到变换输出U的信道维度。块的最终输出是通过使用激活s重新缩放U而获得的：xec＝Fscale(uc，sc)＝sc uc，(4)，其中Xe＝[xe1，xe2，…，xec]，Fscale是指标量sc和特征图uc之间的信道相乘∈ ×宽。

Discussion. The excitation operator maps the inputspecific descriptor z to a set of channel weights. In this regard, SE blocks intrinsically introduce dynamics conditioned on the input, which can be regarded as a selfattention function on channels whose relationships are not confined to the local receptive field the convolutional filters are responsive to. 

讨论. 激励算子将输入特定描述符z映射到一组信道权重。在这方面，SE块本质上引入了以输入为条件的动力学，这可以被视为信道上的自我注意函数，其关系不限于卷积滤波器响应的局部接收场。

Fig. 2. The schema of the original Inception module (left) and the SEInception module (right).
图2.原始初始模块(左)和SEInception模块(右)的模式。

### 3.3 Instantiations 实例化
The SE block can be integrated into standard architectures such as VGGNet [11] by insertion after the non-linearity following each convolution. Moreover, the flexibility of the SE block means that it can be directly applied to transformations beyond standard convolutions. To illustrate this point, we develop SENets by incorporating SE blocks into several examples of more complex architectures, described next.

SE块可以通过在每个卷积之后的非线性之后插入来集成到诸如VGGNet[11]的标准架构中。此外，SE块的灵活性意味着它可以直接应用于标准卷积以外的变换。为了说明这一点，我们通过将SE块合并到更复杂的架构的几个样本中来开发SENet，下面将对此进行描述。

We first consider the construction of SE blocks for Inception networks [5]. Here, we simply take the transformation Ftr to be an entire Inception module (see Fig. 2) and by making this change for each such module in the architecture, we obtain an SE-Inception network. SE blocks can also be used directly with residual networks (Fig. 3 depicts the schema of an SE-ResNet module). Here, the SE block transformation Ftr is taken to be the non-identity branch of a residual module. Squeeze and Excitation both act before summation with the identity branch. Further variants that integrate SE blocks with ResNeXt [19], Inception-ResNet [21], MobileNet [64] and ShuffleNet [65] can be constructed by following similar schemes. For concrete examples of SENet architectures, a detailed description of SE-ResNet-50 and SE-ResNeXt-50 is given in Table 1.

我们首先考虑初始网络的SE块的构造[5]。在这里，我们简单地将转换Ftr取为一个完整的Inception模块(见图2)，通过对架构中的每个这样的模块进行此更改，我们获得了一个SE Inceptions网络。SE块也可以直接用于残差网络(图3描述了SE ResNet模块的模式)。这里，SE块变换Ftr被认为是残差模的非恒等分支。挤压和激发都在与恒等式分支求和之前起作用。将SE块与ResNeXt[19]、Inception ResNet[21]、MobileNet[64]和ShuffleNet[65]集成的其他变体可以通过以下类似方案构建。对于SENet架构的具体样本，表1中给出了SE-ResNet-50和SE-ResNeXt-50的详细描述。

One consequence of the flexible nature of the SE block is that there are several viable ways in which it could be integrated into these architectures. Therefore, to assess sensitivity to the integration strategy used to incorporate SE blocks into a network architecture, we also provide ablation experiments exploring different designs for block inclusion in Section 6.5. 

SE块的灵活特性的一个结果是，有几种可行的方法可以将其集成到这些架构中。因此，为了评估用于将SE块纳入网络架构的集成策略的敏感性，我们还提供了第6.5节中探讨块纳入的不同设计的消融实验。

## 4 MODEL AND COMPUTATIONAL COMPLEXITY 模型和计算复杂性
For the proposed SE block design to be of practical use, it must offer a good trade-off between improved performance and increased model complexity. To illustrate the computational burden associated with the module, we consider a comparison between ResNet-50 and SE-ResNet-50 as an example. ResNet-50 requires ∼3.86 GFLOPs in a single forward pass for a 224 × 224 pixel input image. Each SE block makes use of a global average pooling operation in the squeeze phase and two small FC layers in the excitation phase, followed by an inexpensive channel-wise scaling operation. In the aggregate, when setting the reduction ratio r (introduced in Section 3.2) to 16, SE-ResNet-50 requires ∼3.87 GFLOPs, corresponding to a 0.26% relative increase over the original ResNet-50. In exchange for this slight additional computational burden, the accuracy of SE-ResNet-50 surpasses that of ResNet-50 and indeed, approaches that of a deeper ResNet-101 network requiring ∼7.58 GFLOPs (Table 2).

为了使拟议的SE块设计具有实际用途，它必须在改进的性能和增加的模型复杂度之间提供良好的权衡。为了说明与模块相关的计算负担，我们以ResNet-50和SE-ResNet-50之间的比较为例。ResNet-50需要∼对于224×224像素的输入图像，一次前向通过中的3.86 GFLOP。每个SE块在挤压阶段使用全局平均池操作，在激发阶段使用两个小FC层，然后使用便宜的逐通道缩放操作。总的来说，当将减速比r(第3.2节中介绍)设置为16时，SE-ResNet-50要求∼3.87 GFLOP，与原始ResNet-50相比，相对增加了0.26%。作为交换，SE-ResNet-50的精确度超过了ResNet-500，事实上，接近了需要更深入的ResNet-101网络的精确度∼7.58 GFLOP(表2)。

Fig. 3. The schema of the original Residual module (left) and the SEResNet module (right).
图3.原始残差模块(左)和SEResNet模块(右)的模式。

In practical terms, a single pass forwards and backwards through ResNet-50 takes 190 ms, compared to 209 ms for SE-ResNet-50 with a training minibatch of 256 images (both timings are performed on a server with 8 NVIDIA Titan X GPUs). We suggest that this represents a reasonable runtime overhead, which may be further reduced as global pooling and small inner-product operations receive further optimisation in popular GPU libraries. Due to its importance for embedded device applications, we further benchmark CPU inference time for each model: for a 224 × 224 pixel input image, ResNet-50 takes 164 ms in comparison to 167 ms for SE-ResNet-50. We believe that the small additional computational cost incurred by the SE block is justified by its contribution to model performance.

实际上，通过ResNet-50的一次向前和向后传递需要190 ms，而对于具有256个图像的训练小批量的SE-ResNet-50，则需要209 ms(这两个计时都在具有8个NVIDIA Titan X GPU的服务器上执行)。我们建议，这代表了合理的运行时开销，随着通用GPU库中的全局池和小型内部产品操作得到进一步优化，这可能会进一步减少。由于其对嵌入式设备应用程序的重要性，我们进一步对每个模型的CPU推断时间进行了基准测试：对于224×224像素的输入图像，ResNet-50需要164毫秒，而SE-ResNet-50则需要167毫秒。

We next consider the additional parameters introduced by the proposed SE block. These additional parameters result solely from the two FC layers of the gating mechanism and therefore constitute a small fraction of the total network capacity. Concretely, the total number introduced by the weight parameters of these FC layers is given by: 

接下来，我们考虑拟议SE区块引入的额外参数。这些附加参数仅由选通机制的两个FC层产生，因此构成总网络容量的一小部分。具体而言，这些FC层的权重参数引入的总数量由下式给出：

2 r X S s=1 Ns · Cs 2 , (5) 

where r denotes the reduction ratio, S refers to the number of stages (a stage refers to the collection of blocks operating on feature maps of a common spatial dimension), Cs denotes the dimension of the output channels and Ns denotes the number of repeated blocks for stage s (when bias terms are used in FC layers, the introduced parameters and computational cost are typically negligible). SE-ResNet-50 introduces ∼2.5 million additional parameters beyond the ∼25 million parameters required by ResNet-50, corresponding to a ∼10% increase. In practice, the majority of these parameters come from the final stage of the network, where the excitation operation is performed across the greatest number of channels. However, we found that this comparatively costly final stage of SE blocks could be removed at only a small cost in performance (<0.1% top-5 error on ImageNet) reducing the relative parameter increase to ∼4%, which may prove useful in cases where parameter usage is a key consideration (see Section 6.4 and 7.2 for further discussion). 

其中r表示缩减率，S表示级的数量(一级是指在公共空间维度的特征图上操作的块的集合)，Cs表示输出通道的维度，Ns表示级S的重复块的数量(当在FC层中使用偏置项时，引入的参数和计算成本通常可以忽略不计)。SE-ResNet-50引入∼250万个额外参数∼ResNet-50需要2500万个参数，对应于∼增加10%。在实践中，这些参数中的大部分来自网络的最后阶段，在该阶段，在最大数量的信道上执行激励操作。然而，我们发现，这个相对昂贵的SE块的最后阶段可以在性能上以很小的成本(在ImageNet上＜0.1%的前5个错误)移除，从而将相对参数的增加减少到∼4%，这在参数使用是关键考虑因素的情况下可能会被证明是有用的(参见第6.4节和第7.2节的进一步讨论)。

TABLE 1 (Left) ResNet-50 [13]. (Middle) SE-ResNet-50. (Right) SE-ResNeXt-50 with a 32×4d template. The shapes and operations with specific parameter settings of a residual building block are listed inside the brackets and the number of stacked blocks in a stage is presented outside. The inner brackets following by fc indicates the output dimension of the two fully connected layers in an SE module.

表1(左)ResNet-50[13]。(中)SE-ResNet-50。(右)SE-RresNeXt-50，带32×4d模板。残差构建块的形状和具有特定参数设置的操作列在括号内，一个阶段中的堆叠块数量列在括号外。fc后面的内括号表示SE模块中两个完全连接的层的输出尺寸。

TABLE 2 Single-crop error rates (%) on the ImageNet validation set and complexity comparisons. The original column refers to the results reported in the original papers (the results of ResNets are obtained from the website: https://github.com/Kaiminghe/deep-residual-networks). To enable a fair comparison, we re-train the baseline models and report the scores in the re-implementation column. The SENet column refers to the corresponding architectures in which SE blocks have been added. The numbers in brackets denote the performance improvement over the re-implemented baselines. † indicates that the model has been evaluated on the non-blacklisted subset of the validation set (this is discussed in more detail in [21]), which may slightly improve results. VGG-16 and SE-VGG-16 are trained with batch normalization. 

表2 ImageNet验证集的单次裁剪错误率(%)和复杂性比较。原始栏目是指原始论文中报告的结果(ResNets的结果可从以下网站获得：https://github.com/Kaiminghe/deep-residual-networks). 为了进行公平的比较，我们重新训练基线模型，并在重新实施栏中报告分数。SENet列指的是添加了SE块的相应架构。括号中的数字表示与重新实施的基线相比的绩效改进。†表明该模型已在验证集的非黑名单子集上进行了评估(在[21]中对此进行了更详细的讨论)，这可能会略微改善结果。VGG-16和SE-VGG-16被训练成批归一化。

## 5 EXPERIMENTS
In this section, we conduct experiments to investigate the effectiveness of SE blocks across a range of tasks, datasets and model architectures.

在本节中，我们进行实验，以研究SE块在一系列任务、数据集和模型架构中的有效性。

### 5.1 Image Classification
To evaluate the influence of SE blocks, we first perform experiments on the ImageNet 2012 dataset [10] which comprises 1.28 million training images and 50K validation images from 1000 different classes. We train networks on the training set and report the top-1 and top-5 error on the validation set.

为了评估SE块的影响，我们首先在ImageNet 2012数据集[10]上进行实验，该数据集包含128万个训练图像和来自1000个不同类别的50K个验证图像。我们在训练集上训练网络，并在验证集上报告前1和前5个错误。

Each baseline network architecture and its corresponding SE counterpart are trained with identical optimisation schemes. We follow standard practices and perform data augmentation with random cropping using scale and aspect ratio [5] to a size of 224 × 224 pixels (or 299 × 299 for Inception-ResNet-v2 [21] and SE-Inception-ResNet-v2) and perform random horizontal flipping. Each input image is normalised through mean RGB-channel subtraction. All models are trained on our distributed learning system ROCS which is designed to handle efficient parallel training of large networks. Optimisation is performed using synchronous SGD with momentum 0.9 and a minibatch size of 1024. The initial learning rate is set to 0.6 and decreased by a factor of 10 every 30 epochs. Models are trained for 100 epochs from scratch, using the weight initialisation strategy described in [66]. The reduction ratio r (in Section 3.2) is set to 16 by default (except where stated otherwise).

每个基线网络架构及其对应的SE对应物都使用相同的优化方案进行训练。我们遵循标准实践，使用缩放和纵横比[5]将数据放大到224×224像素(或对于Inception-ResNet-v2[21]和SE-Inception-ReNet-v2为299×299)，并执行随机水平翻转。通过平均RGB通道减法对每个输入图像进行归一化。所有模型都在我们的分布式学习系统ROCS上进行训练，该系统旨在处理大型网络的高效并行训练。使用动量为0.9、小批量大小为1024的同步SGD进行优化。初始学习速率设置为0.6，每30个周期减少10倍。使用[66]中描述的权重初始化策略，从零开始对模型进行100个时期的训练。减速比r(第3.2节)默认设置为16(除非另有规定)。

When evaluating the models we apply centre-cropping so that 224 × 224 pixels are cropped from each image, after its shorter edge is first resized to 256 (299 × 299 from each image whose shorter edge is first resized to 352 for Inception-ResNet-v2 and SE-Inception-ResNet-v2).

在评估模型时，我们应用中心裁剪，以便在将图像的短边首次调整为256(对于Inception-ResNet-v2和SE-Inteption-ReNet-v3，每个图像的短边缘首次调整为352)后，从每个图像中裁剪224×224像素。

TABLE 3. Single-crop error rates (%) on the ImageNet validation set and complexity comparisons. MobileNet refers to “1.0 MobileNet-224” in [64] and ShuffleNet refers to “ShuffleNet 1 × (g = 3)” in [65]. The numbers in brackets denote the performance improvement over the re-implementation. 

表3.ImageNet验证集的单次裁剪错误率(%)和复杂性比较。在[64]中，MobileNet指的是“1.0 MobileNet-224”，而在[65]中，ShuffleNet指“Shuffle Net 1×(g=3)”。括号中的数字表示重新实施后的性能改进。

Fig. 4. Training baseline architectures and their SENet counterparts on ImageNet. SENets exhibit improved optimisation characteristics and produce consistent gains in performance which are sustained throughout the training process. 
图4.在ImageNet上训练基线架构及其SENet对应物。SENets表现出改进的优化特性，并在整个培训过程中产生持续的性能增益。

Network depth. We begin by comparing SE-ResNet against ResNet architectures with different depths and report the results in Table 2. We observe that SE blocks consistently improve performance across different depths with an extremely small increase in computational complexity. Remarkably, SE-ResNet-50 achieves a single-crop top-5 validation error of 6.62%, exceeding ResNet-50 (7.48%) by 0.86% and approaching the performance achieved by the much deeper ResNet-101 network (6.52% top-5 error) with only half of the total computational burden (3.87 GFLOPs vs. 7.58 GFLOPs). This pattern is repeated at greater depth, where SE-ResNet-101 (6.07% top-5 error) not only matches, but outperforms the deeper ResNet-152 network (6.34% top-5 error) by 0.27%. While it should be noted that the SE blocks themselves add depth, they do so in an extremely computationally efficient manner and yield good returns even at the point at which extending the depth of the base architecture achieves diminishing returns. Moreover, we see that the gains are consistent across a range of different network depths, suggesting that the improvements induced by SE blocks may be complementary to those obtained by simply increasing the depth of the base architecture.

网络深度。我们首先将SE ResNet与具有不同深度的ResNet架构进行比较，并将结果报告在表2中。我们观察到，SE块在不同深度上持续提高性能，但计算复杂度的增加非常小。值得注意的是，SE-ResNet-50实现了6.62%的单作物前5位验证误差，超过了ResNet-50(7.48%)0.86%，并且接近了更深层次的ResNet-101网络(6.52%的前5位误差)所实现的性能，仅占总计算负担的一半(3.87 GFLOP对7.58 GFLOP)。这种模式在更大的深度重复，其中SE-ResNet-101(6.07%的前5个错误)不仅匹配，而且比更深的ResNet-152网络(6.34%的前5错误)高0.27%。尽管应该注意，SE区块本身增加了深度，但它们以非常高效的计算方式实现了这一点，即使在扩展基础架构的深度实现收益递减的时候，也能产生良好的收益。此外，我们发现，在不同网络深度范围内，增益是一致的，这表明SE块带来的改进可能与通过简单地增加基础架构的深度而获得的改进是互补的。

Integration with modern architectures. We next study the effect of integrating SE blocks with two further state-ofthe-art architectures, Inception-ResNet-v2 [21] and ResNeXt (using the setting of 32 × 4d) [19], both of which introduce additional computational building blocks into the base network. We construct SENet equivalents of these networks, SE-Inception-ResNet-v2 and SE-ResNeXt (the configuration of SE-ResNeXt-50 is given in Table 1) and report results in Table 2. As with the previous experiments, we observe significant performance improvements induced by the introduction of SE blocks into both architectures. In particular, SE-ResNeXt-50 has a top-5 error of 5.49% which is superior to both its direct counterpart ResNeXt-50 (5.90% top-5 error) as well as the deeper ResNeXt-101 (5.57% top-5 error), a model which has almost twice the total number of parameters and computational overhead. We note a slight difference in performance between our re-implementation of Inception-ResNet-v2 and the result reported in [21]. However, we observe a similar trend with regard to the effect of SE blocks, finding that SE counterpart (4.79% top-5 error) outperforms our reimplemented Inception-ResNet-v2 baseline (5.21% top-5 error) by 0.42% as well as the reported result in [21].

与现代架构的集成。接下来，我们将研究将SE块与另外两种最先进的架构Inception-ResNet-v2[21]和ResNeXt(使用32×。我们构建了这些网络的SENet等价物，SE-Inception-ResNet-v2和SE-ResNeXt(SE-ResNeXt-50的配置如表1所示)，并在表2中报告了结果。与之前的实验一样，我们观察到在两种架构中引入SE块导致的显著性能改进。特别是，SE-ResNeXt-50的前5位误差为5.49%，优于其直接对应的ResNeXt-5(5.90%前5位错误)和更深层次的ResNeXt-101(5.57%前5位)，该模型的参数和计算开销几乎是总参数数的两倍。我们注意到Inception-ResNet-v2的重新实现与[21]中报告的结果之间的性能略有不同。然而，我们观察到关于SE块效应的类似趋势，发现SE对应物(4.79%top 5错误)比我们重新实施的Inception-ResNet-v2基线(5.21%top 5误差)高0.42%，以及[21]中报告的结果。

We also assess the effect of SE blocks when operating on non-residual networks by conducting experiments with the VGG-16 [11] and BN-Inception architecture [6]. To facilitate the training of VGG-16 from scratch, we add Batch Normalization layers after each convolution. We use identical training schemes for both VGG-16 and SE-VGG-16. The results of the comparison are shown in Table 2. Similarly to the results reported for the residual baseline architectures, we observe that SE blocks bring improvements in performance on the non-residual settings.

我们还通过使用VGG-16[11]和BN初始架构[6]进行实验，评估在非残差网络上运行时SE块的影响。为了便于从头开始训练VGG-16，我们在每次卷积之后添加了批处理规范化层。我们对VGG-16和SE-VGG-16使用相同的训练方案。比较结果如表2所示。与残余基线架构报告的结果类似，我们观察到SE块在非残余设置上带来了性能改进。

To provide some insight into influence of SE blocks on the optimisation of these models, example training curves for runs of the baseline architectures and their respective SE counterparts are depicted in Fig. 4. We observe that SE blocks yield a steady improvement throughout the optimisation procedure. Moreover, this trend is fairly consistent across a range of network architectures considered as baselines.

为了深入了解SE块对这些模型优化的影响，图4中描绘了基线架构及其各自SE对应物运行的样本训练曲线。我们观察到SE块在整个优化过程中产生了稳定的改进。此外，这一趋势在一系列被视为基线的网络架构中相当一致。

Mobile setting. Finally, we consider two representative architectures from the class of mobile-optimised networks, MobileNet [64] and ShuffleNet [65]. For these experiments, we used a minibatch size of 256 and slightly less aggressive data augmentation and regularisation as in [65]. We trained the models across 8 GPUs using SGD with momentum (set to 0.9) and an initial learning rate of 0.1 which was reduced by a factor of 10 each time the validation loss plateaued. The total training process required ∼ 400 epochs (enabling us to reproduce the baseline performance of [65]). The results reported in Table 3 show that SE blocks consistently improve the accuracy by a large margin at a minimal increase in computational cost.

移动设置。最后，我们考虑移动优化网络类别中的两个代表性架构：MobileNet[64]和ShuffleNet[65]。对于这些实验，我们使用了256个小批量的数据，并像[65]中那样稍微减少了数据的扩充和规范化。我们使用具有动量(设置为0.9)和初始学习率0.1的SGD在8个GPU上训练模型，每次验证损失稳定时，初始学习率降低10倍。所需的总培训流程∼ 400个时期(使我们能够重现[65]的基线表现)。表3中报告的结果表明，SE块在最小的计算成本增加的情况下，以较大幅度持续提高精度。

TABLE 4. Classification error (%) on CIFAR-10. 

TABLE 5. Classification error (%) on CIFAR-100. 

Additional datasets. We next investigate whether the benefits of SE blocks generalise to datasets beyond ImageNet. We perform experiments with several popular baseline architectures and techniques (ResNet-110 [14], ResNet-164 [14], WideResNet-16-8 [67], Shake-Shake [68] and Cutout [69]) on the CIFAR-10 and CIFAR-100 datasets [70]. These comprise a collection of 50k training and 10k test 32 × 32 pixel RGB images, labelled with 10 and 100 classes respectively. The integration of SE blocks into these networks follows the same approach that was described in Section 3.3. Each baseline and its SENet counterpart are trained with standard data augmentation strategies [24], [71]. During training, images are randomly horizontally flipped and zero-padded on each side with four pixels before taking a random 32 × 32 crop. Mean and standard deviation normalisation is also applied. The setting of the training hyperparameters (e.g. minibatch size, initial learning rate, weight decay) match those suggested by the original papers. We report the performance of each baseline and its SENet counterpart on CIFAR-10 in Table 4 and performance on CIFAR-100 in Table 5. We observe that in every comparison SENets outperform the baseline architectures, suggesting that the benefits of SE blocks are not confined to the ImageNet dataset.

其他数据集。接下来，我们将研究SE块的好处是否适用于ImageNet以外的数据集。我们在CIFAR-10和CIFAR-100数据集[70]上使用几种流行的基线架构和技术(ResNet-110[14]、ResNet-164]、WideResNet-16-8[67]、Shake Shake[68]和Cutout[69])进行实验。这些包括一组50k训练和10k测试的32×32像素RGB图像，分别标记为10和100类。将SE块集成到这些网络中遵循第3.3节中所述的相同方法。每个基线及其SENet对应物均采用标准数据增广策略进行训练[24]，[71]。在训练过程中，图像被随机水平翻转，并在每一侧用四个像素填充零，然后进行随机的32×32裁剪。平均值和标准偏差归一化也适用。训练超参数的设置(例如，小批量大小、初始学习速率、权重衰减)与原始论文所建议的相匹配。我们在表4中报告了每个基线及其SENet对应物在CIFAR-10上的性能，在表5中报告了CIFAR-100上的性能。我们观察到，在每次比较中，SENet都优于基线架构，这表明SE块的好处并不局限于ImageNet数据集。

### 5.2 Scene Classification
We also conduct experiments on the Places365-Challenge dataset [73] for scene classification. This dataset comprises 8 million training images and 36, 500 validation images across 365 categories. Relative to classification, the task of scene understanding offers an alternative assessment of a model’s ability to generalise well and handle abstraction. This is because it often requires the model to handle more complex data associations and to be robust to a greater level of appearance variation.

我们还对Places365挑战数据集[73]进行了场景分类实验。该数据集包括365个类别的800万张训练图像和3650张验证图像。相对于分类，场景理解任务提供了一种对模型的概括能力和处理抽象能力的替代评估。这是因为它通常要求模型处理更复杂的数据关联，并对更大程度的外观变化具有稳健性。

We opted to use ResNet-152 as a strong baseline to assess the effectiveness of SE blocks and follow the training and evaluation protocols described in [72], [74]. In these experiments, models are trained from scratch. We report the results in Table 6, comparing also with prior work. We observe that SE-ResNet-152 (11.01% top-5 error) achieves a lower validation error than ResNet-152 (11.61% top-5 error), providing evidence that SE blocks can also yield improvements for scene classification. This SENet surpasses the previous state-of-the-art model Places-365-CNN [72] which has a top-5 error of 11.48% on this task.

我们选择使用ResNet-152作为强有力的基线来评估SE块的有效性，并遵循[72]、[74]中描述的训练和评估协议。在这些实验中，模型是从头开始训练的。我们在表6中报告了结果，并与先前的工作进行了比较。我们观察到SE-ResNet-152(11.01%的前5个错误)比ResNet-152实现了更低的验证错误(11.61%的前五个错误)，这证明SE块也可以改善场景分类。该SENet超越了先前最先进的Places-365-CN[72]模型，该模型在该任务中的前5个错误率为11.48%。

TABLE 6. Single-crop error rates (%) on Places365 validation set. 

TABLE 7. Faster R-CNN object detection results (%) on COCO minival set.

### 5.3 Object Detection on COCO
We further assess the generalisation of SE blocks on the task of object detection using the COCO dataset [75]. As in previous work [19], we use the minival protocol, i.e., training the models on the union of the 80k training set and a 35k val subset and evaluating on the remaining 5k val subset. Weights are initialised by the parameters of the model trained on the ImageNet dataset. We use the Faster R-CNN [4] detection framework as the basis for evaluating our models and follow the hyperparameter setting described in [76] (i.e., end-to-end training with the ’2x’ learning schedule). Our goal is to evaluate the effect of replacing the trunk architecture (ResNet) in the object detector with SE-ResNet, so that any changes in performance can be attributed to better representations. Table 7 reports the validation set performance of the object detector using ResNet-50, ResNet-101 and their SE counterparts as trunk architectures. SE-ResNet-50 outperforms ResNet-50 by 2.4% (a relative 6.3% improvement) on COCO’s standard AP metric and by 3.1% on AP@IoU=0.5. SE blocks also benefit the deeper ResNet-101 architecture achieving a 2.0% improvement (5.0% relative improvement) on the AP metric. In summary, this set of experiments demonstrate the generalisability of SE blocks. The induced improvements can be realised across a broad range of architectures, tasks and datasets.

我们使用COCO数据集进一步评估SE块在目标检测任务中的通用性[75]。与之前的工作[19]一样，我们使用minival协议，即在80k训练集和35k val子集的并集上训练模型，并对剩余的5k val子集进行评估。权重由在ImageNet数据集上训练的模型参数初始化。我们使用更快的R-CNN[4]检测框架作为评估模型的基础，并遵循[76]中所述的超参数设置(即，使用“2x”学习计划进行端到端培训)。我们的目标是评估用SE ResNet替换对象检测器中的主干架构(ResNet)的效果，以便将性能的任何变化归因于更好的表示。表7报告了使用ResNet-50、ResNet-101及其SE对应物作为主干架构的对象检测器的验证集性能。在COCO的标准AP指标上，SE-ResNet-50优于ResNet-50 2.4%(相对提高6.3%)AP@IoUSE块也有利于更深层次的ResNet-101架构，在AP度量上实现2.0%的改进(5.0%的相对改进)。总之，这组实验证明了SE块的通用性。诱导的改进可以在广泛的架构、任务和数据集中实现。

### 5.4 ILSVRC 2017 Classification Competition
SENets formed the foundation of our submission to the ILSVRC competition where we achieved first place. Our winning entry comprised a small ensemble of SENets that employed a standard multi-scale and multi-crop fusion strategy to obtain a top-5 error of 2.251% on the test set. As part of this submission, we constructed an additional model, SENet-154, by integrating SE blocks with a modified ResNeXt [19] (the details of the architecture are provided in Appendix). We compare this model with prior work on the ImageNet validation set in Table 8 using standard crop sizes (224 × 224 and 320 × 320). We observe that SENet-154 achieves a top-1 error of 18.68% and a top-5 error of 4.47% using a 224 × 224 centre crop evaluation, which represents the strongest reported result.

SENets是我们参加ILSVRC比赛的基础，我们在比赛中获得了第一名。我们的获奖作品包括一个小型SENets集合，该集合采用了标准的多尺度和多作物融合策略，在测试集上获得了2.251%的前5个误差。作为本次提交的一部分，我们通过将SE块与修改后的ResNeXt[19]集成，构建了一个额外的模型SENet-154(架构的详情见附录)。我们使用标准作物大小(224×224和320×320)将该模型与表8中ImageNet验证集的先前工作进行了比较。我们观察到，使用224×224中心作物评估，SENet-154实现了18.68%的前1个错误和4.47%的前5个错误，这代表了最强的报告结果。

TABLE 8. Single-crop error rates (%) of state-of-the-art CNNs on ImageNet validation set with crop sizes 224 × 224 and 320 × 320 / 299 × 299. 224 × 224 320 × 320 / 299 × 299 .
表8.ImageNet验证集上最先进的CNN单次裁剪错误率(%)，裁剪尺寸为224×224和320×320/299×299。

TABLE 9. Comparison (%) with state-of-the-art CNNs on ImageNet validation set using larger crop sizes/additional training data. †This model was trained with a crop size of 320 × 320. 
表9：使用更大作物尺寸/额外培训数据，在ImageNet验证集上与最先进的CNN进行比较(%)。†该模型以320×。

Following the challenge there has been a great deal of further progress on the ImageNet benchmark. For comparison, we include the strongest results that we are currently aware of in Table 9. The best performance using only ImageNet data was recently reported by [79]. This method uses reinforcement learning to develop new policies for data augmentation during training to improve the performance of the architecture searched by [31]. The best overall performance was reported by [80] using a ResNeXt-101 32×48d architecture. This was achieved by pretraining their model on approximately one billion weakly labelled images and finetuning on ImageNet. The improvements yielded by more sophisticated data augmentation [79] and extensive pretraining [80] may be complementary to our proposed changes to the network architecture. 

在挑战之后，ImageNet基准测试有了很大的进步。为了进行比较，我们在表9中列出了目前已知的最强结果。[79]最近报告了仅使用ImageNet数据的最佳性能。该方法使用强化学习来开发训练期间数据增广的新策略，以提高[31]搜索的架构的性能。[80]使用ResNeXt-101 32×48d架构报告了最佳的总体性能。这是通过在大约10亿张弱标记图像上预训练他们的模型并在ImageNet上进行微调来实现的。更复杂的数据增广[79]和广泛的预训练[80]所带来的改进可能是对我们提出的网络架构更改的补充。

## 6 ABLATION STUDY
In this section we conduct ablation experiments to gain a better understanding of the effect of using different configurations on components of the SE blocks. All ablation experiments are performed on the ImageNet dataset on a single machine (with 8 GPUs). ResNet-50 is used as the backbone architecture. We found empirically that on ResNet architectures, removing the biases of the FC layers in the excitation operation facilitates the modelling of channel dependencies, and use this configuration in the following experiments. The data augmentation strategy follows the approach described in Section 5.1. To allow us to study the upper limit of performance for each variant, the learning rate is initialised to 0.1 and training continues until the validation loss plateaus2 (∼300 epochs in total). The learning rate is then reduced by a factor of 10 and then this process is repeated (three times in total). Label-smoothing regularisation [20] is used during training.

在本节中，我们进行消融实验，以更好地了解使用不同配置对SE块组件的影响。所有消融实验都在一台机器(具有8个GPU)上的ImageNet数据集上进行。ResNet-50用作骨干架构。我们根据经验发现，在ResNet架构上，消除激励操作中FC层的偏置有助于信道依赖性的建模，并在以下实验中使用此配置。数据增广策略遵循第5.1节中所述的方法。为了使我们能够研究每个变量的性能上限，将学习率初始化为0.1，并继续培训，直到验证损失达到稳定状态2(∼总共300个时代)。然后将学习率降低10倍，然后重复此过程(总共三次)。训练期间使用标签平滑规则[20]。

TABLE 10. Single-crop error rates (%) on ImageNet and parameter sizes for SE-ResNet-50 at different reduction ratios. Here, original refers to ResNet-50.
表10.ImageNet上的单次裁剪错误率(%)和不同缩减率下SE-ResNet-50的参数大小。这里，原始指的是ResNet-50。

### 6.1 Reduction ratio
The reduction ratio r introduced in Eqn. 5 is a hyperparameter which allows us to vary the capacity and computational cost of the SE blocks in the network. To investigate the trade-off between performance and computational cost mediated by this hyperparameter, we conduct experiments with SE-ResNet-50 for a range of different r values. The comparison in Table 10 shows that performance is robust to a range of reduction ratios. Increased complexity does not improve performance monotonically while a smaller ratio dramatically increases the parameter size of the model. Setting r = 16 achieves a good balance between accuracy and complexity. In practice, using an identical ratio throughout a network may not be optimal (due to the distinct roles performed by different layers), so further improvements may be achievable by tuning the ratios to meet the needs of a given base architecture.

等式中引入的减速比r。5是一个超参数，它允许我们改变网络中SE块的容量和计算成本。为了研究该超参数介导的性能和计算成本之间的权衡，我们使用SE-ResNet-50对不同r值范围进行了实验。表10中的比较表明，性能对一系列减速比是稳健的。增加的复杂度不会单调地提高性能，而较小的比率会显著增加模型的参数大小。设置r＝16可以在准确性和复杂性之间实现良好的平衡。实际上，在整个网络中使用相同的比率可能不是最佳的(由于不同层执行不同的角色)，因此可以通过调整比率以满足给定基础架构的需要来实现进一步的改进。

### 6.2 Squeeze Operator
We examine the significance of using global average pooling as opposed to global max pooling as our choice of squeeze operator (since this worked well, we did not consider more sophisticated alternatives). The results are reported in Table 11. While both max and average pooling are effective, average pooling achieves slightly better performance, justifying its selection as the basis of the squeeze operation. However, we note that the performance of SE blocks is fairly robust to the choice of specific aggregation operator.

我们研究了使用全局平均池而不是全局最大池作为我们选择的挤压运算符的重要性(因为这很好，所以我们没有考虑更复杂的替代方案)。结果如表11所示。虽然最大池和平均池都是有效的，但平均池的性能稍好一些，这证明其选择是挤压操作的基础。然而，我们注意到，SE块的性能对于特定聚合运算符的选择是相当稳健的。

### 6.3 Excitation Operator
We next assess the choice of non-linearity for the excitation mechanism. We consider two further options: ReLU and tanh, and experiment with replacing the sigmoid with these alternative non-linearities. The results are reported in Table 12. We see that exchanging the sigmoid for tanh slightly worsens performance, while using ReLU is dramatically worse and in fact causes the performance of SE-ResNet-50 to drop below that of the ResNet-50 baseline. This suggests that for the SE block to be effective, careful construction of the excitation operator is important.

接下来，我们评估激励机制的非线性选择。我们考虑了另外两个选项：ReLU和tanh，并尝试用这些可选的非线性来代替sigmoid。结果如表12所示。我们发现，将西格玛曲线换成tanh曲线会略微降低性能，而使用ReLU会显著降低性能，事实上会导致SE-ResNet-50的性能低于ResNet-50基线。这表明，为了使SE块有效，仔细构造励磁操作器非常重要。

TABLE 11. Effect of using different squeeze operators in SE-ResNet-50 on ImageNet (error rates %).
表11 SE-ResNet-50中使用不同挤压运算符对ImageNet的影响(错误率%)。

TABLE 12. Effect of using different non-linearities for the excitation operator in SE-ResNet-50 on ImageNet (error rates %).
表12 SE-ResNet-50中激励算子使用不同非线性对ImageNet的影响(错误率%)。

### 6.4 Different stages
We explore the influence of SE blocks at different stages by integrating SE blocks into ResNet-50, one stage at a time.Specifically, we add SE blocks to the intermediate stages: stage 2, stage 3 and stage 4, and report the results in Table 13. We observe that SE blocks bring performance benefits when introduced at each of these stages of the architecture. Moreover, the gains induced by SE blocks at different stages are complementary, in the sense that they can be combined effectively to further bolster network performance.

我们通过将SE块集成到ResNet-50中，一次一个阶段，探索SE块在不同阶段的影响。具体来说，我们将SE块添加到中间阶段：阶段2、阶段3和阶段4，并在表13中报告结果。此外，SE块在不同阶段产生的增益是互补的，因为它们可以有效地组合在一起，以进一步提高网络性能。

### 6.5 Integration strategy
Finally, we perform an ablation study to assess the influence of the location of the SE block when integrating it into existing architectures. In addition to the proposed SE design, we consider three variants: (1) SE-PRE block, in which the SE block is moved before the residual unit; (2) SE-POST block, in which the SE unit is moved after the summation with the identity branch (after ReLU) and (3) SE-Identity block, in which the SE unit is placed on the identity connection in parallel to the residual unit. These variants are illustrated in Figure 5 and the performance of each variant is reported in Table 14. We observe that the SE-PRE, SE-Identity and proposed SE block each perform similarly well, while usage of the SE-POST block leads to a drop in performance. This experiment suggests that the performance improvements produced by SE units are fairly robust to their location, provided that they are applied prior to branch aggregation.

最后，我们进行了消融研究，以评估将SE块整合到现有架构中时SE块位置的影响。除了建议的SE设计，我们考虑三种变体：(1)SE-PRE块，其中SE块在剩余单元之前移动; (2) SE-POST块，其中SE单元在与同一分支求和后(ReLU之后)移动，以及(3)SE同一块，其中，SE单元与剩余单元并联放置在同一连接上。这些变体如图5所示，每个变体的性能如表14所示。我们观察到SE-PRE、SE Identity和提议的SE块的性能都很好，而SE-POST块的使用会导致性能下降。这个实验表明，SE单元产生的性能改进对于其位置来说是相当稳健的，前提是它们在分支聚合之前应用。

TABLE 13. Effect of integrating SE blocks with ResNet-50 at different stages on ImageNet (error rates %).
表13.在不同阶段将SE块与ResNet-50集成对ImageNet的影响(错误率%)。

TABLE 14. Effect of different SE block integration strategies with ResNet-50 on ImageNet (error rates %).
表14.使用ResNet-50的不同SE块集成策略对ImageNet的影响(错误率%)。

TABLE 15. Effect of integrating SE blocks at the 3x3 convolutional layer of each residual branch in ResNet-50 on ImageNet (error rates %).
表15.ResNet-50中每个残余分支的3x3卷积层处的SE块积分对ImageNet的影响(错误率%)。

In the experiments above, each SE block was placed outside the structure of a residual unit. We also construct a variant of the design which moves the SE block inside the residual unit, placing it directly after the 3 × 3 convolutional layer. Since the 3 × 3 convolutional layer possesses fewer channels, the number of parameters introduced by the corresponding SE block is also reduced. The comparison in Table 15 shows that the SE 3×3 variant achieves comparable classification accuracy with fewer parameters than the standard SE block. Although it is beyond the scope of this work, we anticipate that further efficiency gains will be achievable by tailoring SE block usage for specific architectures. 

在上述实验中，每个SE块被放置在残差单元的结构之外。我们还构造了一个设计变体，将SE块移动到残差单元内，将其直接放置在3×3卷积层之后。由于3×3卷积层具有较少的信道，相应SE块引入的参数数量也减少了。表15中的比较表明，SE 3×。尽管这超出了本工作的范围，但我们预计，通过为特定架构定制SE块的使用，可以进一步提高效率。

## 7 ROLE OF SE BLOCKS
Although the proposed SE block has been shown to improve network performance on multiple visual tasks, we would also like to understand the relative importance of the squeeze operation and how the excitation mechanism operates in practice. A rigorous theoretical analysis of the representations learned by deep neural networks remains challenging, we therefore take an empirical approach to examining the role played by the SE block with the goal of attaining at least a primitive understanding of its practical function.

尽管所提出的SE块已被证明可以提高网络在多个视觉任务中的性能，但我们还希望了解挤压操作的相对重要性以及激励机制在实践中的运行方式。对深度神经网络学习到的表示进行严格的理论分析仍然具有挑战性，因此，我们采用经验方法来检查SE块所起的作用，目的是至少对其实际功能有一个初步的了解。

### 7.1 Effect of Squeeze
To assess whether the global embedding produced by the squeeze operation plays an important role in performance, we experiment with a variant of the SE block that adds an equal number of parameters, but does not perform global average pooling. Specifically, we remove the pooling operation and replace the two FC layers with corresponding 1 × 1 convolutions with identical channel dimensions in the excitation operator, namely NoSqueeze, where the excitation output maintains the spatial dimensions as input. In contrast to the SE block, these point-wise convolutions can only remap the channels as a function of the output of a local operator. While in practice, the later layers of a deep network will typically possess a (theoretical) global receptive field, global embeddings are no longer directly accessible throughout the network in the NoSqueeze variant. 

为了评估挤压操作产生的全局嵌入是否在性能中起着重要作用，我们使用SE块的一个变体进行实验，该变体添加了相同数量的参数，但不执行全局平均池。具体地说，我们删除了池操作，并在激励算子(即NoSqueze)中将两个FC层替换为具有相同信道尺寸的对应1×1卷积，其中激励输出保持空间尺寸作为输入。与SE块相反，这些逐点卷积只能根据本地运算符的输出重新映射通道。虽然在实践中，深层网络的后一层通常具有(理论上)全局接受域，但在NoSqueze变体中，全局嵌入不再能在整个网络中直接访问。

Fig. 5. SE block integration designs explored in the ablation study. 
图5.消融研究中探索的SE阻滞整合设计。

Fig. 6. Activations induced by the Excitation operator at different depths in the SE-ResNet-50 on ImageNet. Each set of activations is named according to the following scheme: SE_stageID_blockID. With the exception of the unusual behaviour at SE_5_2, the activations become increasingly class-specific with increasing depth.
图6.ImageNet上SE-ResNet-50中不同深度的激发操作员引发的激活。每组激活根据以下方案命名：SE_stageID_blockID。除了SE_5_2处的异常行为外，随着深度的增加，激活变得越来越具有类别特异性。

TABLE 16. Effect of Squeeze operator on ImageNet (error rates %). 
表16.挤压运算符对ImageNet的影响(错误率%)。

The accuracy and computational complexity of both models are compared to a standard ResNet-50 model in Table 16. We observe that the use of global information has a significant influence on the model performance, underlining the importance of the squeeze operation. Moreover, in comparison to the NoSqueeze design, the SE block allows this global information to be used in a computationally parsimonious manner.

两个模型的精度和计算复杂性与表16中的标准ResNet-50模型进行了比较。我们观察到，全局信息的使用对模型性能有显著影响，突出了挤压操作的重要性。此外，与NoSqueze设计相比，SE块允许以计算简洁的方式使用此全局信息。

### 7.2 Role of Excitation
To provide a clearer picture of the function of the excitation operator in SE blocks, in this section we study example activations from the SE-ResNet-50 model and examine their distribution with respect to different classes and different input images at various depths in the network. In particular, we would like to understand how excitations vary across images of different classes, and across images within a class.

为了更清楚地了解SE块中激发算子的功能，在本节中，我们研究了SE-ResNet-50模型中的样本激活，并检查了它们在网络中不同深度的不同类别和不同输入图像中的分布。特别是，我们想了解不同类别的图像以及同一类别内的图像之间的兴奋是如何变化的。

We first consider the distribution of excitations for different classes. Specifically, we sample four classes from the ImageNet dataset that exhibit semantic and appearance diversity, namely goldfish, pug, plane and cliff (example images from these classes are shown in Appendix). We then draw fifty samples for each class from the validation set and compute the average activations for fifty uniformly sampled channels in the last SE block of each stage (immediately prior to downsampling) and plot their distribution in Fig. 6. For reference, we also plot the distribution of the mean activations across all of the 1000 classes.

我们首先考虑不同类别的激励分布。具体来说，我们从ImageNet数据集中抽取了四个表现出语义和外观多样性的类，即金鱼、哈巴狗、飞机和悬崖(这些类的样本图像见附录)。然后，我们从验证集中为每个类别抽取50个样本，计算每个阶段最后一个SE块(下采样之前)中50个均匀采样通道的平均激活，并在图6中绘制其分布。为了参考，我们还绘制了1000个类别的平均激活分布。

We make the following three observations about the role of the excitation operation. First, the distribution across different classes is very similar at the earlier layers of the network, e.g. SE 2 3. This suggests that the importance of feature channels is likely to be shared by different classes in the early stages. The second observation is that at greater depth, the value of each channel becomes much more class-specific as different classes exhibit different preferences to the discriminative value of features, e.g. SE 4 6 and SE 5 1. These observations are consistent with findings in previous work [81], [82], namely that earlier layer features are typically more general (e.g. class agnostic in the context of the classification task) while later layer features exhibit greater levels of specificity [83]. 

我们对励磁操作的作用进行了以下三点观察。首先，在网络的早期层(例如SE 2 3)，不同类别之间的分布非常相似。这表明，在早期阶段，不同类别可能共享特征频道的重要性。第二个观察结果是，在更大的深度上，由于不同的类别对特征的区别值表现出不同的偏好，例如SE 4 6和SE 5 1，因此每个通道的值变得更具类别特异性，即，较早的层特征通常更一般(例如，在分类任务的上下文中与类无关)。

Fig. 7. Activations induced by Excitation in the different modules of SE-ResNet-50 on image samples from the goldfish and plane classes of ImageNet. The module is named “SE_stageID_blockID”. 
图7 SE-ResNet-50的不同模块对来自金鱼和ImageNet平面类的图像样本的激发诱导的激活。该模块名为“SE_stageID_blockID”。

Next, we observe a somewhat different phenomena in the last stage of the network. SE 5 2 exhibits an interesting tendency towards a saturated state in which most of the activations are close to one. At the point at which all activations take the value one, an SE block reduces to the identity operator. At the end of the network in the SE 5 3 (which is immediately followed by global pooling prior before classifiers), a similar pattern emerges over different classes, up to a modest change in scale (which could be tuned by the classifiers). This suggests that SE 5 2 and SE 5 3 are less important than previous blocks in providing recalibration to the network. This finding is consistent with the result of the empirical investigation in Section 4 which demonstrated that the additional parameter count could be significantly reduced by removing the SE blocks for the last stage with only a marginal loss of performance.

接下来，我们在网络的最后一个阶段观察到一些不同的现象。SE 5 2表现出一种有趣的趋向，即大多数活化接近于饱和状态。在所有激活都取值1的点上，SE块简化为标识运算符。在SE 5 3网络的末尾(紧接着是分类器之前的全局池)，在不同的类别上出现了类似的模式，直到规模的适度变化(可以由分类器调整)。这表明SE 5 2和SE 5 3在向网络提供重新校准方面不如先前的块重要。这一发现与第4节中的经验调查结果一致，该结果表明，通过移除最后阶段的SE块，可以显著减少附加参数计数，而性能损失仅为轻微。

Finally, we show the mean and standard deviations of the activations for image instances within the same class for two sample classes (goldfish and plane) in Fig. 7. We observe a trend consistent with the inter-class visualisation, indicating that the dynamic behaviour of SE blocks varies over both classes and instances within a class. Particularly in the later layers of the network where there is considerable diversity of representation within a single class, the network learns to take advantage of feature recalibration to improve its discriminative performance [84]. In summary, SE blocks produce instance-specific responses which nevertheless function to support the increasingly class-specific needs of the model at different layers in the architecture. 

最后，我们在图7中显示了两个样本类(金鱼和平面)的同一类内的图像实例的激活的平均值和标准偏差。我们观察到与类间可视化一致的趋势，表明SE块的动态行为在类和类内的实例上都不同。特别是在网络的后一层中，在单个类中有相当多的表示形式，网络学习利用特征重新校准来提高其辨别性能[84]。总之，SE块会产生特定于实例的响应，尽管如此，这些响应仍然能够支持架构中不同层的模型日益增长的特定于类的需求。

## 8 CONCLUSION
In this paper we proposed the SE block, an architectural unit designed to improve the representational power of a network by enabling it to perform dynamic channel-wise feature recalibration. A wide range of experiments show the effectiveness of SENets, which achieve state-of-the-art performance across multiple datasets and tasks. In addition, SE blocks shed some light on the inability of previous architectures to adequately model channel-wise feature dependencies. We hope this insight may prove useful for other tasks requiring strong discriminative features. Finally, the feature importance values produced by SE blocks may be of use for other tasks such as network pruning for model compression.

在本文中，我们提出了SE块，这是一种架构单元，旨在通过使网络能够执行动态信道特征重新校准来提高网络的表示能力。广泛的实验表明了SENets的有效性，它在多个数据集和任务中实现了最先进的性能。此外，SE块揭示了以前的架构无法对通道特性依赖性进行充分建模。我们希望这一洞察力可能对其他需要强烈辨别特征的任务有用。最后，SE块产生的特征重要性值可用于其他任务，例如用于模型压缩的网络修剪。

## ACKNOWLEDGMENTS
The authors would like to thank Chao Li and Guangyuan Wang from Momenta for their contributions in the training system optimisation and experiments on CIFAR dataset. We would also like to thank Andrew Zisserman, Aravindh Mahendran and Andrea Vedaldi for many helpful discussions. The work is supported in part by NSFC Grants (61632003, 61620106003, 61672502, 61571439), National Key R&D Program of China (2017YFB1002701), and Macao FDCT Grant (068/2015/A2). Samuel Albanie is supported by EPSRC AIMS CDT EP/L015897/1.

作者要感谢Momenta的Chao Li和Guangyuan Wang在CIFAR数据集的培训系统优化和实验中所做的贡献。我们还要感谢Andrew Zisserman、Aravindh Mahendran和Andrea Vedaldi进行了许多有益的讨论。这项工作得到了国家自然科学基金资助(61632003、61620106003、61672502、61571439)、国家重点研发计划(2017YFB1002701)和澳门FDCT资助(068/2015/A2)的部分支持。Samuel Albanie由EPSRC AIMS CDT EP/L015897/1支持。

## APPENDIX: DETAILS OF SENET-154
SENet-154 is constructed by incorporating SE blocks into a modified version of the 64×4d ResNeXt-152 which extends the original ResNeXt-101 [19] by adopting the block stacking strategy of ResNet-152 [13]. Further differences to the design and training of this model (beyond the use of SE blocks) are as follows: (a) The number of the first 1 × 1 convolutional channels for each bottleneck building block was halved to reduce the computational cost of the model with a minimal decrease in performance. (b) The first 7 × 7 convolutional layer was replaced with three consecutive 3 × 3 convolutional layers. (c) The 1 × 1 down-sampling projection with stride-2 convolution was replaced with a 3 × 3 stride-2 convolution to preserve information. (d) A dropout layer (with a dropout ratio of 0.2) was inserted before the classification layer to reduce overfitting. (e) Labelsmoothing regularisation (as introduced in [20]) was used during training. (f) The parameters of all BN layers were frozen for the last few training epochs to ensure consistency between training and testing. (g) Training was performed with 8 servers (64 GPUs) in parallel to enable large batch sizes (2048). The initial learning rate was set to 1.0.

SENet-154是通过将SE块合并到64×。该模型的设计和训练的进一步差异(除了使用SE块)如下：(a)每个瓶颈构建块的前1×。(b) 第一个7×7卷积层被三个连续的3×。(c) 1×1下采样投影与跨步-2卷积被替换为3×3跨步-2卷积以保存信息。(d) 在分类层之前插入脱落层(脱落率为0.2)以减少过度拟合。(e) 训练期间使用了标记平滑规则化(如[20]中所述)。(f) 所有BN层的参数在最后几个训练时期被冻结，以确保训练和测试之间的一致性。(g) 使用8台服务器(64 GPU)并行执行训练，以实现大批量(2048)。初始学习率设置为1.0。

Fig. 8. Sample images from the four classes of ImageNet used in the experiments described in Sec. 7.2. 
图8.第7.2节所述实验中使用的四类ImageNet的样本图像。

## REFERENCES
1. A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in Conference on Neural Information Processing Systems, 2012.
2. A. Toshev and C. Szegedy, “DeepPose: Human pose estimation
via deep neural networks,” in CVPR, 2014.
3. J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in CVPR, 2015.
4. S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards
real-time object detection with region proposal networks,” in
Conference on Neural Information Processing Systems, 2015.
5. C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov,
D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with
convolutions,” in CVPR, 2015.
6. S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep
network training by reducing internal covariate shift,” in ICML,2015.
7. S. Bell, C. L. Zitnick, K. Bala, and R. Girshick, “Inside-outside net:
Detecting objects in context with skip pooling and recurrent neural
networks,” in CVPR, 2016.
8. A. Newell, K. Yang, and J. Deng, “Stacked hourglass networks for
human pose estimation,” in ECCV, 2016.
9. M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu,
“Spatial transformer networks,” in Conference on Neural Information
Processing Systems, 2015.
10. O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma,
Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and
L. Fei-Fei, “ImageNet large scale visual recognition challenge,”
International Journal of Computer Vision, 2015.
11. K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in ICLR, 2015.
12. S. Santurkar, D. Tsipras, A. Ilyas, and A. Madry, “How does
batch normalization help optimization? (no, it is not about internal
covariate shift),” in Conference on Neural Information Processing
Systems, 2018.
13. K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
image recognition,” in CVPR, 2016.
14. K. He, X. Zhang, S. Ren, and J. Sun, “Identity mappings in deep
residual networks,” in ECCV, 2016.
15. R. K. Srivastava, K. Greff, and J. Schmidhuber, “Training very deep
networks,” in Conference on Neural Information Processing Systems,2015.
16. Y. Chen, J. Li, H. Xiao, X. Jin, S. Yan, and J. Feng, “Dual path
networks,” in Conference on Neural Information Processing Systems,2017.
17. G. Huang, Z. Liu, K. Q. Weinberger, and L. Maaten, “Densely
connected convolutional networks,” in CVPR, 2017.
18. Y. Ioannou, D. Robertson, R. Cipolla, and A. Criminisi, “Deep
roots: Improving CNN efficiency with hierarchical filter groups,”
in CVPR, 2017.
19. S. Xie, R. Girshick, P. Dollar, Z. Tu, and K. He, “Aggregated ´
residual transformations for deep neural networks,” in CVPR,2017.
20. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in CVPR 2016.
21. C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi, “Inceptionv4, inception-resnet and the impact of residual connections on
learning,” in AAAI Conference on Artificial Intelligence, 2016.
22. M. Jaderberg, A. Vedaldi, and A. Zisserman, “Speeding up convolutional neural networks with low rank expansions,” in BMVC,2014.
23. F. Chollet, “Xception: Deep learning with depthwise separable
convolutions,” in CVPR, 2017.
24. M. Lin, Q. Chen, and S. Yan, “Network in network,” in ICLR, 2014.
25. G. F. Miller, P. M. Todd, and S. U. Hegde, “Designing neural
networks using genetic algorithms.” in ICGA, 1989.
26. K. O. Stanley and R. Miikkulainen, “Evolving neural networks
through augmenting topologies,” Evolutionary computation, 2002.
27. J. Bayer, D. Wierstra, J. Togelius, and J. Schmidhuber, “Evolving
memory cell structures for sequence learning,” in ICANN, 2009.
28. R. Jozefowicz, W. Zaremba, and I. Sutskever, “An empirical exploration of recurrent network architectures,” in ICML, 2015.
29. L. Xie and A. L. Yuille, “Genetic CNN,” in ICCV, 2017.
30. E. Real, S. Moore, A. Selle, S. Saxena, Y. L. Suematsu, J. Tan, Q. Le,
and A. Kurakin, “Large-scale evolution of image classifiers,” in
ICML, 2017.
31. E. Real, A. Aggarwal, Y. Huang, and Q. V. Le, “Regularized
evolution for image classifier architecture search,” arXiv preprint
arXiv:1802.01548, 2018.
32. T. Elsken, J. H. Metzen, and F. Hutter, “Efficient multi-objective
neural architecture search via lamarckian evolution,” arXiv
preprint arXiv:1804.09081, 2018.
33. H. Liu, K. Simonyan, and Y. Yang, “DARTS: Differentiable architecture search,” arXiv preprint arXiv:1806.09055, 2018.
34. J. Bergstra and Y. Bengio, “Random search for hyper-parameter
optimization,” JMLR, 2012.
35. C. Liu, B. Zoph, J. Shlens, W. Hua, L.-J. Li, L. Fei-Fei, A. Yuille,
J. Huang, and K. Murphy, “Progressive neural architecture
search,” in ECCV, 2018.
36. R. Negrinho and G. Gordon, “Deeparchitect: Automatically
designing and training deep architectures,” arXiv preprint
arXiv:1704.08792, 2017.
37. S. Saxena and J. Verbeek, “Convolutional neural fabrics,” in Conference on Neural Information Processing Systems, 2016.
38. A. Brock, T. Lim, J. M. Ritchie, and N. Weston, “SMASH: one-shot
model architecture search through hypernetworks,” in ICLR, 2018.
39. B. Baker, O. Gupta, R. Raskar, and N. Naik, “Accelerating neural
architecture search using performance prediction,” in ICLR Workshop, 2018.
40. B. Baker, O. Gupta, N. Naik, and R. Raskar, “Designing neural
network architectures using reinforcement learning,” in ICLR,2017.
41. B. Zoph and Q. V. Le, “Neural architecture search with reinforcement learning,” in ICLR, 2017.
42. B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le, “Learning transferable architectures for scalable image recognition,” in CVPR, 2018.
43. H. Liu, K. Simonyan, O. Vinyals, C. Fernando, and
K. Kavukcuoglu, “Hierarchical representations for efficient
architecture search,” in ICLR, 2018.
44. H. Pham, M. Y. Guan, B. Zoph, Q. V. Le, and J. Dean, “Efficient
neural architecture search via parameter sharing,” in ICML, 2018.
45. M. Tan, B. Chen, R. Pang, V. Vasudevan, and Q. V. Le, “Mnasnet: Platform-aware neural architecture search for mobile,” arXiv
preprint arXiv:1807.11626, 2018.13
46. B. A. Olshausen, C. H. Anderson, and D. C. V. Essen, “A neurobiological model of visual attention and invariant pattern recognition
based on dynamic routing of information,” Journal of Neuroscience,1993.
47. L. Itti, C. Koch, and E. Niebur, “A model of saliency-based visual
attention for rapid scene analysis,” IEEE Transactions on Pattern
Analysis and Machine Intelligence, 1998.
48. L. Itti and C. Koch, “Computational modelling of visual attention,”
Nature reviews neuroscience, 2001.
49. H. Larochelle and G. E. Hinton, “Learning to combine foveal
glimpses with a third-order boltzmann machine,” in Conference
on Neural Information Processing Systems, 2010.
50. V. Mnih, N. Heess, A. Graves, and K. Kavukcuoglu, “Recurrent
models of visual attention,” in Conference on Neural Information
Processing Systems, 2014.
51. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.
Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”
in Conference on Neural Information Processing Systems, 2017.
52. T. Bluche, “Joint line segmentation and transcription for end-toend handwritten paragraph recognition,” in Conference on Neural
Information Processing Systems, 2016.
53. A. Miech, I. Laptev, and J. Sivic, “Learnable pooling with context
gating for video classification,” arXiv:1706.06905, 2017.
54. C. Cao, X. Liu, Y. Yang, Y. Yu, J. Wang, Z. Wang, Y. Huang, L. Wang,
C. Huang, W. Xu, D. Ramanan, and T. S. Huang, “Look and
think twice: Capturing top-down visual attention with feedback
convolutional neural networks,” in ICCV, 2015.
55. K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhudinov,
R. Zemel, and Y. Bengio, “Show, attend and tell: Neural image
caption generation with visual attention,” in ICML, 2015.
56. L. Chen, H. Zhang, J. Xiao, L. Nie, J. Shao, W. Liu, and T. Chua,
“SCA-CNN: Spatial and channel-wise attention in convolutional
networks for image captioning,” in CVPR, 2017.
57. J. S. Chung, A. Senior, O. Vinyals, and A. Zisserman, “Lip reading
sentences in the wild,” in CVPR, 2017.
58. F. Wang, M. Jiang, C. Qian, S. Yang, C. Li, H. Zhang, X. Wang, and
X. Tang, “Residual attention network for image classification,” in
CVPR, 2017.
59. S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional
block attention module,” in ECCV, 2018.
60. J. Yang, K. Yu, Y. Gong, and T. Huang, “Linear spatial pyramid
matching using sparse coding for image classification,” in CVPR,2009.
61. J. Sanchez, F. Perronnin, T. Mensink, and J. Verbeek, “Image classification with the fisher vector: Theory and practice,” International
Journal of Computer Vision, 2013.
62. L. Shen, G. Sun, Q. Huang, S. Wang, Z. Lin, and E. Wu, “Multilevel discriminative dictionary learning with application to large
scale image classification,” IEEE TIP, 2015.
63. V. Nair and G. E. Hinton, “Rectified linear units improve restricted
boltzmann machines,” in ICML, 2010.
64. A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang,
T. Weyand, M. Andreetto, and H. Adam, “MobileNets: Efficient
convolutional neural networks for mobile vision applications,”
arXiv:1704.04861, 2017.
65. X. Zhang, X. Zhou, M. Lin, and J. Sun, “ShuffleNet: An extremely
efficient convolutional neural network for mobile devices,” in
CVPR, 2018.
66. K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers:
Surpassing human-level performance on ImageNet classification,”
in ICCV, 2015.
67. S. Zagoruyko and N. Komodakis, “Wide residual networks,” in
BMVC, 2016.
68. X. Gastaldi, “Shake-shake regularization,” arXiv preprint
arXiv:1705.07485, 2017.
69. T. DeVries and G. W. Taylor, “Improved regularization of
convolutional neural networks with cutout,” arXiv preprint
arXiv:1708.04552, 2017.
70. A. Krizhevsky and G. Hinton, “Learning multiple layers of features from tiny images,” Citeseer, Tech. Rep., 2009.
71. G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger, “Deep
networks with stochastic depth,” in ECCV, 2016.
72. L. Shen, Z. Lin, G. Sun, and J. Hu, “Places401 and places365 models,” https://github.com/lishen-shirley/Places2-CNNs, 2016.
73. B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba,
“Places: A 10 million image database for scene recognition,” IEEE
Transactions on Pattern Analysis and Machine Intelligence, 2017.
74. L. Shen, Z. Lin, and Q. Huang, “Relay backpropagation for effective learning of deep convolutional neural networks,” in ECCV,2016.
75. T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan,
P. Dollar, and C. L. Zitnick, “Microsoft COCO: Common objects in ´
context,” in ECCV, 2014.
76. R. Girshick, I. Radosavovic, G. Gkioxari, P. Dollar, and K. He, “De- ´
tectron,” https://github.com/facebookresearch/detectron, 2018.
77. D. Han, J. Kim, and J. Kim, “Deep pyramidal residual networks,”
in CVPR, 2017.
78. X. Zhang, Z. Li, C. C. Loy, and D. Lin, “Polynet: A pursuit of
structural diversity in very deep networks,” in CVPR, 2017.
79. E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le,
“Autoaugment: Learning augmentation policies from data,” arXiv
preprint arXiv:1805.09501, 2018.
80. D. Mahajan, R. Girshick, V. Ramanathan, K. He, M. Paluri, Y. Li,
A. Bharambe, and L. van der Maaten, “Exploring the limits of
weakly supervised pretraining,” in ECCV, 2018.
81. H. Lee, R. Grosse, R. Ranganath, and A. Y. Ng, “Convolutional
deep belief networks for scalable unsupervised learning of hierarchical representations,” in ICML, 2009.
82. J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How transferable
are features in deep neural networks?” in Conference on Neural
Information Processing Systems, 2014.
83. A. S. Morcos, D. G. Barrett, N. C. Rabinowitz, and M. Botvinick,
“On the importance of single directions for generalization,” in
ICLR, 2018.
84. J. Hu, L. Shen, S. Albanie, G. Sun, and A. Vedaldi, “Gather-excite:
Exploiting feature context in convolutional neural networks,” in
Conference on Neural Information Processing Systems, 2018.
