# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
2014-9-4   https://arxiv.org/abs/1409.1556

## 阅读笔记
* https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
* 主要贡献：3*3卷积重复堆叠

## ABSTRACT
In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small ( 3 × 3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisation and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision. 

在这项工作中，我们研究了在大规模图像识别设置中卷积网络深度对其精度的影响。我们的主要贡献是使用具有非常小(3×3)。这些发现是我们2014年ImageNet挑战赛的基础，我们的团队分别获得了本地化和分类轨道的第一和第二名。我们还表明，我们的表示可以很好地推广到其他数据集，在这些数据集中，它们可以获得最先进的结果。我们已经公开了两个性能最好的ConvNet模型，以便于进一步研究在计算机视觉中使用深度视觉表示。

## 1 INTRODUCTION
Convolutional networks (ConvNets) have recently enjoyed a great success in large-scale image and video recognition (Krizhevsky et al., 2012; Zeiler & Fergus, 2013; Sermanet et al., 2014; Simonyan & Zisserman, 2014) which has become possible due to the large public image repositories, such as ImageNet (Deng et al., 2009), and high-performance computing systems, such as GPUs or large-scale distributed clusters (Dean et al., 2012). In particular, an important role in the advance of deep visual recognition architectures has been played by the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) (Russakovsky et al., 2014), which has served as a testbed for a few generations of large-scale image classification systems, from high-dimensional shallow feature encodings (Perronnin et al., 2010) (the winner of ILSVRC-2011) to deep ConvNets (Krizhevsky et al., 2012) (the winner of ILSVRC-2012).

卷积网络(ConvNets)最近在大规模图像和视频识别方面取得了巨大成功(Krizhevskyet al., 2012;Zeiler&Fergus，2013;Sermanetet al., 2014;Simonyan&Zisserman，2014年)，由于大型公共图像存储库(如ImageNet(Denget al., 2009年)和高性能计算系统，例如GPU或大规模分布式集群(Deanet al., 2012)。特别是，ImageNet大规模视觉识别挑战(ILSVRC)(Russakovskyet al., 2014)在深度视觉识别架构的发展中发挥了重要作用，从高维浅特征编码(Perronninet al., 2010年)(ILSVRC-2011的获胜者)到深度ConvNets(Krizhevskyet al., 2012年)(IL SVRC-2012的获胜器)。

With ConvNets becoming more of a commodity in the computer vision field, a number of attempts have been made to improve the original architecture of Krizhevsky et al. (2012) in a bid to achieve better accuracy. For instance, the best-performing submissions to the ILSVRC2013 (Zeiler & Fergus, 2013; Sermanet et al., 2014) utilised smaller receptive window size and smaller stride of the first convolutional layer. Another line of improvements dealt with training and testing the networks densely over the whole image and over multiple scales (Sermanet et al., 2014; Howard, 2014). In this paper, we address another important aspect of ConvNet architecture design – its depth. To this end, we fix other parameters of the architecture, and steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small ( 3 × 3) convolution filters in all layers.

随着ConvNets在计算机视觉领域越来越成为一种商品，人们已经尝试改进Krizhevskyet al (2012)的原始架构，以达到更好的准确性。例如，提交给ILSVRC2013(Zeiler&Fergus，2013; Sermanet et al.，2014)的表现最好的文件利用了较小的接收窗口大小和第一卷积层的较小步幅。另一系列改进涉及在整个图像和多个尺度上密集地训练和测试网络(Sermanet et al.，2014; Howard，2014)。在本文中，我们讨论了ConvNet架构设计的另一个重要方面——深度。为此，我们固定了架构的其他参数，并通过添加更多卷积层稳步增加网络深度，这是可行的，因为在所有层中使用了非常小的(3×3)卷积滤波器。

As a result, we come up with significantly more accurate ConvNet architectures, which not only achieve the state-of-the-art accuracy on ILSVRC classification and localisation tasks, but are also applicable to other image recognition datasets, where they achieve excellent performance even when used as a part of a relatively simple pipelines (e.g. deep features classified by a linear SVM without fine-tuning). We have released our two best-performing models 1 to facilitate further research.

因此，我们提出了更精确的ConvNet架构，不仅在ILSVRC分类和定位任务上达到了最先进的精度，而且还适用于其他图像识别数据集，其中它们即使用作相对简单的管线的一部分(例如，由线性SVM分类的深度特征。我们发布了两个性能最好的模型1，以便于进一步研究。

The rest of the paper is organised as follows. In Sect. 2, we describe our ConvNet configurations. The details of the image classification training and evaluation are then presented in Sect. 3, and the configurations are compared on the ILSVRC classification task in Sect. 4. Sect. 5 concludes the paper. For completeness, we also describe and assess our ILSVRC-2014 object localisation system in Appendix A, and discuss the generalisation of very deep features to other datasets in Appendix B. Finally, Appendix C contains the list of major paper revisions. 

论文的其余部分组织如下。在Sect。2，我们描述我们的ConvNet配置。图像分类培训和评估的详情见第节。3节中的ILSVRC分类任务比较了这些配置。第4、5总结全文。为了完整性，我们还在附录A中描述和评估了ILSVRC-2014对象定位系统，并在附录B中讨论了对其他数据集的深度特征的概括。最后，附录C包含了主要文件修订清单。

## 2 CONVNET CONFIGURATIONS
To measure the improvement brought by the increased ConvNet depth in a fair setting, all our ConvNet layer configurations are designed using the same principles, inspired by Ciresan et al. (2011); Krizhevsky et al. (2012). In this section, we first describe a generic layout of our ConvNet configurations (Sect. 2.1) and then detail the specific configurations used in the evaluation (Sect. 2.2). Our design choices are then discussed and compared to the prior art in Sect. 2.3.

为了在公平的环境下测量ConvNet深度增加带来的改进，我们所有的ConvNet层配置都是根据Ciresanet al (2011年)的启发，使用相同的原理设计的; Krizhevskyet al (2012)。在本节中，我们首先描述ConvNet配置的一般布局(第2.1节)，然后详细说明评估中使用的具体配置(第2.2节)。然后讨论我们的设计选择，并将其与Sect。2.3.

### 2.1 ARCHITECTURE
During training, the input to our ConvNets is a fixed-size 224 × 224 RGB image. The only preprocessing we do is subtracting the mean RGB value, computed on the training set, from each pixel. The image is passed through a stack of convolutional (conv.) layers, where we use filters with a very small receptive field: 3 × 3 (which is the smallest size to capture the notion of left/right, up/down, center). In one of the configurations we also utilise 1 × 1 convolution filters, which can be seen as a linear transformation of the input channels (followed by non-linearity). The convolution stride is fixed to 1 pixel; the spatial padding of conv. layer input is such that the spatial resolution is preserved after convolution, i.e. the padding is 1 pixel for 3 × 3 conv. layers. Spatial pooling is carried out by five max-pooling layers, which follow some of the conv. layers (not all the conv. layers are followed by max-pooling). Max-pooling is performed over a 2 × 2 pixel window, with stride 2.

在训练期间，ConvNets的输入是固定大小的224×224 RGB图像。我们所做的唯一预处理是从每个像素中减去在训练集上计算的平均RGB值。图像通过一堆卷积(conv.)层，我们使用具有非常小的接收场的滤波器：3×3(这是捕捉左/右、上/下、中心概念的最小尺寸)。在其中一种配置中，我们还使用了1×1卷积滤波器，这可以看作是输入通道的线性变换(随后是非线性)。卷积步幅固定为1像素; conv.层输入的空间填充使得在卷积之后保留空间分辨率，即对于3×。空间池由五个最大池层执行，它们跟随一些conv.层(并非所有conv.图层都跟随最大池)。最大池在2×2像素窗口上执行，步幅为2。

A stack of convolutional layers (which has a different depth in different architectures) is followed by three Fully-Connected (FC) layers: the first two have 4096 channels each, the third performs 1000- way ILSVRC classification and thus contains 1000 channels (one for each class). The final layer is the soft-max layer. The configuration of the fully connected layers is the same in all networks.

一个卷积层堆栈(在不同的架构中具有不同的深度)后面是三个完全连接(FC)层：前两个层每个有4096个信道，第三个层执行1000路ILSVRC分类，因此包含1000个信道(每个类一个)。最后一层是软最大层。在所有网络中，全连接层的配置是相同的。

All hidden layers are equipped with the rectification (ReLU (Krizhevsky et al., 2012)) non-linearity. We note that none of our networks (except for one) contain Local Response Normalisation (LRN) normalisation (Krizhevsky et al., 2012): as will be shown in Sect. 4, such normalisation does not improve the performance on the ILSVRC dataset, but leads to increased memory consumption and computation time. Where applicable, the parameters for the LRN layer are those of (Krizhevsky et al., 2012).

所有隐藏层都配备了校正(ReLU(Krizhevskyet al., 2012))非线性。我们注意到，我们的网络(除一个网络外)均不包含局部响应规范化(LRN)规范化(Krizhevskyet al., 2012年)：如第。4，这种归一化不会提高ILSVRC数据集的性能，但会增加内存消耗和计算时间。如适用，LRN层的参数为(Krizhevskyet al., 2012年)。

### 2.2 CONFIGURATIONS
The ConvNet configurations, evaluated in this paper, are outlined in Table 1, one per column. In the following we will refer to the nets by their names (A–E). All configurations follow the generic design presented in Sect. 2.1, and differ only in the depth: from 11 weight layers in the network A (8 conv. and 3 FC layers) to 19 weight layers in the network E (16 conv. and 3 FC layers). The width of conv. layers (the number of channels) is rather small, starting from 64 in the first layer and then increasing by a factor of 2 after each max-pooling layer, until it reaches 512.

本文中评估的ConvNet配置如表1所示，每列一个。在下文中，我们将用它们的名称(A–E)来指代网络。所有配置均遵循第。并且仅在深度上不同：从网络A中的11个权重层(8个conv.和3个FC层)到网络E中的19个权重层。conv.层的宽度(通道数)相当小，从第一层的64开始，然后在每个最大池层之后增加2倍，直到达到512。

In Table 2 we report the number of parameters for each configuration. In spite of a large depth, the number of weights in our nets is not greater than the number of weights in a more shallow net with larger conv. layer widths and receptive fields (144M weights in (Sermanet et al., 2014)).

在表2中，我们报告了每个配置的参数数量。尽管深度较大，但我们的网中的权重数不大于具有较大的转换层宽度和接收场的较浅网中的权数((Sermanet et al.，2014)中的144M权重)。

### 2.3 DISCUSSION
Our ConvNet configurations are quite different from the ones used in the top-performing entries of the ILSVRC-2012 (Krizhevsky et al., 2012) and ILSVRC-2013 competitions (Zeiler & Fergus, 2013; Sermanet et al., 2014). Rather than using relatively large receptive fields in the first conv. layers (e.g. 11×11 with stride 4 in (Krizhevsky et al., 2012), or 7×7 with stride 2 in (Zeiler & Fergus, 2013; Sermanet et al., 2014)), we use very small 3 × 3 receptive fields throughout the whole net, which are convolved with the input at every pixel (with stride 1). It is easy to see that a stack of two 3×3 conv. layers (without spatial pooling in between) has an effective receptive field of 5×5; three such layers have a 7 × 7 effective receptive field. So what have we gained by using, for instance, a stack of three 3×3 conv. layers instead of a single 7×7 layer? First, we incorporate three non-linear rectification layers instead of a single one, which makes the decision function more discriminative. Second, we decrease the number of parameters: assuming that both the input and the output of a three-layer 3 × 3 convolution stack has C channels, the stack is parametrised by 3  3 2C 2  = 27C 2 weights; at the same time, a single 7 × 7 conv. layer would require 7 2C 2 = 49C 2 parameters, i.e. 81% more. This can be seen as imposing a regularisation on the 7 × 7 conv. filters, forcing them to have a decomposition through the 3 × 3 filters (with non-linearity injected in between).

我们的ConvNet配置与ILSVRC-2012(Krizhevskyet al., 2012)和ILSVRC-2003比赛(Zeiler&Fergus，2013; Sermanetet al., 2014)中表现最好的参赛作品中使用的配置非常不同。与在第一个卷积层中使用相对较大的感受野(例如，步幅为4的11×11(Krizhevskyet al., 2012年)，或步幅为2英寸的7×7(Zeiler&Fergus，2013;Sermanetet al., 2014年))不同，我们在整个网络中使用非常小的3×3感受野，它们与每个像素处的输入(步幅为1)卷积。很容易看出，两个3×; 三个这样的层具有7×7的有效感受野。那么，我们通过使用三个3×3转换层的堆栈而不是一个7×7层获得了什么呢？首先，我们合并了三个非线性校正层，而不是单个校正层，这使得决策函数更具辨别力。第二，我们减少了参数的数量：假设三层3×; 同时，单个7×7 conv.层将需要72C2＝49C2参数，即81%以上。这可以看作是对7×7转换滤波器施加了规则化，迫使它们通过3×3滤波器进行分解(其间注入了非线性)。

Table 1: ConvNet configurations (shown in columns). The depth of the configurations increases from the left (A) to the right (E), as more layers are added (the added layers are shown in bold). The convolutional layer parameters are denoted as “convhreceptive field sizei-hnumber of channelsi”. The ReLU activation function is not shown for brevity.
表1：ConvNet配置(以列显示)。配置的深度从左(A)到右(E)增加，因为添加了更多层(添加的层以粗体显示)。卷积层参数表示为“卷积场大小和信道数”。为简洁起见，未显示ReLU激活功能。

Table 2: Number of parameters (in millions).
表2：参数数量(百万)。

The incorporation of 1 × 1 conv. layers (configuration C, Table 1) is a way to increase the nonlinearity of the decision function without affecting the receptive fields of the conv. layers. Even though in our case the 1 × 1 convolution is essentially a linear projection onto the space of the same dimensionality (the number of input and output channels is the same), an additional non-linearity is introduced by the rectification function. It should be noted that 1×1 conv. layers have recently been utilised in the “Network in Network” architecture of Lin et al. (2014).

1×1转换层(配置C，表1)的加入是一种在不影响转换层接收场的情况下增加决策函数非线性的方法。即使在我们的情况下，1×1卷积本质上是在相同维度(输入和输出通道的数量相同)的空间上的线性投影，但校正函数引入了额外的非线性。应注意的是，最近Linet al (2014)的“网络中的网络”架构中使用了1×1 conv.层。

Small-size convolution filters have been previously used by Ciresan et al. (2011), but their nets are significantly less deep than ours, and they did not evaluate on the large-scale ILSVRC dataset. Goodfellow et al. (2014) applied deep ConvNets (11 weight layers) to the task of street number recognition, and showed that the increased depth led to better performance. GoogLeNet (Szegedy et al., 2014), a top-performing entry of the ILSVRC-2014 classification task, was developed independently of our work, but is similar in that it is based on very deep ConvNets (22 weight layers) and small convolution filters (apart from 3 × 3, they also use 1 × 1 and 5 × 5 convolutions). Their network topology is, however, more complex than ours, and the spatial resolution of the feature maps is reduced more aggressively in the first layers to decrease the amount of computation. As will be shown in Sect. 4.5, our model is outperforming that of Szegedy et al. (2014) in terms of the single-network classification accuracy. 

Ciresanet al (2011年)以前曾使用过小尺寸卷积滤波器，但它们的网络深度明显低于我们的网络，并且没有在大规模ILSVRC数据集上进行评估。Goodfellowet al (2014)将深度ConvNets(11个权重层)应用于街道编号识别任务，并表明深度的增加会带来更好的性能。GoogLeNet(Szegedy et al.，2014)是ILSVRC-2014分类任务中表现最好的条目，独立于我们的工作开发，但类似之处在于它基于非常深的ConvNets(22个权重层)和小卷积滤波器(除了3×3，它们还使用1×1和5×5卷积)。然而，他们的网络拓扑比我们的更复杂，并且在第一层中更积极地降低了特征图的空间分辨率，以减少计算量。如第节所示。4.5，我们的模型在单网络分类精度方面优于Szegedyet al (2014)的模型。

## 3 CLASSIFICATION FRAMEWORK
In the previous section we presented the details of our network configurations. In this section, we describe the details of classification ConvNet training and evaluation.

在上一节中，我们介绍了网络配置的详情。在本节中，我们将详细介绍分类ConvNet训练和评估。

### 3.1 TRAINING
The ConvNet training procedure generally follows Krizhevsky et al. (2012) (except for sampling the input crops from multi-scale training images, as explained later). Namely, the training is carried out by optimising the multinomial logistic regression objective using mini-batch gradient descent (based on back-propagation (LeCun et al., 1989)) with momentum. The batch size was set to 256, momentum to 0.9. The training was regularised by weight decay (the L2 penalty multiplier set to 5 · 10−4 ) and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5). The learning rate was initially set to 10−2 , and then decreased by a factor of 10 when the validation set accuracy stopped improving. In total, the learning rate was decreased 3 times, and the learning was stopped after 370K iterations (74 epochs). We conjecture that in spite of the larger number of parameters and the greater depth of our nets compared to (Krizhevsky et al., 2012), the nets required less epochs to converge due to (a) implicit regularisation imposed by greater depth and smaller conv. filter sizes; (b) pre-initialisation of certain layers.

ConvNet训练程序通常遵循Krizhevskyet al (2012)(除了从多尺度训练图像中对输入作物进行采样，如下文所述)。即,通过使用小批量梯度下降(基于反向传播(LeCun et al.，1989))和动量优化多项式logistic回归目标来进行训练。批量大小设置为256，动量设置为0.9。训练通过权重衰减进行调整(L2惩罚乘数设置为5.10−4)和前两个全连接层的压降调节(压降比设置为0.5)。学习率最初设定为10−然后当验证设置精度停止提高时减小10倍。总的来说，学习率降低了3倍，在370K次迭代(74个周期)后停止学习。我们推测，尽管与(Krizhevskyet al., 2012年)相比，我们的网络的参数数量和深度更大，但由于(a)更大的深度和更小的卷积滤波器尺寸带来的隐式正则化，网络需要更少的时间来收敛; (b) 某些层的预初始化。

The initialisation of the network weights is important, since bad initialisation can stall learning due to the instability of gradient in deep nets. To circumvent this problem, we began with training the configuration A (Table 1), shallow enough to be trained with random initialisation. Then, when training deeper architectures, we initialised the first four convolutional layers and the last three fullyconnected layers with the layers of net A (the intermediate layers were initialised randomly). We did not decrease the learning rate for the pre-initialised layers, allowing them to change during learning. For random initialisation (where applicable), we sampled the weights from a normal distribution with the zero mean and 10−2 variance. The biases were initialised with zero. It is worth noting that after the paper submission we found that it is possible to initialise the weights without pre-training by using the random initialisation procedure of Glorot & Bengio (2010).

网络权重的初始化很重要，因为由于深度网络中梯度的不稳定性，错误的初始化可能会导致学习停滞。为了避免这个问题，我们首先训练配置A(表1)，足够浅，可以用随机初始化进行训练。然后，当训练更深层次的架构时，我们用网络A的层初始化了前四个卷积层和最后三个完全连接的层(中间层是随机初始化的)。我们没有降低预初始化层的学习率，让它们在学习过程中发生变化。对于随机初始化(如适用)，我们从具有零均值和10−2方差。偏差的初始值为零。值得注意的是，在提交论文后，我们发现可以使用Gllot&Bengio(2010)的随机初始化程序，在不进行预训练的情况下初始化权重。

To obtain the fixed-size 224×224 ConvNet input images, they were randomly cropped from rescaled training images (one crop per image per SGD iteration). To further augment the training set, the crops underwent random horizontal flipping and random RGB colour shift (Krizhevsky et al., 2012). Training image rescaling is explained below.

为了获得固定大小的224×224。为了进一步增强训练集，作物进行了随机水平翻转和随机RGB颜色偏移(Krizhevskyet al., 2012年)。下面解释训练图像缩放。

Training image size. Let S be the smallest side of an isotropically-rescaled training image, from which the ConvNet input is cropped (we also refer to S as the training scale). While the crop size is fixed to 224 × 224, in principle S can take on any value not less than 224: for S = 224 the crop will capture whole-image statistics, completely spanning the smallest side of a training image; for S ≫ 224 the crop will correspond to a small part of the image, containing a small object or an object part.

训练图像大小。设S为各向同性重缩放训练图像的最小边，从中裁剪ConvNet输入(我们也称S为训练尺度)。当裁剪大小固定为224×224时，原则上S可以取不小于224的任何值：对于S＝224，裁剪将捕获整个图像统计，完全跨越训练图像的最小边; 对于S≫ 则裁剪将对应于包含小对象或对象部分的图像的小部分。

We consider two approaches for setting the training scale S. The first is to fix S, which corresponds to single-scale training (note that image content within the sampled crops can still represent multiscale image statistics). In our experiments, we evaluated models trained at two fixed scales: S = 256 (which has been widely used in the prior art (Krizhevsky et al., 2012; Zeiler & Fergus, 2013;Sermanet et al., 2014)) and S = 384. Given a ConvNet configuration, we first trained the network using S = 256. To speed-up training of the S = 384 network, it was initialised with the weights pre-trained with S = 256, and we used a smaller initial learning rate of 10−3 .

我们考虑两种方法来设置训练尺度S。第一种方法是固定S，这对应于单尺度训练(注意，采样作物中的图像内容仍然可以表示多尺度图像统计)。在我们的实验中，我们评估了在两个固定尺度下训练的模型：S＝256(这在现有技术中已被广泛使用(Krizhevskyet al., 2012; Zeiler&Fergus，2013; Sermanetet al., 2014))和S＝384。给定ConvNet配置，我们首先使用S=256训练网络。为了加速S＝382网络的训练，我们使用预先训练的权重进行初始化，我们使用了较小的初始学习率10−3.

The second approach to setting S is multi-scale training, where each training image is individually rescaled by randomly sampling S from a certain range [Smin, Smax] (we used Smin = 256 and Smax = 512). Since objects in images can be of different size, it is beneficial to take this into account during training. This can also be seen as training set augmentation by scale jittering, where a single model is trained to recognise objects over a wide range of scales. For speed reasons, we trained multi-scale models by fine-tuning all layers of a single-scale model with the same configuration, pre-trained with fixed S = 384.

设置S的第二种方法是多尺度训练，其中通过从特定范围[Smin，Smax](我们使用Smin＝256和Smax＝512)随机采样S来分别重新缩放每个训练图像。由于图像中的对象可以具有不同的大小，因此在训练过程中考虑到这一点是有益的。这也可以被视为通过尺度抖动来增强训练集，即训练单个模型以识别大范围尺度上的对象。出于速度的原因，我们通过微调具有相同配置的单尺度模型的所有层来训练多尺度模型，并预先训练固定S＝384。

### 3.2 TESTING
At test time, given a trained ConvNet and an input image, it is classified in the following way. First, it is isotropically rescaled to a pre-defined smallest image side, denoted as Q (we also refer to it as the test scale). We note that Q is not necessarily equal to the training scale S (as we will show in Sect. 4, using several values of Q for each S leads to improved performance). Then, the network is applied densely over the rescaled test image in a way similar to (Sermanet et al., 2014). Namely, the fully-connected layers are first converted to convolutional layers (the first FC layer to a 7 × 7 conv. layer, the last two FC layers to 1 × 1 conv. layers). The resulting fully-convolutional net is then applied to the whole (uncropped) image. The result is a class score map with the number of channels equal to the number of classes, and a variable spatial resolution, dependent on the input image size. Finally, to obtain a fixed-size vector of class scores for the image, the class score map is spatially averaged (sum-pooled). We also augment the test set by horizontal flipping of the images; the soft-max class posteriors of the original and flipped images are averaged to obtain the final scores for the image.

在测试时，给定一个经过训练的ConvNet和一个输入图像，它按以下方式分类。首先，它被各向同性地重新缩放到预定义的最小图像侧，表示为Q(我们也将其称为测试比例)。我们注意到，Q不一定等于训练量表S(正如我们将在第4节中所示，为每个S使用几个Q值可以提高性能)。然后，以类似于(Sermanet et al.，2014)的方式在重新缩放的测试图像上密集地应用网络。即，首先将完全连接的层转换为卷积层(第一个FC层转换为7×7卷积层，最后两个FC层为1×1卷积层)。然后将得到的完全卷积网络应用于整个(未裁剪)图像。结果是一个类别分数图，通道数等于类别数，空间分辨率可变，取决于输入图像大小。最后，为了获得图像的固定大小的类得分向量，对类得分图进行空间平均(和池)。我们还通过水平翻转图像来增强测试集; 对原始图像和翻转图像的软最大类后验进行平均，以获得图像的最终得分。

Since the fully-convolutional network is applied over the whole image, there is no need to sample multiple crops at test time (Krizhevsky et al., 2012), which is less efficient as it requires network re-computation for each crop. At the same time, using a large set of crops, as done by Szegedy et al. (2014), can lead to improved accuracy, as it results in a finer sampling of the input image compared to the fully-convolutional net. Also, multi-crop evaluation is complementary to dense evaluation due to different convolution boundary conditions: when applying a ConvNet to a crop, the convolved feature maps are padded with zeros, while in the case of dense evaluation the padding for the same crop naturally comes from the neighbouring parts of an image (due to both the convolutions and spatial pooling), which substantially increases the overall network receptive field, so more context is captured. While we believe that in practice the increased computation time of multiple crops does not justify the potential gains in accuracy, for reference we also evaluate our networks using 50 crops per scale (5 × 5 regular grid with 2 flips), for a total of 150 crops over 3 scales, which is comparable to 144 crops over 4 scales used by Szegedy et al. (2014).

由于全卷积网络应用于整个图像，因此无需在测试时对多个作物进行采样(Krizhevskyet al., 2012年)，这效率较低，因为它需要对每个作物进行网络重新计算。同时，正如Szegedyet al (2014)所做的那样，使用大量作物可以提高精度，因为与全卷积网络相比，它可以对输入图像进行更精细的采样。此外，由于卷积边界条件不同，多作物评估与密集评估是互补的：当将卷积网络应用于作物时，卷积的特征图用零填充，而在密集评估的情况下，相同作物的填充自然来自图像的相邻部分(由于卷积和空间池)，这显著地增加了整个网络接受场，因此捕获了更多的上下文。虽然我们认为，在实践中，多作物计算时间的增加并不能证明精度的潜在提高，但我们还使用每种尺度50种作物(5×。

### 3.3 IMPLEMENTATION DETAILS
Our implementation is derived from the publicly available C++ Caffe toolbox (Jia, 2013) (branched out in December 2013), but contains a number of significant modifications, allowing us to perform training and evaluation on multiple GPUs installed in a single system, as well as train and evaluate on full-size (uncropped) images at multiple scales (as described above). Multi-GPU training exploits data parallelism, and is carried out by splitting each batch of training images into several GPU batches, processed in parallel on each GPU. After the GPU batch gradients are computed, they are averaged to obtain the gradient of the full batch. Gradient computation is synchronous across the GPUs, so the result is exactly the same as when training on a single GPU.

我们的实现源于公开的C++Caffe工具箱(Jia，2013年)(2013年12月推出)，但包含了一些重要的修改，允许我们在单个系统中安装的多个GPU上执行培训和评估，以及在多个尺度上对全尺寸(未裁剪)图像进行培训和评估(如上所述)。多GPU训练利用数据并行性，通过将每批训练图像分割为若干GPU批次来执行，并在每个GPU上并行处理。计算GPU批次梯度后，将其平均以获得整个批次的梯度。梯度计算是跨GPU同步的，因此结果与在单个GPU上训练时完全相同。

While more sophisticated methods of speeding up ConvNet training have been recently proposed (Krizhevsky, 2014), which employ model and data parallelism for different layers of the net, we have found that our conceptually much simpler scheme already provides a speedup of 3.75 times on an off-the-shelf 4-GPU system, as compared to using a single GPU. On a system equipped with four NVIDIA Titan Black GPUs, training a single net took 2–3 weeks depending on the architecture. 

虽然最近提出了更复杂的加速ConvNet训练的方法(Krizhevsky，2014)，该方法采用了网络不同层的模型和数据并行，但我们发现，与使用单个GPU相比，我们的概念上简单得多的方案已经在现成的4-GPU系统上提供了3.75倍的加速。在配备四个NVIDIA Titan Black GPU的系统上，根据架构，训练单个网络需要2-3周时间。

## 4 CLASSIFICATION EXPERIMENTS
Dataset. In this section, we present the image classification results achieved by the described ConvNet architectures on the ILSVRC-2012 dataset (which was used for ILSVRC 2012–2014 challenges). The dataset includes images of 1000 classes, and is split into three sets: training (1.3M images), validation (50K images), and testing (100K images with held-out class labels). The classification performance is evaluated using two measures: the top-1 and top-5 error. The former is a multi-class classification error, i.e. the proportion of incorrectly classified images; the latter is the main evaluation criterion used in ILSVRC, and is computed as the proportion of images such that the ground-truth category is outside the top-5 predicted categories.

数据集。在本节中，我们将介绍所述ConvNet架构在ILSVRC-2012数据集(用于ILSVRC2012-2014挑战)上实现的图像分类结果。该数据集包括1000个类的图像，并分为三组：训练(1.3M个图像)、验证(50K个图像)和测试(100K个图像，带有固定的类标签)。分类性能使用两种度量方法进行评估：前1和前5错误。前者是多类分类错误，即分类错误图像的比例; 后者是ILSVRC中使用的主要评估标准，并被计算为图像的比例，使得地面真实类别在前5个预测类别之外。

For the majority of experiments, we used the validation set as the test set. Certain experiments were also carried out on the test set and submitted to the official ILSVRC server as a “VGG” team entry to the ILSVRC-2014 competition (Russakovsky et al., 2014).

对于大多数实验，我们使用验证集作为测试集。还对测试集进行了某些实验，并将其作为“VGG”团队参赛作品提交给ILSVRC官方服务器，参加ILSVRC-2014比赛(Russakovskyet al., 2014年)。

## 4.1 SINGLE SCALE EVALUATION
We begin with evaluating the performance of individual ConvNet models at a single scale with the layer configurations described in Sect. 2.2. The test image size was set as follows: Q = S for fixed S, and Q = 0.5(Smin + Smax) for jittered S ∈ [Smin, Smax]. The results of are shown in Table 3.

我们首先评估单个ConvNet模型在单个尺度上的性能，并使用第节中描述的层配置。2.2.测试图像大小设置如下：对于固定S，Q=S，对于抖动S，Q=0.5(Smin+Smax)∈ [Smin，Smax]。的结果如表3所示。

First, we note that using local response normalisation (A-LRN network) does not improve on the model A without any normalisation layers. We thus do not employ normalisation in the deeper architectures (B–E).

首先，我们注意到，在没有任何归一化层的情况下，使用局部响应归一化(A-LRN网络)并不能改善模型A。因此，我们在更深层次的架构(B-E)中不采用标准化。

Second, we observe that the classification error decreases with the increased ConvNet depth: from 11 layers in A to 19 layers in E. Notably, in spite of the same depth, the configuration C (which contains three 1 × 1 conv. layers), performs worse than the configuration D, which uses 3 × 3 conv. layers throughout the network. This indicates that while the additional non-linearity does help (C is better than B), it is also important to capture spatial context by using conv. filters with non-trivial receptive fields (D is better than C). The error rate of our architecture saturates when the depth reaches 19 layers, but even deeper models might be beneficial for larger datasets. We also compared the net B with a shallow net with five 5 × 5 conv. layers, which was derived from B by replacing each pair of 3 × 3 conv. layers with a single 5 × 5 conv. layer (which has the same receptive field as explained in Sect. 2.3). The top-1 error of the shallow net was measured to be 7% higher than that of B (on a center crop), which confirms that a deep net with small filters outperforms a shallow net with larger filters.

其次，我们观察到，分类误差随着ConvNet深度的增加而减小：从A中的11层到E中的19层。值得注意的是，尽管深度相同，配置C(包含3个1×1 conv.层)的性能比配置D(在整个网络中使用3×3 conv.层数)差。这表明，虽然额外的非线性确实有帮助(C比B好)，但通过使用具有非平凡接收场的conv.滤波器(D比C好)来捕捉空间背景也很重要。当深度达到19层时，我们的架构的错误率就会饱和，但更深层的模型可能对更大的数据集有益。我们还将网B与具有5个5×5 conv.层的浅网进行了比较，该浅网是通过用单个5×5 conv.层(其具有第2.3节中所述的相同接收场)替换每对3×。浅网的前1误差被测量为比B(在中心作物上)高7%，这证实了具有小过滤器的深网优于具有较大过滤器的浅网。

Finally, scale jittering at training time (S ∈ [256; 512]) leads to significantly better results than training on images with fixed smallest side (S = 256 or S = 384), even though a single scale is used at test time. This confirms that training set augmentation by scale jittering is indeed helpful for capturing multi-scale image statistics.

最后，在训练时间(S∈ [256; 512])比在具有固定最小边(S＝256或S＝384)的图像上进行训练的结果要好得多。这证实了通过尺度抖动增强训练集确实有助于捕获多尺度图像统计。

Table 3: ConvNet performance at a single test scale.
表3:ConvNet在单个测试规模下的性能。

### 4.2 MULTI-SCALE EVALUATION
Having evaluated the ConvNet models at a single scale, we now assess the effect of scale jittering at test time. It consists of running a model over several rescaled versions of a test image (corresponding to different values of Q), followed by averaging the resulting class posteriors. Considering that a large discrepancy between training and testing scales leads to a drop in performance, the models trained with fixed S were evaluated over three test image sizes, close to the training one: Q = {S − 32, S, S + 32}. At the same time, scale jittering at training time allows the network to be applied to a wider range of scales at test time, so the model trained with variable S ∈ [Smin; Smax] was evaluated over a larger range of sizes Q = {Smin, 0.5(Smin + Smax), Smax}. 

在单个尺度上评估了ConvNet模型之后，我们现在评估了测试时尺度抖动的影响。它包括在测试图像的多个重新缩放版本(对应于不同的Q值)上运行模型，然后对得到的类后验进行平均。考虑到训练和测试尺度之间的巨大差异会导致性能下降，使用固定S训练的模型在三个测试图像尺寸上进行评估，接近于训练图像尺寸：Q＝{S− 32，S，S+32}。同时，训练时的尺度抖动允许网络在测试时应用于更大范围的尺度，因此使用变量S训练模型∈ [Smin; Smax]在更大范围的尺寸Q＝{Smin，0.5(Smin+Smax)，Smax}上进行评估。

The results, presented in Table 4, indicate that scale jittering at test time leads to better performance (as compared to evaluating the same model at a single scale, shown in Table 3). As before, the deepest configurations (D and E) perform the best, and scale jittering is better than training with a fixed smallest side S. Our best single-network performance on the validation set is 24.8%/7.5% top-1/top-5 error (highlighted in bold in Table 4). On the test set, the configuration E achieves 7.3% top-5 error.

表4所示的结果表明，测试时的标度抖动导致了更好的性能(与在单个标度下评估相同模型相比，如表3所示)。如前所述，最深的配置(D和E)性能最好，规模抖动优于固定最小边S的训练。在验证集上，我们的最佳单网络性能为24.8%/7.5%top 1/top 5错误(表4中以粗体突出显示)。在测试集上，配置E实现了7.3%的前5个错误。

Table 4: ConvNet performance at multiple test scales.
表4:ConvNet在多个测试规模下的性能。

### 4.3 MULTI-CROP EVALUATION
In Table 5 we compare dense ConvNet evaluation with mult-crop evaluation (see Sect. 3.2 for details). We also assess the complementarity of the two evaluation techniques by averaging their softmax outputs. As can be seen, using multiple crops performs slightly better than dense evaluation, and the two approaches are indeed complementary, as their combination outperforms each of them.

在表5中，我们将密集ConvNet评估与多作物评估进行了比较(详见第3.2节)。我们还通过平均其softmax输出来评估两种评估技术的互补性。可以看出，使用多种作物比密集评估表现稍好，而且这两种方法确实是互补的，因为它们的组合优于每种方法。

As noted above, we hypothesize that this is due to a different treatment of convolution boundary conditions.

如上所述，我们假设这是由于对卷积边界条件的不同处理。

Table 5: ConvNet evaluation techniques comparison. In all experiments the training scale S was sampled from [256; 512], and three test scales Q were considered: {256, 384, 512}.
表5:ConvNet评估技术比较。在所有实验中，从[256; 512]中抽取训练量表S，并考虑三个测试量表Q：{256，384，512}。

### 4.4 CONVNET FUSION
Up until now, we evaluated the performance of individual ConvNet models. In this part of the experiments, we combine the outputs of several models by averaging their soft-max class posteriors. This improves the performance due to complementarity of the models, and was used in the top ILSVRC submissions in 2012 (Krizhevsky et al., 2012) and 2013 (Zeiler & Fergus, 2013; Sermanet et al., 2014).

到目前为止，我们评估了单个ConvNet模型的性能。在这部分实验中，我们通过平均软最大类后验来组合几个模型的输出。由于模型的互补性，这提高了性能，并在2012年(Krizhevskyet al., 2012年)和2013年(Zeiler&Fergus，2013;Sermanetet al., 2014年)的ILSVRC顶级提交中使用。

The results are shown in Table 6. By the time of ILSVRC submission we had only trained the single-scale networks, as well as a multi-scale model D (by fine-tuning only the fully-connected layers rather than all layers). The resulting ensemble of 7 networks has 7.3% ILSVRC test error. After the submission, we considered an ensemble of only two best-performing multi-scale models (configurations D and E), which reduced the test error to 7.0% using dense evaluation and 6.8% using combined dense and multi-crop evaluation. For reference, our best-performing single model achieves 7.1% error (model E, Table 5).

结果如表6所示。在ILSVRC提交时，我们只训练了单尺度网络以及多尺度模型D(通过仅微调完全连接的层而不是所有层)。结果7个网络的集成具有7.3%的ILSVRC测试误差。提交后，我们只考虑了两个性能最佳的多尺度模型(配置D和E)的集合，使用密集评估将测试误差降低到7.0%，使用密集和多作物组合评估将测试错误降低到6.8%。作为参考，我们性能最好的单一模型实现了7.1%的误差(模型E，表5)。

### 4.5 COMPARISON WITH THE STATE OF THE ART
Finally, we compare our results with the state of the art in Table 7. In the classification task of ILSVRC-2014 challenge (Russakovsky et al., 2014), our “VGG” team secured the 2nd place with 7.3% test error using an ensemble of 7 models. After the submission, we decreased the error rate to  6.8% using an ensemble of 2 models.

最后，我们将我们的结果与表7中的最新水平进行了比较。在ILSVRC-2014挑战的分类任务中(Russakovskyet al., 2014年)，我们的“VGG”团队使用7个模型的集合以7.3%的测试误差获得了第二名。提交后，我们使用2个模型的集合将错误率降低到6.8%。

Table 6: Multiple ConvNet fusion results. 
表6：多个ConvNet融合结果。

As can be seen from Table 7, our very deep ConvNets significantly outperform the previous generation of models, which achieved the best results in the ILSVRC-2012 and ILSVRC-2013 competitions. Our result is also competitive with respect to the classification task winner (GoogLeNet with 6.7% error) and substantially outperforms the ILSVRC-2013 winning submission Clarifai, which achieved 11.2% with outside training data and 11.7% without it. This is remarkable, considering that our best result is achieved by combining just two models – significantly less than used in most ILSVRC submissions. In terms of the single-net performance, our architecture achieves the best result (7.0% test error), outperforming a single GoogLeNet by 0.9%. Notably, we did not depart from the classical ConvNet architecture of LeCun et al. (1989), but improved it by substantially increasing the depth.

从表7可以看出，我们的深度ConvNets显著优于上一代模型，后者在ILSVRC-2012和ILSVRC-2003比赛中取得了最佳成绩。我们的结果在分类任务获胜者(GoogleLeNet，错误率为6.7%)方面也很有竞争力，并大大优于ILSVRC-2013获奖提交的Clarifai，后者在外部培训数据的情况下达到11.2%，没有外部培训数据时达到11.7%。考虑到我们的最佳结果是通过组合两个模型实现的，这一点值得注意——大大低于大多数ILSVRC提交文件中使用的模型。就单网性能而言，我们的架构实现了最佳结果(7.0%的测试误差)，比单个GoogleLeNet高0.9%。值得注意的是，我们没有偏离LeCunet al (1989)的经典ConvNet架构，而是通过大幅增加深度来改进它。

Table 7: Comparison with the state of the art in ILSVRC classification. Our method is denoted as “VGG”. Only the results obtained without outside training data are reported.
表7：与ILSVRC分类的最新技术进行比较。我们的方法表示为“VGG”。只报告没有外部培训数据的结果。

## 5 CONCLUSION
In this work we evaluated very deep convolutional networks (up to 19 weight layers) for largescale image classification. It was demonstrated that the representation depth is beneficial for the classification accuracy, and that state-of-the-art performance on the ImageNet challenge dataset can be achieved using a conventional ConvNet architecture (LeCun et al., 1989; Krizhevsky et al., 2012) with substantially increased depth. In the appendix, we also show that our models generalise well to a wide range of tasks and datasets, matching or outperforming more complex recognition pipelines built around less deep image representations. Our results yet again confirm the importance of depth in visual representations.

在这项工作中，我们评估了用于大规模图像分类的非常深的卷积网络(多达19个权重层)。研究表明，表示深度有助于提高分类精度，并且使用传统的ConvNet架构(LeCunet al., 1989; Krizhevskyet al., 2012)可以在深度显著增加的情况下实现ImageNet挑战数据集的最新性能。在附录中，我们还展示了我们的模型能够很好地推广到广泛的任务和数据集，匹配或优于围绕深度较低的图像表示构建的更复杂的识别管道。我们的结果再次证实了深度在视觉表现中的重要性。

## ACKNOWLEDGEMENTS
This work was supported by ERC grant VisRec no. 228180. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the GPUs used for this research. 

这项工作得到了ERC赠款VisRec编号228180的支持。我们感谢NVIDIA Corporation捐赠用于此研究的GPU。

## REFERENCES 

## A
### A LOCALISATION
In the main body of the paper we have considered the classification task of the ILSVRC challenge, and performed a thorough evaluation of ConvNet architectures of different depth. In this section, we turn to the localisation task of the challenge, which we have won in 2014 with 25.3% error. It can be seen as a special case of object detection, where a single object bounding box should be predicted for each of the top-5 classes, irrespective of the actual number of objects of the class. For this we adopt the approach of Sermanet et al. (2014), the winners of the ILSVRC-2013 localisation challenge, with a few modifications. Our method is described in Sect. A.1 and evaluated in Sect. A.2.

#### A.1 LOCALISATION CONVNET
To perform object localisation, we use a very deep ConvNet, where the last fully connected layer predicts the bounding box location instead of the class scores. A bounding box is represented by a 4-D vector storing its center coordinates, width, and height. There is a choice of whether the bounding box prediction is shared across all classes (single-class regression, SCR (Sermanet et al., 2014)) or is class-specific (per-class regression, PCR). In the former case, the last layer is 4-D, while in the latter it is 4000-D (since there are 1000 classes in the dataset). Apart from the last bounding box prediction layer, we use the ConvNet architecture D (Table 1), which contains 16 weight layers and was found to be the best-performing in the classification task (Sect. 4).

在本文的主体部分，我们考虑了ILSVRC挑战的分类任务，并对不同深度的ConvNet架构进行了全面评估。在本节中，我们将讨论挑战的本地化任务，我们在2014年以25.3%的错误赢得了挑战。这可以看作是对象检测的一种特殊情况，在这种情况下，应该为前5个类中的每一个预测单个对象边界框，而不管该类的实际对象数量。为此，我们采用了Sermanetet al (2014年)的方法，即ILSVRC-2013本地化挑战的获胜者，并进行了一些修改。我们的方法在Sect。A、 1节中进行了评估。A、 2。

Training. Training of localisation ConvNets is similar to that of the classification ConvNets (Sect. 3.1). The main difference is that we replace the logistic regression objective with a Euclidean loss, which penalises the deviation of the predicted bounding box parameters from the ground-truth.

为了执行对象定位，我们使用一个非常深的ConvNet，其中最后一个完全连接的层预测边界框的位置，而不是类分数。边界框由存储其中心坐标、宽度和高度的4-D向量表示。可以选择是在所有类之间共享边界框预测(单类回归，SCR(Sermanet et al.，2014))还是特定于类(每类回归，PCR)。在前一种情况下，最后一层是4-D，而在后一种情况中是4000-D(因为数据集中有1000个类)。除了最后一个边界框预测层之外，我们使用ConvNet架构D(表1)，它包含16个权重层，被发现在分类任务中表现最佳(第4节)。

We trained two localisation models, each on a single scale: S = 256 and S = 384 (due to the time constraints, we did not use training scale jittering for our ILSVRC-2014 submission). Training was initialised with the corresponding classification models (trained on the same scales), and the initial learning rate was set to 10−3 . We explored both fine-tuning all layers and fine-tuning only the first two fully-connected layers, as done in (Sermanet et al., 2014). The last fully-connected layer was initialised randomly and trained from scratch.

训练本地化ConvNets的训练类似于分类ConvNet的训练(第3.1节)。主要区别在于，我们用欧几里德损失代替了逻辑回归目标，从而惩罚了预测边界框参数与地面真实值的偏差。

Testing. We consider two testing protocols. The first is used for comparing different network modifications on the validation set, and considers only the bounding box prediction for the ground truth class (to factor out the classification errors). The bounding box is obtained by applying the network only to the central crop of the image.

测试。我们考虑两种测试协议。第一个用于比较验证集上的不同网络修改，并仅考虑地面真值类的边界框预测(以排除分类错误)。边界框是通过将网络仅应用于图像的中心裁剪来获得的。

The second, fully-fledged, testing procedure is based on the dense application of the localisation ConvNet to the whole image, similarly to the classification task (Sect. 3.2). The difference is that instead of the class score map, the output of the last fully-connected layer is a set of bounding box predictions. To come up with the final prediction, we utilise the greedy merging procedure of Sermanet et al. (2014), which first merges spatially close predictions (by averaging their coordinates), and then rates them based on the class scores, obtained from the classification ConvNet. When several localisation ConvNets are used, we first take the union of their sets of bounding box predictions, and then run the merging procedure on the union. We did not use the multiple pooling offsets technique of Sermanet et al. (2014), which increases the spatial resolution of the bounding box predictions and can further improve the results.

第二个完全成熟的测试程序基于对整个图像密集应用定位ConvNet，类似于分类任务(第3.2节)。不同之处在于，最后一个全连接层的输出是一组边界框预测，而不是类得分图。为了得出最终预测，我们使用了Sermanetet al (2014)的贪婪合并程序，该程序首先合并空间上相近的预测(通过平均它们的坐标)，然后根据分类ConvNet获得的类分数对它们进行评级。当使用多个本地化ConvNets时，我们首先对它们的边界框预测集进行并集，然后对并集运行合并过程。我们没有使用Sermanetet al (2014)的多池偏移技术，这提高了边界框预测的空间分辨率，并可以进一步改善结果。

#### A.2 LOCALISATION EXPERIMENTS
In this section we first determine the best-performing localisation setting (using the first test protocol), and then evaluate it in a fully-fledged scenario (the second protocol). The localisation error is measured according to the ILSVRC criterion (Russakovsky et al., 2014), i.e. the bounding box prediction is deemed correct if its intersection over union ratio with the ground-truth bounding box is above 0.5.

在本节中，我们首先确定性能最佳的本地化设置(使用第一个测试协议)，然后在完全成熟的场景(第二个协议)中对其进行评估。根据ILSVRC标准(Russakovskyet al., 2014)测量定位误差，即，如果边界框预测与地面真值边界框的交集比大于0.5，则认为边界框预测是正确的。

Settings comparison. As can be seen from Table 8, per-class regression (PCR) outperforms the class-agnostic single-class regression (SCR), which differs from the findings of Sermanet et al. (2014), where PCR was outperformed by SCR. We also note that fine-tuning all layers for the localisation task leads to noticeably better results than fine-tuning only the fully-connected layers (as done in (Sermanet et al., 2014)). In these experiments, the smallest images side was set to S = 384; the results with S = 256 exhibit the same behaviour and are not shown for brevity.

设置比较。从表8中可以看出，每类回归(PCR)优于未知单类回归(SCR)，这与Sermanetet al (2014)的发现不同，后者的PCR优于SCR。我们还注意到，与仅微调完全连接的层相比，微调本地化任务的所有层会产生明显更好的结果(如(Sermanet et al.，2014)中所做)。在这些实验中，最小图像侧被设置为S＝384; S＝256的结果表现出相同的行为，并且为了简洁起见。

Table 8: Localisation error for different modifications with the simplified testing protocol: the bounding box is predicted from a single central image crop, and the ground-truth class is used. All ConvNet layers (except for the last one) have the configuration D (Table 1), while the last layer performs either single-class regression (SCR) or per-class regression (PCR).

表8：使用简化测试协议进行不同修改时的定位误差：从单个中心图像裁剪预测边界框，并使用地面真值类。所有ConvNet层(除了最后一层)都具有配置D(表1)，而最后一层执行单类回归(SCR)或逐类回归(PCR)。

Fully-fledged evaluation. Having determined the best localisation setting (PCR, fine-tuning of all layers), we now apply it in the fully-fledged scenario, where the top-5 class labels are predicted using our best-performing classification system (Sect. 4.5), and multiple densely-computed bounding box predictions are merged using the method of Sermanet et al. (2014). As can be seen from Table 9, application of the localisation ConvNet to the whole image substantially improves the results compared to using a center crop (Table 8), despite using the top-5 predicted class labels instead of the ground truth. Similarly to the classification task (Sect. 4), testing at several scales and combining the predictions of multiple networks further improves the performance.

全面评估。在确定了最佳本地化设置(PCR，所有层的微调)后，我们现在将其应用于完全成熟的场景中，其中使用我们性能最佳的分类系统(第4.5节)预测前5类标签，并使用Sermanetet al (2014)的方法合并多个密集计算的边界框预测。从表9可以看出，与使用中心裁剪(表8)相比，将局部ConvNet应用于整个图像显著改善了结果，尽管使用了前5个预测类标签而不是地面真值。类似于分类任务(第4节)，在多个尺度上进行测试并结合多个网络的预测，可以进一步提高性能。

Table 9: Localisation error. 

Comparison with the state of the art. We compare our best localisation result with the state of the art in Table 10. With 25.3% test error, our “VGG” team won the localisation challenge of ILSVRC-2014 (Russakovsky et al., 2014). Notably, our results are considerably better than those of the ILSVRC-2013 winner Overfeat (Sermanet et al., 2014), even though we used less scales and did not employ their resolution enhancement technique. We envisage that better localisation performance can be achieved if this technique is incorporated into our method. This indicates the performance advancement brought by our very deep ConvNets – we got better results with a simpler localisation method, but a more powerful representation.

与最先进技术进行比较。我们将我们的最佳本地化结果与表10中的最先进技术相比较。我们的“VGG”团队以25.3%的测试误差赢得了ILSVRC-2014的本地化挑战(Russakovskyet al., 2014)。值得注意的是，我们的结果比ILSVRC-2013获奖者Overfeat(Sermanet et al.，2014)的结果要好得多，尽管我们使用了较少的尺度，并且没有使用它们的分辨率增强技术。我们设想，如果将此技术纳入我们的方法，可以实现更好的定位性能。这表明了我们非常深入的ConvNets带来的性能提升——我们使用更简单的本地化方法获得了更好的结果，但使用了更强大的表示。

### B GENERALISATION OF VERY DEEP FEATURES 非常深层特征的概括
In the previous sections we have discussed training and evaluation of very deep ConvNets on the ILSVRC dataset. In this section, we evaluate our ConvNets, pre-trained on ILSVRC, as feature extractors on other, smaller, datasets, where training large models from scratch is not feasible due to over-fitting. Recently, there has been a lot of interest in such a use case (Zeiler & Fergus, 2013; Donahue et al., 2013; Razavian et al., 2014; Chatfield et al., 2014), as it turns out that deep image representations, learnt on ILSVRC, generalise well to other datasets, where they have outperformed hand-crafted representations by a large margin. Following that line of work, we investigate if our models lead to better performance than more shallow models utilised in the state-of-the-art methods.In this evaluation, we consider two models with the best classification performance on ILSVRC (Sect. 4) – configurations “Net-D” and “Net-E” (which we made publicly available).

在前面的章节中，我们讨论了ILSVRC数据集上非常深的ConvNets的训练和评估。在本节中，我们将在ILSVRC上预先训练的ConvNets评估为其他较小数据集上的特征提取器，在这些数据集中，由于过度拟合，从头开始训练大模型是不可行的。最近，人们对这种用例产生了浓厚的兴趣(Zeiler&Fergus，2013; Donahueet al., 2013; Razavianet al., 2014; Chatfieldet al., 2014)，因为事实证明，在ILSVRC上学习的深度图像表示可以很好地推广到其他数据集，在这些数据集中，深度图像表示大大优于手工绘制的表示。按照这一工作路线，我们调查我们的模型是否比最先进方法中使用的更浅的模型带来更好的性能。在本次评估中，我们考虑了ILSVRC(第4节)中分类性能最好的两种模型——配置“Net-D”和“Net-E”(我们公开提供)。

Table 10: Comparison with the state of the art in ILSVRC localisation. Our method is denoted as “VGG”.
表10：与ILSVRC本地化技术水平的比较。我们的方法表示为“VGG”。

To utilise the ConvNets, pre-trained on ILSVRC, for image classification on other datasets, we remove the last fully-connected layer (which performs 1000-way ILSVRC classification), and use 4096-D activations of the penultimate layer as image features, which are aggregated across multiple locations and scales. The resulting image descriptor is L2-normalised and combined with a linear SVM classifier, trained on the target dataset. For simplicity, pre-trained ConvNet weights are kept fixed (no fine-tuning is performed).

为了利用在ILSVRC上预先训练的ConvNets对其他数据集进行图像分类，我们移除最后一个完全连接的层(它执行1000次ILSVRC分类)，并使用倒数第二层的4096-D激活作为图像特征，这些图像特征在多个位置和尺度上聚合。得到的图像描述符被L2归一化并与在目标数据集上训练的线性SVM分类器组合。为了简单起见，预先训练的ConvNet权重保持固定(不进行微调)。

Aggregation of features is carried out in a similar manner to our ILSVRC evaluation procedure (Sect. 3.2). Namely, an image is first rescaled so that its smallest side equals Q, and then the network is densely applied over the image plane (which is possible when all weight layers are treated as convolutional). We then perform global average pooling on the resulting feature map, which produces a 4096-D image descriptor. The descriptor is then averaged with the descriptor of a horizontally flipped image. As was shown in Sect. 4.2, evaluation over multiple scales is beneficial, so we extract features over several scales Q. The resulting multi-scale features can be either stacked or pooled across scales. Stacking allows a subsequent classifier to learn how to optimally combine image statistics over a range of scales; this, however, comes at the cost of the increased descriptor dimensionality. We return to the discussion of this design choice in the experiments below. We also assess late fusion of features, computed using two networks, which is performed by stacking their respective image descriptors.

以与ILSVRC评估程序(第3.2节)类似的方式进行特征聚合。即,首先重新缩放图像，使其最小边等于Q，然后在图像平面上密集应用网络(当所有权重层都被视为卷积时，这是可能的)。然后，我们对得到的特征图执行全局平均池，这将生成4096-D图像描述符。然后将描述符与水平翻转图像的描述符进行平均。如Sect。4.2，在多个尺度上进行评估是有益的，因此我们提取了多个尺度Q上的特征。所得的多尺度特征可以跨尺度叠加或合并。堆叠允许后续分类器学习如何在一定范围内优化组合图像统计; 然而，这是以增加描述符维度为代价的。我们将在下面的实验中重新讨论这个设计选择。我们还评估了使用两个网络计算的特征的后期融合，这是通过堆叠它们各自的图像描述符来实现的。

Table 11: Comparison with the state of the art in image classification on VOC-2007, VOC-2012,Caltech-101, and Caltech-256. Our models are denoted as “VGG”. Results marked with * were achieved using ConvNets pre-trained on the extended ILSVRC dataset (2000 classes).
表11：与VOC-2007、VOC-2012、Caltech-101和Caltech-256图像分类的最新技术进行比较。我们的模型表示为“VGG”。标记为*的结果是使用在扩展的ILSVRC数据集(2000个类)上预先训练的ConvNets实现的。

Image Classification on VOC-2007 and VOC-2012. We begin with the evaluation on the image classification task of PASCAL VOC-2007 and VOC-2012 benchmarks (Everingham et al., 2015). These datasets contain 10K and 22.5K images respectively, and each image is annotated with one or several labels, corresponding to 20 object categories. The VOC organisers provide a pre-defined split into training, validation, and test data (the test data for VOC-2012 is not publicly available; instead, an official evaluation server is provided). Recognition performance is measured using mean average precision (mAP) across classes.

VOC-2007和VOC-2012的图像分类。我们首先对PASCAL VOC-2007与VOC-2012基准的图像分类任务进行评估(Everinghamet al., 2015)。这些数据集分别包含10K和22.5K图像，每个图像都用一个或多个标签标注，对应于20个对象类别。VOC组织者提供预先定义的培训、验证和测试数据(VOC-2012的测试数据不公开，而是提供官方评估服务器)。识别性能是使用跨类平均精度(mAP)来测量的。

Notably, by examining the performance on the validation sets of VOC-2007 and VOC-2012, we found that aggregating image descriptors, computed at multiple scales, by averaging performs similarly to the aggregation by stacking. We hypothesize that this is due to the fact that in the VOC dataset the objects appear over a variety of scales, so there is no particular scale-specific semantics which a classifier could exploit. Since averaging has a benefit of not inflating the descriptor dimensionality, we were able to aggregated image descriptors over a wide range of scales: Q ∈ {256, 384, 512, 640, 768}. It is worth noting though that the improvement over a smaller range of {256, 384, 512} was rather marginal (0.3%).

值得注意的是，通过检查VOC-2007和VOC-2012验证集的性能，我们发现在多个尺度上计算的图像描述符的平均聚合性能与堆叠聚合性能类似。我们假设这是由于在VOC数据集中，对象出现在各种尺度上，因此没有分类器可以利用的特定尺度语义。由于平均化有一个好处，即不会膨胀描述符的维度，因此我们能够在很大范围内聚合图像描述符：Q∈ {256, 384, 512, 640, 768}. 值得注意的是，在较小的{256，384，512}范围内的改进相当小(0.3%)。

The test set performance is reported and compared with other approaches in Table 11. Our networks “Net-D” and “Net-E” exhibit identical performance on VOC datasets, and their combination slightly improves the results. Our methods set the new state of the art across image representations, pretrained on the ILSVRC dataset, outperforming the previous best result of Chatfield et al. (2014) by more than 6%. It should be noted that the method of Wei et al. (2014), which achieves 1% better mAP on VOC-2012, is pre-trained on an extended 2000-class ILSVRC dataset, which includes additional 1000 categories, semantically close to those in VOC datasets. It also benefits from the fusion with an object detection-assisted classification pipeline.

表11中报告了测试集性能，并与其他方法进行了比较。我们的网络“Net-D”和“Net-E”在VOC数据集上表现出相同的性能，它们的组合略微改善了结果。我们的方法在ILSVRC数据集上预处理的图像表示方面开创了新的技术水平，比Chatfieldet al (2014年)之前的最佳结果高出6%以上。应注意的是，Weiet al (2014)的方法在VOC-2012上实现了1%的mAP，该方法在扩展的2000级ILSVRC数据集上进行了预训练，该数据集包括额外的1000个类别，在语义上与VOC数据集中的类别接近。它还受益于与目标检测辅助分类管道的融合。

Image Classification on Caltech-101 and Caltech-256. In this section we evaluate very deep features on Caltech-101 (Fei-Fei et al., 2004) and Caltech-256 (Griffin et al., 2007) image classification benchmarks. Caltech-101 contains 9K images labelled into 102 classes (101 object categories and a background class), while Caltech-256 is larger with 31K images and 257 classes. A standard evaluation protocol on these datasets is to generate several random splits into training and test data and report the average recognition performance across the splits, which is measured by the mean class recall (which compensates for a different number of test images per class). Following Chatfield et al. (2014); Zeiler & Fergus (2013); He et al. (2014), on Caltech-101 we generated 3 random splits into training and test data, so that each split contains 30 training images per class, and up to 50 test images per class. On Caltech-256 we also generated 3 splits, each of which contains 60 training images per class (and the rest is used for testing). In each split, 20% of training images were used as a validation set for hyper-parameter selection.

Caltech-101和Caltech-256上的图像分类。在本节中，我们评估了Caltech-102(Fei Feiet al., 2004)和Caltech 256(Griffinet al., 2007)图像分类基准上的深度特征。Caltech-101包含标记为102类(101个对象类别和一个背景类别)的9K图像，而Caltech-256更大，有31K图像和257类。对这些数据集的标准评估协议是生成训练和测试数据的若干随机分割，并报告分割的平均识别性能，该平均识别性能由平均类召回(它补偿每个类的不同数量的测试图像)测量。遵循Chatfieldet al (2014); Zeiler&Fergus(2013); Heet al (2014年)，在Caltech-101上，我们生成了3个随机分割为训练和测试数据，因此每个分割包含每个类30个训练图像，每个类最多50个测试图像。在Caltech-256上，我们还生成了3个分割，每个分割包含每个类60个训练图像(其余用于测试)。在每次分割中，20%的训练图像被用作超参数选择的验证集。

We found that unlike VOC, on Caltech datasets the stacking of descriptors, computed over multiple scales, performs better than averaging or max-pooling. This can be explained by the fact that in Caltech images objects typically occupy the whole image, so multi-scale image features are semantically different (capturing the whole object vs. object parts), and stacking allows a classifier to exploit such scale-specific representations. We used three scales Q ∈ {256, 384, 512}.

我们发现，与VOC不同的是，在加州理工学院的数据集上，在多个尺度上计算的描述符堆叠比平均或最大池更好。这可以通过以下事实来解释：在加州理工学院的图像中，对象通常占据整个图像，因此多尺度图像特征在语义上是不同的(捕捉整个对象与对象部分)，堆叠允许分类器利用这种特定尺度的表示。我们使用了三个等级Q∈ {256, 384, 512}.

Our models are compared to each other and the state of the art in Table 11. As can be seen, the deeper 19-layer Net-E performs better than the 16-layer Net-D, and their combination further improves the performance. On Caltech-101, our representations are competitive with the approach of He et al. (2014), which, however, performs significantly worse than our nets on VOC-2007. On Caltech-256, our features outperform the state of the art (Chatfield et al., 2014) by a large margin (8.6%).

我们的模型在表11中进行了相互比较和最新技术。可以看出，较深的19层Net-E比16层Net-D性能更好，它们的组合进一步提高了性能。在Caltech-101上，我们的表现与Heet al (2014年)的方法具有竞争力，然而，Heet al 在VOC-2007上的表现明显不如我们的网络。在Caltech-256上，我们在功能上的表现远远优于最先进的技术(Chatfieldet al., 2014年)。

Action Classification on VOC-2012. We also evaluated our best-performing image representation (the stacking of Net-D and Net-E features) on the PASCAL VOC-2012 action classification task (Everingham et al., 2015), which consists in predicting an action class from a single image, given a bounding box of the person performing the action. The dataset contains 4.6K training images, labelled into 11 classes. Similarly to the VOC-2012 object classification task, the performance is measured using the mAP. We considered two training settings: (i) computing the ConvNet features on the whole image and ignoring the provided bounding box; (ii) computing the features on the whole image and on the provided bounding box, and stacking them to obtain the final representation. The results are compared to other approaches in Table 12. Our representation achieves the state of art on the VOC action classification task even without using the provided bounding boxes, and the results are further improved when using both images and bounding boxes. Unlike other approaches, we did not incorporate any task-specific heuristics, but relied on the representation power of very deep convolutional features.

VOC-2012的动作分类。我们还评估了PASCAL VOC-2012动作分类任务(Everinghamet al., 2015)中表现最佳的图像表示(Net-D和Net-E特征的叠加)，该任务包括在给定执行动作的人的边界框的情况下，从单个图像预测动作类别。数据集包含4.6K个训练图像，标记为11个类。与VOC-2012对象分类任务类似，使用mAP测量性能。我们考虑了两种训练设置：(i)计算整个图像上的ConvNet特征并忽略提供的边界框; (ii)计算整个图像和所提供的边界框上的特征并将其叠加以获得最终表示。结果与表12中的其他方法进行了比较。即使不使用提供的边界框，我们的表示也达到了VOC动作分类任务的最新水平，并且当同时使用图像和边界框时，结果得到了进一步改善。与其他方法不同，我们没有结合任何特定任务的启发式，而是依赖于非常深的卷积特征的表示能力。

Other Recognition Tasks. Since the public release of our models, they have been actively used by the research community for a wide range of image recognition tasks, consistently outperforming more shallow representations. For instance, Girshick et al. (2014) achieve the state of the object detection results by replacing the ConvNet of Krizhevsky et al. (2012) with our 16-layer model. Similar gains over a more shallow architecture of Krizhevsky et al. (2012) have been observed in semantic segmentation (Long et al., 2014), image caption generation (Kiros et al., 2014; Karpathy & Fei-Fei, 2014), texture and material recognition (Cimpoi et al., 2014; Bell et al., 2014).

其他识别任务。自我们的模型公开发布以来，研究社区一直在积极地将其用于广泛的图像识别任务，始终优于较浅的表示。例如，Girshicket al (2014年)通过用我们的16层模型替换Krizhevskyet al (2012年)的ConvNet来实现目标检测结果的状态。在语义分割(Longet al., 2014)、图像字幕生成(Kiroset al., 2014; Karpathy和Fei Fei，2014)以及纹理和材料识别(Cimpoiet al., 2014，Bellet al., 2014年)中，也观察到了Krizhevskyet al (2012)更浅架构的类似成果。

Table 12: Comparison with the state of the art in single-image action classification on VOC2012. Our models are denoted as “VGG”. Results marked with * were achieved using ConvNets pre-trained on the extended ILSVRC dataset (1512 classes).
表12:VOC2012单图像动作分类与现有技术的比较。我们的模型表示为“VGG”。标记为*的结果是使用在扩展的ILSVRC数据集(1512个类)上预先训练的ConvNets实现的。

### C PAPER REVISIONS
Here we present the list of major paper revisions, outlining the substantial changes for the convenience of the reader. 
* v1 Initial version. Presents the experiments carried out before the ILSVRC submission. 
* v2 Adds post-submission ILSVRC experiments with training set augmentation using scale jittering, which improves the performance. 
* v3 Adds generalisation experiments (Appendix B) on PASCAL VOC and Caltech image classification datasets. The models used for these experiments are publicly available. 
* v4 The paper is converted to ICLR-2015 submission format. Also adds experiments with multiple crops for classification. 
* v6 Camera-ready ICLR-2015 conference paper. Adds a comparison of the net B with a shallow net and the results on PASCAL VOC action classification benchmark.

在这里，我们列出了主要的论文修订清单，概述了为方便读者所做的实质性修改。
* v1 初始版本。介绍了ILSVRC提交之前进行的实验。
* v2 添加提交后ILSVRC实验，使用缩放抖动增加训练集，从而提高性能。
* v3 增加了PASCAL VOC和加州理工学院图像分类数据集的归纳实验(附录B)。用于这些实验的模型是公开的。
* v4 论文转换为ICLR-2015提交格式。还增加了多种剪裁的分类实验。
* v6 相机就绪ICLR-2015会议论文。添加网B与浅网的比较以及PASCAL VOC动作分类基准的结果。