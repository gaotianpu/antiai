# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
EfficientNet:卷积神经网络模型缩放的再思考 2019-5-28 https://arxiv.org/abs/1905.11946

## 阅读笔记
* [pytorch实现](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)
* 模型缩放

## Abstract
Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.

卷积神经网络(ConvNets)通常是在固定的资源预算下开发的，如果有更多的资源可用，则会进行扩展以获得更好的精度。在本文中，我们系统地研究了模型缩放，并发现仔细平衡网络深度、宽度和分辨率可以获得更好的性能。基于这一观察，我们提出了一种新的缩放方法，使用简单但高效的复合系数均匀缩放深度/宽度/分辨率的所有维度。我们证明了这种方法在扩展MobileNets和ResNet上的有效性。

To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing
ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at https: //github.com/tensorflow/tpu/tree/ master/models/official/efficientnet.

为了更进一步，我们使用神经架构搜索来设计一个新的基线网络，并将其放大以获得一系列模型，称为EfficientNets，它们比以前的ConvNets实现了更好的准确性和效率。特别是，我们的EfficientNet-B7在ImageNet上实现了最先进的84.3%的顶级精度，同时比现有最佳ConvNet小8.4倍，推理速度快6.1倍。我们的EfficientNets在CIFAR-100(91.7%)、Flowers(98.8%)和其他3个传输学习数据集上也传输良好，并达到了最先进的精度，参数数量级更少。源代码位于https://github。com/tensorflow/tpu/tree/master/models/official/efficientnet。

## 1. Introduction
Scaling up ConvNets is widely used to achieve better accuracy. For example, ResNet (He et al., 2016) can be scaled up from ResNet-18 to ResNet-200 by using more layers; Recently, GPipe (Huang et al., 2018) achieved 84.3% ImageNet top-1 accuracy by scaling up a baseline model four time larger. However, the process of scaling up ConvNets has never been well understood and there are currently many ways to do it. The most common way is to scale up ConvNets by their depth (He et al., 2016) or width (Zagoruyko & Komodakis, 2016). Another less common, but increasingly popular, method is to scale up models by image resolution (Huang et al., 2018). In previous work, it is common to scale only one of the three dimensions – depth, width, and image size. Though it is possible to scale two or three dimensions arbitrarily, arbitrary scaling requires tedious manual tuning and still often yields sub-optimal accuracy and efficiency.

缩放ConvNets被广泛用于实现更好的精度。例如，通过使用更多层，ResNet(Heet al., 2016)可以从ResNet-18扩展到ResNet-200; 最近，GPipe(Huang et al.，2018)通过将基线模型放大四倍，实现了84.3%的ImageNet顶级精度。然而，扩展ConvNets的过程从未被很好地理解，目前有很多方法可以做到这一点。最常见的方法是通过深度(He et al.，2016)或宽度(Zagoruyko&Komodakis，2016)扩大ConvNets。另一种不太常见但越来越流行的方法是通过图像分辨率放大模型(Huanget al., 2018)。在以前的工作中，通常只缩放三个维度中的一个 —— 深度、宽度和图像大小。尽管可以任意缩放二维或三维，但任意缩放需要繁琐的手动调整，并且通常会产生次优的精度和效率。

Figure 1. Model Size vs. ImageNet Accuracy. All numbers are for single-crop, single-model. Our EfficientNets significantly outperform other ConvNets. In particular, EfficientNet-B7 achieves new state-of-the-art 84.3% top-1 accuracy but being 8.4x smaller and 6.1x faster than GPipe. EfficientNet-B1 is 7.6x smaller and 5.7x faster than ResNet-152. Details are in Table 2 and 4.
图1.模型尺寸与ImageNet精度。所有数字均为单一剪裁、单一模型。我们的EfficientNets显著优于其他ConvNets。特别是，EfficientNet-B7实现了最先进的84.3%的顶级精度，但比GPipe小8.4倍，快6.1倍。EfficientNet-B1比ResNet-152小7.6倍，快5.7倍。详情见表2和表4。

In this paper, we want to study and rethink the process of scaling up ConvNets. In particular, we investigate the central question: is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency? Our empirical study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio. Based on this observation, we propose a simple yet effective compound scaling method. Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. For example, if we want to use 2N times more computational resources, then we can simply increase the network depth by αN , width by βN , and image size by γN , where α, β, γ are constant coefficients determined by a small grid search on the original small model. Figure 2 illustrates the difference between our scaling method and conventional methods.

在本文中，我们想研究并重新思考ConvNets的扩展过程。特别是，我们调查了一个核心问题：是否有一种原则性的方法来扩大ConvNets，以实现更好的准确性和效率？我们的实证研究表明，平衡网络宽度/深度/分辨率的所有维度是至关重要的，令人惊讶的是，通过简单地以恒定比例缩放每个维度，可以实现这种平衡。基于这一观察，我们提出了一种简单而有效的复合缩放方法。与任意缩放这些因素的传统实践不同，我们的方法使用一组固定的缩放系数均匀缩放网络宽度、深度和分辨率。例如，如果我们想使用2N倍的计算资源，那么我们可以简单地将网络深度增加αN，宽度增加βN，图像大小增加γN，其中α、β、γ是由原始小模型上的小网格搜索确定的常数。图2说明了我们的缩放方法和传统方法之间的区别。

Figure 2. Model Scaling. (a) is a baseline network example; (b)-(d) are conventional scaling that only increases one dimension of network width, depth, or resolution. (e) is our proposed compound scaling method that uniformly scales all three dimensions with a fixed ratio. 
图2.模型缩放。(a) 是基线网络样本; (b) -(d)是仅增加网络宽度、深度或分辨率一维的常规缩放。(e) 是我们提出的以固定比例均匀缩放所有三维的复合缩放方法。

Intuitively, the compound scaling method makes sense because if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image. In fact, previous theoretical (Raghu et al., 2017; Lu et al., 2018) and empirical results (Zagoruyko & Komodakis, 2016) both show that there exists certain relationship between network width and depth, but to our best knowledge, we are the first to empirically quantify the relationship among all three dimensions of network width, depth, and resolution.

直观地说，复合缩放方法是有意义的，因为如果输入图像更大，那么网络需要更多的层来增加感受野，需要更多的通道来在更大的图像上捕获更精细的图案。事实上，先前的理论(Raghuet al., 2017; Luet al., 2018)和实证结果(Zagoruyko&Komodakis，2016)都表明网络宽度和深度之间存在一定的关系，但据我们所知，我们是第一个实证量化网络宽度、深度和分辨率这三个维度之间关系的人。

We demonstrate that our scaling method work well on existing MobileNets (Howard et al., 2017; Sandler et al., 2018) and ResNet (He et al., 2016). Notably, the effectiveness of model scaling heavily depends on the baseline network; to go even further, we use neural architecture search (Zoph & Le, 2017; Tan et al., 2019) to develop a new baseline network, and scale it up to obtain a family of models, called EfficientNets. Figure 1 summarizes the ImageNet performance, where our EfficientNets significantly outperform other ConvNets. In particular, our EfficientNet-B7 surpasses the best existing GPipe accuracy (Huang et al., 2018), but using 8.4x fewer parameters and running 6.1x faster on inference. Compared to the widely used ResNet-50 (He et al., 2016), our EfficientNet-B4 improves the top-1 accuracy from 76.3% to 83.0% (+6.7%) with similar FLOPS. Besides ImageNet, EfficientNets also transfer well and achieve stateof-the-art accuracy on 5 out of 8 widely used datasets, while reducing parameters by up to 21x than existing ConvNets.

我们证明，我们的缩放方法在现有的MobileNets(Howardet al., 2017; Sandleret al., 2018)和ResNet(Heet al., 2016)上运行良好。值得注意的是，模型缩放的有效性在很大程度上取决于基线网络; 为了更进一步，我们使用神经架构搜索(Zoph&Le，2017; Tanet al., 2019)开发了一个新的基线网络，并将其放大以获得一系列模型，称为EfficientNets。图1总结了ImageNet的性能，其中我们的EfficientNets显著优于其他ConvNets。特别是，我们的EfficientNet-B7超过了现有的最佳GPipe精度(Huanget al., 2018)，但使用的参数少了8.4倍，推理速度快了6.1倍。与广泛使用的ResNet-50(He et al.，2016)相比，我们的EfficientNet-B4使用类似的FLOPS将顶级精度从76.3%提高到83.0%(+6.7%)。除了ImageNet，EfficientNets还可以在8个广泛使用的数据集中的5个上传输良好并达到最先进的精度，同时比现有的ConvNets减少了多达21倍的参数。

## 2. Related Work
ConvNet Accuracy: Since AlexNet (Krizhevsky et al., 2012) won the 2012 ImageNet competition, ConvNets have become increasingly more accurate by going bigger: while the 2014 ImageNet winner GoogleNet (Szegedy et al., 2015) achieves 74.8% top-1 accuracy with about 6.8M parameters, the 2017 ImageNet winner SENet (Hu et al., 2018) achieves 82.7% top-1 accuracy with 145M parameters. Recently,
GPipe (Huang et al., 2018) further pushes the state-of-the-art ImageNet top-1 validation accuracy to 84.3% using 557M parameters: it is so big that it can only be trained with a specialized pipeline parallelism library by partitioning the network and spreading each part to a different accelerator. While these models are mainly designed for ImageNet, recent studies have shown better ImageNet models also perform better across a variety of transfer learning datasets (Kornblith et al., 2019), and other computer vision tasks such as object detection (He et al., 2016; Tan et al., 2019). Although higher accuracy is critical for many applications, we have already hit the hardware memory limit, and thus further accuracy gain needs better efficiency.

ConvNet精度：自从AlexNet(Krizhevskyet al., 2012年)赢得2012年ImageNet比赛以来，ConvNets通过扩大规模变得越来越准确：2014年ImageNetwork获奖者GoogleNet(Szegedyet al., 2015年)以约68M个参数实现了74.8%的顶级准确度，2017年ImageNet获奖者SENet(Huet al., 2018年)以145M个参数达到了82.7%的顶级精度。不久前GPipe(Huang et al.，2018)使用557M参数进一步将最先进的ImageNet top 1验证精度提高到84.3%：它太大了，只能通过划分网络并将每个部分扩展到不同的加速器来使用专用的管道并行库进行训练。尽管这些模型主要是为ImageNet设计的，但最近的研究表明，更好的ImageNet模型在各种迁移学习数据集(Kornblithet al., 2019)和其他计算机视觉任务(如对象检测)中也表现得更好(Heet al., 2016; Tanet al., 201)。尽管更高的精度对于许多应用来说至关重要，但我们已经达到了硬件内存限制，因此进一步提高精度需要更好的效率。

ConvNet Efficiency: Deep ConvNets are often overparameterized. Model compression (Han et al., 2016; He et al., 2018; Yang et al., 2018) is a common way to reduce model size by trading accuracy for efficiency. As mobile phones become ubiquitous, it is also common to handcraft efficient mobile-size ConvNets, such as SqueezeNets (Iandola et al., 2016; Gholami et al., 2018), MobileNets (Howard et al., 2017; Sandler et al., 2018), and ShuffleNets(Zhang et al., 2018; Ma et al., 2018). Recently, neural architecture search becomes increasingly popular in designing efficient mobile-size ConvNets (Tan et al., 2019; Cai et al., 2019), and achieves even better efficiency than hand-crafted mobile ConvNets by extensively tuning the network width, depth, convolution kernel types and sizes. However, it is unclear how to apply these techniques for larger models that have much larger design space and much more expensive tuning cost. In this paper, we aim to study model efficiency for super large ConvNets that surpass state-of-the-art accuracy. To achieve this goal, we resort to model scaling.

ConvNet效率：深度ConvNets通常被过度参数化。模型压缩(Hanet al., 2016; Heet al., 2018; Yanget al., 2018)是通过以精度换取效率来减小模型大小的常见方法。随着手机变得无处不在，手工制作高效的移动尺寸ConvNets也很常见，例如SqueezeNets(Iandolaet al., 2016; Gholmaet al., 2018)、MobileNets(Howardet al., 2017; Sandleret al., 2018年)和ShuffleNets(Zhanget al., 2018; Maet al., 2018)。最近，神经架构搜索在设计高效移动大小的ConvNets中越来越流行(Tanet al., 2019; Caiet al., 201)，并通过广泛调整网络宽度、深度、卷积内核类型和大小，实现了比手工制作的移动ConvNet更好的效率。然而，目前尚不清楚如何将这些技术应用于具有更大设计空间和更昂贵调整成本的更大模型。在本文中，我们旨在研究超大型ConvNets的模型效率，该模型的精度超过了最先进水平。为了实现这个目标，我们求助于模型缩放。

Model Scaling: There are many ways to scale a ConvNet for different resource constraints: ResNet (He et al., 2016) can be scaled down (e.g., ResNet-18) or up (e.g.,ResNet-200) by adjusting network depth (#layers), while WideResNet (Zagoruyko & Komodakis, 2016) and MobileNets (Howard et al., 2017) can be scaled by network width (#channels). It is also well-recognized that bigger input image size will help accuracy with the overhead of more FLOPS. Although prior studies (Raghu et al., 2017; Lin & Jegelka, 2018; Sharir & Shashua, 2018; Lu et al., 2018) have shown that network depth and width are both important for ConvNets’ expressive power, it still remains an open question of how to effectively scale a ConvNet to achieve better efficiency and accuracy. Our work systematically and empirically studies ConvNet scaling for all three dimensions of network width, depth, and resolutions.

模型缩放：有多种方法可以针对不同的资源限制来缩放ConvNet：ResNet(Heet al., 2016)可以通过调整网络深度(#层)来缩小(例如，ResNet-18)或增大(例如，ResNet-200)，而WideResNet(Zagoruyko&Komodakis，2016)和MobileNets(Howardet al., 2017)可以通过网络宽度(#信道)来缩放。人们还清楚地认识到，更大的输入图像尺寸将有助于提高准确性，同时增加FLOPS的开销。尽管先前的研究(Raghuet al., 2017; Lin&Jegelka，2018; Sharir&Shashua，2018; Luet al., 2018)表明，网络深度和宽度对ConvNets的表达能力都很重要，但如何有效地扩展ConvNet以实现更好的效率和准确性仍然是一个悬而未决的问题。我们的工作系统和经验地研究了网络宽度、深度和分辨率三个维度的ConvNet缩放。

## 3. Compound Model Scaling
In this section, we will formulate the scaling problem, study different approaches, and propose our new scaling method.

### 3.1. Problem Formulation
A ConvNet Layer i can be defined as a function: Yi = Fi(Xi), where Fi is the operator, Yi is output tensor, Xi is input tensor, with tensor shape h Hi , Wi , Cii 1 , where Hi and Wi are spatial dimension and Ci is the channel dimension. A ConvNet N can be represented by a list of composed layers: N = Fk  ...  F2  F1(X1) = J j=1...k Fj (X1). In practice, ConvNet layers are often partitioned into multiple stages and all layers in each stage share the same architecture: for example, ResNet (He et al., 2016) has five stages, and all layers in each stage has the same convolutional type except the first layer performs down-sampling. Therefore, we can define a ConvNet as:

N = K i=1...s FiLi  Xh Hi,Wi,Cii  (1) 

where FiLi denotes layer Fi is repeated Li times in stage i, h Hi , Wi , Cii denotes the shape of input tensor X of layer i. Figure 2(a) illustrate a representative ConvNet, where the spatial dimension is gradually shrunk but the channel dimension is expanded over layers, for example, from initial input shape h 224, 224, 3i to final output shape h 7, 7, 512i .

1For the sake of simplicity, we omit batch dimension. 

Unlike regular ConvNet designs that mostly focus on finding the best layer architecture Fi , model scaling tries to expand the network length (Li), width (Ci), and/or resolution (Hi , Wi) without changing Fi predefined in the baseline network. By fixing Fi , model scaling simplifies the design problem for new resource constraints, but it still remains a large design space to explore different Li , Ci , Hi , Wi for each layer. In order to further reduce the design space, we restrict that all layers must be scaled uniformly with constant ratio. Our target is to maximize the model accuracy for any given resource constraints, which can be formulated as an optimization problem: 

max d,w,r Accuracy N (d, w, r) 

s.t. N (d, w, r) = K i=1...s Fˆd·ˆLi i  Xh r· ˆHi,r· ˆWi,w·Cˆii 

Memory(N ) ≤ target memory

FLOPS(N ) ≤ target flops (2) 

where w, d, r are coefficients for scaling network width, depth, and resolution; Fˆi, ˆLi, ˆHi, ˆWi, Cˆi are predefined parameters in baseline network (see Table 1 as an example).

### 3.2. Scaling Dimensions 缩放维度
Figure 3. Scaling Up a Baseline Model with Different Network Width (w), Depth (d), and Resolution (r) Coefficients. Bigger networks with larger width, depth, or resolution tend to achieve higher accuracy, but the accuracy gain quickly saturate after reaching 80%, demonstrating the limitation of single dimension scaling. Baseline network is described in Table 1.
图3.用不同的网络宽度(w)、深度(d)和分辨率(r)系数放大基线模型。具有更大宽度、深度或分辨率的更大网络往往实现更高的精度，但精度增益在达到80%后迅速饱和，这表明了一维缩放的局限性。基线网络如表1所示。

The main difficulty of problem 2 is that the optimal d, w, r depend on each other and the values change under different resource constraints. Due to this difficulty, conventional methods mostly scale ConvNets in one of these dimensions:

问题2的主要困难在于，最优d、w、r相互依赖，并且值在不同的资源约束下发生变化。由于这一困难，传统方法大多在以下维度之一对ConvNets进行缩放：

Depth (d): Scaling network depth is the most common way used by many ConvNets (He et al., 2016; Huang et al., 2017; Szegedy et al., 2015; 2016). The intuition is that deeper ConvNet can capture richer and more complex features, and generalize well on new tasks. However, deeper networks are also more difficult to train due to the vanishing gradient problem (Zagoruyko & Komodakis, 2016). Although several techniques, such as skip connections (He et al., 2016) and batch normalization (Ioffe & Szegedy, 2015), alleviate the training problem, the accuracy gain of very deep network diminishes: for example, ResNet-1000 has similar accuracy as ResNet-101 even though it has much more layers. Figure 3 (middle) shows our empirical study on scaling a baseline model with different depth coefficient d, further suggesting the diminishing accuracy return for very deep ConvNets.

深度(d)：缩放网络深度是许多ConvNets最常用的方式(Heet al., 2016; Huanget al., 2017; Szegedyet al., 2015; 2016)。直觉是，更深层次的ConvNet可以捕获更丰富、更复杂的特征，并能很好地概括新任务。然而，由于梯度消失问题，更深的网络也更难训练(Zagoruyko&Komodakis，2016)。尽管一些技术，如跳跃连接(He et al.，2016)和批归一化(Ioffe&Szegedy，2015)可以缓解训练问题，但深度网络的精度增益会降低：例如，ResNet-1000具有与ResNet-101相似的精度，尽管它有更多的层。图3(中间)显示了我们对具有不同深度系数d的基线模型进行缩放的实证研究，进一步表明了非常深的ConvNets的精度回报递减。

Width (w): Scaling network width is commonly used for small size models (Howard et al., 2017; Sandler et al., 2018; Tan et al., 2019)(2 In some literature, scaling number of channels is called “depth multiplier”, which means the same as our width coefficient w. ) . As discussed in (Zagoruyko & Komodakis, 2016), wider networks tend to be able to capture more fine-grained features and are easier to train. However, extremely wide but shallow networks tend to have difficulties in capturing higher level features. Our empirical results in Figure 3 (left) show that the accuracy quickly saturates when networks become much wider with larger w.

宽度(w)：缩放网络宽度通常用于小尺寸模型(Howardet al., 2017; Sandleret al., 2018; Tanet al., 2019(在一些文献中，通道的缩放数被称为“深度乘数”，这意味着与我们的宽度系数w相同。). 如(Zagoruyko&Komodakis，2016)所述，更宽的网络往往能够捕获更细粒度的特征，并且更容易训练。然而，极宽但浅的网络往往难以捕获更高级别的特征。我们在图3(左)中的经验结果表明，当网络变得更宽且w更大时，精度很快饱和。

Resolution (r): With higher resolution input images, ConvNets can potentially capture more fine-grained patterns. Starting from 224x224 in early ConvNets, modern ConvNets tend to use 299x299 (Szegedy et al., 2016) or 331x331 (Zoph et al., 2018) for better accuracy. Recently, GPipe (Huang et al., 2018) achieves state-of-the-art ImageNet accuracy with 480x480 resolution. Higher resolutions, such as 600x600, are also widely used in object detection ConvNets (He et al., 2017; Lin et al., 2017). Figure 3 (right) shows the results of scaling network resolutions, where indeed higher resolutions improve accuracy, but the accuracy gain diminishes for very high resolutions (r = 1.0 denotes resolution 224x224 and r = 2.5 denotes resolution 560x560).

分辨率(r)：使用更高分辨率的输入图像，ConvNets可以潜在地捕获更细粒度的模式。从早期ConvNets的224x224开始，现代ConvNet倾向于使用299x299(Szegedyet al., 2016)或331x331(Zophet al., 2018)以获得更好的准确性。最近，GPipe(Huanget al., 2018)以480x480分辨率实现了最先进的ImageNet精度。更高的分辨率，如600x600，也广泛用于对象检测ConvNets(Heet al., 2017; Linet al., 2018)。图3(右)显示了缩放网络分辨率的结果，其中更高的分辨率确实提高了精度，但对于非常高的分辨率，精度增益会降低(r=1.0表示分辨率224x224，r=2.5表示分辨率560x560)。

The above analyses lead us to the first observation:
Observation 1 – Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.

以上分析使我们得出第一个观察结果：
观察1–放大网络宽度、深度或分辨率的任何维度都可以提高精度，但对于更大的模型，精度增益会降低。

### 3.3. Compound Scaling 复合缩放
We empirically observe that different scaling dimensions are not independent. Intuitively, for higher resolution images, we should increase network depth, such that the larger receptive fields can help capture similar features that include more pixels in bigger images. Correspondingly, we should also increase network width when resolution is higher, in order to capture more fine-grained patterns with more pixels in high resolution images. These intuitions suggest that we need to coordinate and balance different scaling dimensions rather than conventional single-dimension scaling.

我们根据经验观察到，不同的缩放维度不是独立的。直观地说，对于更高分辨率的图像，我们应该增加网络深度，这样更大的感受野可以帮助捕捉类似的特征，包括更大图像中的更多像素。相应地，当分辨率更高时，我们也应该增加网络宽度，以便在高分辨率图像中捕获具有更多像素的更细粒度图案。这些直觉表明，我们需要协调和平衡不同的缩放维度，而不是传统的一维缩放。

Figure 4. Scaling Network Width for Different Baseline Networks. Each dot in a line denotes a model with different width coefficient (w). All baseline networks are from Table 1. The first baseline network (d=1.0, r=1.0) has 18 convolutional layers with resolution 224x224, while the last baseline (d=2.0, r=1.3) has 36 layers with resolution 299x299. 
图4.不同基线网络的缩放网络宽度。一行中的每个点表示具有不同宽度系数(w)的模型。所有基线网络均来自表1。第一个基线网络(d=1.0，r=1.0)具有18个卷积层，分辨率为224x224，而最后一个基线(d=2.0，r=1.3)具有36个层，分辨率299x299。

To validate our intuitions, we compare width scaling under different network depths and resolutions, as shown in Figure 4. If we only scale network width w without changing depth (d=1.0) and resolution (r=1.0), the accuracy saturates quickly. With deeper (d=2.0) and higher resolution (r=2.0), width scaling achieves much better accuracy under the same FLOPS cost. These results lead us to the second observation:

为了验证我们的直觉，我们比较了不同网络深度和分辨率下的宽度缩放，如图4所示。如果我们只缩放网络宽度w而不改变深度(d=1.0)和分辨率(r=1.0)，则精度会很快饱和。随着深度(d=2.0)和分辨率(r=2.0)的提高，宽度缩放在相同的FLOPS成本下实现了更好的精度。这些结果使我们得出第二个观察结果：

Observation 2 – In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.

观察2——为了追求更好的准确性和效率，在ConvNet缩放过程中平衡网络宽度、深度和分辨率的所有维度至关重要。

In fact, a few prior work (Zoph et al., 2018; Real et al., 2019) have already tried to arbitrarily balance network width and depth, but they all require tedious manual tuning.

事实上，一些先前的工作(Zophet al., 2018; Realet al., 2019)已经试图任意平衡网络宽度和深度，但它们都需要繁琐的手动调整。

In this paper, we propose a new compound scaling method, which use a compound coefficient φ to uniformly scales network width, depth, and resolution in a principled way: 

在本文中，我们提出了一种新的复合缩放方法，该方法使用复合系数φ以原则性的方式均匀缩放网络宽度、深度和分辨率：

depth: d = αφ 

width: w = βφ 

resolution: r = γφ 

s.t. α · β2 · γ2 ≈ 2 α ≥ 1, β ≥ 1, γ ≥ 1 (3) 

where α, β, γ are constants that can be determined by a small grid search. Intuitively, φ is a user-specified coeffi- cient that controls how many more resources are available for model scaling, while α, β, γ specify how to assign these extra resources to network width, depth, and resolution respectively. Notably, the FLOPS of a regular convolution op is proportional to d, w2, r2 , i.e., doubling network depth will double FLOPS, but doubling network width or resolution will increase FLOPS by four times. Since convolution ops usually dominate the computation cost in ConvNets, scaling a ConvNet with equation 3 will approximately increase total FLOPS by  α · β2 · γ2 φ . In this paper, we constraint α · β2 · γ2 ≈ 2 such that for any new φ, the total FLOPS will approximately(3 FLOPS may differ from theoretical value due to rounding.) increase by 2φ.

其中α、β、γ是可以通过小网格搜索确定的常数。直观地说，φ是一个用户指定的系数，用于控制可用于模型缩放的更多资源，而α、β、γ分别指定如何将这些额外资源分配给网络宽度、深度和分辨率。值得注意的是，规则卷积运算的FLOPS与d、w2、r2成比例，即，网络深度加倍将使FLOPS加倍，但网络宽度或分辨率加倍将使FROPS增加四倍。由于卷积运算通常支配ConvNets的计算成本，使用等式3缩放ConvNet将使总FLOPS大约增加α·β2·γ2φ。在本文中，我们约束α·β2·γ2≈ 使得对于任何新的φ，总FLOPS将大约增加2φ(由于四舍五入，FLOPS可能与理论值不同)。

## 4. EfficientNet Architecture
Since model scaling does not change layer operators Fˆi in baseline network, having a good baseline network is also critical. We will evaluate our scaling method using existing ConvNets, but in order to better demonstrate the effectiveness of our scaling method, we have also developed a new mobile-size baseline, called EfficientNet.

由于模型缩放不会改变基线网络中的层运算符Fˆi，因此具有良好的基线网络也是至关重要的。我们将使用现有的ConvNets评估我们的缩放方法，但为了更好地证明缩放方法的有效性，我们还开发了一个新的移动大小基线，称为EfficientNet。

Inspired by (Tan et al., 2019), we develop our baseline network by leveraging a multi-objective neural architecture search that optimizes both accuracy and FLOPS. Specifi- cally, we use the same search space as (Tan et al., 2019), and use ACC(m)×[F LOP S(m)/T]w as the optimization goal, where ACC(m) and F LOP S(m) denote the accuracy and FLOPS of model m, T is the target FLOPS and w=-0.07 is a hyperparameter for controlling the trade-off between accuracy and FLOPS. Unlike (Tan et al., 2019; Cai et al., 2019), here we optimize FLOPS rather than latency since we are not targeting any specific hardware device. Our search produces an efficient network, which we name EfficientNet-B0. Since we use the same search space as (Tan et al., 2019), the architecture is similar to MnasNet, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M). Table 1 shows the architecture of EfficientNet-B0. Its main building block is mobile inverted bottleneck MBConv (Sandler et al., 2018; Tan et al., 2019), to which we also add squeeze-and-excitation optimization (Hu et al., 2018).

受(Tanet al., 2019)启发，我们利用多目标神经架构搜索来开发基线网络，该搜索优化了准确性和FLOPS。具体而言，我们使用与(Tanet al., 2019)相同的搜索空间，并使用ACC(m)×[F LOP S(m)/T]w作为优化目标，其中ACC(m)和F LOP S(m)表示模型m的精度和FLOPS，T是目标FLOPS并且w=-0.07是用于控制精度和FLOP之间的权衡的超参数。与(Tanet al., 2019; Caiet al., 2018)不同，这里我们优化了FLOPS而不是延迟，因为我们没有针对任何特定的硬件设备。我们的搜索产生了一个高效的网络，我们将其命名为EfficientNet-B0。由于我们使用与(Tan et al.，2019)相同的搜索空间，所以架构与MnasNet类似，只是我们的EfficientNet-B由于FLOPS目标更大(我们的FLOPS目标为400M)而稍大。表1显示了EfficientNet-B0的架构。它的主要构建块是移动反向瓶颈MBConv(Sandleret al., 2018; Tanet al., 2019)，我们还添加了挤压和激励优化(Huet al., 2018)。

Table 1. EfficientNet-B0 baseline network – Each row describes a stage i with ˆLi layers, with input resolution h ˆHi, ˆWii and output channels Cˆi. Notations are adopted from equation 2.
表1.EfficientNet-B0基线网络–每行描述了一个具有ˆLi层的阶段i，输入分辨率为h \710 Hi、\710; Wii和输出通道为Cᮼi。符号采用等式2。

Starting from the baseline EfficientNet-B0, we apply our compound scaling method to scale it up with two steps:
* STEP 1: we first fix φ = 1, assuming twice more resources available, and do a small grid search of α, β, γ based on Equation 2 and 3. In particular, we find the best values for EfficientNet-B0 are α = 1.2, β = 1.1, γ = 1.15, under constraint of α · β2 · γ2 ≈ 2. 
* STEP 2: we then fix α, β, γ as constants and scale up baseline network with different φ using Equation 3, to obtain EfficientNet-B1 to B7 (Details in Table 2).

从基线EfficientNet-B0开始，我们应用我们的复合缩放方法，通过两个步骤进行缩放：
* 步骤1：我们首先确定φ=1，假设可用资源增加两倍，并根据等式2和3对α、β、γ进行小网格搜索。特别是，在α·β2·γ2的约束下，我们发现EfficientNet-B0的最佳值为α=1.2、β=1.1、γ=1.15≈ 2.
* 步骤2：然后我们将α、β、γ固定为常数，并使用等式3将不同φ的基线网络放大，以获得EfficientNet-B1至B7(详情见表2)。

Notably, it is possible to achieve even better performance by searching for α, β, γ directly around a large model, but the search cost becomes prohibitively more expensive on larger models. Our method solves this issue by only doing search once on the small baseline network (step 1), and then use the same scaling coefficients for all other models (step 2).

值得注意的是，通过直接在大型模型周围搜索α、β、γ，可以获得更好的性能，但在大型模型上搜索成本变得昂贵得令人望而却步。我们的方法解决了这个问题，只在小基线网络上搜索一次(步骤1)，然后对所有其他模型使用相同的缩放系数(步骤2)。

## 5. Experiments
In this section, we will first evaluate our scaling method on existing ConvNets and the new proposed EfficientNets.

### 5.1. Scaling Up MobileNets and ResNets
As a proof of concept, we first apply our scaling method to the widely-used MobileNets (Howard et al., 2017; Sandler et al., 2018) and ResNet (He et al., 2016). Table 3 shows the ImageNet results of scaling them in different ways. Compared to other single-dimension scaling methods, our compound scaling method improves the accuracy on all these models, suggesting the effectiveness of our proposed scaling method for general existing ConvNets.

Table 2. EfficientNet Performance Results on ImageNet (Russakovsky et al., 2015). All EfficientNet models are scaled from our baseline EfficientNet-B0 using different compound coefficient φ in Equation 3. ConvNets with similar top-1/top-5 accuracy are grouped together for efficiency comparison. Our scaled EfficientNet models consistently reduce parameters and FLOPS by an order of magnitude (up to 8.4x parameter reduction and up to 16x FLOPS reduction) than existing ConvNets.

 
Table 3. Scaling Up MobileNets and ResNet.
 

Table 4. Inference Latency Comparison – Latency is measured with batch size 1 on a single core of Intel Xeon CPU E5-2690.


Figure 5. FLOPS vs. ImageNet Accuracy – Similar to Figure 1 except it compares FLOPS rather than model size.

### 5.2. ImageNet Results for EfficientNet
We train our EfficientNet models on ImageNet using similar settings as (Tan et al., 2019): RMSProp optimizer with decay 0.9 and momentum 0.9; batch norm momentum 0.99;


Table 5. EfficientNet Performance Results on Transfer Learning Datasets. Our scaled EfficientNet models achieve new state-of-theart accuracy for 5 out of 8 datasets, with 9.6x fewer parameters on average.


Figure 6. Model Parameters vs. Transfer Learning Accuracy – All models are pretrained on ImageNet and finetuned on new datasets. weight decay 1e-5; initial learning rate 0.256 that decays by 0.97 every 2.4 epochs. We also use SiLU (Swish-1) activation (Ramachandran et al., 2018; Elfwing et al., 2018; Hendrycks & Gimpel, 2016), AutoAugment (Cubuk et al., 2019), and stochastic depth (Huang et al., 2016) with survival probability 0.8. As commonly known that bigger models need more regularization, we linearly increase dropout (Srivastava et al., 2014) ratio from 0.2 for EfficientNet-B0 to 0.5 for B7. We reserve 25K randomly picked images from the training set as a minival set, and perform early stopping on this minival; we then evaluate the earlystopped checkpoint on the original validation set to report the final validation accuracy.

Table 2 shows the performance of all EfficientNet models that are scaled from the same baseline EfficientNet-B0. Our EfficientNet models generally use an order of magnitude fewer parameters and FLOPS than other ConvNets with similar accuracy. In particular, our EfficientNet-B7 achieves 84.3% top1 accuracy with 66M parameters and 37B FLOPS, being more accurate but 8.4x smaller than the previous best GPipe (Huang et al., 2018). These gains come from both better architectures, better scaling, and better training settings that are customized for EfficientNet.

Figure 1 and Figure 5 illustrates the parameters-accuracy and FLOPS-accuracy curve for representative ConvNets, where our scaled EfficientNet models achieve better accuracy with much fewer parameters and FLOPS than other ConvNets. Notably, our EfficientNet models are not only small, but also computational cheaper. For example, our EfficientNet-B3 achieves higher accuracy than ResNeXt- 101 (Xie et al., 2017) using 18x fewer FLOPS.

To validate the latency, we have also measured the inference latency for a few representative CovNets on a real CPU as shown in Table 4, where we report average latency over 20 runs. Our EfficientNet-B1 runs 5.7x faster than the widely used ResNet-152, while EfficientNet-B7 runs about 6.1x faster than GPipe (Huang et al., 2018), suggesting our EfficientNets are indeed fast on real hardware.


Figure 7. Class Activation Map (CAM) (Zhou et al., 2016) for Models with different scaling methods- Our compound scaling method allows the scaled model (last column) to focus on more relevant regions with more object details. Model details are in Table 7.

Table 6. Transfer Learning Datasets.


### 5.3. Transfer Learning Results for EfficientNet
We have also evaluated our EfficientNet on a list of commonly used transfer learning datasets, as shown in Table 6. We borrow the same training settings from (Kornblith et al., 2019) and (Huang et al., 2018), which take ImageNet pretrained checkpoints and finetune on new datasets.

Table 5 shows the transfer learning performance: (1) Compared to public available models, such as NASNet-A (Zoph et al., 2018) and Inception-v4 (Szegedy et al., 2017), our Ef- ficientNet models achieve better accuracy with 4.7x average (up to 21x) parameter reduction. (2) Compared to stateof-the-art models, including DAT (Ngiam et al., 2018) that dynamically synthesizes training data and GPipe (Huang et al., 2018) that is trained with specialized pipeline parallelism, our EfficientNet models still surpass their accuracy in 5 out of 8 datasets, but using 9.6x fewer parameters

Figure 6 compares the accuracy-parameters curve for a variety of models. In general, our EfficientNets consistently achieve better accuracy with an order of magnitude fewer parameters than existing models, including ResNet (He et al., 2016), DenseNet (Huang et al., 2017), Inception (Szegedy et al., 2017), and NASNet (Zoph et al., 2018).

## 6. Discussion
To disentangle the contribution of our proposed scaling method from the EfficientNet architecture, Figure 8 compares the ImageNet performance of different scaling methods for the same EfficientNet-B0 baseline network. In general, all scaling methods improve accuracy with the cost of more FLOPS, but our compound scaling method can further improve accuracy, by up to 2.5%, than other singledimension scaling methods, suggesting the importance of our proposed compound scaling.

Figure 8. Scaling Up EfficientNet-B0 with Different Methods.

Table 7. Scaled Models Used in Figure 7.

In order to further understand why our compound scaling method is better than others, Figure 7 compares the class activation map (Zhou et al., 2016) for a few representative models with different scaling methods. All these models are scaled from the same baseline, and their statistics are shown in Table 7. Images are randomly picked from ImageNet validation set. As shown in the figure, the model with compound scaling tends to focus on more relevant regions with more object details, while other models are either lack of object details or unable to capture all objects in the images. 

## 7. Conclusion
In this paper, we systematically study ConvNet scaling and identify that carefully balancing network width, depth, and resolution is an important but missing piece, preventing us from better accuracy and efficiency. To address this issue, we propose a simple and highly effective compound scaling method, which enables us to easily scale up a baseline ConvNet to any target resource constraints in a more principled way, while maintaining model efficiency. Powered by this compound scaling method, we demonstrate that a mobilesize EfficientNet model can be scaled up very effectively, surpassing state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPS, on both ImageNet and five commonly used transfer learning datasets.

## Acknowledgements
We thank Ruoming Pang, Vijay Vasudevan, Alok Aggarwal, Barret Zoph, Hongkun Yu, Jonathon Shlens, Raphael Gontijo Lopes, Yifeng Lu, Daiyi Peng, Xiaodan Song, Samy Bengio, Jeff Dean, and the Google Brain team for their help.

## Appendix
Since 2017, most research papers only report and compare ImageNet validation accuracy; this paper also follows this convention for better comparison. In addition, we have also verified the test accuracy by submitting our predictions on the 100k test set images to http://image-net.org; results are in Table 8. As expected, the test accuracy is very close to the validation accuracy.

Table 8. ImageNet Validation vs. Test Top-1/5 Accuracy.

## References
