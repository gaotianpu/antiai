# ImageNet Classification with Deep Convolutional
基于深度卷积的ImageNet分类 原文：https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

## 阅读笔记
* [pytorch实现](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py)

## Abstract
We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called “dropout” that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.

我们训练了一个大型深度卷积神经网络，将ImageNet LSVRC-2010比赛中120万张高分辨率图像分为1000个不同类别。在测试数据中，我们获得了37.5%和17.0%的top-1和top-5错误率，这比以前的SOTA要好得多。该神经网络有6000万个参数和65万个神经元，由五个卷积层组成，其中一些层之后是最大池化层，还有三个完全连接的层，最后是1000路softmax。为了加快训练速度，我们使用了非饱和神经元和非常有效的GPU实现卷积运算。为了减少全连接层中的过拟合，我们采用了最近开发的称为“dropout”的正则化方法，该方法被证明非常有效。我们还在ILSVRC-2012竞赛中加入了该模型的一个变体，并获得了15.3%的top-5测试集错误率，而第二名的测试集错误率为26.2%。

## 1 Introduction
Current approaches to object recognition make essential use of machine learning methods. To improve their performance, we can collect larger datasets, learn more powerful models, and use better techniques for preventing overfitting. Until recently, datasets of labeled images were relatively small — on the order of tens of thousands of images (e.g., NORB [16], Caltech-101/256 [8, 9], and CIFAR-10/100 [12]). Simple recognition tasks can be solved quite well with datasets of this size, especially if they are augmented with label-preserving transformations. For example, the currentbest error rate on the MNIST digit-recognition task (<0.3%) approaches human performance [4]. But objects in realistic settings exhibit considerable variability, so to learn to recognize them it is necessary to use much larger training sets. And indeed, the shortcomings of small image datasets have been widely recognized (e.g., Pinto et al. [21]), but it has only recently become possible to collect labeled datasets with millions of images. The new larger datasets include LabelMe [23], which consists of hundreds of thousands of fully-segmented images, and ImageNet [6], which consists of over 15 million labeled high-resolution images in over 22,000 categories.

当前的目标识别方法主要使用机器学习方法。为了提高它们的性能，我们可以收集更大的数据集，学习更强大的模型，并使用更好的技术来防止过拟合。直到最近，标记图像的数据集还是相对较小的，大约有成千上万个图像(例如，NORB[16]、Caltech-101/256[8、9]和CIFAR-10/100[12])。对于这种规模的数据集，简单的识别任务可以很好地解决，特别是如果使用标签保持转换对它们进行增强的话。例如，MNIST数字识别任务的当前最佳错误率(<0.3%)接近人类能力[4]。但现实场景中的对象表现出相当大的可变性，因此要学会识别它们，必须使用更大的训练集。事实上，小型图像数据集的缺点已被广泛认识(例如，Pintoet al [21])，但直到最近才有可能收集具有数百万图像的标注数据集。新的更大的数据集包括LabelMe[23]，它由数十万张完全分割的图像组成，ImageNet[6]，它包括超过1500万张标记的高分辨率图像，分为22000多个类别。

To learn about thousands of objects from millions of images, we need a model with a large learning capacity. However, the immense complexity of the object recognition task means that this problem cannot be specified even by a dataset as large as ImageNet, so our model should also have lots of prior knowledge to compensate for all the data we don’t have. Convolutional neural networks (CNNs) constitute one such class of models [16, 11, 13, 18, 15, 22, 26]. Their capacity can be controlled by varying their depth and breadth, and they also make strong and mostly correct assumptions about the nature of images (namely, stationarity of statistics and locality of pixel dependencies). Thus, compared to standard feedforward neural networks with similarly-sized layers, CNNs have much fewer connections and parameters and so they are easier to train, while their theoretically-best performance is likely to be only slightly worse.

要从数以百万计的图像中了解数以千计的物体，我们需要一个具有巨大学习能力的模型。然而，物体识别任务的巨大复杂性意味着，即使是像ImageNet这样大的数据集也无法说明这个问题，因此我们的模型也应该有很多先验知识来弥补我们没有的所有数据。卷积神经网络(CNN)就是这样一类模型[16，11，13，18，15，22，26]。它们的容量可以通过改变深度和宽度来控制，而且它们还对图像的性质(即统计数据的平稳性和像素的局部相关性)做出了强力且最正确的假设。因此，与具有相似大小层的标准前馈神经网络相比，CNN的连接和参数要少得多，因此更容易训练，而其理论上的最佳性能可能只稍差。

Despite the attractive qualities of CNNs, and despite the relative efficiency of their local architecture, they have still been prohibitively expensive to apply in large scale to high-resolution images. Luckily, current GPUs, paired with a highly-optimized implementation of 2D convolution, are powerful enough to facilitate the training of interestingly-large CNNs, and recent datasets such as ImageNet contain enough labeled examples to train such models without severe overfitting.

尽管CNN具有吸引人的品质，尽管其局部架构相对高效，但大规模应用于高分辨率图像的成本仍然高得令人望而却步。幸运的是，当前的GPU与高度优化的2D卷积实现相结合，功能强大，足以促进有趣的大型CNN的训练，最近的数据集(如ImageNet)包含足够的标记样本，以训练此类模型，而不会出现严重的过拟合。

The specific contributions of this paper are as follows: we trained one of the largest convolutional neural networks to date on the subsets of ImageNet used in the ILSVRC-2010 and ILSVRC-2012 competitions [2] and achieved by far the best results ever reported on these datasets. We wrote a highly-optimized GPU implementation of 2D convolution and all the other operations inherent in training convolutional neural networks, which we make available publicly1 . Our network contains a number of new and unusual features which improve its performance and reduce its training time, which are detailed in Section 3. The size of our network made overfitting a significant problem, even with 1.2 million labeled training examples, so we used several effective techniques for preventing overfitting, which are described in Section 4. Our final network contains five convolutional and three fully-connected layers, and this depth seems to be important: we found that removing any convolutional layer (each of which contains no more than 1% of the model’s parameters) resulted in inferior performance.

本文的具体贡献如下：我们对ILSVRC-2010和ILSVRC-2012比赛中使用的ImageNet子集训练了迄今为止最大的卷积神经网络[2]，并取得了迄今为止在这些数据集上报道的最佳结果。我们编写了一个高度优化的2D卷积GPU实现和训练卷积神经网络所固有的所有其他操作，并将其公开1。我们的网络包含许多新的和不寻常的功能，这些功能提高了其性能并缩短了训练时间，详见第3节。我们网络规模使过拟合成为一个重要问题，即使有120万个标记训练样本，我们也使用了一些有效的技术来防止过拟合，如第4节所述。我们的最终网络包含五个卷积和三个完全连接的层，这个深度似乎很重要：我们发现删除任何卷积层(每个卷积层包含的模型参数不超过1%)都会导致性能下降。

In the end, the network’s size is limited mainly by the amount of memory available on current GPUs and by the amount of training time that we are willing to tolerate. Our network takes between five and six days to train on two GTX 580 3GB GPUs. All of our experiments suggest that our results can be improved simply by waiting for faster GPUs and bigger datasets to become available. 

最后，网络的大小主要受到当前GPU上可用内存量和我们愿意接受的训练时间的限制。我们的网络需要在两个GTX 580 3GB GPU上进行5~6天时间的训练。我们所有的实验都表明，在更快的GPU和更大的数据集的加持下，我们的结果就仍可以得到改进。

## 2 The Dataset
ImageNet is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories. The images were collected from the web and labeled by human labelers using Amazon’s Mechanical Turk crowd-sourcing tool. Starting in 2010, as part of the Pascal Visual Object Challenge, an annual competition called the ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) has been held. ILSVRC uses a subset of ImageNet with roughly 1000 images in each of 1000 categories. In all, there are roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images.

ImageNet是一个包含1500多万个标签高分辨率图像的数据集，这些图像属于大约22000个类别。这些图片是从网上收集的，并由人类贴标者使用亚马逊的Mechanical Turk众包工具进行标记。从2010年开始，作为Pascal视觉对象挑战赛的一部分，每年举行一次名为ImageNet大规模视觉识别挑战赛(ILSVRC)的比赛。ILSVRC使用ImageNet的子集，在1000个类别中的每个类别中大约有1000个图像。总共大约有120万张训练图像、50000张验证图像和150000张测试图像。

ILSVRC-2010 is the only version of ILSVRC for which the test set labels are available, so this is the version on which we performed most of our experiments. Since we also entered our model in the ILSVRC-2012 competition, in Section 6 we report our results on this version of the dataset as well, for which test set labels are unavailable. On ImageNet, it is customary to report two error rates: top-1 and top-5, where the top-5 error rate is the fraction of test images for which the correct label is not among the five labels considered most probable by the model.

ILSVRC-2010是唯一有测试集标签的ILSVRC版本，因此这是我们进行大多数实验的版本。由于我们也在ILSVRC-2012竞赛中输入了我们的模型，因此在第6节中，我们也报告了关于该版本数据集的结果，因为测试集标签不可用。在ImageNet上，通常报告两个错误率：top-1和top-5，其中top-5错误率是测试图像中正确标签不在模型认为最可能的五个标签中的部分。

ImageNet consists of variable-resolution images, while our system requires a constant input dimensionality. Therefore, we down-sampled the images to a fixed resolution of 256 × 256. Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then cropped out the central 256×256 patch from the resulting image. We did not pre-process the images in any other way, except for subtracting the mean activity over the training set from each pixel. So we trained our network on the (centered) raw RGB values of the pixels. 

ImageNet由分辨率不固定的图像组成，而我们的系统需要恒定的输入维度。因此，我们将图像降采样到256×256的固定分辨率。给定一个矩形图像，我们首先重新缩放图像，使较短的边的长度为256，然后从生成的图像中裁剪出中心256×256分块。除了从每个像素减去训练集上的平均活动度？之外，我们没有用任何其他方法对图像进行预处理。因此，我们根据像素的(居中)原始RGB值训练网络。

## 3 The Architecture
The architecture of our network is summarized in Figure 2. It contains eight learned layers — five convolutional and three fully-connected. Below, we describe some of the novel or unusual features of our network’s architecture. Sections 3.1-3.4 are sorted according to our estimation of their importance, with the most important first. 

我们的网络架构如图2所示。它包含8个学习层 —— 5个卷积层和3个全连接层。下面，我们将介绍我们网络架构的一些新颖或不寻常的特征。第3.1-3.4节根据我们对其重要性的估计进行排序，最重要的放在第一位。

### 3.1 ReLU Nonlinearity
The standard way to model a neuron’s output f as a function of its input x is with f(x) = tanh(x) or $f(x) = (1 + e^{−x})^{−1}$ . In terms of training time with gradient descent, these saturating nonlinearities are much slower than the non-saturating nonlinearity f(x) = max(0, x). Following Nair and Hinton [20], we refer to neurons with this nonlinearity as Rectified Linear Units (ReLUs). Deep convolutional neural networks with ReLUs train several times faster than their equivalents with tanh units. This is demonstrated in Figure 1, which shows the number of iterations required to reach 25% training error on the CIFAR-10 dataset for a particular four-layer convolutional network. This plot shows that we would not have been able to experiment with such large neural networks for this work if we had used traditional saturating neuron models.

将神经元的输出f建模为其输入x的函数的标准方法是f(x) = tanh(x) 或 $f(x) = (1 + e^{−x})^{−1}$。就梯度下降的训练时间而言，这些饱和非线性比非饱和非线性f(x) = max(0, x) 慢得多。继Nair和Hinton[20]之后，我们将具有这种非线性的神经元称为校正线性单位(ReLUs)。带有ReLU的深度卷积神经网络的训练速度比具有tanh单位的等效网络快几倍。图1展示了这一点，它显示了在CIFAR-10数据集上为特定的四层卷积网络达到25%训练误差所需的迭代次数。这张图表明，如果我们使用传统的饱和神经元模型，我们将无法用如此大的神经网络进行实验。

Figure 1: A four-layer convolutional neural network with ReLUs (solid line) reaches a 25% training error rate on CIFAR-10 six times faster than an equivalent network with tanh neurons (dashed line). The learning rates for each network were chosen independently to make training as fast as possible. No regularization of any kind was employed. The magnitude of the effect demonstrated here varies with network architecture, but networks with ReLUs consistently learn several times faster than equivalents with saturating neurons.
图1：具有ReLUs(实线)的四层卷积神经网络在CIFAR-10上达到25%的训练错误率，比具有tanh神经元的等效网络(虚线)快六倍。每个网络的学习速率都是独立选择的，以便尽可能快地进行训练。没有采用任何形式的正则化。这里展示的效果的大小因网络架构而异，但具有ReLU的网络学习速度始终比具有饱和神经元的网络快数倍。

We are not the first to consider alternatives to traditional neuron models in CNNs. For example, Jarrett et al. [11] claim that the nonlinearity f(x) = |tanh(x)| works particularly well with their type of contrast normalization followed by local average pooling on the Caltech-101 dataset. However, on this dataset the primary concern is preventing overfitting, so the effect they are observing is different from the accelerated ability to fit the training set which we report when using ReLUs. Faster learning has a great influence on the performance of large models trained on large datasets.

我们不是第一个考虑在CNN中替代传统神经元模型的人。例如，Jarrettet al [11]声称，非线性 f(x) = |tanh(x)| 对于其对比度归一化(contrast normalization)类型，以及Caltech-101数据集上的局部平均池化，尤其有效。然而，在这个数据集上，主要关注的是防止过拟合，因此他们观察到的效果不同于我们在使用ReLUs时报告的适应训练集的加速能力。快速学习对在大型数据集上训练大型模型的性能有很大影响。

### 3.2 Training on Multiple GPUs
A single GTX 580 GPU has only 3GB of memory, which limits the maximum size of the networks that can be trained on it. It turns out that 1.2 million training examples are enough to train networks which are too big to fit on one GPU. Therefore we spread the net across two GPUs. Current GPUs are particularly well-suited to cross-GPU parallelization, as they are able to read from and write to one another’s memory directly, without going through host machine memory. The parallelization scheme that we employ essentially puts half of the kernels (or neurons) on each GPU, with one additional trick: the GPUs communicate only in certain layers. This means that, for example, the kernels of layer 3 take input from all kernel maps in layer 2. However, kernels in layer 4 take input only from those kernel maps in layer 3 which reside on the same GPU. Choosing the pattern of connectivity is a problem for cross-validation, but this allows us to precisely tune the amount of communication until it is an acceptable fraction of the amount of computation.

单个GTX 580 GPU只有3GB内存，这限制了可以在其上训练的网络的最大大小。事实证明，120万个训练样本足以训练一个GPU无法容纳的网络。因此，我们将网络扩展到两个GPU。当前的GPU特别适合跨GPU并行化，因为它们能够直接读取和写入彼此的内存，而无需经过主机内存。我们采用的并行化方案本质上把一半的内核(或神经元)放在每个GPU上，还有一个额外的技巧：GPU只在特定的层中通信。这意味着，例如，第3层的内核从第2层的所有内核映射获取输入。但是，第4层的内核仅从位于同一GPU上的第3层内核映射获取输出。选择连接模式是交叉验证的一个问题，但这允许我们精确地调整通信量，直到它成为计算量的可接受部分。

The resultant architecture is somewhat similar to that of the “columnar” CNN employed by Cire¸san et al. [5], except that our columns are not independent (see Figure 2). This scheme reduces our top-1 and top-5 error rates by 1.7% and 1.2%, respectively, as compared with a net with half as many kernels in each convolutional layer trained on one GPU. The two-GPU net takes slightly less time to train than the one-GPU net (2 The one-GPU net actually has the same number of kernels as the two-GPU net in the final convolutional layer. This is because most of the net’s parameters are in the first fully-connected layer, which takes the last convolutional layer as input. So to make the two nets have approximately the same number of parameters, we did not halve the size of the final convolutional layer (nor the fully-conneced layers which follow). Therefore this comparison is biased in favor of the one-GPU net, since it is bigger than “half the size” of the two-GPU net.)

合成的架构与Cire，sanet al [5]使用的“柱状”CNN的架构有些相似，只是我们的柱不是独立的(见图2)。与在一个GPU上训练的每个卷积层中有一半内核的网络相比，该方案将我们的top-1和top-5错误率分别降低了1.7%和1.2%。两个GPU网络的训练时间略少于一个GPU网。(2 一个GPU网络实际上与最后卷积层中的两个GPU网具有相同的内核数。这是因为网络的大多数参数都在第一个完全连接的层中，该层将最后一个卷积层作为输入。因此，为了使两个网络具有大致相同的参数数目，我们没有将最终卷积层(也没有将随后的全连接层)的大小减半。因此，这种比较偏向于一个GPU网络，因为它大于两个GPU网的“一半大小”。)

### 3.3 Local Response Normalization 局部响应归一化
ReLUs have the desirable property that they do not require input normalization to prevent them from saturating. If at least some training examples produce a positive input to a ReLU, learning will happen in that neuron. However, we still find that the following local normalization scheme aids generalization. Denoting by ai x,y the activity of a neuron computed by applying kernel i at position (x, y) and then applying the ReLU nonlinearity, the response-normalized activity bi x,y is given by the expression 

ReLU具有理想的特性，它们不需要输入规一化来防止饱和。如果至少有一些训练样本对ReLU产生了积极的输入，那么学习将在该神经元中发生。然而，我们仍然发现以下局部归一化方案有助于推广。用ai x，y表示通过在位置(x，y)应用核i，然后应用ReLU非线性计算的神经元活动，响应归一化活动bi x，y由表达式给出

$b^i_{x,y} = a^i_{x,y}/(k + α \sum_{j=max(0,i−n/2)}^{min(N−1,i+n/2)}(a^j_{x,y})^2)^β $

where the sum runs over n “adjacent” kernel maps at the same spatial position, and N is the total number of kernels in the layer. The ordering of the kernel maps is of course arbitrary and determined before training begins. This sort of response normalization implements a form of lateral inhibition inspired by the type found in real neurons, creating competition for big activities amongst neuron outputs computed using different kernels. The constants k, n, α, and β are hyper-parameters whose values are determined using a validation set; we used k = 2, n = 5, $α = 10^{−4}$ , and β = 0.75. We applied this normalization after applying the ReLU nonlinearity in certain layers (see Section 3.5).

其中，总和在相同空间位置的n个“相邻”内核映射上运行，n是层中内核的总数。当然，内核映射的顺序是任意的，并在训练开始之前确定。这种响应标准化实现了一种受真实神经元类型启发的侧向抑制形式，在使用不同核计算的神经元输出之间形成了对大活动的竞争。常数k、n、α和β是超参数，其值通过验证集确定; 我们使用k = 2, n = 5, $α = 10^{−4}$, β = 0.75。在某些层中应用ReLU非线性后，我们应用了这种归一化(见第3.5节)。

This scheme bears some resemblance to the local contrast normalization scheme of Jarrett et al. [11], but ours would be more correctly termed “brightness normalization”, since we do not subtract the mean activity. Response normalization reduces our top-1 and top-5 error rates by 1.4% and 1.2%, respectively. We also verified the effectiveness of this scheme on the CIFAR-10 dataset: a four-layer CNN achieved a 13% test error rate without normalization and 11% with normalization (3 We cannot describe this network in detail due to space constraints, but it is specified precisely by the code and parameter files provided here: http://code.google.com/p/cuda-convnet/.)

该方案与Jarrettet al [11]的局部对比度归一化方案有些相似，但我们的方案更准确地称为“亮度归一化”，因为我们不减去平均活度。响应规一化将我们的top和top-5错误率分别降低了1.4%和1.2%。我们还验证了该方案在CIFAR-10数据集上的有效性：四层CNN在未进行标准化的情况下达到13%的测试错误率，在进行标准化时达到11%。(3 由于空间限制，我们无法详细描述该网络，但此处提供的代码和参数文件对其进行了精确说明：http://code.google.com/p/cuda-convnet/.4.)

### 3.4 Overlapping Pooling 重叠池化
Pooling layers in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map. Traditionally, the neighborhoods summarized by adjacent pooling units do not overlap (e.g., [17, 11, 4]). To be more precise, a pooling layer can be thought of as consisting of a grid of pooling units spaced s pixels apart, each summarizing a neighborhood of size z × z centered at the location of the pooling unit. If we set s = z, we obtain traditional local pooling as commonly employed in CNNs. If we set s < z, we obtain overlapping pooling. This is what we use throughout our network, with s = 2 and z = 3. This scheme reduces the top-1 and top-5 error rates by 0.4% and 0.3%, respectively, as compared with the non-overlapping scheme s = 2, z = 2, which produces output of equivalent dimensions. We generally observe during training that models with overlapping pooling find it slightly more difficult to overfit.

CNN中的池化层总结了同一核映射中相邻神经元组的输出。传统上，由相邻联营单位汇总的社区？并不重叠(例如，[17，11，4])。更准确地说，池化层可以被认为是由间隔s个像素的池化单元网格组成的，每个网格汇总了以池化单元位置为中心的大小为z×z的邻域。如果设置s=z，我们将获得CNN中常用的传统局部池化。如果我们设置s z，我们将获得重叠池化。这就是我们在整个网络中使用的方法，其中s=2和z=3。与产生等效尺寸输出的非重叠方案s=2、z=2相比，该方案将top-1和top-5错误率分别减少0.4%和0.3%。我们通常在训练期间观察到，具有重叠池化的模型越难过拟合。

### 3.5 Overall Architecture
Now we are ready to describe the overall architecture of our CNN. As depicted in Figure 2, the net contains eight layers with weights; the first five are convolutional and the remaining three are fullyconnected. The output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels. Our network maximizes the multinomial logistic regression objective, which is equivalent to maximizing the average across training cases of the log-probability of the correct label under the prediction distribution.

现在，我们已经准备好描述CNN的总体架构。如图2所示，该网包含八层，带有权重; 前五个是卷积的，其余三个是完全连接的。最后一个完全连接的层的输出被馈送到1000路softmax，它在1000个类标签上产生分布。我们的网络最大化了多项式logistic回归目标，这相当于最大化了预测分布下正确标签的对数概率的跨训练案例的平均值。

Figure 2: An illustration of the architecture of our CNN, explicitly showing the delineation of responsibilities between the two GPUs. One GPU runs the layer-parts at the top of the figure while the other runs the layer-parts at the bottom. The GPUs communicate only at certain layers. The network’s input is 150,528-dimensional, and the number of neurons in the network’s remaining layers is given by 253,440–186,624–64,896–64,896–43,264– 4096–4096–1000. 
图2：美国有线电视新闻网(CNN)架构示意图，明确显示了两个GPU之间的职责划分。一个GPU运行图形顶部的层部件，而另一个运行图形底部的层部件。GPU仅在特定层通信。网络的输入是150528维，网络剩余层中的神经元数量为253440–186624–64896–64896-43264–4096–4096-1000。

The kernels of the second, fourth, and fifth convolutional layers are connected only to those kernel maps in the previous layer which reside on the same GPU (see Figure 2). The kernels of the third convolutional layer are connected to all kernel maps in the second layer. The neurons in the fullyconnected layers are connected to all neurons in the previous layer. Response-normalization layers follow the first and second convolutional layers. Max-pooling layers, of the kind described in Section 3.4, follow both response-normalization layers as well as the fifth convolutional layer. The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer.

第二、第四和第五卷积层的内核仅与位于同一GPU上的前一层中的内核映射相连接(见图2)。第三卷积层的核与第二层的所有核映射相连。全连接层中的神经元与前一层中的所有神经元连接。响应规一化层遵循第一和第二卷积层。第3.4节中描述的最大池化层遵循响应归一化层和第五卷积层。ReLU非线性应用于每个卷积和全连接层的输出。

The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3 with a stride of 4 pixels (this is the distance between the receptive field centers of neighboring neurons in a kernel map). The second convolutional layer takes as input the (response-normalized and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5 × 5 × 48. The third, fourth, and fifth convolutional layers are connected to one another without any intervening pooling or normalization layers. The third convolutional layer has 384 kernels of size 3 × 3 × 256 connected to the (normalized, pooled) outputs of the second convolutional layer. The fourth convolutional layer has 384 kernels of size 3 × 3 × 192 , and the fifth convolutional layer has 256 kernels of size 3 × 3 × 192. The fully-connected layers have 4096 neurons each. 

第一卷积层过滤224×224×3输入图像，96个11×11×3大小的核，步幅为4像素(这是核图中相邻神经元感受野中心之间的距离)。第二卷积层将第一卷积层的(响应归一化和池化)输出作为输入，并用256个大小为5×5×48的内核对其进行过滤。第三、第四和第五卷积层相互连接，没有任何中间的池化或归一化层。第三卷积层有384个大小为3×3×256的内核，连接到第二卷积层的(归一化、合并)输出。第四卷积层有384个3×3×192大小的核，第五卷积层则有256个3×3X192大小的核。全连接层各有4096个神经元。



## 4 Reducing Overfitting 减少过拟合
Our neural network architecture has 60 million parameters. Although the 1000 classes of ILSVRC make each training example impose 10 bits of constraint on the mapping from image to label, this turns out to be insufficient to learn so many parameters without considerable overfitting. Below, we describe the two primary ways in which we combat overfitting.

我们的神经网络架构有6000万个参数。尽管ILSVRC的1000个类使每个训练样本对从图像到标签的映射施加10位约束，但事实证明，这不足以学习如此多的参数，而不会出现过拟合。下面，我们描述了两种主要的减少过拟合的方法。

### 4.1 Data Augmentation
The easiest and most common method to reduce overfitting on image data is to artificially enlarge the dataset using label-preserving transformations (e.g., [25, 4, 5]). We employ two distinct forms of data augmentation, both of which allow transformed images to be produced from the original images with very little computation, so the transformed images do not need to be stored on disk. In our implementation, the transformed images are generated in Python code on the CPU while the GPU is training on the previous batch of images. So these data augmentation schemes are, in effect, computationally free.

减少图像数据过拟合的最简单也是最常见的方法是使用标签保留变换(例如[25，4，5])人工放大数据集。我们采用了两种不同的数据增广形式，这两种形式都允许用很少的计算从原始图像生成转换图像，因此转换后的图像不需要存储在磁盘上。在我们的实现中，转换后的图像是在CPU上用Python代码生成的，而GPU则在前一批图像上进行训练。因此，这些数据增广方案实际上是无需计算的。

The first form of data augmentation consists of generating image translations and horizontal reflections. We do this by extracting random 224 × 224 patches (and their horizontal reflections) from the 256×256 images and training our network on these extracted patches(4This is the reason why the input images in Figure 2 are 224 × 224 × 3-dimensional.). This increases the size of our training set by a factor of 2048, though the resulting training examples are, of course, highly interdependent. Without this scheme, our network suffers from substantial overfitting, which would have forced us to use much smaller networks. At test time, the network makes a prediction by extracting five 224 × 224 patches (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all), and averaging the predictions made by the network’s softmax layer on the ten patches.

第一种数据增广形式包括生成图像平移和水平反射。我们通过从256×256个图像中随机提取224×224个分块(及其水平反射)，并在这些提取的分块上训练我们的网络4来实现这一点(4 这就是为什么图2中的输入图像是224×224×3维的原因。)。这将使我们的训练集的大小增加了2048倍，尽管得到的训练样本当然是高度相互依赖的。如果没有这个方案，我们的网络将遭受严重的过拟合，这将迫使我们使用更小的网络。在测试时，网络通过提取五个224×224分块(四个角分块和中心分块)及其水平反射(因此总共十个分块)进行预测，并平均网络的softmax层对十个面团的预测。

The second form of data augmentation consists of altering the intensities of the RGB channels in training images. Specifically, we perform PCA on the set of RGB pixel values throughout the ImageNet training set. To each training image, we add multiples of the found principal components, with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1. Therefore to each RGB image pixel Ixy = [IR xy, IG xy, IB xy]T we add the following quantity: 

第二种数据增广形式包括改变训练图像中RGB通道的强度。具体来说，我们在整个ImageNet训练集中对RGB像素值集执行PCA。对于每个训练图像，我们将找到的主成分的倍数相加，其大小与相应的特征值成比例，乘以从平均值为零且标准偏差为0.1的高斯中提取的随机变量。因此，对于每个RGB图像像素Ixy=[IR xy，IG xy，IB xy]T，我们添加以下数量：

[p1, p2, p3][α1λ1, α2λ2, α3λ3]T 

where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance matrix of RGB pixel values, respectively, and αi is the aforementioned random variable. Each αi is drawn only once for all the pixels of a particular training image until that image is used for training again, at which point it is re-drawn. This scheme approximately captures an important property of natural images, namely, that object identity is invariant to changes in the intensity and color of the illumination. This scheme reduces the top-1 error rate by over 1%.

其中pi和λi分别是RGB像素值的3×3协方差矩阵的第i个特征向量和特征值，αi是上述随机变量。对于特定训练图像的所有像素，每个αi只绘制一次，直到该图像再次用于训练，然后重新绘制。该方案近似地捕获了自然图像的一个重要特性，即物体身份对照明强度和颜色的变化是不变的。该方案将top-1错误率降低了1%以上。

### 4.2 Dropout
Combining the predictions of many different models is a very successful way to reduce test errors [1, 3], but it appears to be too expensive for big neural networks that already take several days to train. There is, however, a very efficient version of model combination that only costs about a factor of two during training. The recently-introduced technique, called “dropout” [10], consists of setting to zero the output of each hidden neuron with probability 0.5. The neurons which are “dropped out” in this way do not contribute to the forward pass and do not participate in backpropagation. So every time an input is presented, the neural network samples a different architecture, but all these architectures share weights. This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons. It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons. At test time, we use all the neurons but multiply their outputs by 0.5, which is a reasonable approximation to taking the geometric mean of the predictive distributions produced by the exponentially-many dropout networks.

将许多不同模型的预测结合起来是减少测试误差的一种非常成功的方法[1,3]，但对于已经需要几天时间来训练的大型神经网络来说，这种方法似乎过于昂贵。然而，有一种非常有效的模型组合，在训练期间只需花费大约两倍的成本。最近引入的技术称为“dropout”[10]，包括将每个隐藏神经元的输出设置为零，概率为0.5。以这种方式“dropout”的神经元不参与正向传递，也不参与反向传播。因此，每次出现输入时，神经网络都会对不同的架构进行采样，但所有这些架构都共享权重。这项技术减少了神经元的复杂协同适应，因为神经元不能依赖于特定的其他神经元的存在。因此，它被迫学习更健壮的特征，这些特征与其他神经元的许多不同随机子集相结合是有用的。在测试时，我们使用所有神经元，但将它们的输出乘以0.5，这是一个合理的近似值，用于计算由指数级多个dropout网络生成的预测分布的几何平均值。

We use dropout in the first two fully-connected layers of Figure 2. Without dropout, our network exhibits substantial overfitting. Dropout roughly doubles the number of iterations required to converge.

我们在图2的前两个完全连接的层中使用了dropout。如果没有dropout，我们的网络会出现严重的过拟合。Dropout大约使收敛所需的迭代次数加倍。

## 5 Details of learning
We trained our models using stochastic gradient descent with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005. We found that this small amount of weight decay was important for the model to learn. In other words, weight decay here is not merely a regularizer: it reduces the model’s training error. The update rule for weight w was 

我们使用随机梯度下降法对模型进行训练，批量为128个样本，动量为0.9，权重衰减为0.0005。我们发现，少量的权重衰减对模型的学习很重要。换句话说，这里的权重衰减不仅仅是一个正则化器：它减少了模型的训练误差。权重w的更新规则为

vi+1 := 0.9 · vi − 0.0005 ·  · wi −  · ∂L ∂w  wi Di 

wi+1 := wi + vi+1 

where i is the iteration index, v is the momentum variable,  is the learning rate, and D ∂L ∂w wiE Di is the average over the ith batch Di of the derivative of the objective with respect to w, evaluated at wi.

其中i是迭代指数，v是动量变量，是学习速率，D∂L∂w wiE Di是第i批Di中目标相对于w的导数的平均值，在wi处评估。

We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01. We initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1. This initialization accelerates the early stages of learning by providing the ReLUs with positive inputs. We initialized the neuron biases in the remaining layers with the constant 0. We used an equal learning rate for all layers, which we adjusted manually throughout training. The heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and 6 reduced three times prior to termination. We trained the network for roughly 90 cycles through the training set of 1.2 million images, which took five to six days on two NVIDIA GTX 580 3GB GPUs. 

我们从标准偏差为0.01的零均值高斯分布中初始化每个层中的权重。我们用常数1初始化第二、第四和第五卷积层以及完全连接的隐藏层中的神经元偏差。这个初始化通过向ReLU提供正输入来加速学习的早期阶段。我们用常数0初始化其余层中的神经元偏差。我们对所有层使用相同的学习速率，在整个训练过程中手动调整。我们遵循的启发式方法是，当验证错误率不再随着当前学习率而改善时，将学习率除以10。学习率初始化为0.01，终止前降低了三倍。我们通过120万张图像的训练集对网络进行了大约90个周期的训练，在两个NVIDIA GTX 580 3GB GPU上训练了5到6天。

## 6 Results
Our results on ILSVRC-2010 are summarized in Table 1. Our network achieves top-1 and top-5 test set error rates of 37.5% and 17.0%5 . The best performance achieved during the ILSVRC- 2010 competition was 47.1% and 28.2% with an approach that averages the predictions produced from six sparse-coding models trained on different features [2], and since then the best published results are 45.7% and 25.7% with an approach that averages the predictions of two classi- fiers trained on Fisher Vectors (FVs) computed from two types of densely-sampled features [24].

表1总结了我们在ILSVRC-2010上的结果。我们的网络达到了37.5%和17.0%的top-1和top-5测试集错误率。在ILSVRC-2010比赛中取得的最佳性能分别为47.1%和28.2%，采用的方法是平均六个稀疏编码模型对不同特征进行训练的预测[2]，自那时以来，最佳公布结果分别为45.7%和25.7%，采用的方法是根据两种密集采样特征计算的Fisher向量(FVs)训练的两个分类器的预测平均值[24]。

Table 1: Comparison of results on ILSVRC- 2010 test set. In italics are best results achieved by others.
表1:ILSVRC-2010测试集的结果比较。斜体字表示其他人取得的最佳结果。

We also entered our model in the ILSVRC-2012 competition and report our results in Table 2. Since the ILSVRC-2012 test set labels are not publicly available, we cannot report test error rates for all the models that we tried. In the remainder of this paragraph, we use validation and test error rates interchangeably because in our experience they do not differ by more than 0.1% (see Table 2). The CNN described in this paper achieves a top-5 error rate of 18.2%. Averaging the predictions of five similar CNNs gives an error rate of 16.4%. Training one CNN, with an extra sixth convolutional layer over the last pooling layer, to classify the entire ImageNet Fall 2011 release (15M images, 22K categories), and then “fine-tuning” it on ILSVRC-2012 gives an error rate of 16.6%. Averaging the predictions of two CNNs that were pre-trained on the entire Fall 2011 release with the aforementioned five CNNs gives an error rate of 15.3%. The second-best contest entry achieved an error rate of 26.2% with an approach that averages the predictions of several classifiers trained on FVs computed from different types of densely-sampled features [7].

我们还在ILSVRC-2012竞赛中输入了我们的模型，并在表2中报告了我们的结果。由于ILSVRC-2002测试集标签不公开，我们无法报告我们尝试的所有模型的测试错误率。在本段的其余部分，我们交替使用验证和测试错误率，因为根据我们的经验，它们之间的差异不超过0.1%(见表2)。本文中描述的CNN达到了18.2%的top-5位错误率。对五个类似CNN的预测进行平均后，错误率为16.4%。训练一个CNN，在最后一个池化层上额外增加第六个卷积层，对整个ImageNet 2011年秋季发布版(15M图像，22K类别)进行分类，然后在ILSVRC-2012上对其进行“微调”，错误率为16.6%。使用上述五个CNN对2011年秋季发布的两个CNN进行了预训练，平均预测的错误率为15.3%。第二名参赛选手的错误率为26.2%，采用的方法是对根据不同类型的密集采样特征计算的FV训练的几个分类器的预测进行平均[7]。

Table 2: Comparison of error rates on ILSVRC-2012 validation and test sets. In italics are best results achieved by others. Models with an asterisk* were “pre-trained” to classify the entire ImageNet 2011 Fall release. See Section 6 for details.

表2:ILSVRC-2012验证集和测试集的错误率比较。斜体字表示其他人取得的最佳结果。带有星号*的模型经过了“预先训练”，可以对整个ImageNet 2011秋季版本进行分类。详见第6节。

Finally, we also report our error rates on the Fall 2009 version of ImageNet with 10,184 categories and 8.9 million images. On this dataset we follow the convention in the literature of using half of the images for training and half for testing. Since there is no established test set, our split necessarily differs from the splits used by previous authors, but this does not affect the results appreciably. Our top-1 and top-5 error rates on this dataset are 67.4% and 40.9%, attained by the net described above but with an additional, sixth convolutional layer over the last pooling layer. The best published results on this dataset are 78.1% and 60.9% [19].

最后，我们还报告了2009年秋季版ImageNet的错误率，该版本包含10184个类别和890万张图像。在这个数据集上，我们遵循文献中的惯例，使用一半的图像进行训练，一半用于测试。由于没有确定的测试集，我们的分割必然与以前作者使用的分割不同，但这不会显著影响结果。我们在该数据集上的top-1和top-5错误率分别为67.4%和40.9%，通过上述网络实现，但在最后一个池化层上还有一个额外的第六卷积层。该数据集的最佳发布结果为78.1%和60.9%[19]。

### 6.1 Qualitative Evaluations
Figure 3 shows the convolutional kernels learned by the network’s two data-connected layers. The network has learned a variety of frequency- and orientation-selective kernels, as well as various colored blobs. Notice the specialization exhibited by the two GPUs, a result of the restricted connectivity described in Section 3.5. The kernels on GPU 1 are largely color-agnostic, while the kernels on on GPU 2 are largely color-specific. This kind of specialization occurs during every run and is independent of any particular random weight initialization (modulo a renumbering of the GPUs). 

图3显示了网络的两个数据连接层所学习的卷积核。该网络已经学习了各种频率和方向选择内核，以及各种彩色斑点。注意两个GPU显示的特殊性，这是第3.5节中描述的受限连接的结果。GPU 1上的内核基本上不区分颜色，而GPU 2上的内核则基本上特定于颜色。这种专门化在每次运行期间发生，并且独立于任何特定的随机权重初始化(GPU重新编号的模)。

Figure 3: 96 convolutional kernels of size 11×11×3 learned by the first convolutional layer on the 224×224×3 input images. The top 48 kernels were learned on GPU 1 while the bottom 48 kernels were learned on GPU 2. See Section 6.1 for details.
图3:224×224×3输入图像上第一卷积层学习的96个卷积核，大小为11×11×3。前48个内核在GPU 1上学习，后48个内核则在GPU 2上学习。有关详情，请参阅第6.1节。

5The error rates without averaging predictions over ten patches as described in Section 4.1 are 39.0% and 18.3%. 7

5如第4.1节所述，在十个分块上不进行平均预测的错误率分别为39.0%和18.3%。7.

Figure 4: (Left) Eight ILSVRC-2010 test images and the five labels considered most probable by our model. The correct label is written under each image, and the probability assigned to the correct label is also shown with a red bar (if it happens to be in the top 5). (Right) Five ILSVRC-2010 test images in the first column. The remaining columns show the six training images that produce feature vectors in the last hidden layer with the smallest Euclidean distance from the feature vector for the test image.

图4：(左)八张ILSVRC-2010测试图像和我们的模型认为最可能的五个标签。正确的标签写在每个图像下，分配给正确标签的概率也用红色条显示(如果恰好在top-5位)。(右)第一列中的五幅ILSVRC-2010测试图像。其余的列显示了六个训练图像，它们在最后一个隐藏层中生成特征向量，与测试图像的特征向量之间的欧氏距离最小。

In the left panel of Figure 4 we qualitatively assess what the network has learned by computing its top-5 predictions on eight test images. Notice that even off-center objects, such as the mite in the top-left, can be recognized by the net. Most of the top-5 labels appear reasonable. For example, only other types of cat are considered plausible labels for the leopard. In some cases (grille, cherry) there is genuine ambiguity about the intended focus of the photograph.

在图4的左侧面板中，我们通过计算八张测试图像上的前五个预测，定性评估网络所学到的知识。请注意，即使是偏离中心的对象，例如左上方的螨虫，也可以被网络识别。前五大标签中的大多数看起来都很合理。例如，只有其他类型的猫被认为是豹的合理标签。在某些情况下(格栅、樱桃色)，照片的预期焦点确实不明确。

Another way to probe the network’s visual knowledge is to consider the feature activations induced by an image at the last, 4096-dimensional hidden layer. If two images produce feature activation vectors with a small Euclidean separation, we can say that the higher levels of the neural network consider them to be similar. Figure 4 shows five images from the test set and the six images from the training set that are most similar to each of them according to this measure. Notice that at the pixel level, the retrieved training images are generally not close in L2 to the query images in the first column. For example, the retrieved dogs and elephants appear in a variety of poses. We present the results for many more test images in the supplementary material.

探索网络视觉知识的另一种方法是考虑由最后一个4096维隐藏层的图像引发的特征激活。如果两幅图像产生的特征激活向量具有较小的欧几里德分离，我们可以说神经网络的高层认为它们是相似的。图4显示了测试集中的五幅图像和训练集中的六幅图像，根据这个度量，这两幅图像中的每一幅都最相似。请注意，在像素级别，二级缓存中检索的训练图像通常与第一列中的查询图像不太接近。例如，取回的狗和大象以各种姿势出现。我们在补充材料中展示了更多测试图像的结果。

Computing similarity by using Euclidean distance between two 4096-dimensional, real-valued vectors is inefficient, but it could be made efficient by training an auto-encoder to compress these vectors to short binary codes. This should produce a much better image retrieval method than applying autoencoders to the raw pixels [14], which does not make use of image labels and hence has a tendency to retrieve images with similar patterns of edges, whether or not they are semantically similar. 

使用两个4096维实值向量之间的欧氏距离计算相似性是低效的，但可以通过训练自动编码器将这些向量压缩为短二进制码来提高效率。与对原始像素应用自动编码器相比，这应该产生一种更好的图像检索方法[14]，该方法不使用图像标签，因此倾向于检索具有相似边缘模式的图像，无论它们在语义上是否相似。

## 7 Discussion
Our results show that a large, deep convolutional neural network is capable of achieving recordbreaking results on a highly challenging dataset using purely supervised learning. It is notable that our network’s performance degrades if a single convolutional layer is removed. For example, removing any of the middle layers results in a loss of about 2% for the top-1 performance of the network. So the depth really is important for achieving our results.

我们的结果表明，一个大的深度卷积神经网络能够在一个具有高度挑战性的数据集上使用纯监督学习获得破记录结果。值得注意的是，如果删除单个卷积层，我们的网络性能会降低。例如，删除任何中间层都会导致网络顶级性能损失约2%。因此，深度对于实现我们的结果真的很重要。

To simplify our experiments, we did not use any unsupervised pre-training even though we expect that it will help, especially if we obtain enough computational power to significantly increase the size of the network without obtaining a corresponding increase in the amount of labeled data. Thus far, our results have improved as we have made our network larger and trained it longer but we still have many orders of magnitude to go in order to match the infero-temporal pathway of the human visual system. Ultimately we would like to use very large and deep convolutional nets on video sequences where the temporal structure provides very helpful information that is missing or far less obvious in static images. 8

为了简化我们的实验，我们没有使用任何无监督的预训练，尽管我们希望它会有所帮助，特别是如果我们获得足够的计算能力，以显著增加网络的大小，而没有相应增加标注数据的数量。到目前为止，我们的结果已经得到改善，因为我们已经使我们的网络更大，训练时间更长，但为了匹配人类视觉系统的下时间路径，我们还有许多数量级的工作要做。最终，我们希望在视频序列上使用非常大和深度的卷积网络，其中时间结构提供了非常有用的信息，而这些信息在静态图像中缺失或不太明显。8.

## References
1. R.M. Bell and Y. Koren. Lessons from the netflix prize challenge. ACM SIGKDD Explorations Newsletter, 9(2):75–79, 2007.
2. A. Berg, J. Deng, and L. Fei-Fei. Large scale visual recognition challenge 2010. www.imagenet.org/challenges. 2010.
3. L. Breiman. Random forests. Machine learning, 45(1):5–32, 2001.
4. D. Cire¸san, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. Arxiv preprint arXiv:1202.2745, 2012.
5. D.C. Cire¸san, U. Meier, J. Masci, L.M. Gambardella, and J. Schmidhuber. High-performance neural networks for visual object classification. Arxiv preprint arXiv:1102.0183, 2011.
6. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.
7. J. Deng, A. Berg, S. Satheesh, H. Su, A. Khosla, and L. Fei-Fei. ILSVRC-2012, 2012. URL http://www.image-net.org/challenges/LSVRC/2012/.
8. L. Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. Computer Vision and Image Understanding, 106(1):59–70, 2007.
9. G. Griffin, A. Holub, and P. Perona. Caltech-256 object category dataset. Technical Report 7694, California Institute of Technology, 2007. URL http://authors.library.caltech.edu/7694.
10. G.E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R.R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580, 2012.
11. K. Jarrett, K. Kavukcuoglu, M. A. Ranzato, and Y. LeCun. What is the best multi-stage architecture for object recognition? In International Conference on Computer Vision, pages 2146–2153. IEEE, 2009.
12. A. Krizhevsky. Learning multiple layers of features from tiny images. Master’s thesis, Department of Computer Science, University of Toronto, 2009.
13. A. Krizhevsky. Convolutional deep belief networks on cifar-10. Unpublished manuscript, 2010.
14. A. Krizhevsky and G.E. Hinton. Using very deep autoencoders for content-based image retrieval. In ESANN, 2011.
15. Y. Le Cun, B. Boser, J.S. Denker, D. Henderson, R.E. Howard, W. Hubbard, L.D. Jackel, et al. Handwritten digit recognition with a back-propagation network. In Advances in neural information processing systems, 1990.
16. Y. LeCun, F.J. Huang, and L. Bottou. Learning methods for generic object recognition with invariance to pose and lighting. In Computer Vision and Pattern Recognition, 2004. CVPR 2004. Proceedings of the 2004 IEEE Computer Society Conference on, volume 2, pages II–97. IEEE, 2004.
17. Y. LeCun, K. Kavukcuoglu, and C. Farabet. Convolutional networks and applications in vision. In Circuits and Systems (ISCAS), Proceedings of 2010 IEEE International Symposium on, pages 253–256. IEEE, 2010.
18. H. Lee, R. Grosse, R. Ranganath, and A.Y. Ng. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In Proceedings of the 26th Annual International Conference on Machine Learning, pages 609–616. ACM, 2009.
19. T. Mensink, J. Verbeek, F. Perronnin, and G. Csurka. Metric Learning for Large Scale Image Classifi- cation: Generalizing to New Classes at Near-Zero Cost. In ECCV - European Conference on Computer Vision, Florence, Italy, October 2012.
20. V. Nair and G. E. Hinton. Rectified linear units improve restricted boltzmann machines. In Proc. 27th International Conference on Machine Learning, 2010.
21. N. Pinto, D.D. Cox, and J.J. DiCarlo. Why is real-world visual object recognition hard? PLoS computational biology, 4(1):e27, 2008.
22. N. Pinto, D. Doukhan, J.J. DiCarlo, and D.D. Cox. A high-throughput screening approach to discovering good forms of biologically inspired visual representation. PLoS computational biology, 5(11):e1000579, 2009.
23. B.C. Russell, A. Torralba, K.P. Murphy, and W.T. Freeman. Labelme: a database and web-based tool for image annotation. International journal of computer vision, 77(1):157–173, 2008.
24. J. Sánchez and F. Perronnin. High-dimensional signature compression for large-scale image classification. In Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on, pages 1665–1672. IEEE, 2011.
25. P.Y. Simard, D. Steinkraus, and J.C. Platt. Best practices for convolutional neural networks applied to visual document analysis. In Proceedings of the Seventh International Conference on Document Analysis and Recognition, volume 2, pages 958–962, 2003.
26. S.C. Turaga, J.F. Murray, V. Jain, F. Roth, M. Helmstaedter, K. Briggman, W. Denk, and H.S. Seung. Convolutional networks can learn to generate affinity graphs for image segmentation. Neural Computation, 22(2):511–538, 2010. 9
