# Focal Loss for Dense Object Detection
密集目标检测的焦点损失 2017.8.7 https://arxiv.org/abs/1708.02002

## 阅读笔记
* 解决样本不均衡问题：根据样本分类难易程度，调整分类损失函数的权重。
* RetinaNet

## Abstract
The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors. Code is at: https://github.com/facebookresearch/Detectron .

迄今为止精度最高的目标检测器基于R-CNN推广的两阶段方法，其中分类器应用于候选目标位置的稀疏集合。相比之下，在可能的目标位置的常规密集采样上应用的单级检测器有可能更快更简单，但迄今为止精度一直落后于两级检测器。在本文中，我们调查了为什么会出现这种情况。我们发现，在密集检测器训练过程中遇到的极端前景/背景类别不平衡是主要原因。我们建议通过重塑标准交叉熵损失来解决这类别不平衡问题，使其降低分配给分类良好的样本的损失的权重。我们新颖的焦点损失将训练重点放在一组稀少的困难样本上，并防止训练过程中大量容易的负面信息压倒检测器。为了评估损失的有效性，我们设计并训练了一个简单的密集检测器，我们称之为RetinaNet。我们的结果表明，当使用焦点损失进行训练时，RetinaNet能够匹配以前的单级检测器的速度，同时超过所有现有最先进的两级检测器的精度。代码位于：https://github.com/facebookresearch/Detectron .

## 1. Introduction
Current state-of-the-art object detectors are based on a two-stage, proposal-driven mechanism. As popularized in the R-CNN framework [11], the first stage generates a sparse set of candidate object locations and the second stage classifies each candidate location as one of the foreground classes or as background using a convolutional neural network. Through a sequence of advances [10, 28, 20, 14], this two-stage framework consistently achieves top accuracy on the challenging COCO benchmark [21].

当前最先进的目标检测器是基于两阶段、候选驱动机制的。如R-CNN框架[11]中所推广的，第一阶段生成候选目标位置的稀疏集合，第二阶段使用卷积神经网络将每个候选位置分类为前景类之一或背景。经过一系列推进工作[10、28、20、14]，该两阶段框架在具有挑战性的COCO基准上始终达到最高精度[21]。

Despite the success of two-stage detectors, a natural question to ask is: could a simple one-stage detector achieve similar accuracy? One stage detectors are applied over a regular, dense sampling of object locations, scales, and aspect ratios. Recent work on one-stage detectors, such as YOLO [26, 27] and SSD [22, 9], demonstrates promising results, yielding faster detectors with accuracy within 10- 40% relative to state-of-the-art two-stage methods.

尽管两级检测器取得了成功，但一个自然的问题是：简单的单级检测器能否达到类似的精度？单级检测器应用于目标位置、比例和纵横比的常规密集采样。最近关于单级检测器的研究，如YOLO[26，27]和SSD[22，9]，证明了有希望的结果，与最先进的两级方法相比，产生了更快的检测器，准确度在10-40%以内。

This paper pushes the envelop further: we present a onestage object detector that, for the first time, matches the state-of-the-art COCO AP of more complex two-stage detectors, such as the Feature Pyramid Network (FPN) [20] or Mask R-CNN [14] variants of Faster R-CNN [28]. To achieve this result, we identify class imbalance during training as the main obstacle impeding one-stage detector from achieving state-of-the-art accuracy and propose a new loss function that eliminates this barrier.

本文进一步推进了包络：我们提出了一种单级目标检测器，它首次与更复杂的两级检测器的最先进COCO AP相匹配，如特征金字塔网络(FPN)[20]或Faster R-CNN[14]的Mask R-CNN[28]变体。为了实现这一结果，我们将训练期间的类别不平衡确定为阻碍单级检测器实现最先进精度的主要障碍，并提出了一种新的损失函数，消除了这一障碍。

Class imbalance is addressed in R-CNN-like detectors by a two-stage cascade and sampling heuristics. The proposal stage (e.g., Selective Search [35], EdgeBoxes [39], DeepMask [24, 25], RPN [28]) rapidly narrows down the number of candidate object locations to a small number (e.g., 1-2k), filtering out most background samples. In the second classification stage, sampling heuristics, such as a fixed foreground-to-background ratio (1:3), or online hard example mining (OHEM) [31], are performed to maintain a manageable balance between foreground and background.

类R-CNN检测器通过两级级联和采样启发式解决了类别不平衡问题。建议阶段(例如，选择性搜索[35]、EdgeBox[39]、DeepMask[24，25]、RPN[28])迅速将候选目标位置的数量缩小到一小部分(例如，1-2k)，过滤掉大多数背景样本。在第二分类阶段，执行采样启发式，例如固定的前景与背景比(1:3)或在线困难样本挖掘(OHEM)[31]，以保持前景与背景之间的可管理平衡。

In contrast, a one-stage detector must process a much larger set of candidate object locations regularly sampled across an image. In practice this often amounts to enumerating ∼100k locations that densely cover spatial positions, scales, and aspect ratios. While similar sampling heuristics may also be applied, they are inefficient as the training procedure is still dominated by easily classified background examples. This inefficiency is a classic problem in object detection that is typically addressed via techniques such as bootstrapping [33, 29] or hard example mining [37, 8, 31].

相比之下，单级检测器必须处理一组更大的候选目标位置，这些候选目标位置在图像中定期采样。实际上，这通常相当于枚举密集覆盖空间位置、比例和纵横比的~100k个位置。虽然也可以应用类似的采样启发式，但它们效率低下，因为训练过程仍然由容易分类的背景样本主导。这种低效率是目标检测中的一个典型问题，通常通过自举[33，29]或困难样本挖掘[37，8，31]等技术来解决。

In this paper, we propose a new loss function that acts as a more effective alternative to previous approaches for dealing with class imbalance. The loss function is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases, see Figure 1. Intuitively, this scaling factor can automatically down-weight the contribution of easy examples during training and rapidly focus the model on hard examples. Experiments show that our proposed Focal Loss enables us to train a high-accuracy, one-stage detector that significantly outperforms the alternatives of training with the sampling heuristics or hard example mining, the previous state-ofthe-art techniques for training one-stage detectors. Finally, we note that the exact form of the focal loss is not crucial, and we show other instantiations can achieve similar results.

在本文中，我们提出了一种新的损失函数，它可以作为处理类别不平衡问题的更有效的替代方法。损失函数是一个动态缩放的交叉熵损失，当正确类别的置信度增加时，缩放因子衰减为零，见图1。直观地说，这个缩放因子可以自动降低训练过程中简单样本的贡献，并快速将模型集中在困难样本上。实验表明，我们提出的Focal Loss使我们能够训练一个高精度、单级检测器，它显著优于采样启发式或困难样本挖掘训练的替代方案，这是以前训练单级检测器的最先进技术。最后，我们注意到焦点损失的确切形式并不重要，我们表明其他实例也可以获得类似的结果。

Figure 1. We propose a novel loss we term the Focal Loss that adds a factor $(1 − p_t)^γ$ to the standard cross entropy criterion. Setting γ > 0 reduces the relative loss for well-classified examples ($p_t$ > .5), putting more focus on hard, misclassified examples. As our experiments will demonstrate, the proposed focal loss enables training highly accurate dense object detectors in the presence of vast numbers of easy background examples.
图1.我们提出了一种新的损失，我们称之为焦点损失，它增加了一个因子$(1 − p_t)^γ$符合标准交叉熵准则。设置γ>0可以减少分类良好的样本的相对损失($p_t$>.5)，将更多的注意力放在难以分类的错误样本上。正如我们的实验将证明的那样，所提出的焦点损失能够在大量简单背景样本的情况下训练高精度的密集目标检测器。

To demonstrate the effectiveness of the proposed focal loss, we design a simple one-stage object detector called RetinaNet, named for its dense sampling of object locations in an input image. Its design features an efficient in-network feature pyramid and use of anchor boxes. It draws on a variety of recent ideas from [22, 6, 28, 20]. RetinaNet is effi- cient and accurate; our best model, based on a ResNet-101-FPN backbone, achieves a COCO test-dev AP of 39.1 while running at 5 fps, surpassing the previously best published single-model results from both one and two-stage detectors, see Figure 2.

为了证明所提出的焦点损失的有效性，我们设计了一种简单的单级目标检测器，称为RetinaNet，因其对输入图像中目标位置的密集采样而得名。它的设计特点是高效的网络特征金字塔和锚框的使用。它借鉴了[22，6，28，20]中的各种最新想法。RetinaNet高效准确; 我们的最佳模型基于ResNet-101-FPN主干，在运行速度为5fps的情况下实现了39.1的COCO测试开发AP，超过了之前发表的单级和两级检测器的最佳单一模型结果，见图2。

Figure 2. Speed (ms) versus accuracy (AP) on COCO test-dev. Enabled by the focal loss, our simple one-stage RetinaNet detector outperforms all previous one-stage and two-stage detectors, including the best reported Faster R-CNN [28] system from [20]. We show variants of RetinaNet with ResNet-50-FPN (blue circles) and ResNet-101-FPN (orange diamonds) at five scales (400-800 pixels). Ignoring the low-accuracy regime (AP<25), RetinaNet forms an upper envelope of all current detectors, and an improved variant (not shown) achieves 40.8 AP. Details are given in §5.
图2.COCO测试的速度(ms)与精度(AP)。由于焦点损失，我们简单的单级RetinaNet检测器优于所有以前的单级和两级检测器，包括[20]中报道的Faster R-CNN[28]系统。我们以五种比例(400-800像素)显示了具有ResNet-50-FPN(蓝色圆圈)和ResNet-101-FPN(橙色菱形)的RetinaNet变体。忽略低精度状态(AP＜25)，RetinaNet形成了所有电流检测器的上包络，改进的变型(未示出)实现了40.8 AP。详情见§5。

## 2. Related Work
### Classic Object Detectors: 
The sliding-window paradigm, in which a classifier is applied on a dense image grid, has a long and rich history. One of the earliest successes is the classic work of LeCun et al. who applied convolutional neural networks to handwritten digit recognition [19, 36]. Viola and Jones [37] used boosted object detectors for face detection, leading to widespread adoption of such models. The introduction of HOG [4] and integral channel features [5] gave rise to effective methods for pedestrian detection. DPMs [8] helped extend dense detectors to more general object categories and had top results on PASCAL [7] for many years. While the sliding-window approach was the leading detection paradigm in classic computer vision, with the resurgence of deep learning [18], two-stage detectors, described next, quickly came to dominate object detection.

经典目标检测器：滑动窗口范例，其中分类器应用于密集的图像网格，有着悠久而丰富的历史。最早的成功之一是LeCunet al 的经典工作，他将卷积神经网络应用于手写数字识别[19，36]。Viola和Jones[37]使用增强型目标检测器进行人脸检测，导致此类模型的广泛采用。HOG[4]和整体通道特征[5]的引入为行人检测带来了有效的方法。DPM[8]帮助将密集检测器扩展到更一般的目标类别，并在PASCAL[7]上取得了多年的最佳结果。虽然滑动窗口方法是经典计算机视觉中的领先检测范式，但随着深度学习的复兴[18]，接下来将描述的两阶段检测器很快主导了目标检测。

### Two-stage Detectors: 
The dominant paradigm in modern object detection is based on a two-stage approach. As pioneered in the Selective Search work [35], the first stage generates a sparse set of candidate proposals that should contain all objects while filtering out the majority of negative locations, and the second stage classifies the proposals into foreground classes / background. R-CNN [11] upgraded the second-stage classifier to a convolutional network yielding large gains in accuracy and ushering in the modern era of object detection. R-CNN was improved over the years, both in terms of speed [15, 10] and by using learned object proposals [6, 24, 28]. Region Proposal Networks (RPN) integrated proposal generation with the second-stage classifier into a single convolution network, forming the Faster RCNN framework [28]. Numerous extensions to this framework have been proposed, e.g. [20, 31, 32, 16, 14].

两阶段检测器：现代目标检测的主要范式基于两阶段方法。如选择性搜索工作[35]中所开创的，第一阶段生成一组稀疏的候选提案，该提案应包含所有目标，同时过滤掉大多数负面位置，第二阶段将提案分类为前景类/背景。R-CNN[11]将第二阶段分类器升级为卷积网络，从而提高了精度，开创了现代目标检测时代。多年来，R-CNN在速度[15，10]和使用学习目标提议[6，24，28]方面都有所改进。区域候选网络(RPN)将提案生成与第二阶段分类器集成到单个卷积网络中，形成了Faster RCNN框架[28]。已经提出了对该框架的许多扩展，例如[20，31，32，16，14]。

### One-stage Detectors: 
OverFeat [30] was one of the first modern one-stage object detector based on deep networks. More recently SSD [22, 9] and YOLO [26, 27] have renewed interest in one-stage methods. These detectors have been tuned for speed but their accuracy trails that of twostage methods. SSD has a 10-20% lower AP, while YOLO focuses on an even more extreme speed/accuracy trade-off. See Figure 2. Recent work showed that two-stage detectors can be made fast simply by reducing input image resolution and the number of proposals, but one-stage methods trailed in accuracy even with a larger compute budget [17]. In contrast, the aim of this work is to understand if one-stage detectors can match or surpass the accuracy of two-stage detectors while running at similar or faster speeds.

单级检测器：OverFeat[30]是第一个基于深度网络的现代单级目标检测器之一。最近，SSD[22，9]和YOLO[26，27]重新对一步法产生了兴趣。这些检测器已经调整了速度，但其精度落后于两阶段方法。SSD的AP降低了10-20%，而YOLO专注于更极端的速度/精度权衡。参见图2。最近的工作表明，两级检测器可以通过降低输入图像分辨率和提案数量来快速制作，但即使计算预算更大，单级方法的精度也很低[17]。相比之下，这项工作的目的是了解在以相似或更快的速度运行时，单级检测器是否能够达到或超过两级检测器的精度。

The design of our RetinaNet detector shares many similarities with previous dense detectors, in particular the concept of ‘anchors’ introduced by RPN [28] and use of features pyramids as in SSD [22] and FPN [20]. We emphasize that our simple detector achieves top results not based on innovations in network design but due to our novel loss. 

我们的RetinaNet检测器的设计与以前的密集检测器有许多相似之处，特别是RPN[28]引入的“锚”概念以及SSD[22]和FPN[20]中使用的特征金字塔。我们强调，我们的简单检测器获得最佳结果并非基于网络设计的创新，而是由于我们的新颖的损失函数。

### Class Imbalance: 
Both classic one-stage object detection methods, like boosted detectors [37, 5] and DPMs [8], and more recent methods, like SSD [22], face a large class imbalance during training. These detectors evaluate 104- 105 candidate locations per image but only a few locations contain objects. This imbalance causes two problems: (1) training is inefficient as most locations are easy negatives that contribute no useful learning signal; (2) en masse, the easy negatives can overwhelm training and lead to degenerate models. A common solution is to perform some form of hard negative mining [33, 37, 8, 31, 22] that samples hard examples during training or more complex sampling/reweighing schemes [2]. In contrast, we show that our proposed focal loss naturally handles the class imbalance faced by a one-stage detector and allows us to efficiently train on all examples without sampling and without easy negatives overwhelming the loss and computed gradients.

类别不平衡：无论是经典的单级目标检测方法，如增强检测器[37，5]和DPM[8]，还是更新的方法，如SSD[22]，在训练过程中都面临较大的类别不平衡。这些检测器评估每幅图像的104-105个候选位置，但只有少数位置包含目标。这种不平衡导致了两个问题：(1)训练效率低下，因为大多数地点都是容易产生负面影响的，没有产生有用的学习信号; (2) 总的来说，简单的负面影响可能会压倒训练，导致模型退化。一种常见的解决方案是执行某种形式的困难负样本挖掘[33，37，8，31，22]，在训练或更复杂的采样/重新加权方案[2]期间对困难样本进行采样。相反，我们表明，我们提出的焦点损失自然地处理了单级检测器所面临的类别不平衡，并允许我们在没有采样的情况下有效地训练所有样本，并且不会出现压倒损失和计算梯度的简单负面。

### Robust Estimation: 
There has been much interest in designing robust loss functions (e.g., Huber loss [13]) that reduce the contribution of outliers by down-weighting the loss of examples with large errors (hard examples). In contrast, rather than addressing outliers, our focal loss is designed to address class imbalance by down-weighting inliers (easy examples) such that their contribution to the total loss is small even if their number is large. In other words, the focal loss performs the opposite role of a robust loss: it focuses training on a sparse set of hard examples.

稳健估计：人们对设计稳健损失函数(例如Huber损失[13])非常感兴趣，该函数通过对具有较大误差的样本(困难样本)的损失进行加权来减少异常值的贡献。相比之下，我们的焦点损失不是解决异常值，而是通过降低内部值的权重(简单样本)来解决类别不平衡问题，即使内部值的数量很大，它们对总损失的贡献也很小。换句话说，焦点损失与稳健损失的作用相反：它将训练集中在一组稀疏的困难样本上。

## 3. Focal Loss
The Focal Loss is designed to address the one-stage object detection scenario in which there is an extreme imbalance between foreground and background classes during training (e.g., 1:1000). We introduce the focal loss starting from the cross entropy (CE) loss for binary classification(1 Extending the focal loss to the multi-class case is straightforward and works well; for simplicity we focus on the binary loss in this work.):

Focal Loss旨在解决一阶段目标检测场景，在该场景中，训练期间前景和背景类之间存在极端不平衡(例如，1:1000)。我们从二元分类的交叉熵(CE)损失开始引入焦点损失(1将焦点损失扩展到多类情况很简单，效果很好; 为了简单起见，我们将重点放在这项工作中的二分类损失上。)：

CE(p, y) = { − log(p) if y = 1 , − log(1 − p) otherwise. } (1)

In the above y ∈ {±1} specifies the ground-truth class and p ∈ [0, 1] is the model’s estimated probability for the class with label y = 1. For notational convenience, we define $p_t$: 

在上述中y∈{±1}指定真值类和p∈[0，1]是标签为y=1的类的模型估计概率。为了便于表示，我们定义了$p_t$

$p_t$ = { p if y = 1 , 1 − p otherwise, }(2) 

and rewrite CE(p, y) = CE($p_t$) = − log($p_t$).

The CE loss can be seen as the blue (top) curve in Figure 1. One notable property of this loss, which can be easily seen in its plot, is that even examples that are easily classified ($p_t$ >> .5) incur a loss with non-trivial magnitude. When summed over a large number of easy examples, these small loss values can overwhelm the rare class. 

CE损失如图1中的蓝色(顶部)曲线所示。这种损失的一个显著特点是，即使是易分类样本($p_t$>>.5)也会产生非平凡量级的损失。当通过大量简单样本进行汇总时，这些小损失值可能会压倒罕见类别。

### 3.1. Balanced Cross Entropy
A common method for addressing class imbalance is to introduce a weighting factor α ∈ [0, 1] for class 1 and 1−α for class −1. In practice α may be set by inverse class frequency or treated as a hyperparameter to set by cross validation. For notational convenience, we define $α_t$ analogously to how we defined$p_t$. We write the α-balanced CE loss as:

解决类别不平衡的常用方法是引入加权因子α∈[0，1]用于分类1,1−α用于-1分类. 在实践中，α可以通过反类频率来设置，或者被视为通过交叉验证来设置的超参数。为了便于记法，我们将$α_t$定义为类似于我们定义$p_t$的方式。我们将α平衡CE损失写为：

$CE(p_t) = −α_tlog(p_t)$. (3)

This loss is a simple extension to CE that we consider as an experimental baseline for our proposed focal loss.

这种损失是对CE的简单扩展，我们认为这是我们提出的焦点损失的实验基线。

### 3.2. Focal Loss Definition
As our experiments will show, the large class imbalance encountered during training of dense detectors overwhelms the cross entropy loss. Easily classified negatives comprise the majority of the loss and dominate the gradient. While α balances the importance of positive/negative examples, it does not differentiate between easy/hard examples. Instead, we propose to reshape the loss function to down-weight easy examples and thus focus training on hard negatives.

正如我们的实验将显示的那样，在密集检测器训练过程中遇到的巨大的类别不平衡压倒了交叉熵损失。容易分类的负样本构成了大部分损失，并主导了梯度。虽然α平衡了正面/负面样本的重要性，但它不区分简单/困难样本。相反，我们建议重塑损失函数以减轻简易样本的权重，从而将训练重点放在困难样本上。

More formally, we propose to add a modulating factor $(1 − p_t)^γ$ to the cross entropy loss, with tunable focusing parameter γ ≥ 0. We define the focal loss as:

更正式地说，我们建议在交叉熵损失上增加一个调制因子$(1 − p_t)^γ$与，具有可调聚焦参数γ≥0. 我们将焦点损失定义为：

$FL(p_t) = −(1 − p_t)^γ log(p_t)$. (4)

The focal loss is visualized for several values of γ ∈ [0, 5] in Figure 1. We note two properties of the focal loss. (1) When an example is misclassified and $p_t$ is small, the modulating factor is near 1 and the loss is unaffected. As $p_t$ → 1, the factor goes to 0 and the loss for well-classified examples is down-weighted. (2) The focusing parameter γ smoothly adjusts the rate at which easy examples are downweighted. When γ = 0, FL is equivalent to CE, and as γ is increased the effect of the modulating factor is likewise increased (we found γ = 2 to work best in our experiments).

焦点损失在图1中可视化为γ ∈ [0, 5]。我们注意到焦点损失的两个特性。(1) 当一个样本被错误分类并且$p_t$很小时，调制因子接近1并且损失不受影响。作为$p_t$→ 1，因子变为0，分类良好的样本的损失被降权。(2) 聚焦参数γ平滑地调整了简化样本的比率。当γ=0时，FL等同于CE，并且随着γ的增加，调制因子的影响也随之增加(我们发现γ=2在我们的实验中效果最佳)。

Intuitively, the modulating factor reduces the loss contribution from easy examples and extends the range in which an example receives low loss. For instance, with γ = 2, an example classified with $p_t$ = 0.9 would have 100× lower loss compared with CE and with $p_t$≈ 0.968 it would have 1000× lower loss. This in turn increases the importance of correcting misclassified examples (whose loss is scaled down by at most 4× for $p_t$ ≤ .5 and γ = 2).

直观地，调制因子减少了来自简单样本的损失贡献，并扩展了样本接收低损失的范围。例如，当γ=2时，与CE和pt相比，分类为$p_t$=0.9的样本的损失将低100倍，当 $p_t$≈ 0.968，则损失将降低1000倍。这反过来增加了纠正错误分类样本的重要性(对于$p_t$，其损失最多减少4倍≤ .γ=2)。

In practice we use an α-balanced variant of the focal loss:

在实践中，我们使用焦点损失的α平衡变量：

$FL(p_t) = −αt(1 − p_t)^γ log(p_t)$. (5)

We adopt this form in our experiments as it yields slightly improved accuracy over the non-α-balanced form. Finally, we note that the implementation of the loss layer combines the sigmoid operation for computing p with the loss computation, resulting in greater numerical stability.

我们在实验中采用了这种形式，因为它比非α平衡形式的精度略有提高。最后，我们注意到损失层的实现将计算p的sigmoid运算与损失计算相结合，从而提高了数值稳定性。

While in our main experimental results we use the focal loss definition above, its precise form is not crucial. In the appendix we consider other instantiations of the focal loss and demonstrate that these can be equally effective.

虽然在我们的主要实验结果中，我们使用了上面的焦点损失定义，但其精确形式并不重要。在附录中，我们考虑了焦点损失的其他实例，并证明这些实例同样有效。

### 3.3. Class Imbalance and Model Initialization 类失衡和模型初始化
Binary classification models are by default initialized to have equal probability of outputting either y = −1 or 1. Under such an initialization, in the presence of class imbalance, the loss due to the frequent class can dominate total loss and cause instability in early training. To counter this, we introduce the concept of a ‘prior’ for the value of p estimated by the model for the rare class (foreground) at the start of training. We denote the prior by π and set it so that the model’s estimated p for examples of the rare class is low, e.g. 0.01. We note that this is a change in model initialization (see §4.1) and not of the loss function. We found this to improve training stability for both the cross entropy and focal loss in the case of heavy class imbalance.

默认情况下，二分类模型被初始化为具有相等的概率输出y=−1或1。在这种初始化下，在类失衡的情况下，由于高频分类而导致的损失可能会主导总损失，并导致早期训练的不稳定性。为了应对这种情况，我们引入了“先验”的概念，用于在训练开始时由罕见类(前景)的模型估计的p值。我们用π表示先验，并对其进行设置，以使罕见类样本的模型估计p为低，例如0.01。我们注意到，这是模型初始化的变化(见§4.1)，而不是损失函数的变化。我们发现，这可以改善重类失衡情况下交叉熵和焦点损失的训练稳定性。

### 3.4. Class Imbalance and Two-stage Detectors 类失衡和两级探测器
Two-stage detectors are often trained with the cross entropy loss without use of α-balancing or our proposed loss. Instead, they address class imbalance through two mechanisms: (1) a two-stage cascade and (2) biased minibatch sampling. The first cascade stage is an object proposal mechanism [35, 24, 28] that reduces the nearly infinite set of possible object locations down to one or two thousand. Importantly, the selected proposals are not random, but are likely to correspond to true object locations, which removes the vast majority of easy negatives. When training the second stage, biased sampling is typically used to construct minibatches that contain, for instance, a 1:3 ratio of positive to negative examples. This ratio is like an implicit α- balancing factor that is implemented via sampling. Our proposed focal loss is designed to address these mechanisms in a one-stage detection system directly via the loss function.

两级检测器通常使用交叉熵损失来训练，而不使用α平衡或我们提出的损失。相反，他们通过两种机制来解决类别失衡问题：(1)两级级联和(2)有偏差的小批量抽样。第单级级联是一个目标建议机制[35，24，28]，它将几乎无限多的可能目标位置减少到一到两千个。重要的是，所选择的方案不是随机的，但很可能与真实的目标位置相对应，这消除了绝大多数简单的负面影响。在训练第二阶段时，通常使用偏差抽样来构建小批次，例如，正反比为1:3的样本。这个比率就像一个通过采样实现的隐式α平衡因子。我们提出的焦点损失旨在通过损失函数直接在单级检测系统中解决这些机制。

## 4. RetinaNet Detector
RetinaNet is a single, unified network composed of a backbone network and two task-specific subnetworks. The backbone is responsible for computing a convolutional feature map over an entire input image and is an off-the-self convolutional network. The first subnet performs convolutional object classification on the backbone’s output; the second subnet performs convolutional bounding box regression. The two subnetworks feature a simple design that we propose specifically for one-stage, dense detection, see Figure 3. While there are many possible choices for the details of these components, most design parameters are not particularly sensitive to exact values as shown in the experiments. We describe each component of RetinaNet next.

RetinaNet是一个单一的、统一的网络，由一个主干网和两个任务特定的子网组成。主干负责计算整个输入图像上的卷积特征图，并且是一个非自卷积网络。第一个子网对主干网的输出执行卷积目标分类; 第二子网执行卷积边框回归。这两个子网络的特点是我们专门为单级密集检测提出的简单设计，见图3。尽管这些组件的细节有很多可能的选择，但大多数设计参数对实验中显示的精确值并不特别敏感。接下来我们将介绍RetinaNet的各个组件。

### Feature Pyramid Network Backbone: 
We adopt the Feature Pyramid Network (FPN) from [20] as the backbone network for RetinaNet. In brief, FPN augments a standard convolutional network with a top-down pathway and lateral connections so the network efficiently constructs a rich, multi-scale feature pyramid from a single resolution input image, see Figure 3(a)-(b). Each level of the pyramid can be used for detecting objects at a different scale. FPN improves multi-scale predictions from fully convolutional networks (FCN) [23], as shown by its gains for RPN [28] and DeepMask-style proposals [24], as well at two-stage detectors such as Fast R-CNN [10] or Mask R-CNN [14].

特征金字塔网络主干：我们采用[20]中的特征金字塔网络(FPN)作为RetinaNet的主干网络。简言之，FPN通过自上而下的路径和横向连接增强了标准卷积网络，因此该网络可以从单分辨率输入图像高效构建丰富的多尺度特征金字塔，见图3(a)-(b)。金字塔的每单级都可以用于检测不同尺度的目标。FPN改进了全卷积网络(FCN)[23]的多尺度预测，如RPN[28]和DeepMask风格建议[24]以及两级检测器(如Fast R-CNN[10]或Mask R-CNN[24])的增益所示。

Following [20], we build FPN on top of the ResNet architecture [16]. We construct a pyramid with levels P3 through P7, where l indicates pyramid level (Pl has resolution 2l lower than the input). As in [20] all pyramid levels have C = 256 channels. Details of the pyramid generally follow [20] with a few modest differences(2 RetinaNet uses feature pyramid levels P3 to P7, where P3 to P5 are computed from the output of the corresponding ResNet residual stage (C3 through C5) using top-down and lateral connections just as in [20], P6 is obtained via a 3×3 stride-2 conv on C5, and P7 is computed by applying ReLU followed by a 3×3 stride-2 conv on P6. This differs slightly from [20]: (1) we don’t use the high-resolution pyramid level P2 for computational reasons, (2) P6 is computed by strided convolution instead of downsampling, and (3) we include P7 to improve large object detection. These minor modifications improve speed while maintaining accuracy. ). While many design choices are not crucial, we emphasize the use of the FPN backbone is; preliminary experiments using features from only the final ResNet layer yielded low AP.

在[20]之后，我们在ResNet架构之上构建FPN[16]。我们构建了一个具有级别P3到P7的金字塔，其中l表示金字塔级别(Pl的分辨率比输入低2l)。如[20]中所示，所有金字塔级都具有C＝256个通道。金字塔的细节通常遵循[20]，但有一些适度的差异(2 RetinaNet使用特征金字塔级别P3至P7，其中P3到P5是根据相应ResNet残差级(C3至C5)的输出计算的，使用自上而下和横向连接，如[20]所示，P6是通过C5上的3×3跨2转换获得的，而P7是通过在P6上应用ReLU和3×3跨步-2卷积来计算的。这与[20]略有不同：(1)出于计算原因，我们没有使用高分辨率金字塔级别P2，(2)P6是通过跨步卷积而不是下采样来计算的，(3)我们包括P7以改进大目标检测。这些微小的修改提高了速度，同时保持了准确性。虽然许多设计选择并不重要，但我们强调FPN主干的使用是; 使用仅来自最终ResNet层的特征的初步实验产生了低AP。

### Anchors: 
We use translation-invariant anchor boxes similar to those in the RPN variant in [20]. The anchors have areas of 322 to 5122 on pyramid levels P3 to P7, respectively. As in [20], at each pyramid level we use anchors at three aspect ratios {1:2, 1:1, 2:1}. For denser scale coverage than in [20], at each level we add anchors of sizes {20, 21/3, 22/3} of the original set of 3 aspect ratio anchors. This improve AP in our setting. In total there are A = 9 anchors per level and across levels they cover the scale range 32 - 813 pixels with respect to the network’s input image.

锚：我们使用与[20]中的RPN变体中的锚框类似的平移不变锚框。锚分别在金字塔层P3至P7上具有322至5122的面积。如[20]所述，在每个金字塔级别，我们使用三种纵横比{1:2、1:1、2:1}的锚。对于比[20]中更密集的尺度覆盖，在每个级别上，我们添加了尺寸为{20，21/3，22/3}的锚，这是原始的3个纵横比锚的集合。这改善了我们环境中的AP。总的来说，每个级别有A=9个锚，并且在级别之间，相对于网络的输入图像，它们覆盖了32-813像素的尺度范围。

Each anchor is assigned a length K one-hot vector of classification targets, where K is the number of object classes, and a 4-vector of box regression targets. We use the assignment rule from RPN [28] but modified for multiclass detection and with adjusted thresholds. Specifically, anchors are assigned to ground-truth object boxes using an intersection-over-union (IoU) threshold of 0.5; and to background if their IoU is in [0, 0.4). As each anchor is assigned to at most one object box, we set the corresponding entry in its length K label vector to 1 and all other entries to 0.

每个锚被分配一个长度K，一个分类目标的one-hot向量，其中K是目标类的编号，一个边框回归目标的4向量。我们使用了RPN[28]中的分配规则，但针对多类检测进行了修改，并调整了阈值。具体地，使用0.5的交集超过联合(IoU)阈值将锚分配给地面真实目标框; 如果它们的IoU在[0，0.4)中，则将其设置为背景。由于每个锚被分配给最多一个目标框，因此我们将其长度K标签向量中的相应条目设置为1，将所有其他条目设置为0。

If an anchor is unassigned, which may happen with overlap in [0.4, 0.5), it is ignored during training. Box regression targets are computed as the offset between each anchor and its assigned object box, or omitted if there is no assignment. 

如果一个锚点未分配，这可能发生在[0.4，0.5)中的重叠，则在训练过程中忽略它。边框回归目标计算为每个锚点与其指定目标框之间的偏移，如果没有分配，则忽略。

Figure 3. The one-stage RetinaNet network architecture uses a Feature Pyramid Network (FPN) [20] backbone on top of a feedforward ResNet architecture [16] (a) to generate a rich, multi-scale convolutional feature pyramid (b). To this backbone RetinaNet attaches two subnetworks, one for classifying anchor boxes (c) and one for regressing from anchor boxes to ground-truth object boxes (d). The network design is intentionally simple, which enables this work to focus on a novel focal loss function that eliminates the accuracy gap between our one-stage detector and state-of-the-art two-stage detectors like Faster R-CNN with FPN [20] while running at faster speeds.
图3：单级RetinaNet网络架构使用前馈ResNet架构[16](a)之上的特征金字塔网络(FPN)[20]主干来生成丰富的多尺度卷积特征金字塔(b)。RetinaNet连接了两个子网，一个子网用于对锚框(c)进行分类，一个子网络用于从锚框回归到真实目标框(d)。网络设计有意简单，这使得本工作能够专注于一种新颖的焦点损失函数，消除了我们的单级探测器和最先进的两级探测器(如带FPN的Faster R-CNN[20])之间的精度差距，同时以更快的速度运行。

### Classification Subnet: 
The classification subnet predicts the probability of object presence at each spatial position for each of the A anchors and K object classes. This subnet is a small FCN attached to each FPN level; parameters of this subnet are shared across all pyramid levels. Its design is simple. Taking an input feature map with C channels from a given pyramid level, the subnet applies four 3×3 conv layers, each with C filters and each followed by ReLU activations, followed by a 3×3 conv layer with KA filters. Finally sigmoid activations are attached to output the KA binary predictions per spatial location, see Figure 3 (c). We use C = 256 and A = 9 in most experiments.

分类子网：分类子网预测A个锚和K个目标类在每个空间位置的目标存在概率。此子网是连接到每个FPN级别的小型FCN; 此子网的参数在所有金字塔级别上共享。它的设计很简单。从给定的金字塔级别获取具有C个通道的输入特征图，子网应用四个3×3 conv层，每个层具有C个过滤器，每个层之后是ReLU激活，然后是具有KA过滤器的3×3 conv层。最后，附加sigmoid激活以输出每个空间位置的KA二进制预测，见图3(c)。我们在大多数实验中使用C=256和A=9。

In contrast to RPN [28], our object classification subnet is deeper, uses only 3×3 convs, and does not share parameters with the box regression subnet (described next). We found these higher-level design decisions to be more important than specific values of hyperparameters.

与RPN[28]相比，我们的目标分类子网更深，仅使用3×3个conv，并且不与边框回归子网共享参数(如下所述)。我们发现这些更高级别的设计决策比超参数的特定值更重要。

### Box Regression Subnet: 
In parallel with the object classi- fication subnet, we attach another small FCN to each pyramid level for the purpose of regressing the offset from each anchor box to a nearby ground-truth object, if one exists. The design of the box regression subnet is identical to the classification subnet except that it terminates in 4A linear outputs per spatial location, see Figure 3 (d). For each of the A anchors per spatial location, these 4 outputs predict the relative offset between the anchor and the groundtruth box (we use the standard box parameterization from RCNN [11]). We note that unlike most recent work, we use a class-agnostic bounding box regressor which uses fewer parameters and we found to be equally effective. The object classification subnet and the box regression subnet, though sharing a common structure, use separate parameters.

边框回归子网：与目标分类子网并行，我们将另一个小FCN附加到每个金字塔级别，以便将每个锚框到附近地面真实目标(如果存在)的偏移回归。箱式回归子网的设计与分类子网相同，不同的是它在每个空间位置终止于4A线性输出，见图3(d)。对于每个空间位置的每个A锚，这4个输出预测了锚和地面真值框之间的相对偏移(我们使用RCNN[11]中的标准框参数化)。我们注意到，与最近的工作不同，我们使用了一个使用较少参数的类未知边界框回归器，我们发现它同样有效。目标分类子网和边框回归子网

### 4.1. Inference and Training
#### Inference: 
RetinaNet forms a single FCN comprised of a ResNet-FPN backbone, a classification subnet, and a box regression subnet, see Figure 3. As such, inference involves simply forwarding an image through the network. To improve speed, we only decode box predictions from at most 1k top-scoring predictions per FPN level, after thresholding detector confidence at 0.05. The top predictions from all levels are merged and non-maximum suppression with a threshold of 0.5 is applied to yield the final detections.

RetinaNet形成了一个由ResNet FPN主干网、分类子网和边框回归子网组成的单个FCN，见图3。因此，推理仅涉及通过网络发图像。为了提高速度，在将检测器置信度阈值设置为0.05之后，我们仅对每个FPN级别最多1k个最高得分预测中的框预测进行解码。合并所有级别的最高预测，并应用阈值为0.5的非最大抑制来产生最终检测。

#### Focal Loss: 
We use the focal loss introduced in this work as the loss on the output of the classification subnet. As we will show in §5, we find that γ = 2 works well in practice and the RetinaNet is relatively robust to γ ∈ [0.5, 5]. We emphasize that when training RetinaNet, the focal loss is applied to all ∼100k anchors in each sampled image. This stands in contrast to common practice of using heuristic sampling (RPN) or hard example mining (OHEM, SSD) to select a small set of anchors (e.g., 256) for each minibatch. The total focal loss of an image is computed as the sum of the focal loss over all ∼100k anchors, normalized by the number of anchors assigned to a ground-truth box. We perform the normalization by the number of assigned anchors, not total anchors, since the vast majority of anchors are easy negatives and receive negligible loss values under the focal loss. Finally we note that α, the weight assigned to the rare class, also has a stable range, but it interacts with γ making it necessary to select the two together (see Tables 1a and 1b). In general α should be decreased slightly as γ is increased (for γ = 2, α = 0.25 works best).

焦点损失：我们使用本工作中引入的焦点损失作为分类子网输出的损失。如我们将在§5中所示，我们发现γ=2在实践中工作良好，并且RetinaNet对γ∈ [0.5, 5]. 我们强调，在训练RetinaNet时，焦点损失适用于所有∼每个采样图像中有100k个锚。这与使用启发式采样(RPN)或困难样本挖掘(OHEM，SSD)为每个小批量选择一小组锚(例如，256)的常见做法形成对比。图像的总焦点损失计算为全部焦点损失的总和∼100k个锚点，由分配给真值框的锚点数量归一化。我们通过指定锚的数量而不是总锚的数量来执行标准化，因为绝大多数锚都是简单的负值，并且在焦点损失下接收到的损失值可以忽略不计。最后，我们注意到，分配给稀有类的权重α也有一个稳定的范围，但它与γ相互作用，因此有必要将两者一起选择(见表1a和1b)。通常，随着γ的增加，α应略微减小(对于γ=2，α=0.25效果最好)。

#### Initialization: 
We experiment with ResNet-50-FPN and ResNet-101-FPN backbones [20]. The base ResNet-50 and ResNet-101 models are pre-trained on ImageNet1k; we use the models released by [16]. New layers added for FPN are initialized as in [20]. All new conv layers except the final one in the RetinaNet subnets are initialized with bias b = 0 and a Gaussian weight fill with σ = 0.01. For the final conv layer of the classification subnet, we set the bias initialization to b = − log((1 − π)/π), where π specifies that at the start of training every anchor should be labeled as foreground with confidence of ∼π. We use π = .01 in all experiments, although results are robust to the exact value. As explained in §3.3, this initialization prevents the large number of background anchors from generating a large, destabilizing loss value in the first iteration of training.

初始化：我们使用ResNet-50-FPN和ResNet-101-FPN主干进行实验[20]。基本ResNet-50和ResNet-101模型在ImageNet1k上预先训练; 我们使用[16]发布的模型。为FPN添加的新层如[20]所示进行初始化。除RetinaNet子网中的最后一层外，所有新的conv层都使用偏差b=0和σ=0.01的高斯权重填充进行初始化。对于分类子网的最后一个conv层，我们将偏差初始化设置为 b=−log((1−π)/π)，其中π指定在训练开始时，每个锚应标记为前景，置信度为∼π. 我们在所有实验中使用π=.01，尽管结果对精确值是稳健的。如§3.3所述，该初始化可防止大量背景锚在第一次训练迭代中产生较大的不稳定损失值。

Table 1. Ablation experiments for RetinaNet and Focal Loss (FL). All models are trained on trainval35k and tested on minival unless noted. If not specified, default values are: γ = 2; anchors for 3 scales and 3 aspect ratios; ResNet-50-FPN backbone; and a 600 pixel train and test image scale. (a) RetinaNet with α-balanced CE achieves at most 31.1 AP. (b) In contrast, using FL with the same exact network gives a 2.9 AP gain and is fairly robust to exact γ/α settings. (c) Using 2-3 scale and 3 aspect ratio anchors yields good results after which point performance saturates. (d) FL outperforms the best variants of online hard example mining (OHEM) [31, 22] by over 3 points AP. (e) Accuracy/Speed trade-off of RetinaNet on test-dev for various network depths and image scales (see also Figure 2). 

表1.RetinaNet和焦点损失(FL)的消融实验。除非另有说明，所有模型均在trainval35k上进行训练，并在minival上进行测试。如果未指定，默认值为：γ=2; 3个比例和3个纵横比的锚; ResNet-50-FPN主干; 以及600像素序列和测试图像尺度。(a) α平衡CE的RetinaNet最多可达到31.1 AP。(b) 相反，使用具有相同精确网络的FL可获得2.9 AP增益，并且对精确的γ/α设置相当稳健。(c) 使用2-3个比例和3个纵横比锚定器产生良好的结果，之后性能饱和。(d) FL优于在线困难样本挖掘(OHEM)的最佳变体[31，22]，AP超过3点。(e) RetinaNet在各种网络深度和图像比例的测试开发中的精度/速度权衡(另请参见图2)。

#### Optimization: 
RetinaNet is trained with stochastic gradient descent (SGD). We use synchronized SGD over 8 GPUs with a total of 16 images per minibatch (2 images per GPU). Unless otherwise specified, all models are trained for 90k iterations with an initial learning rate of 0.01, which is then divided by 10 at 60k and again at 80k iterations. We use horizontal image flipping as the only form of data augmentation unless otherwise noted. Weight decay of 0.0001 and momentum of 0.9 are used. The training loss is the sum the focal loss and the standard smooth L1 loss used for box regression [10]. Training time ranges between 10 and 35 hours for the models in Table 1e.

优化：使用随机梯度下降(SGD)训练RetinaNet。我们在8个GPU上使用同步的SGD，每个小批量总共16个图像(每个GPU 2个图像)。除非另有规定，否则所有模型都以0.01的初始学习率进行90k次迭代训练，然后在60k次迭代时再除以10，在80k次迭代中再除以10。除非另有说明，我们使用水平图像翻转作为数据增广的唯一形式。使用0.0001的重量衰减和0.9的动量。训练损失是用于边框回归的焦点损失和标准平滑L1损失之和[10]。表1e中模型的训练时间范围为10至35小时。

## 5. Experiments
We present experimental results on the bounding box detection track of the challenging COCO benchmark [21]. For training, we follow common practice [1, 20] and use the COCO trainval35k split (union of 80k images from train and a random 35k subset of images from the 40k image val split). We report lesion and sensitivity studies by evaluating on the minival split (the remaining 5k images from val). For our main results, we report COCO AP on the test-dev split, which has no public labels and requires use of the evaluation server.

我们在具有挑战性的COCO基准的边框检测追踪上给出了实验结果[21]。对于训练，我们遵循常见做法[1，20]并使用COCO trainval35k分割(来自训练的80k图像和来自40k图像val分割的随机35k图像子集的联合)。我们报告了通过评估最小val分裂(val的剩余5k图像)进行的损伤和敏感性研究。对于我们的主要结果，我们报告了COCOAP的测试开发拆分，它没有公共标签，需要使用评估服务器。

### 5.1. Training Dense Detection
We run numerous experiments to analyze the behavior of the loss function for dense detection along with various optimization strategies. For all experiments we use depth 50 or 101 ResNets [16] with a Feature Pyramid Network (FPN) [20] constructed on top. For all ablation studies we use an image scale of 600 pixels for training and testing.

我们运行了大量实验来分析密集检测的损失函数的行为以及各种优化策略。对于所有实验，我们使用深度为50或101的ResNets[16]，顶部构建了特征金字塔网络(FPN)[20]。对于所有消融研究，我们使用600像素的图像比例进行训练和测试。

#### Network Initialization: 
Our first attempt to train RetinaNet uses standard cross entropy (CE) loss without any modifications to the initialization or learning strategy. This fails quickly, with the network diverging during training. However, simply initializing the last layer of our model such that the prior probability of detecting an object is π = .01 (see §4.1) enables effective learning. Training RetinaNet with ResNet-50 and this initialization already yields a respectable AP of 30.2 on COCO. Results are insensitive to the exact value of π so we use π = .01 for all experiments. 

网络初始化：我们第一次尝试训练RetinaNet使用标准交叉熵(CE)损失，而不需要对初始化或学习策略进行任何修改。这很快就失败了，因为网络在训练过程中出现了分歧。然而，简单地初始化我们模型的最后一层，使得检测到目标的先验概率为π=.01(参见§4.1)，就可以实现有效的学习。使用ResNet-50训练RetinaNet，并且此初始化已经在COCO上产生了30.2的可观AP。结果对π的精确值不敏感，因此我们使用π=.01进行所有实验。

Figure 4. Cumulative distribution functions of the normalized loss for positive and negative samples for different values of γ for a converged model. The effect of changing γ on the distribution of the loss for positive examples is minor. For negatives, however, increasing γ heavily concentrates the loss on hard examples, focusing nearly all attention away from easy negatives.

图4.收敛模型中不同γ值正负样本归一化损失的累积分布函数。对于正例，改变γ对损失分布的影响很小。然而，对于负片，增加γ值会使损失严重集中在困难的样本上，几乎所有的注意力都集中在简单的负片上。

#### Balanced Cross Entropy: 
Our next attempt to improve learning involved using the α-balanced CE loss described in §3.1. Results for various α are shown in Table 1a. Setting α = .75 gives a gain of 0.9 points AP.

平衡交叉熵：我们下一次尝试使用§3.1中描述的α平衡CE损失来改进学习。各种α的结果如表1a所示。设置α=.75可获得0.9点AP增益。

#### Focal Loss: 
Results using our proposed focal loss are shown in Table 1b. The focal loss introduces one new hyperparameter, the focusing parameter γ, that controls the strength of the modulating term. When γ = 0, our loss is equivalent to the CE loss. As γ increases, the shape of the loss changes so that “easy” examples with low loss get further discounted, see Figure 1. FL shows large gains over CE as γ is increased. With γ = 2, FL yields a 2.9 AP improvement over the α-balanced CE loss.

焦点损失：表1b显示了使用我们提出的焦点损失的结果。焦点损失引入了一个新的超参数，即聚焦参数γ，它控制调制项的强度。当γ=0时，我们的损失等于CE损失。随着γ的增加，损失的形状发生了变化，因此低损失的“简单”样本会进一步降低，见图1。随着γ的增大，FL比CE有较大的增益。当γ=2时，FL比α平衡CE损失提高2.9 AP。

For the experiments in Table 1b, for a fair comparison we find the best α for each γ. We observe that lower α’s are selected for higher γ’s (as easy negatives are downweighted, less emphasis needs to be placed on the positives). Overall, however, the benefit of changing γ is much larger, and indeed the best α’s ranged in just [.25,.75] (we tested α ∈ [.01, .999]). We use γ = 2.0 with α = .25 for all experiments but α = .5 works nearly as well (.4 AP lower).

对于表1b中的实验，为了公平比较，我们发现每个γ的最佳α。我们观察到，较低的α值被选择用于较高的γ值(因为容易的负值被下调，所以不需要太强调正值)。然而，总的来说，改变γ的益处要大得多，实际上最好的α的范围仅为[25，.75](我们测试了α∈ [.01, .999]). 对于所有实验，我们使用γ=2.0，α=.25，但α=.5几乎同样有效(0.4AP更低)。

#### Analysis of the Focal Loss: 
To understand the focal loss better, we analyze the empirical distribution of the loss of a converged model. For this, we take take our default ResNet- 101 600-pixel model trained with γ = 2 (which has 36.0 AP). We apply this model to a large number of random images and sample the predicted probability for ∼107 negative windows and ∼105 positive windows. Next, separately for positives and negatives, we compute FL for these samples, and normalize the loss such that it sums to one. Given the normalized loss, we can sort the loss from lowest to highest and plot its cumulative distribution function (CDF) for both positive and negative samples and for different settings for γ (even though model was trained with γ = 2).

焦点损失分析：为了更好地理解焦点损失，我们分析了收敛模型的损失的经验分布。为此，我们采用我们的默认ResNet-101 600像素模型，该模型使用γ=2(其AP为36.0)进行训练。我们将该模型应用于大量随机图像，并对预测概率进行采样∼107个负窗口和∼105个阳性窗口。接下来，分别针对正值和负值，我们计算这些样本的FL，并将损失归一化，使其总和为1。给定归一化损失，我们可以从最低到最高对损失进行排序，并绘制正样本和负样本以及γ的不同设置的累积分布函数(CDF)(即使模型是用γ=2训练的)。

Cumulative distribution functions for positive and negative samples are shown in Figure 4. If we observe the positive samples, we see that the CDF looks fairly similar for different values of γ. For example, approximately 20% of the hardest positive samples account for roughly half of the positive loss, as γ increases more of the loss gets concentrated in the top 20% of examples, but the effect is minor.

阳性和阴性样本的累积分布函数如图4所示。如果我们观察阳性样本，我们会发现不同γ值的CDF看起来相当相似。例如，大约20%的最硬的阳性样品约占阳性损失的一半，因为γ增加，更多的损失集中在前20%的样品中，但影响很小。

The effect of γ on negative samples is dramatically different. For γ = 0, the positive and negative CDFs are quite similar. However, as γ increases, substantially more weight becomes concentrated on the hard negative examples. In fact, with γ = 2 (our default setting), the vast majority of the loss comes from a small fraction of samples. As can be seen, FL can effectively discount the effect of easy negatives, focusing all attention on the hard negative examples.

γ对阴性样品的影响有很大不同。对于γ=0，正负CDF非常相似。然而，随着γ的增加，更多的重量集中在硬负片上。事实上，当γ=2(我们的默认设置)时，绝大多数损失来自一小部分样本。可以看出，FL可以有效地淡化简单负面的影响，将所有注意力集中在硬负面的样本上。

#### Online Hard Example Mining (OHEM): 
[31] proposed to improve training of two-stage detectors by constructing minibatches using high-loss examples. Specifically, in OHEM each example is scored by its loss, non-maximum suppression (nms) is then applied, and a minibatch is constructed with the highest-loss examples. The nms threshold and batch size are tunable parameters. Like the focal loss, OHEM puts more emphasis on misclassified examples, but unlike FL, OHEM completely discards easy examples. We also implement a variant of OHEM used in SSD [22]: after applying nms to all examples, the minibatch is constructed to enforce a 1:3 ratio between positives and negatives to help ensure each minibatch has enough positives.

在线困难样本挖掘(OHEM)：[31]建议通过使用高损失样本构建小批量来改进两阶段检测器的训练。具体地说，在OHEM中，每个样本都通过其损失进行评分，然后应用非最大抑制(nms)，并用损失最高的样本构建一个小批次。nms阈值和批大小是可调参数。与焦点损失一样，OHEM更强调错误分类的样本，但与FL不同，OHEM完全放弃了简单的样本。我们还实现了SSD中使用的OHEM的一个变体[22]：在将nms应用于所有样本之后，构建小批次以强制执行正负比1:3，以帮助确保每个小批次具有足够的正值。

We test both OHEM variants in our setting of one-stage detection which has large class imbalance. Results for the original OHEM strategy and the ‘OHEM 1:3’ strategy for selected batch sizes and nms thresholds are shown in Table 1d. These results use ResNet-101, our baseline trained with FL achieves 36.0 AP for this setting. In contrast, the best setting for OHEM (no 1:3 ratio, batch size 128, nms of .5) achieves 32.8 AP. This is a gap of 3.2 AP, showing FL is more effective than OHEM for training dense detectors. We note that we tried other parameter setting and variants for OHEM but did not achieve better results.

我们在我们的单级检测设置中测试两种OHEM变体，该设置具有较大的类别不平衡。表1d显示了原始OHEM策略和“OHEM 1:3”策略对选定批次大小和nms阈值的结果。这些结果使用ResNet-101，我们使用FL训练的基线在该设置下达到36.0 AP。相比之下，OHEM的最佳设置(没有1:3的比例，批次大小128，nms为.5)达到32.8 AP。这是3.2 AP的差距，表明FL比OHEM更有效地训练密集探测器。我们注意到，我们尝试了OHEM的其他参数设置和变体，但没有获得更好的结果。

#### Hinge Loss: 
Finally, in early experiments, we attempted to train with the hinge loss [13] on $p_t$, which sets loss to 0 above a certain value of $p_t$. However, this was unstable and we did not manage to obtain meaningful results. Results exploring alternate loss functions are in the appendix. 

Hinge损失：最后，在早期的实验中，我们尝试用$p_t$上的Hinge损失[13]进行训练，这将损失设置为0，高于$p_t$的某个值。然而，这是不稳定的，我们未能获得有意义的结果。附录中列出了探索替代损失函数的结果。

Table 2. Object detection single-model results (bounding box AP), vs. state-of-the-art on COCO test-dev. We show results for our RetinaNet-101-800 model, trained with scale jitter and for 1.5× longer than the same model from Table 1e. Our model achieves top results, outperforming both one-stage and two-stage models. For a detailed breakdown of speed versus accuracy see Table 1e and Figure 2.

表2.目标检测单个模型结果(边界框AP)与COCO测试版上的最新技术。我们展示了使用比例抖动训练的RetinaNet-101-800模型的结果，该模型比表1e中的相同模型长1.5倍。我们的模型取得了最好的结果，优于一阶段和两阶段模型。有关速度与精度的详细细分，请参见表1e和图2。

### 5.2. Model Architecture Design
Anchor Density: One of the most important design factors in a one-stage detection system is how densely it covers the space of possible image boxes. Two-stage detectors can classify boxes at any position, scale, and aspect ratio using a region pooling operation [10]. In contrast, as one-stage detectors use a fixed sampling grid, a popular approach for achieving high coverage of boxes in these approaches is to use multiple ‘anchors’ [28] at each spatial position to cover boxes of various scales and aspect ratios.

锚密度：单级检测系统中最重要的设计因素之一是它覆盖可能的图像框空间的密度。两级探测器可以使用区域池操作对任何位置、比例和纵横比的边框进行分类[10]。相比之下，由于单级探测器使用固定的采样网格，在这些方法中实现边框高覆盖率的一种流行方法是在每个空间位置使用多个“锚”[28]来覆盖不同尺度和纵横比的边框。

We sweep over the number of scale and aspect ratio anchors used at each spatial position and each pyramid level in FPN. We consider cases from a single square anchor at each location to 12 anchors per location spanning 4 sub-octave scales (2 k/4 , for k ≤ 3) and 3 aspect ratios [0.5, 1, 2]. Results using ResNet-50 are shown in Table 1c. A surprisingly good AP (30.3) is achieved using just one square anchor. However, the AP can be improved by nearly 4 points (to 34.0) when using 3 scales and 3 aspect ratios per location.
We used this setting for all other experiments in this work.

我们扫描了FPN中每个空间位置和每个金字塔级别使用的比例和纵横比锚的数量。我们考虑从每个位置的单个方形锚到每个位置的12个锚跨越4个次八度音阶(2k/4，对于k≤ 3) 和3个纵横比[0.5，1，2]。使用ResNet-50的结果如表1c所示。仅使用一个方形锚就实现了令人惊讶的良好AP(30.3)。然而，当每个位置使用3个尺度和3个纵横比时，AP可以提高近4个点(达到34.0)。在这项工作中，我们将此设置用于所有其他实验。

Finally, we note that increasing beyond 6-9 anchors did not shown further gains. Thus while two-stage systems can classify arbitrary boxes in an image, the saturation of performance w.r.t. density implies the higher potential density of two-stage systems may not offer an advantage.

最后，我们注意到，增加超过6-9个锚并没有显示出进一步的收益。因此，虽然两级系统可以对图像中的任意框进行分类，但性能相对密度的饱和意味着两级系统的较高潜在密度可能不会带来优势。

Speed versus Accuracy: Larger backbone networks yield higher accuracy, but also slower inference speeds. Likewise for input image scale (defined by the shorter image side). We show the impact of these two factors in Table 1e. In Figure 2 we plot the speed/accuracy trade-off curve for RetinaNet and compare it to recent methods using public numbers on COCO test-dev. The plot reveals that RetinaNet, enabled by our focal loss, forms an upper envelope over all existing methods, discounting the low-accuracy regime. RetinaNet with ResNet-101-FPN and a 600 pixel image scale (which we denote by RetinaNet-101-600 for simplicity) matches the accuracy of the recently published ResNet- 101-FPN Faster R-CNN [20], while running in 122 ms per image compared to 172 ms (both measured on an Nvidia M40 GPU). Using larger scales allows RetinaNet to surpass the accuracy of all two-stage approaches, while still being faster. For faster runtimes, there is only one operating point (500 pixel input) at which using ResNet-50-FPN improves over ResNet-101-FPN. Addressing the high frame rate regime will likely require special network design, as in [27], and is beyond the scope of this work. We note that after publication, faster and more accurate results can now be obtained by a variant of Faster R-CNN from [12].

速度与准确性：较大的主干网络产生更高的准确性，但推理速度较慢。同样，对于输入图像比例(由较短的图像侧定义)。我们在表1e中显示了这两个因素的影响。在图2中，我们绘制了RetinaNet的速度/精度折衷曲线，并将其与最近使用COCO test-dev上的公共数字的方法进行了比较。该图显示，RetinaNet，由我们的焦点损失启用，在所有现有方法中形成了一个较高的包络线，不考虑低精度的情况。具有ResNet-101-FPN和600像素图像标度的RetinaNet(为了简单起见，我们将其表示为RetinaNet-101-600)与最近发布的ResNet-101-FPN Faster R-CNN[20]的精度相匹配，而每幅图像的运行时间为122 ms，而不是172ms(均在Nvidia M40 GPU上测量)。使用更大的尺度可以让RetinaNet超越所有两阶段方法的精度，同时速度仍然更快。对于更快的运行时间，只有一个操作点(500像素输入)使用ResNet-50-FPN比ResNet-101-FPN提高。如[27]所述，解决高帧速率体制可能需要特殊的网络设计，这超出了本工作的范围。我们注意到，发表后，现在可以通过[12]中faster R-CNN的变体获得更快、更准确的结果。

### 5.3. Comparison to State of the Art
We evaluate RetinaNet on the challenging COCO dataset and compare test-dev results to recent state-of-the-art methods including both one-stage and two-stage models. Results are presented in Table 2 for our RetinaNet-101-800 model trained using scale jitter and for 1.5× longer than the models in Table 1e (giving a 1.3 AP gain). Compared to existing one-stage methods, our approach achieves a healthy 5.9 point AP gap (39.1 vs. 33.2) with the closest competitor, DSSD [9], while also being faster, see Figure 2. Compared to recent two-stage methods, RetinaNet achieves a 2.3 point gap above the top-performing Faster R-CNN model based on Inception-ResNet-v2-TDM [32]. Plugging in ResNeXt- 32x8d-101-FPN [38] as the RetinaNet backbone further improves results another 1.7 AP, surpassing 40 AP on COCO.

我们在具有挑战性的COCO数据集上评估RetinaNet，并将测试开发结果与最新的先进方法(包括一阶段和两阶段模型)进行比较。表2给出了我们使用尺度抖动训练的RetinaNet-101-800模型的结果，该模型比表1e中的模型长1.5倍(给出1.3 AP增益)。与现有的一阶段方法相比，我们的方法与最接近的竞争对手DSSD[9]实现了健康的5.9点AP差距(39.1对33.2)，同时速度更快，见图2。与最近的两阶段方法相比相比，RetinaNet实现了2.3点差距，高于基于Inception-ResNet-v2-TDM[32]的表现最佳的faster-R-CNN模型。将ResNeXt-32x8d-101-FPN[38]作为RetinaNet主干进一步提高了1.7个AP的性能，超过了COCO上的40个AP。


## 6. Conclusion
In this work, we identify class imbalance as the primary obstacle preventing one-stage object detectors from surpassing top-performing, two-stage methods. To address this, we propose the focal loss which applies a modulating term to the cross entropy loss in order to focus learning on hard negative examples. Our approach is simple and highly effective. We demonstrate its efficacy by designing a fully convolutional one-stage detector and report extensive experimental analysis showing that it achieves stateof-the-art accuracy and speed. Source code is available at https://github.com/facebookresearch/Detectron [12]. 

在这项工作中，我们将类不平衡确定为阻止单级目标检测器超越性能最好的两级方法的主要障碍。为了解决这一问题，我们提出了将调制项应用于交叉熵损失的焦点损失，以便将学习集中在硬反例上。我们的方法简单有效。我们通过设计一个完全卷积的单级检测器来证明其有效性，并报告了大量的实验分析，表明其达到了最先进的精度和速度。源代码位于https://github.com/facebookresearch/Detectron [12].

Figure 5. Focal loss variants compared to the cross entropy as a function of xt = yx. Both the original FL and alternate variant FL∗ reduce the relative loss for well-classified examples (xt > 0). 
图5.焦点损失变量与交叉熵的比较，作为xt=yx的函数。原始FL和备选FL∗ 减少分类良好的样本的相对损失(xt>0)。

Table 3. Results of FL and FL∗ versus CE for select settings.
表3.FL和FL结果∗ 而CE用于选择设置。

## Appendix A: Focal Loss*
The exact form of the focal loss is not crucial. We now show an alternate instantiation of the focal loss that has similar properties and yields comparable results. The following also gives more insights into properties of the focal loss.

焦点损失的确切形式并不重要。我们现在展示了焦点损失的另一个实例，它具有相似的属性，并产生了可比较的结果。以下内容还提供了对焦点损失特性的更多见解。

We begin by considering both cross entropy (CE) and the focal loss (FL) in a slightly different form than in the main text. Specifically, we define a quantity $x_t$ as follows: 

我们首先以与正文略有不同的形式来考虑交叉熵(CE)和焦点损失(FL)。具体来说，我们定义一个数量$x_t$如下：

$x_t = yx$, (6) 

where y ∈ {±1} specifies the ground-truth class as before. We can then write$p_t$= σ($x_t$) (this is compatible with the definition of$p_t$in Equation 2). An example is correctly classified when $x_t$ > 0, in which case $p_t$ > .5.

其中y∈ ｛±1｝如前所述指定真实值等级。然后我们可以写出$p_t$=σ($x_t$)(这与方程2中$p_t$的定义兼容)。当$x_t$>0时，样本被正确分类，在这种情况下$p_t$>.5。

We can now define an alternate form of the focal loss in terms of $x_t$ . We define $p^∗_t$ and $FL^∗$ as follows: 

我们现在可以用$x_t$定义焦点损失的另一种形式^∗_t$和$FL^∗$ 如下所示：

$p^∗_t = σ(γx_t + β)$, (7)

$FL^∗ = − log(p^∗_t )/γ$. (8)

$FL^∗$ has two parameters, γ and β, that control the steepness and shift of the loss curve. We plot $FL^∗$ for two selected settings of γ and β in Figure 5 alongside CE and FL. As can be seen, like FL, $FL^∗$ with the selected parameters diminishes the loss assigned to well-classified examples.

$FL^∗$ 具有两个参数，γ和β，控制损耗曲线的陡度和偏移。我们绘制$FL^∗$ 图5中的γ和β以及CE和FL的两个选定设置。如图所示，如FL、$FL^∗$ 利用所选择的参数减少分配给分类良好的样本的损失。

We trained RetinaNet-50-600 using identical settings as before but we swap out FL for $FL^∗$ with the selected parameters. These models achieve nearly the same AP as those trained with FL, see Table 3. In other words, $FL^∗$ is a reasonable alternative for the FL that works well in practice. 

我们使用与之前相同的设置训练RetinaNet-50-600，但我们将FL换成$FL^∗$ 使用选定的参数。这些模型获得的AP与接受FL训练的模型几乎相同，见表3。换句话说，$FL^∗$ 是FL的合理替代方案，在实践中效果良好。

Figure 6. Derivates of the loss functions from Figure 5 w.r.t. x. 
图6.图5 w.r.t.x中损失函数的推导。

Figure 7. Effectiveness of $FL^∗$ with various settings γ and β. The plots are color coded such that effective settings are shown in blue.
图7.$FL的有效性^∗$ 具有不同的设置γ和β。绘图采用颜色编码，有效设置以蓝色显示。

We found that various γ and β settings gave good results. In Figure 7 we show results for RetinaNet-50-600 with $FL^∗$ for a wide set of parameters. The loss plots are color coded such that effective settings (models converged and with AP over 33.5) are shown in blue. We used α = .25 in all experiments for simplicity. As can be seen, losses that reduce weights of well-classified examples ($x_t$ > 0) are effective.

我们发现，不同的γ和β设置给出了良好的结果。在图7中，我们显示了使用$FL的RetinaNet-50-600的结果^∗$ 用于广泛的参数集。损失图是彩色编码的，因此有效设置(模型聚合且AP超过33.5)以蓝色显示。为了简单起见，我们在所有实验中都使用α=0.25。可以看出，降低分类良好的样本($x_t$>0)权重的损失是有效的。

More generally, we expect any loss function with similar properties as FL or $FL^∗$ to be equally effective.
更一般地，我们期望任何损失函数具有与FL或$FL相似的性质^∗$ 同样有效。

## Appendix B: Derivatives
For reference, derivates for CE, FL, and $FL^∗$ w.r.t. x are: 

$\frac{dCE}{dx} = y(p_t − 1)$ (9) 

$\frac{dFL}{dx} = y(1 − p_t)^γ(γp_t log(p_t) + p_t − 1)$ (10) 

$\frac{dFL^∗}{dx} = y(p^∗_t − 1)$ (11)

Plots for selected settings are shown in Figure 6. For all loss functions, the derivative tends to -1 or 0 for high-confidence predictions. However, unlike CE, for effective settings of both FL and $FL^∗$ , the derivative is small as soon as $x_t$ > 0. 9 .

## References
1. S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Insideoutside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016. 6
2. S. R. Bulo, G. Neuhold, and P. Kontschieder. Loss maxpooling for semantic image segmentation. In CVPR, 2017. 3
3. J. Dai, Y. Li, K. He, and J. Sun. R-FCN: Object detection via region-based fully convolutional networks. In NIPS, 2016. 1
4. N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In CVPR, 2005. 2
5. P. Doll´ar, Z. Tu, P. Perona, and S. Belongie. Integral channel features. In BMVC, 2009. 2, 3
6. D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov. Scalable object detection using deep neural networks. In CVPR, 2014. 2
7. M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes (VOC) Challenge. IJCV, 2010. 2
8. P. F. Felzenszwalb, R. B. Girshick, and D. McAllester. Cascade object detection with deformable part models. In CVPR, 2010. 2, 3
9. C.-Y. Fu, W. Liu, A. Ranga, A. Tyagi, and A. C. Berg. DSSD: Deconvolutional single shot detector. arXiv:1701.06659, 2016. 1, 2, 8
10. R. Girshick. Fast R-CNN. In ICCV, 2015. 1, 2, 4, 6, 8
11. R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014. 1, 2, 5
12. R. Girshick, I. Radosavovic, G. Gkioxari, P. Doll´ar, and K. He. Detectron. https://github.com/ facebookresearch/detectron, 2018. 8
13. T. Hastie, R. Tibshirani, and J. Friedman. The elements of statistical learning. Springer series in statistics Springer, Berlin, 2008. 3, 7
14. K. He, G. Gkioxari, P. Doll´ar, and R. Girshick. Mask RCNN. In ICCV, 2017. 1, 2, 4
15. K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV. 2014. 2
16. K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016. 2, 4, 5, 6, 8
17. J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara, A. Fathi, I. Fischer, Z. Wojna, Y. Song, S. Guadarrama, and K. Murphy. Speed/accuracy trade-offs for modern convolutional object detectors. In CVPR, 2017. 2, 8
18. A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012. 2
19. Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989. 2
20. T.-Y. Lin, P. Doll´ar, R. Girshick, K. He, B. Hariharan, and S. Belongie. Feature pyramid networks for object detection. In CVPR, 2017. 1, 2, 4, 5, 6, 8
21. T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Doll´ar, and C. L. Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014. 1, 6
22. W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed. SSD: Single shot multibox detector. In ECCV, 2016. 1, 2, 3, 6, 7, 8
23. J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015. 4
24. P. O. Pinheiro, R. Collobert, and P. Dollar. Learning to segment object candidates. In NIPS, 2015. 2, 4
25. P. O. Pinheiro, T.-Y. Lin, R. Collobert, and P. Doll´ar. Learning to refine object segments. In ECCV, 2016. 2
26. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. In CVPR, 2016. 1, 2
27. J. Redmon and A. Farhadi. YOLO9000: Better, faster, stronger. In CVPR, 2017. 1, 2, 8
28. S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015. 1, 2, 4, 5, 8
29. H. Rowley, S. Baluja, and T. Kanade. Human face detection in visual scenes. Technical Report CMU-CS-95-158R, Carnegie Mellon University, 1995. 2
30. P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014. 2
31. A. Shrivastava, A. Gupta, and R. Girshick. Training regionbased object detectors with online hard example mining. In CVPR, 2016. 2, 3, 6, 7
32. A. Shrivastava, R. Sukthankar, J. Malik, and A. Gupta. Beyond skip connections: Top-down modulation for object detection. arXiv:1612.06851, 2016. 2, 8
33. K.-K. Sung and T. Poggio. Learning and Example Selection for Object and Pattern Detection. In MIT A.I. Memo No. 1521, 1994. 2, 3
34. C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. In AAAI Conference on Artificial Intelligence, 2017. 8
35. J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders. Selective search for object recognition. IJCV, 2013. 2, 4
36. R. Vaillant, C. Monrocq, and Y. LeCun. Original approach for the localisation of objects in images. IEE Proc. on Vision, Image, and Signal Processing, 1994. 2
37. P. Viola and M. Jones. Rapid object detection using a boosted cascade of simple features. In CVPR, 2001. 2, 3
38. S. Xie, R. Girshick, P. Doll´ar, Z. Tu, and K. He. Aggregated residual transformations for deep neural networks. In CVPR, 2017. 8
39. C. L. Zitnick and P. Doll´ar. Edge boxes: Locating object proposals from edges. In ECCV, 2014. 2 10