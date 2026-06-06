# YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications
YOLOv6:工业应用的单阶段目标检测框架 2022-9-7 原文：https://arxiv.org/abs/2209.02976

## Abstract
For years, YOLO series have been de facto industry-level standard for efficient object detection. The YOLO community has prospered overwhelmingly to enrich its use in a multitude of hardware platforms and abundant scenarios. In this technical report, we strive to push its limits to the next level, stepping forward with an unwavering mindset for industry application. Considering the diverse requirements for speed and accuracy in the real environment, we extensively examine the up-to-date object detection advancements either from industry or academy. Specifically, we heavily assimilate ideas from recent network design, training strategies, testing techniques, quantization and optimization methods. On top of this, we integrate our thoughts and practice to build a suite of deployment-ready networks at various scales to accommodate diversified use cases. With the generous permission of YOLO authors, we name it YOLOv6. We also express our warm welcome to users and contributors for further enhancement. For a glimpse of performance, our YOLOv6-N hits 35.9% AP on COCO dataset at a throughput of 1234 FPS on an NVIDIA Tesla T4 GPU. YOLOv6-S strikes 43.5% AP at 495 FPS, outperforming other mainstream detectors at the same scale (YOLOv5-S, YOLOX-S and PPYOLOE-S). Our quantized version of YOLOv6-S even brings a new state-of-theart 43.3% AP at 869 FPS. Furthermore, YOLOv6-M/L also achieves better accuracy performance (i.e., 49.5%/52.3%) than other detectors with the similar inference speed. We carefully conducted experiments to validate the effectiveness of each component. Our code is made available at https://github.com/meituan/YOLOv6 . 

多年来，YOLO系列已成为高效目标检测事实上的行业级标准。YOLO社区发展迅速，在众多硬件平台和丰富的场景中丰富了它的使用。在本技术报告中，我们努力将其局限性推向下一个层次，以坚定的行业应用心态向前迈进。考虑到现实环境中对速度和精度的不同要求，我们广泛研究了工业界或学术界的最新目标检测进展。具体而言，我们大量吸收了近期网络设计、训练策略、测试技术、量化和优化方法的思想。除此之外，我们整合了我们的思想和实践，构建了一套不同规模的可部署网络，以适应多样化的用例。在YOLO作者的慷慨许可下，我们将其命名为YOLOv6。我们也对用户和贡献者的进一步改进表示热烈欢迎。为了了解性能，我们的YOLOv6-N在COCO数据集上达到35.9%的AP，在NVIDIA Tesla T4 GPU上的吞吐量为1234 FPS。YOLOv6-S以495 FPS的速度命中43.5%的AP，在相同规模上优于其他主流检测器(YOLOv5-S、YOLOX-S和PPYOLOE-S)。我们的量化版本的YOLOv6-S甚至以869 FPS的速度带来了43.3%AP的新状态。此外，YOLOv6-M/L也比推理速度相近的其他检测器获得了更好的精度性能(即49.5%/52.3%)。我们仔细地进行了实验，以验证每个组件的有效性。我们的代码位于https://github.com/meituan/YOLOv6 .

Figure 1: Comparison of state-of-the-art efficient object detectors. Both latency and throughput (at a batch size of 32) are given for a handy reference. All models are test with TensorRT 7 except that the quantized model is with TensorRT 8.
图1：最先进高效目标检测器的比较。延迟和吞吐量(批次大小为32)都是为了方便参考。所有模型都使用TensorRT 7进行测试，但量化模型使用Tensor RT 8。

## 1. Introduction
YOLO series have been the most popular detection frameworks in industrial applications, for its excellent balance between speed and accuracy. Pioneering works of YOLO series are YOLOv1-3 [32–34], which blaze a new trail of one-stage detectors along with the later substantial improvements. YOLOv4 [1] reorganized the detection framework into several separate parts (backbone, neck and head), and verified bag-of-freebies and bag-of-specials at the time to design a framework suitable for training on a single GPU. At present, YOLOv5 [10], YOLOX [7], PPYOLOE [44] and YOLOv7 [42] are all the competing candidates for efficient detectors to deploy. Models at different sizes are commonly obtained through scaling techniques.

YOLO系列因其在速度和精度之间的出色平衡而成为工业应用中最流行的检测框架。YOLO系列的先驱作品是YOLOv1-3[32–34]，以及后来的实质性改进, 它们开创了一条单阶段检测器的新路径。YOLOv4[1]将检测框架重新组织为几个单独的部分(主干、颈部和头部)，并在当时验证了BoF和BoS，以设计一个适合在单个GPU上进行训练的框架。目前，YOLOv5[10]、YOLOX[7]、PPYOLOE[44]和YOLOv7[42]都是部署高效检测器的竞争对手。不同尺寸的模型通常通过缩放技术获得。

In this report, we empirically observed several important factors that motivate us to refurnish the YOLO framework: (1) Reparameterization from RepVGG [3] is a superior technique that is not yet well exploited in detection. We also notice that simple model scaling for RepVGG blocks becomes impractical, for which we consider that the elegant consistency of the network design between small and large networks is unnecessary. The plain single-path architecture is a better choice for small networks, but for larger models, the exponential growth of the parameters and the computation cost of the single-path architecture makes it infeasible; (2) Quantization of reparameterization-based detectors also requires meticulous treatment, otherwise it would be intractable to deal with performance degradation due to its heterogeneous configuration during training and inference. (3) Previous works [7, 10, 42, 44] tend to pay less attention to deployment, whose latencies are commonly compared on high-cost machines like V100. There is a hardware gap when it comes to real serving environment. Typically, lowpower GPUs like Tesla T4 are less costly and provide rather good inference performance. (4) Advanced domain-specific strategies like label assignment and loss function design need further verifications considering the architectural variance; (5) For deployment, we can tolerate the adjustments of the training strategy that improve the accuracy performance but not increase inference costs, such as knowledge distillation.

在本报告中，我们从经验上观察到了促使我们重新完善YOLO框架的几个重要因素：
1. RepVGG的重新参数化[3]是一种在检测中尚未得到很好利用的高级技术。我们还注意到，RepVGG块的简单模型缩放变得不切实际，为此，我们认为小型和大型网络之间的网络设计没有必要保持优雅的一致性。对于小型网络来说，简单的单路径架构是更好的选择，但对于较大的模型，参数的指数增长和单路径架构的计算成本使其不可行; 
2. 基于再参数化的检测器的量化也需要仔细处理，否则在训练和推理过程中，由于其异构配置，处理性能退化将是困难的。
3. 以前的工作[7、10、42、44]往往不太关注部署，通常在V100等高成本机器上比较部署的延迟。在实际服务环境中存在硬件差距。通常，像特斯拉T4这样的低功耗GPU成本较低，并且提供了相当好的推理性能。
4. 考虑到架构差异，标签分配和损失函数设计等高级领域特定策略需要进一步验证; 
5. 对于部署，我们可以容忍调整训练策略，以提高准确性性能，但不会增加推理成本，例如知识蒸馏。

With the aforementioned observations in mind, we bring the birth of YOLOv6, which accomplishes so far the best trade-off in terms of accuracy and speed. We show the comparison of YOLOv6 with other peers at a similar scale in Fig. 1. To boost inference speed without much performance degradation, we examined the cutting-edge quantization methods, including post-training quantization (PTQ) and quantization-aware training (QAT), and accommodate them in YOLOv6 to achieve the goal of deployment-ready networks.

考虑到上述观察结果，我们推出了YOLOv6，它在准确性和速度方面实现了迄今为止最好的权衡。我们在图1中显示了YOLOv6与类似规模的其他对等网络的比较。为了在不大幅降低性能的情况下提高推理速度，我们研究了最前沿的量化方法，包括训练后量化(PTQ)和量化感知训练(QAT)，并在YOLOv6中调整它们，以实现可部署网络的目标。

We summarize the main aspects of YOLOv6 as follows:
* We refashion a line of networks of different sizes tailored for industrial applications in diverse scenarios. The architectures at different scales vary to achieve the best speed and accuracy trade-off, where small models feature a plain single-path backbone and large models are built on efficient multi-branch blocks.
* We imbue YOLOv6 with a self-distillation strategy, performed both on the classification task and the regression task. Meanwhile, we dynamically adjust the knowledge from the teacher and labels to help the student model learn knowledge more efficiently during all training phases.
* We broadly verify the advanced detection techniques for label assignment, loss function and data augmentation techniques and adopt them selectively to further boost the performance.
* We reform the quantization scheme for detection with the help of RepOptimizer [2] and channel-wise distillation [36], which leads to an ever-fast and accurate detector with 43.3% COCO AP and a throughput of 869 FPS at a batch size of 32.

我们将YOLOv6的主要方面总结如下：
* 我们重新构建了一系列不同规模的网络，以适应不同场景中的工业应用。不同规模的架构各不相同，以实现最佳的速度和精度权衡，其中小型模型具有普通的单路径主干，大型模型建立在高效的多分支块上。
* 我们给YOLOv6注入了一个自蒸馏策略，在分类任务和回归任务上都执行了该策略。同时，我们动态调整来自教师模型和标签的知识，以帮助学生模型在所有训练阶段更有效地学习知识。
* 我们广泛验证了标签分配、损失函数和数据增广技术的高级检测技术，并选择性地采用这些技术以进一步提高性能。
* 我们在RepOptimizer[2]和通道蒸馏[36]的帮助下对检测的量化方案进行了改进，从而得到了一个快速准确的检测器，COCO AP占43.3%，批次大小为32时的吞吐量为869 FPS。

## 2. Method
The renovated design of YOLOv6 consists of the following components, network design, label assignment, loss function, data augmentation, industry-handy improvements, and quantization and deployment:
* Network Design: Backbone: Compared with other mainstream architectures, we find that RepVGG [3] backbones are equipped with more feature representation power in small networks at a similar inference speed, whereas it can hardly be scaled to obtain larger models due to the explosive growth of the parameters and computational costs. In this regard, we take RepBlock [3] as the building block of our small networks. For large models, we revise a more efficient CSP [43] block, named CSPStackRep Block. Neck: The neck of YOLOv6 adopts PAN topology [24] following YOLOv4 and YOLOv5. We enhance the neck with RepBlocks or CSPStackRep Blocks to have RepPAN. Head: We simplify the decoupled head to make it more efficient, called Efficient Decoupled Head.
* Label Assignment: We evaluate the recent progress of label assignment strategies [5, 7, 18, 48, 51] on YOLOv6 through numerous experiments, and the results indicate that TAL [5] is more effective and training-friendly.
* Loss Function: The loss functions of the mainstream anchor-free object detectors contain classification loss, box regression loss and object loss. For each loss, we systematically experiment it with all available techniques and finally select VariFocal Loss [50] as our classification loss and SIoU [8]/GIoU [35] Loss as our regression loss.
* Industry-handy improvements: We introduce additional common practice and tricks to improve the performance including self-distillation and more training epochs. For self-distillation, both classification and box regression are respectively supervised by the teacher model. The distillation of box regression is made possible thanks to DFL [20]. In addition, the proportion of information from the soft and hard labels is dynamically declined via cosine decay, which helps the student selectively acquire knowledge at different phases during the training process. In addition, we encounter the problem of the impaired performance without adding extra gray borders at evaluation, for which we provide some remedies.
* Quantization and deployment: To cure the performance degradation in quantizing reparameterizationbased models, we train YOLOv6 with RepOptimizer [2] to obtain PTQ-friendly weights. We further adopt QAT with channel-wise distillation [36] and graph optimization to pursue extreme performance. Our quantized YOLOv6-S hits a new state of the art with 42.3% AP and a throughput of 869 FPS (batch size=32).

YOLOv6的重新设计包括以下组件：网络设计、标签分配、损失函数、数据增广、行业便利改进以及量化和部署：
* 网络设计：主干：与其他主流架构相比，我们发现RepVGG[3]主干在小型网络中以相似的推理速度具有更强的特征表示能力，而由于参数和计算成本的爆炸式增长，很难进行缩放以获得更大的模型。在这方面，我们将RepBlock[3]作为我们小型网络的构建块。对于大型模型，我们修改了一个更有效的CSP[43]块，名为CSPStackRep块。颈部：YOLOv6的颈部采用PAN拓扑[24]，紧跟YOLOv4和YOLOv5。我们使用RepBlocks或CSPStackRep Blocks来增强颈部，使其具有RepPAN。头部：我们简化了解耦头部，使其更高效，称为高效解耦头部。
* 标签分配：我们通过大量实验评估了YOLOv6上标签分配策略[5、7、18、48、51]的最新进展，结果表明TAL[5]更有效，更便于训练。
* 损失函数：主流无锚目标检测器的损失函数包括分类损失、边框回归损失和目标损失。对于每种损失，我们使用所有可用的技术进行了系统的实验，最后选择VariFocal损失[50]作为我们的分类损失，选择SIoU[8]/GIoU[35]损失作为我们的回归损失。
* 行业方便的改进：我们引入了额外的常见做法和技巧来提高性能，包括自蒸馏和更多的训练时间。对于自蒸馏，分类和边框回归都分别由教师模型监督。得益于DFL，边框回归的提取成为可能[20]。此外，软标签和硬标签的信息比例通过余弦衰减动态下降，这有助于学生在训练过程中的不同阶段选择性地获取知识。此外，我们在评估时遇到了性能受损的问题，但没有添加额外的灰色边界，对此我们提供了一些补救措施。
* 量化和部署：为了解决基于重新参数化的量化模型时性能下降的问题，我们使用RepOptimizer[2]训练YOLOv6以获得PTQ友好的权重。我们进一步采用QAT与通道式蒸馏[36]和图优化，以追求极致的性能。我们的量化YOLOv6-S达到了最新的技术水平，AP为42.3%，吞吐量为869 FPS(批次大小=32)。

### 2.1. Network Design
A one-stage object detector is generally composed of the following parts: a backbone, a neck and a head. The backbone mainly determines the feature representation ability, meanwhile, its design has a critical influence on the inference efficiency since it carries a large portion of computation cost. The neck is used to aggregate the low-level physical features with high-level semantic features, and then build up pyramid feature maps at all levels. The head consists of several convolutional layers, and it predicts final detection results according to multi-level features assembled by the neck. It can be categorized as anchorbased and anchor-free, or rather parameter-coupled head and parameter-decoupled head from the structure’s perspective.

一级目标检测器一般由以下部分组成：主干、颈部和头部。主干结构主要决定特征表示能力，同时，它的设计对推理效率有着至关重要的影响，因为它承担了很大一部分的计算成本。颈部用于将低级物理特征与高级语义特征聚合，然后在所有级别构建金字塔特征图。头部由几个卷积层组成，它根据颈部集合的多级特征预测最终检测结果。从结构的角度来看，它可以分为锚固型和无锚固型，或者更确切地说是参数耦合型头部和参数解耦型头部。

In YOLOv6, based on the principle of hardwarefriendly network design [3], we propose two scaled reparameterizable backbones and necks to accommodate models at different sizes, as well as an efficient decoupled head with the hybrid-channel strategy. The overall architecture of YOLOv6 is shown in Fig. 2.

在YOLOv6中，基于硬件友好的网络设计原则[3]，我们提出了两个可缩放的可重新参数化主干和颈部，以适应不同大小的模型，以及一个高效的混合信道策略解耦头。YOLOv6的总体架构如图2所示。

Figure 2: The YOLOv6 framework (N and S are shown). Note for M/L, RepBlocks is replaced with CSPStackRep.
图2:YOLOv6框架(显示了N和S)。注意，对于M/L，RepBlocks替换为CSPStackRep。

#### 2.1.1 Backbone
As mentioned above, the design of the backbone network has a great impact on the effectiveness and efficiency of the detection model. Previously, it has been shown that multibranch networks [13, 14, 38, 39] can often achieve better classification performance than single-path ones [15, 37], but often it comes with the reduction of the parallelism and results in an increase of inference latency. On the contrary, plain single-path networks like VGG [37] take the advantages of high parallelism and less memory footprint, leading to higher inference efficiency. Lately in RepVGG [3], a structural re-parameterization method is proposed to decouple the training-time multi-branch topology with an inference-time plain architecture to achieve a better speedaccuracy trade-off.

如上所述，骨干网的设计对检测模型的有效性和效率有很大影响。此前，研究表明，多分支网络[13、14、38、39]通常比单路径网络[15、37]具有更好的分类性能，但通常伴随着并行度的降低，并导致推理延迟的增加。相反，像VGG[37]这样的普通单路径网络具有高并行性和较少内存占用的优点，从而提高了推理效率。最近在RepVGG[3]中，提出了一种结构重新参数化方法，将训练时间多分支拓扑与推理时间平面结构解耦，以实现更好的速度精度权衡。

Inspired by the above works, we design an efficient re-parameterizable backbone denoted as EfficientRep. For small models, the main component of the backbone is RepBlock during the training phase, as shown in Fig. 3 (a). And each RepBlock is converted to stacks of 3 × 3 convolutional layers (denoted as RepConv) with ReLU activation functions during the inference phase, as shown in Fig. 3 (b).Typically a 3×3 convolution is highly optimized on mainstream GPUs and CPUs and it enjoys higher computational density. Consequently, EfficientRep Backbone sufficiently 3 utilizes the computing power of the hardware, resulting in a significant decrease in inference latency while enhancing the representation ability in the meantime.

受上述工作的启发，我们设计了一个高效的可重新参数化主干，称为EfficientRep。对于小型模型，主干的主要组件是训练阶段的RepBlock，如图3(a)所示。在推理阶段，每个RepBlock被转换为具有ReLU激活函数的3×3卷积层(表示为RepConv)的堆栈，如图3(b)所示。通常，3×3卷积在主流GPU和CPU上进行了高度优化，具有更高的计算密度。因此，EfficientRep Backbone 3充分利用了硬件的计算能力，从而显著降低了推理延迟，同时增强了表示能力。

However, we notice that with the model capacity further expanded, the computation cost and the number of parameters in the single-path plain network grow exponentially. To achieve a better trade-off between the computation burden and accuracy, we revise a CSPStackRep Block to build the backbone of medium and large networks. As shown in Fig. 3 (c), CSPStackRep Block is composed of three 1×1 convolution layers and a stack of sub-blocks consisting of two RepVGG blocks [3] or RepConv (at training or inference respectively) with a residual connection. Besides, a cross stage partial (CSP) connection is adopted to boost performance without excessive computation cost. Compared with CSPRepResStage [45], it comes with a more succinct outlook and considers the balance between accuracy and speed.

然而，我们注意到，随着模型容量的进一步扩大，单路径平面网络的计算成本和参数数量呈指数级增长。为了在计算负担和精度之间取得更好的平衡，我们修改了CSPStackRep块来构建中型和大型网络的主干。如图3(c)所示，CSPStackRep块由三个1×1卷积层和一堆子块组成，子块由两个RepVGG块[3]或RepConv(分别在训练或推理时)组成，具有残差连接。此外，还采用了跨级部分(CSP)连接来提高性能，而不需要过多的计算开销。与CSPRepResStage[45]相比，它具有更简洁的外观，并考虑了准确性和速度之间的平衡。

Figure 3: (a) RepBlock is composed of a stack of RepVGG blocks with ReLU activations at training. (b) During inference time, RepVGG block is converted to RepConv. (c) CSPStackRep Block comprises three 1×1 convolutional layers and a stack of sub-blocks of double RepConvs following the ReLU activations with a residual connection.
图3：(a)RepBlock由训练时激活ReLU的RepVGG块堆栈组成。(b)在推理期间，RepVGG块转换为RepConv。(c)CSPStackRep块包括三个1×1卷积层和一个双RepConv子块堆栈，这些子块在ReLU激活后具有残差连接。

#### 2.1.2 Neck
In practice, the feature integration at multiple scales has been proved to be a critical and effective part of object detection [9, 21, 24, 40]. We adopt the modified PAN topology [24] from YOLOv4 [1] and YOLOv5 [10] as the base of our detection neck. In addition, we replace the CSPBlock used in YOLOv5 with RepBlock (for small models) or CSPStackRep Block (for large models) and adjust the width and depth accordingly. The neck of YOLOv6 is denoted as Rep-PAN.

在实践中，多尺度的特征集成已被证明是目标检测的关键和有效部分[9，21，24，40]。我们采用YOLOv4[1]和YOLOv5[10]中修改的PAN拓扑[24]作为我们的检测颈部的基础。此外，我们将YOLOv5中使用的CSPBlock替换为RepBlock(适用于小型模型)或CSPStackRepBlok(适用于大型模型)，并相应调整宽度和深度。YOLOv6的颈部表示为Rep-PAN。

#### 2.1.3 Head
Efficient decoupled head. The detection head of YOLOv5 is a coupled head with parameters shared between the classification and localization branches, while its counterparts in FCOS [41] and YOLOX [7] decouple the two branches, and additional two 3×3 convolutional layers are introduced in each branch to boost the performance.

高效解耦头. YOLOv5的检测头是一个耦合头，在分类和定位分支之间共享参数，而FCOS[41]和YOLOX[7]中的检测头将这两个分支解耦，并且在每个分支中引入额外的两个3×3卷积层以提高性能。

In YOLOv6, we adopt a hybrid-channel strategy to build a more efficient decoupled head. Specifically, we reduce the number of the middle 3×3 convolutional layers to only one. The width of the head is jointly scaled by the width multiplier for the backbone and the neck. These modifications further reduce computation costs to achieve a lower inference latency.

在YOLOv6中，我们采用混合信道策略来构建更高效的解耦头。具体来说，我们将中间3×3卷积层的数量减少到只有一个。头部的宽度由主干和颈部的宽度倍增器共同缩放。这些修改进一步降低了计算成本，以实现更低的推理延迟。

Anchor-free detectors stand out because of their better generalization ability and simplicity in decoding prediction results. The time cost of its post-processing is substantially reduced. There are two types of anchorfree detectors: anchor point-based [7, 41] and keypointbased [16, 46, 53]. In YOLOv6, we adopt the anchor pointbased paradigm, whose box regression branch actually predicts the distance from the anchor point to the four sides of the bounding boxes.

无锚检测器因其更好的泛化能力和解码预测结果的简单性而脱颖而出。它的后处理时间成本大大减少。无锚检测器有两种类型：基于锚点的检测器[7，41]和基于关键点的检测器[16，46，53]。在YOLOv6中，我们采用了基于锚点的范式，其边框回归分支实际上预测了从锚点到边界框四边的距离。

### 2.2. Label Assignment
Label assignment is responsible for assigning labels to predefined anchors during the training stage. Previous work has proposed various label assignment strategies ranging from simple IoU-based strategy and inside ground-truth method [41] to other more complex schemes [5, 7, 18, 48, 51].

标签分配负责在训练阶段将标签分配给预定义的锚。之前的工作提出了各种标签分配策略，从简单的基于IoU的策略和内幕真相方法[41]到其他更复杂的方案[5、7、18、48、51]。

SimOTA. OTA [6] considers the label assignment in object detection as an optimal transmission problem. It defines positive/negative training samples for each ground-truth object from a global perspective. SimOTA [7] is a simplified version of OTA [6], which reduces additional hyperparameters and maintains the performance. SimOTA was utilized as the label assignment method in the early version of YOLOv6. However, in practice, we find that introducing SimOTA will slow down the training process. And it is not rare to fall into unstable training. Therefore, we desire a replacement for SimOTA.

SimOTA. OTA[6]认为目标检测中的标签分配是一个最优传输问题。它从全局角度为每个地面真相对象定义了正/负训练样本。SimOTA[7]是OTA[6]的简化版本，它减少了额外的超参数并保持了性能。在早期版本的YOLOv6中，SimOTA被用作标签分配方法。然而，在实践中，我们发现引入SimOTA会减缓训练过程。而且，陷入不稳定训练的情况并不罕见。因此，我们希望更换SimOTA。

Task alignment learning Task. Task Alignment Learning (TAL) was first proposed in TOOD [5], in which a unified metric of classification score and predicted box quality is designed. The IoU is replaced by this metric to assign object labels. To a certain extent, the problem of the misalignment of tasks (classification and box regression) is alleviated.

任务对齐学习。对齐学习(TAL)最早在TOOD[5]中提出，其中设计了一个统一的分类分数和预测框质量度量。IoU被此指标替换，以分配对象标签。在一定程度上，任务错位(分类和边框回归)的问题得到了缓解。

The other main contribution of TOOD is about the task aligned head (T-head). T-head stacks convolutional layers to build interactive features, on top of which the Task-Aligned Predictor (TAP) is used. PP-YOLOE [45] improved Thead by replacing the layer attention in T-head with the 4 lightweight ESE attention, forming ET-head. However, we find that the ET-head will deteriorate the inference speed in our models and it comes with no accuracy gain. Therefore, we retain the design of our Efficient decoupled head.

TOOD的另一个主要贡献是关于任务导向型头(T型头)。T型头堆叠卷积层以构建交互特征，在其上使用任务对齐预测(TAP)。PP-YOLOE[45]改进了Thead，将T型头部的分层注意力替换为4个轻型ESE注意力，形成了ET型头部。然而，我们发现ET头会降低我们模型中的推理速度，并且没有精度增益。因此，我们保留了高效解耦磁头的设计。

Furthermore, we observed that TAL could bring more performance improvement than SimOTA and stabilize the training. Therefore, we adopt TAL as our default label assignment strategy in YOLOv6.

此外，我们观察到，TAL可以比SimOTA带来更多绩效改进，并稳定训练。因此，我们在YOLOv6中采用TAL作为我们的默认标签分配策略。

### 2.3. Loss Functions
Object detection contains two sub-tasks: classification and localization, corresponding to two loss functions: classification loss and box regression loss. For each sub-task, there are various loss functions presented in recent years. In this section, we will introduce these loss functions and describe how we select the best ones for YOLOv6.

目标检测包含两个子任务：分类和定位，对应两个损失函数：分类损失和边框回归损失。对于每个子任务，近年来提出了各种损失函数。在本节中，我们将介绍这些损失函数，并描述如何为YOLOv6选择最佳函数。

#### 2.3.1 Classification Loss
Improving the performance of the classifier is a crucial part of optimizing detectors. Focal Loss [22] modified the traditional cross-entropy loss to solve the problems of class imbalance either between positive and negative examples, or hard and easy samples. To tackle the inconsistent usage of the quality estimation and classification between training and inference, Quality Focal Loss (QFL) [20] further extended Focal Loss with a joint representation of the classification score and the localization quality for the supervision in classification. Whereas VariFocal Loss (VFL) [50] is rooted from Focal Loss [22], but it treats the positive and negative samples asymmetrically. By considering positive and negative samples at different degrees of importance, it balances learning signals from both samples. Poly Loss [17] decomposes the commonly used classification loss into a series of weighted polynomial bases. It tunes polynomial coefficients on different tasks and datasets, which is proved better than Cross-entropy Loss and Focal Loss through experiments.

提高分类器的性能是优化检测器的关键部分。Focal Loss[22]对传统的交叉熵损失进行了改进，以解决正负样本或难易样本之间的类不平衡问题。为了解决训练和推理之间质量评估和分类使用不一致的问题，质量焦点损失(QFL)[20]进一步扩展了焦点损失，并将分类分数和局部化质量联合表示，用于分类监督。尽管VariFocal Loss(VFL)[50]来源于Focal Loss[22]，但它对正负样本的处理是不对称的。通过考虑不同重要性的正负样本，它平衡了来自两个样本的学习信号。Poly Loss[17]将常用的分类损失分解为一系列加权多项式基。该算法在不同的任务和数据集上调整多项式系数，通过实验证明其优于交叉熵损失和焦点损失。

We assess all these advanced classification losses on YOLOv6 to finally adopt VFL [50].

我们评估了YOLOv6上所有这些高级分类损失，最终采用VFL[50]。

#### 2.3.2 Box Regression Loss
Box regression loss provides significant learning signals localizing bounding boxes precisely. L1 loss is the original box regression loss in early works. Progressively, a variety of well-designed box regression losses have sprung up, such as IoU-series loss [8,11,35,47,52,52] and probability loss [20].

边框回归损失提供了精确定位边界边框的重要学习信号。L1损失是早期工作中的原始边框回归损失。逐渐地，出现了各种设计良好的边框回归损失，例如IoU系列损失[8,11,35,47,52,52]和概率损失[20]。

IoU-series Loss IoU loss [47] regresses the four bounds of a predicted box as a whole unit. It has been proved to be effective because of its consistency with the evaluation metric. There are many variants of IoU, such as GIoU [35], DIoU [52], CIoU [52], α-IoU [11] and SIoU [8], etc, forming relevant loss functions. We experiment with GIoU, CIoU and SIoU in this work. And SIoU is applied to YOLOv6-N and YOLOv6-T, while others use GIoU.

IoU系列Loss IoU Loss[47]将预测框的四个边界作为一个整体进行回归。由于其与评估指标的一致性，已被证明是有效的。IoU有许多变体，如GIoU[35]、DIoU[52]、CIoU[52]、α-IoU[11]和SIoU[8]等，形成了相关的损失函数。在这项工作中，我们使用GIoU、CIoU和SIoU进行了实验。SIoU适用于YOLOv6-N和YOLOv6-T，而其他使用GIoU。

Probability Loss. Distribution Focal Loss (DFL) [20] simplifies the underlying continuous distribution of box locations as a discretized probability distribution. It considers ambiguity and uncertainty in data without introducing any other strong priors, which is helpful to improve the box localization accuracy especially when the boundaries of the ground-truth boxes are blurred. Upon DFL, DFLv2 [19] develops a lightweight sub-network to leverage the close correlation between distribution statistics and the real localization quality, which further boosts the detection performance. However, DFL usually outputs 17× more regression values than general box regression, leading to a substantial overhead. The extra computation cost significantly hinders the training of small models. Whilst DFLv2 further increases the computation burden because of the extra sub-network. In our experiments, DFLv2 brings similar performance gain to DFL on our models. Consequently, we only adopt DFL in YOLOv6-M/L. Experimental details can be found in Section 3.3.3.

概率损失。分布焦点损失(DFL)[20]将边框子位置的基本连续分布简化为离散概率分布。它在不引入任何其他强先验的情况下考虑数据的模糊性和不确定性，这有助于提高边框子定位精度，特别是当地面真值边框子的边界模糊时。基于DFL，DFLv2[19]开发了一个轻量级子网络，以利用分布统计数据和真实定位质量之间的密切关联，从而进一步提高检测性能。然而，DFL通常输出比常规边框回归多17倍的回归值，这会导致大量开销。额外的计算成本严重阻碍了小模型的训练。同时，由于额外的子网，DFLv2进一步增加了计算负担。在我们的实验中，DFLv2在我们的模型上为DFL带来了类似的性能提升。因此，我们仅在YOLOv6-M/L中采用DFL。实验细节见第3.3.3节。

#### 2.3.3 Object Loss
Object loss was first proposed in FCOS [41] to reduce the score of low-quality bounding boxes so that they can be filtered out in post-processing. It was also used in YOLOX [7] to accelerate convergence and improve network accuracy. As an anchor-free framework like FCOS and YOLOX, we have tried object loss into YOLOv6. Unfortunately, it doesn’t bring many positive effects. Details are given in Section 3.

目标损失最初是在FCOS中提出的[41]，目的是减少低质量边界框的分数，以便在后期处理中过滤掉它们。它还用于YOLOX[7]，以加速收敛并提高网络精度。作为一个像FCOS和YOLOX这样的无锚框架，我们在YOLOv6中尝试了目标损失。不幸的是，它没有带来多少积极影响。详情见第3节。

### 2.4. Industry-handy improvements
The following tricks come ready to use in real practice. They are not intended for a fair comparison but steadily produce performance gain without much tedious effort.

下面的技巧可以在实际操作中使用。它们不是为了进行公平的比较，而是不需要太多繁琐的工作就可以稳定地提高性能。

#### 2.4.1 More training epochs
Empirical results have shown that detectors have a progressing performance with more training time. We extended the training duration from 300 epochs to 400 epochs to reach a better convergence.

实验结果表明，随着训练时间的延长，检测器的性能不断提高。我们将训练时间从300个时代延长到400个时代，以实现更好的融合。

#### 2.4.2 Self-distillation 自蒸馏
To further improve the model accuracy while not introducing much additional computation cost, we apply the clas5 sical knowledge distillation technique minimizing the KLdivergence between the prediction of the teacher and the student. We limit the teacher to be the student itself but pretrained, hence we call it self-distillation. Note that the KL-divergence is generally utilized to measure the difference between data distributions. However, there are two sub-tasks in object detection, in which only the classification task can directly utilize knowledge distillation based on KL-divergence. Thanks to DFL loss [20], we can perform it on box regression as well. The knowledge distillation loss can then be formulated as:

为了进一步提高模型精度，同时不引入太多额外的计算成本，我们应用经典知识蒸馏技术，将教师和学生的预测之间的KL差异降至最低。我们将教师限制为学生本身，但要经过预先训练，因此我们称之为自我升华。注意，KL散度通常用于测量数据分布之间的差异。然而，在目标检测中有两个子任务，其中只有分类任务可以直接利用基于KL散度的知识蒸馏。由于DFL损失[20]，我们也可以在框回归上执行它。知识蒸馏损失可表示为：

$L_{KD} = KL(p^{cls}_t ||p^{cls}_s ) + KL(p^{reg}_t ||p^{reg}_s )$ ,  (1) 

where $p^{cls}_t$ and $p^{cls}_s$ are class prediction of the teacher model and the student model respectively, and accordingly $p^{reg}_t$ and $p^{reg}_s$ are box regression predictions. The overall loss function is now formulated as:

其中，$p^{cls}_t$和$p^{cls}_s$分别是教师模型和学生模型的分类预测，$p^{reg}_t$和$p^{reg}_s$是边框子回归预测。总损失函数现在表示为：

$L_{total} = L_{det} + αL_{KD}$ , (2) 

where $L_{det}$ is the detection loss computed with predictions and labels. The hyperparameter α is introduced to balance two losses. In the early stage of training, the soft labels from the teacher are easier to learn. As the training continues, the performance of the student will match the teacher so that the hard labels will help students more. Upon this, we apply cosine weight decay to α to dynamically adjust the information from hard labels and soft ones from the teacher. We conducted detailed experiments to verify the effect of self-distillation on YOLOv6, which will be discussed in Section 3.

其中，$L_{det}$是使用预测和标签计算的检测损失。引入超参数α来平衡两个损失。在训练的早期阶段，老师的软标签更容易学习。随着训练的继续，学生的表现将与老师相匹配，因此硬标签将对学生有更多帮助。在此基础上，我们将余弦权重衰减应用于α，以动态调整来自硬标签和来自老师的软标签的信息。我们进行了详细的实验来验证自蒸馏对YOLOv6的影响，这将在第3节中讨论。

#### 2.4.3 Gray border of images 图像的灰色边框
We notice that a half-stride gray border is put around each image when evaluating the model performance in the implementations of YOLOv5 [10] and YOLOv7 [42]. Although no useful information is added, it helps in detecting the objects near the edge of the image. This trick also applies in YOLOv6.

我们注意到，在评估YOLOv5[10]和YOLOv7[42]实现中的模型性能时，每个图像周围都有一个半步灰色边框。虽然没有添加有用的信息，但它有助于检测图像边缘附近的对象。这个技巧也适用于YOLOv6。

However, the extra gray pixels evidently reduce the inference speed. Without the gray border, the performance of YOLOv6 deteriorates, which is also the case in [10, 42]. We postulate that the problem is related to the gray borders padding in Mosaic augmentation [1, 10]. Experiments on turning mosaic augmentations off during last epochs [7] (aka. fade strategy) are conducted for verification. In this regard, we change the area of gray border and resize the image with gray borders directly to the target image size. Combining these two strategies, our models can maintain or even boost the performance without the degradation of inference speed.

然而，额外的灰度像素明显降低了推理速度。如果没有灰色边界，YOLOv6的性能就会恶化，[10，42]中的情况也是如此。我们假设这个问题与Mosaic增强中的灰色边界填充有关[1,10]。在最后一个时期内关闭马赛克增强的实验[7](又名衰减策略)进行了验证。在这方面，我们改变了灰色边框的区域，并将带有灰色边框的图像直接调整为目标图像的大小。结合这两种策略，我们的模型可以在不降低推理速度的情况下保持甚至提高性能。

### 2.5. Quantization and Deployment 量化和部署
For industrial deployment, it has been common practice to adopt quantization to further speed up runtime without much performance compromise. Post-training quantization (PTQ) directly quantizes the model with only a small calibration set. Whereas quantization-aware training (QAT) further improves the performance with the access to the training set, which is typically used jointly with distillation. However, due to the heavy use of re-parameterization blocks in YOLOv6, previous PTQ techniques fail to produce high performance, while it is hard to incorporate QAT when it comes to matching fake quantizers during training and inference. We here demonstrate the pitfalls and our cures during deployment.

对于工业部署，通常采用量化来进一步加快运行时，而不会降低性能。训练后量化(PTQ)直接量化模型，只需一个小的校准集。而量化感知训练(QAT)通过访问训练集(通常与提取结合使用)进一步提高了性能。然而，由于在YOLOv6中大量使用重新参数化块，以前的PTQ技术无法产生高性能，而在训练和推理过程中匹配伪量化器时，很难加入QAT。我们在此演示部署期间的缺陷和解决方法。

#### 2.5.1 Reparameterizing Optimizer
RepOptimizer [2] proposes gradient re-parameterization at each optimization step. This technique also well solves the quantization problem of reparameterization-based models. We hence reconstruct the re-parameterization blocks of YOLOv6 in this fashion and train it with RepOptimizer to obtain PTQ-friendly weights. The distribution of feature map is largely narrowed (e.g. Fig. 4, more in B.1), which greatly benefits the quantization process, see Sec 3.5.1 for results.

RepOptimizer[2]建议在每个优化步骤中重新参数化梯度。该技术也很好地解决了基于重参数化模型的量化问题。因此，我们以这种方式重建YOLOv6的重新参数化块，并使用RepOptimizer对其进行训练，以获得PTQ友好的权重。特征图的分布范围大大缩小(如图4，更多内容见B.1)，这大大有利于量化过程，结果见第3.5.1节。

Figure 4: Improved activation distribution of YOLOv6-S trained with RepOptimizer.
图4:使用RepOptimizer训练的YOLOv6-S的激活分布得到改进。

#### 2.5.2 Sensitivity Analysis 敏感性分析
We further improve the PTQ performance by partially converting quantization-sensitive operations into float computation. To obtain the sensitivity distribution, several metrics are commonly used, mean-square error (MSE), signal-noise ratio (SNR) and cosine similarity. Typically for comparison, one can pick the output feature map (after the activation of a certain layer) to calculate these metrics with and without quantization. As an alternative, it is also viable to 6 compute validation AP by switching quantization on and off for the certain layer [29].

通过将量化敏感操作部分转换为浮点计算，我们进一步提高了PTQ性能。为了获得灵敏度分布，常用的度量有：均方误差(MSE)、信噪比(SNR)和余弦相似性。通常，为了进行比较，可以选择输出特征映射(在激活某个层之后)来计算这些量化和不量化的度量。作为替代方案，通过为特定层打开和关闭量化来计算验证AP也是可行的[29]。

We compute all these metrics on the YOLOv6-S model trained with RepOptimizer and pick the top-6 sensitive layers to run in float. The full chart of sensitivity analysis can be found in B.2.

我们在用RepOptimizer训练的YOLOv6-S模型上计算所有这些指标，并选择前6个敏感层以浮动方式运行。敏感性分析的完整图表见B.2。

#### 2.5.3 Quantization-aware Training with Channel-wise Distillation 量化感知训练与通道蒸馏
In case PTQ is insufficient, we propose to involve quantization-aware training (QAT) to boost quantization performance. To resolve the problem of the inconsistency of fake quantizers during training and inference, it is necessary to build QAT upon the RepOptimizer. Besides, channelwise distillation [36] (later as CW Distill) is adapted within the YOLOv6 framework, shown in Fig. 5. This is also a self-distillation approach where the teacher network is the student itself in FP32-precision. See experiments in Sec 3.5.1.

在PTQ不足的情况下，我们建议使用量化感知训练(QAT)来提高量化性能。为了解决训练和推理过程中伪量化器不一致的问题，有必要在RepOptimizer上构建QAT。此外，通道蒸馏[36](后来称为CW蒸馏)适用于YOLOv6框架，如图5所示。这也是一种自蒸馏方法，其中教师网络是FP32精度的学生本身。参见第3.5.1节中的实验。

Figure 5: Schematic of YOLOv6 channel-wise distillation in QAT.
图5:QAT中YOLOv6通道蒸馏示意图。

## 3. Experiments
### 3.1. Implementation Details
We use the same optimizer and the learning schedule as YOLOv5 [10], i.e. stochastic gradient descent (SGD) with momentum and cosine decay on learning rate. Warm-up, grouped weight decay strategy and the exponential moving average (EMA) are also utilized. We adopt two strong data augmentations (Mosaic [1,10] and Mixup [49]) following [1,7,10]. A complete list of hyperparameter settings can be found in our released code. We train our models on the COCO 2017 [23] training set, and the accuracy is evaluated on the COCO 2017 validation set. All our models are trained on 8 NVIDIA A100 GPUs, and the speed performance is measured on an NVIDIA Tesla T4 GPU with TensorRT version 7.2 unless otherwise stated. And the speed performance measured with other TensorRT versions or on other devices is demonstrated in Appendix A.

我们使用与YOLOv5[10]相同的优化器和学习计划，即随机梯度下降(SGD)，具有动量和余弦衰减的学习速率。预热、分组权重衰减策略和指数移动平均(EMA)也被使用。我们在[1,7,10]之后采用了两种强大的数据增广(Mosaic[1,10]和Mixup[49])。在我们发布的代码中可以找到超参数设置的完整列表。我们在COCO 2017[23]训练集上训练我们的模型，并在COCO 17验证集上评估其准确性。我们所有的模型都经过8台NVIDIA A100 GPU的训练，除非另有说明，否则速度性能是在配备TensorRT 7.2版的NVIDIA-Tesla T4 GPU上测量的。附录A显示了使用其他TensorRT版本或其他设备测量的速度性能。

### 3.2. Comparisons
Considering that the goal of this work is to build networks for industrial applications, we primarily focus on the speed performance of all models after deployment, including throughput (FPS at a batch size of 1 or 32) and the GPU latency, rather than FLOPs or the number of parameters. We compare YOLOv6 with other state-of-the-art detectors of YOLO series, including YOLOv5 [10], YOLOX [7], PPYOLOE [45] and YOLOv7 [42]. Note that we test the speed performance of all official models with FP16-precision on the same Tesla T4 GPU with TensorRT [28]. The performance of YOLOv7-Tiny is re-evaluated according to their open-sourced code and weights at the input size of 416 and 640. Results are shown in Table 1 and Fig. 1. Compared with YOLOv5-N/YOLOv7-Tiny (input size=416), our YOLOv6-N has significantly advanced by 7.9%/2.6% respectively. It also comes with the best speed performance in terms of both throughput and latency. Compared with YOLOX-S/PPYOLOE-S, YOLOv6-S can improve AP by 3.0%/0.4% with higher speed. We compare YOLOv5-S and YOLOv7-Tiny (input size=640) with YOLOv6-T, our method is 2.9% more accurate and 73/25 FPS faster with a batch size of 1. YOLOv6-M outperforms YOLOv5-M by 4.2% higher AP with a similar speed, and it achieves 2.7%/0.6% higher AP than YOLOX-M/PPYOLOE-M at a higher speed. Besides, it is more accurate and faster than YOLOv5-L. YOLOv6-L is 2.8%/1.1% more accurate than YOLOX-L/PPYOLOE-L under the same latency constraint. We additionally provide a faster version of YOLOv6-L by replacing SiLU with ReLU (denoted as YOLOv6-L-ReLU). It achieves 51.7% AP with a latency of 8.8 ms, outperforming YOLOX-L/PPYOLOE-L/YOLOv7 in both accuracy and speed.

考虑到这项工作的目标是为工业应用构建网络，我们主要关注部署后所有模型的速度性能，包括吞吐量(FPS批量大小为1或32)和GPU延迟，而不是FLOP或参数数量。我们将YOLOv6与YOLO系列的其他SOTA检测器进行了比较，包括YOLOv5[10]、YOLOX[7]、PPYOLOE[45]和YOLOv7[42]。请注意，我们使用TensorRT在同一台Tesla T4 GPU上测试了所有官方模型FP16精度的速度性能[28]。根据其开源代码和输入大小为416和640时的权重，对YOLOv7 Tiny的性能进行了重新评估。结果如表1和图1所示。与YOLOv 5-N/YOLOv-7 Tiny(输入大小=416)相比，我们的YOLOv6-N分别显著提高了7.9%/2.6%。它还具有吞吐量和延迟方面的最佳速度性能。与YOLOX-S/PPYOLOE-S相比，YOLOv6-S可以以更高的速度将AP提高3.0%/0.4%。我们将YOLOv5-S和YOLOv7 Tiny(输入大小=640)与YOLOv6-T进行比较，我们的方法在批次大小为1的情况下，精确度提高2.9%，速度提高73/25 FPS。YOLOv6-M的AP比YOLOv5-M高4.2%，速度相近，AP比YOLOX-M/PPYOLOEM高2.7%/0.6%。此外，它比YOLOv5-L更准确、更快。在相同的延迟限制下，YOLOv6-L比YOLOX-L/PPYOLOE-L精确2.8%/1.1%。我们还提供了更快的YOLOv6-L版本，将SiLU替换为ReLU(表示为YOLOv 6-L-ReLU)。它以8.8毫秒的延迟达到51.7%的AP，在准确性和速度上都优于YOLOX-L/PPYOLOE-L/YOLOv7。

### 3.3. Ablation Study
#### 3.3.1 Network
Backbone and neck We explore the influence of singlepath structure and multi-branch structure on backbones and necks, as well as the channel coefficient (denoted as CC) of CSPStackRep Block. All models described in this part adopt TAL as the label assignment strategy, VFL as the classification loss, and GIoU with DFL as the regression loss. Results are shown in Table 2. We find that the optimal network structure for models at different sizes should come up with different solutions.

主干和颈部。我们探讨了单路径结构和多分支结构对主干和颈部的影响，以及CSPStackRep区块的通道系数(表示为CC)。本部分描述的所有模型均采用TAL作为标签分配策略，VFL作为分类损失，GIoU和DFL作为回归损失。结果如表2所示。我们发现，不同规模模型的最佳网络结构应该有不同的解决方案。

For YOLOv6-N, the single-path structure outperforms the multi-branch structure in terms of both accuracy and speed. Although the single-path structure has more FLOPs and parameters than the multi-branch structure, it could run faster due to a relatively lower memory footprint and a higher degree of parallelism. For YOLOv6-S, the two block styles bring similar performance. When it comes to larger models, multi-branch structure achieves better performance in accuracy and speed. And we finally select multi-branch with a channel coefficient of 2/3 for YOLOv6-M and 1/2 for YOLOv6-L.

对于YOLOv6-N，单路径结构在精度和速度上都优于多分支结构。虽然与多分支结构相比，单路径结构具有更多的FLOP和参数，但由于内存占用量相对较低，并行度较高，因此可以运行得更快。对于YOLOv6-S，两种块样式带来了相似的性能。对于较大的模型，多分支结构在精度和速度上都取得了更好的性能。最后，我们为YOLOv6-M选择了通道系数为2/3的多分支，为YOLOv6-L选择了1/2的多分支。

Table 1: Comparisons with other YOLO-series detectors on COCO 2017 val. FPS and latency are measured in FP16-precision on a Tesla T4 in the same environment with TensorRT. All our models are trained for 300 epochs without pre-training or any external data. Both the accuracy and the speed performance of our models are evaluated with the input resolution of 640×640. ‘‡’ represents that the proposed self-distillation method is utilized. ‘∗’ represents the re-evaluated result of the released model through the official code. 

表1：与其他YOLO系列检测器在COCO 2017 val.FPS和延迟方面的比较是在与TensorRT相同的环境下，在特斯拉T4上以FP16精度测量的。我们所有的模型都经过300个时代的训练，没有预先训练或任何外部数据。我们以640×640的输入分辨率评估了模型的精度和速度性能。“➢”表示使用了提议的自蒸馏方法。”∗’ 表示通过官方代码重新评估发布模型的结果。

Furthermore, we study the influence of width and depth of the neck on YOLOv6-L. Results in Table 3 show that the slender neck performs 0.2% better than the wide-shallow neck with the similar speed.

此外，我们还研究了颈部宽度和深度对YOLOv6-L的影响。表3中的结果表明，在相同的速度下，细长颈部的表现优于宽浅颈部0.2%。

Combinations of convolutional layers and activation functions YOLO series adopted a wide range of activation functions, ReLU [27], LReLU [25], Swish [31], SiLU [4], Mish [26] and so on. Among these activation functions, SiLU is the most used. Generally speaking, SiLU performs with better accuracy and does not cause too much extra computation cost. However, when it comes to industrial applications, especially for deploying models with TensorRT [28] acceleration, ReLU has a greater speed advantage because of its fusion into convolution.

卷积层和激活函数的组合YOLO系列采用了多种激活函数，ReLU[27]、LReLU[25]、Swish[31]、SiLU[4]、Mish[26]等。在这些激活函数中，SiLU是最常用的。一般来说，SiLU具有更好的精度，不会导致太多额外的计算成本。然而，当涉及到工业应用时，尤其是在部署具有TensorRT[28]加速的模型时，ReLU由于融合到卷积中而具有更大的速度优势。

Table 2: Ablation study on backbones and necks. YOLOv6-L here is equipped with ReLU.
表2：主干和颈部的消融研究。这里的YOLOv6-L配备了ReLU。

Moreover, we further verify the effectiveness of combinations of RepConv/ordinary convolution (denoted as Conv) and ReLU/SiLU/LReLU in networks of different sizes to achieve a better trade-off. As shown in Table 4, Conv with SiLU performs the best in accuracy while the combination of RepConv and ReLU achieves a better trade-off. We suggest users adopt RepConv with ReLU in latency-sensitive applications. We choose to use RepConv/ReLU combination in YOLOv6-N/T/S/M for higher inference speed and use the Conv/SiLU combination in the large model YOLOv6-L to speed up training and improve performance.

此外，我们还进一步验证了不同规模网络中RepConv/普通卷积(表示为Conv)和ReLU/SiLU/LReLU组合的有效性，以实现更好的权衡。如表4所示，使用SiLU的Conv在精度上表现最佳，而RepConv和ReLU的组合实现了更好的权衡。我们建议用户在对延迟敏感的应用程序中采用带有ReLU的RepConv。我们选择在YOLOv6-N/T/S/M中使用RepConv/ReLU组合以获得更高的推理速度，在大型模型YOLOv 6-L中使用Conv/SiLU组合来加快训练和提高性能。

Table 3: Ablation study on the neck settings of YOLOv6-L. SiLU is selected as the activation function.
表3:YOLOv6-L颈部设置的消融研究。选择SiLU作为激活功能。

Table 4: Ablation study on combinations of different types of convolutional layers (denoted as Conv.) and activation layers (denoted as Act.). 
表4：不同类型卷积层(表示为Conv.)和激活层(表示为由Act.)组合的消融研究。

Miscellaneous design We also conduct a series of ablation on other network parts mentioned in Section 2.1 based on YOLOv6-N. We choose YOLOv5-N as the baseline and add other components incrementally. Results are shown in Table 5. Firstly, with decoupled head (denoted as DH), our model is 1.4% more accurate with 5% increase in time cost. Secondly, we verify that the anchor-free paradigm is 51% faster than the anchor-based one for its 3× less predefined anchors, which results in less dimensionality of the output. Further, the unified modification of the backbone (EfficientRep Backbone) and the neck (Rep-PAN neck), denoted as EB+RN, brings 3.6% AP improvements, and runs 21% faster. Finally, the optimized decoupled head (hybrid channels, HC) brings 0.2% AP and 6.8% FPS improvements in accuracy and speed respectively.

其他设计我们还根据YOLOv6-N对第2.1节中提到的其他网络部件进行了一系列消融。我们选择YOLOv5-N作为基线，并逐步添加其他组件。结果如表5所示。首先，对于解耦头(表示为DH)，我们的模型精度提高了1.4%，时间成本增加了5%。其次，我们验证了无锚点范式比基于锚点的范式快51%，因为它的预定义锚点少了3×1，这导致输出的维数更少。此外，主干(EfficientRep backbone)和颈部(Rep PAN颈部)(表示为EB+RN)的统一修改带来3.6%的AP改进，运行速度提高21%。最后，优化的解耦头(混合信道，HC)在精度和速度上分别提高了0.2%的AP和6.8%的FPS。

#### 3.3.2 Label Assignment
In Table 6, we analyze the effectiveness of mainstream label assign strategies. Experiments are conducted on YOLOv6-N. As expected, we observe that SimOTA and TAL are the best two strategies. Compared with the ATSS, SimOTA can increase AP by 2.0%, and TAL brings 0.5% higher AP than SimOTA. Considering the stable training and better accuracy performance of TAL, we adopt TAL as our label assignment strategy.

在表6中，我们分析了主流标签分配策略的有效性。实验在YOLOv6-N上进行。正如所料，我们观察到，SimOTA和TAL是最好的两种策略。与ATSS相比，SimOTA可以增加2.0%的AP，而TAL带来的AP比SimOTA高0.5%。考虑到好未来稳定的训练和更好的准确性，我们采用好未来作为我们的标签分配策略。

Table 5: Ablation study on all network designs in an incremental way. FPS is tested with FP16-precision and batchsize=32 on Tesla T4 GPUs.
表5：以递增方式对所有网络设计进行的消融研究。FPS在Tesla T4 GPU上以FP16精度和批次大小=32进行测试。

Table 6: Comparisons of label assignment methods.
表6：标签分配方法的比较。

Table 7: Comparisons of label assignment methods in warm-up stage. 
表7：预热阶段标签分配方法的比较。

In addition, the implementation of TOOD [5] adopts ATSS [51] as the warm-up label assignment strategy during the early training epochs. We also retain the warm-up strategy and further make some explorations on it. Details are shown in Table 7, and we can find that without warm-up or warmed up by other strategies (i.e., SimOTA) it can also achieve the similar performance.

此外，TOOD[5]的实施采用ATSS[51]作为早期训练阶段的预热标签分配策略。我们还保留了热身策略，并对其进行了进一步的探索。详情如表7所示，我们可以发现，如果不预热或通过其他策略(即SimOTA)预热，它也可以实现类似的性能。

#### 3.3.3 Loss functions
In the object detection framework, the loss function is composed of a classification loss, a box regression loss and an optional object loss, which can be formulated as follows:

在目标检测框架中，损失函数由分类损失、边框回归损失和可选对象损失组成，其公式如下：

$L_{det} = L_{cls} + λL_{reg} + µL{obj} $, (3) 

where $L_{cls}, L_{reg} and L_{obj}$ are classification loss, regression loss and object loss. λ and µ are hyperparameters. 

其中$L_{cls}, L_{reg}和L_{obj}$是分类损失、回归损失和目标损失。λ和µ是超参数。

Table 8: Ablation study on classification loss functions.
表8：分类损失函数的消融研究。

In this subsection, we evaluate each loss function on YOLOv6. Unless otherwise specified, the baselines for YOLOv6-N, YOLOv6-S and YOLOv6-M are 35.0%, 42.9% and 48.0% trained with TAL, Focal Loss and GIoU Loss.

在本小节中，我们评估了YOLOv6上的每个损失函数。除非另有规定，否则YOLOv6-N、YOLOv6-S和YOLOv6-M的基线分别为35.0%、42.9%和48.0%，分别接受了TAL、Focal loss和GIoU loss的训练。

Classification Loss We experiment Focal Loss [22], Poly loss [17], QFL [20] and VFL [50] on YOLOv6-N/S/M. As can be seen in Table 8, VFL brings 0.2%/0.3%/0.1% AP improvements on YOLOv6-N/S/M respectively compared with Focal Loss. We choose VFL as the classification loss function.

分类损失我们在YOLOv6-N/S/M上实验了焦损[22]、聚损[17]、QFL[20]和VFL[50]。如表8所示，与Focal Loss相比，VFL在YOLOv6-N/S/M上分别带来了0.2%/0.3%/0.1%的AP改善。我们选择VFL作为分类损失函数。

Regression Loss IoU-series and probability loss functions are both experimented with on YOLOv6-N/S/M.

回归损失IoU序列和概率损失函数都在YOLOv6-N/S/M上进行了实验。

The latest IoU-series losses are utilized in YOLOv6-N/S/M. Experiment results in Table 9 show that SIoU Loss outperforms others for YOLOv6-N and YOLOv6-T, while CIoU Loss performs better on YOLOv6-M.

YOLOv6-N/S/M中使用了最新的IoU系列损失。表9中的实验结果表明，对于YOLOv6-N和YOLOv6-T，SIoU Loss的表现优于其他产品，而对于YOLOv6-M，CIoU Lost的表现更好。

For probability losses, as listed in Table 10, introducing DFL can obtain 0.2%/0.1%/0.2% performance gain for YOLOv6-N/S/M respectively. However, the inference speed is greatly affected for small models. Therefore, DFL is only introduced in YOLOv6-M/L.

对于概率损失，如表10所示，对于YOLOv6-N/S/M，引入DFL可以分别获得0.2%/0.1%/0.2%的性能增益。然而，对于小模型，推理速度会受到很大影响。因此，DFL仅在YOLOv6-M/L中引入。

Object Loss. Object loss is also experimented with YOLOv6, as shown in Table 11. From Table 11, we can see that object loss has negative effects on YOLOv6-N/S/M networks, where the maximum decrease is 1.1% AP on YOLOv6-N. The negative gain may come from the conflict between the object branch and the other two branches in TAL. Specifically, in the training stage, IoU between predicted boxes and ground-truth ones, as well as classification scores are used to jointly build a metric as the criteria to assign labels. However, the introduced object branch extends the number of tasks to be aligned from two to three, which obviously increases the difficulty. Based on the experimental results and this analysis, the object loss is then discarded in YOLOv6.

对象损失。对象损失也用YOLOv6进行了实验，如表11所示。从表11可以看出，对象损失对YOLOv6-N/S/M网络有负面影响，其中YOLOv6-N上的最大减少量为1.1%AP。负收益可能来自目标分支与TAL中其他两个分支之间的冲突。具体来说，在训练阶段，预测框和地面真实框之间的IoU以及分类分数被用于联合构建一个度量，作为分配标签的标准。然而，引入的对象分支将要对齐的任务数量从两个扩展到了三个，这明显增加了难度。根据实验结果和分析，在YOLOv6中丢弃对象损失。

Table 9: Ablation study on IoU-series box regression loss functions. The classification loss is VFL [50].
表9:IoU系列边框回归损失函数的消融研究。分类损失为VFL[50]。

Table 10: Ablation study on probability loss functions.
表10：概率损失函数的消融研究。

Table 11: Effectiveness of object loss. 
表11：对象损失的有效性。

### 3.4. Industry-handy improvements 行业方便的改进
More training epochs In practice, more training epochs is a simple and effective way to further increase the accuracy. Results of our small models trained for 300 and 400 epochs are shown in Table 12. We observe that training for longer epochs substantially boosts AP by 0.4%, 0.6%, 0.5% for YOLOv6-N, T, S respectively. Considering the acceptable cost and the produced gain, it suggests that training for 400 epochs is a better convergence scheme for YOLOv6. 

更多训练时间在实践中，更多训练时间是进一步提高准确性的简单而有效的方法。表12显示了我们针对300和400个时间段训练的小型模型的结果。我们观察到，对于YOLOv6-N、T、S而言，较长时间段的训练可显著提高AP，分别提高0.4%、0.6%和0.5%。考虑到可接受的成本和产生的收益，它表明400个周期的训练是YOLOv6的一个更好的收敛方案。

Table 12: Experiments of more training epochs on small models.
表12：在小模型上进行更多训练阶段的实验。

Table 13: Ablation study on the self-distillation.
表13：自蒸馏的消融研究。

Self-distillation We conducted detailed experiments to verify the proposed self-distillation method on YOLOv6-L. As can be seen in Table 13, applying the self-distillation only on the classification branch can bring 0.4% AP improvement. Furthermore, we simply perform the selfdistillation on the box regression task to have 0.3% AP increase. The introduction of weight decay boosts the model by 0.6% AP.

自蒸馏我们在YOLOv6-L上进行了详细的实验，以验证所提出的自蒸馏方法。如表13所示，仅在分级分支上应用自蒸馏可使AP提高0.4%。此外，我们只需对边框子回归任务执行自蒸馏，使AP增加0.3%。权重衰减的引入使模型提高了0.6%AP。

Gray border of images In Section 2.4.3, we introduce a strategy to solve the problem of performance degradation without extra gray borders. Experimental results are shown in Table 14. In these experiments, YOLOv6-N and YOLOv6-S are trained for 400 epochs and YOLOv6-M for 300 epochs. It can be observed that the accuracy of YOLOv6-N/S/M is lowered by 0.4%/0.5%/0.7% without Mosaic fading when removing the gray border. However, the performance degradation becomes 0.2%/0.5%/0.5% when adopting Mosaic fading, from which we find that, on the one hand, the problem of performance degradation is mitigated. On the other hand, the accuracy of small models (YOLOv6-N/S) is improved whether we pad gray borders or not. Moreover, we limit the input images to 634×634 and add gray borders by 3 pixels wide around the edges (more results can be found in Appendix C). With this strategy, the size of the final images is the expected 640×640. The results in Table 14 indicate that the final performance of YOLOv6-N/S/M is even 0.2%/0.3%/0.1% more accurate with the final image size reduced from 672 to 640.

图像的灰度边界在第2.4.3节中，我们介绍了一种在没有额外灰度边界的情况下解决性能下降问题的策略。实验结果如表14所示。在这些实验中，YOLOv6-N和YOLOv 6-S训练了400个周期，YOLOv6-M训练了300个周期。可以观察到，在去除灰色边界时，YOLOv6-N/S/M的精度降低了0.4%/0.5%/0.7%，而没有马赛克褪色。然而，当采用Mosaic衰落时，性能下降为0.2%/0.5%/0.5%，我们发现，一方面，性能下降的问题得到缓解。另一方面，无论是否填充灰色边界，小模型(YOLOv6-N/S)的精度都会得到提高。此外，我们将输入图像限制为634×634，并在边缘周围添加3像素宽的灰色边界(更多结果见附录C)。使用此策略，最终图像的大小为640×640。表14中的结果表明，YOLOv6-N/S/M的最终性能甚至提高了0.2%/0.3%/0.1%，最终图像大小从672减小到640。

Table 14: Experimental results about the strategies for solving the problem of the performance degradation without extra gray border.
表14：关于解决性能退化问题的策略的实验结果，没有额外的灰色边界。

### 3.5. Quantization Results 量化结果
We take YOLOv6-S as an example to validate our quantization method. The following experiment is on both two releases. The baseline model is trained for 300 epochs.

我们以YOLOv6-S为例来验证我们的量化方法。下面的实验针对两个版本。基线模型经过300个时代的训练。

#### 3.5.1 PTQ
The average performance is substantially improved when the model is trained with RepOptimizer, see Table 15. RepOptimizer is in general faster and nearly identical.

使用RepOptimizer训练模型时，平均性能显著提高，如表15所示。RepOptimimizer通常速度更快，几乎相同。

Table 15: PTQ performance of YOLOv6s trained with RepOptimizer.
表15：使用RepOptimizer训练的YOLOv6的PTQ性能。

#### 3.5.2 QAT
For v1.0, we apply fake quantizers to non-sensitive layers obtained from Section 2.5.2 to perform quantization-aware training and call it partial QAT. We compare the result with 11 full QAT in Table 16. Partial QAT leads to better accuracy with a slightly reduced throughput.

对于v1.0，我们将伪量化器应用于从2.5.2节获得的非敏感层，以执行量化感知训练，并称之为部分QAT。我们将结果与表16中的11个完全QAT进行了比较。部分QAT可以提高精度，但吞吐量略有降低。

Table 16: QAT performance of YOLOv6-S (v1.0) under different settings.
表16:YOLOv6-S(v1.0)在不同设置下的QAT性能。

Due to the removal of quantization-sensitive layers in v2.0 release, we directly use full QAT on YOLOv6-S trained with RepOptimizer. We eliminate inserted quantizers through graph optimization to obtain higher accuracy and faster speed. We compare the distillation-based quantization results from PaddleSlim [30] in Table 17. Note our quantized version of YOLOv6-S is the fastest and the most accurate, also see Fig. 1.

由于v2.0版本中去除了量化敏感层，我们直接在经过RepOptimizer训练的YOLOv6-S上使用完整的QAT。我们通过图形优化消除插入量化器，以获得更高的精度和更快的速度。我们比较了表17中PaddleSlim[30]的基于蒸馏的量化结果。注意，我们的量化版本YOLOv6-S是最快、最准确的，见图1。

Table 17: QAT performance of YOLOv6-S (v2.0) compared with other quantized detectors. ‘∗ ’: based on v1.0 release. ‘† ’: We tested with TensorRT 8 on Tesla T4 with a batch size of 1 and 32.
表17:YOLOv6-S(v2.0)与其他量化检测器相比的QAT性能。”∗ ’: 基于v1.0版本。“†”：我们在特斯拉T4上使用TensorRT 8进行了测试，批次大小为1和32。

## 4. Conclusion
In a nutshell, with the persistent industrial requirements in mind, we present the current form of YOLOv6, carefully examining all the advancements of components of object detectors up to date, meantime instilling our thoughts and practices. The result surpasses other available real-time detectors in both accuracy and speed. For the convenience of the industrial deployment, we also supply a customized quantization method for YOLOv6, rendering an ever-fast detector out-of-box. We sincerely thank the academic and industrial community for their brilliant ideas and endeavors. In the future, we will continue expanding this project to meet higher standards and more demanding scenarios.

简而言之，考虑到持续的工业需求，我们提出了YOLOv6的当前形式，仔细检查了目标检测器组件的所有最新进展，同时灌输了我们的思想和实践。结果在精度和速度上都优于其他可用的实时检测器。为了便于工业部署，我们还为YOLOv6提供了一种定制的量化方法，使其成为开箱即用的快速检测器。我们衷心感谢学术界和工业界的杰出想法和努力。未来，我们将继续扩大该项目，以满足更高的标准和更高的要求。

## References
1.  Alexey Bochkovskiy, Chien-Yao Wang, and HongYuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020. 2, 4, 6, 7
2.  Xiaohan Ding, Honghao Chen, Xiangyu Zhang, Kaiqi Huang, Jungong Han, and Guiguang Ding. Reparameterizing your optimizers rather than architectures. arXiv preprint arXiv:2205.15242, 2022. 2, 3, 6
3.  Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13733–13742, 2021. 2, 3, 4
4.  Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoidweighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11, 2018. 8
5.  Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R Scott, and Weilin Huang. Tood: Task-aligned one-stage object detection. In ICCV, 2021. 2, 4, 9
6.  Zheng Ge, Songtao Liu, Zeming Li, Osamu Yoshie, and Jian Sun. Ota: Optimal transport assignment for object detection. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 303–312, 2021. 4
7.  Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430, 2021. 2, 4, 5, 6, 7, 8, 9, 15
8.  Zhora Gevorgyan. Siou loss: More powerful learning for bounding box regression. arXiv preprint arXiv:2205.12740, 2022. 3, 5, 10
9.  Golnaz Ghiasi, Tsung-Yi Lin, and Quoc V Le. Nas-fpn: Learning scalable feature pyramid architecture for object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7036–7045, 2019. 4
10.  Jocher Glenn. YOLOv5 release v6.1. https://github. com/ultralytics/yolov5/releases/tag/v6. 1, 2022. 2, 4, 6, 7, 8, 15
11.  Jiabo He, Sarah Erfani, Xingjun Ma, James Bailey, Ying Chi, and Xian-Sheng Hua. α-iou: A family of power intersection over union losses for bounding box regression. Advances in Neural Information Processing Systems, 34:20230–20242, 2021. 5
12.  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
13.  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European conference on computer vision, pages 630–645. Springer,2016. 3
14.  Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4700–4708, 2017. 3
15.  Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012. 3
16.  Hei Law and Jia Deng. Cornernet: Detecting objects as paired keypoints. In Proceedings of the European conference on computer vision (ECCV), pages 734–750, 2018. 4 12
17.  Zhaoqi Leng, Mingxing Tan, Chenxi Liu, Ekin Dogus Cubuk, Xiaojie Shi, Shuyang Cheng, and Dragomir Anguelov. Polyloss: A polynomial expansion perspective of classification loss functions. arXiv preprint arXiv:2204.12511, 2022. 5, 10
18.  Shuai Li, Chenhang He, Ruihuang Li, and Lei Zhang. A dual weighting label assignment scheme for object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9387–9396, June 2022. 2, 4, 9
19.  Xiang Li, Wenhai Wang, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss v2: Learning reliable localization quality estimation for dense object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11632–11641, 2021. 5, 10
20.  Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection. Advances in Neural Information Processing Systems, 33:21002–21012, 2020. 3, 5, 6, 10
21.  Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, ´ Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017. 4
22.  Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In ´ Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017. 5, 10
23.  Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence ´ Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 7
24.  Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. Path aggregation network for instance segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8759–8768, 2018. 2, 4
25.  Andrew L Maas, Awni Y Hannun, Andrew Y Ng, et al. Rectifier nonlinearities improve neural network acoustic models. In Proc. icml, volume 30, page 3. Citeseer, 2013. 8
26.  Diganta Misra. Mish: A self regularized non-monotonic neural activation function. arXiv preprint arXiv:1908.08681, 4(2):10–48550, 2019. 8
27.  Vinod Nair and Geoffrey E Hinton. Rectified linear units improve restricted boltzmann machines. In Icml, 2010. 8
28.  NVIDIA. TensorRT. https://developer.nvidia. com/tensorrt, 2018. 7, 8
29.  NVIDIA. pytorch-quantization’s documentation. https://docs.nvidia.com/deeplearning/ tensorrt/pytorch-quantization-toolkit/ docs/index.html, 2021. 7
30.  PaddleSlim. PaddleSlim documentation. https: //github.com/PaddlePaddle/PaddleSlim/ tree/develop/example/auto_compression/ pytorch_yolo_series, 2022. 12
31.  Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. arXiv preprint arXiv:1710.05941, 2017. 8
32.  Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779–788, 2016. 2
33.  Joseph Redmon and Ali Farhadi. Yolo9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7263–7271, 2017. 2
34.  Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018. 2
35.  Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 658–666,2019. 3, 5, 10
36.  Changyong Shu, Yifan Liu, Jianfei Gao, Zheng Yan, and Chunhua Shen. Channel-wise knowledge distillation for dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5311–5320,2021. 2, 3, 7
37.  Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 3
38.  Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1–9, 2015. 3
39.  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016. 3
40.  Mingxing Tan, Ruoming Pang, and Quoc V Le. Efficientdet: Scalable and efficient object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10781–10790, 2020. 4
41.  Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. FCOS: Fully convolutional one-stage object detection. In Proc. Int. Conf. Computer Vision (ICCV), 2019. 4, 5
42.  Chien-Yao Wang, Alexey Bochkovskiy, and HongYuan Mark Liao. Yolov7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696, 2022. 2, 6, 7, 8, 15
43.  Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. Cspnet: A new backbone that can enhance learning capability of cnn. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 390–391, 2020. 2
44.  Shangliang Xu, Xinxin Wang, Wenyu Lv, Qinyao Chang, Cheng Cui, Kaipeng Deng, Guanzhong Wang, Qingqing Dang, Shengyu Wei, Yuning Du, et al. Pp-yoloe: An evolved version of yolo. arXiv preprint arXiv:2203.16250, 2022. 2 13
45.  Shangliang Xu, Xinxin Wang, Wenyu Lv, Qinyao Chang, Cheng Cui, Kaipeng Deng, Guanzhong Wang, Qingqing Dang, Shengyu Wei, Yuning Du, et al. Pp-yoloe: An evolved version of yolo. arXiv preprint arXiv:2203.16250, 2022. 4, 7, 8, 15
46.  Ze Yang, Shaohui Liu, Han Hu, Liwei Wang, and Stephen Lin. Reppoints: Point set representation for object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9657–9666, 2019. 4
47.  Jiahui Yu, Yuning Jiang, Zhangyang Wang, Zhimin Cao, and Thomas Huang. Unitbox: An advanced object detection network. In Proceedings of the 24th ACM international conference on Multimedia, pages 516–520, 2016. 5
48.  Mohsen Zand, Ali Etemad, and Michael A. Greenspan. Objectbox: From centers to boxes for anchor-free object detection. ArXiv, abs/2207.06985, 2022. 2, 4, 9
49.  Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017. 7
50.  Haoyang Zhang, Ying Wang, Feras Dayoub, and Niko Sunderhauf. Varifocalnet: An iou-aware dense object detector. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8514–8523, 2021. 3, 5, 10
51.  Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z. Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In CVPR, 2020. 2, 4, 9
52.  Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. Distance-iou loss: Faster and better learning for bounding box regression. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 12993–13000, 2020. 5, 10
53.  Xingyi Zhou, Dequan Wang, and Philipp Krahenb ¨ uhl. Ob- ¨ jects as points. arXiv preprint arXiv:1904.07850, 2019. 4 14 

## Appendix
### A. Detailed Latency and Throughput Benchmark 详细的延迟和吞吐量基准
#### A.1. Setup
Unless otherwise stated, all the reported latency is mea- sured on an NVIDIA Tesla T4 GPU with TensorRT ver- sion 7.2.1.6. Due to the large variance of the hardware and software settings, we re-measure latency and throughput of all the models under the same configuration (both hardware and software). For a handy reference, we also switch Ten- sorRT versions (Table 18) for consistency check. Latency on a V100 GPU (Table 19) is included for a convenient comparison. This gives us a full spectrum view of state- of-the-art detectors.

除非另有说明，所有报告的延迟都是在TensorRT版本为7.2.1.6的NVIDIA Tesla T4 GPU上测量的。由于硬件和软件设置差异较大，我们重新测量了相同配置(硬件和软件)下所有模型的延迟和吞吐量。为了方便参考，我们还切换了Ten-sorRT版本(表18)进行一致性检查。为了便于比较，V100 GPU上的延迟(表19)也包括在内。这为我们提供了最先进检测器的全光谱视图。

#### A.2. T4 GPU Latency Table with TensorRT 8 带TensorRT 8的T4 GPU延迟表
See Table 18. The throughput of YOLOv6 models still emulates their peers.

请参见表18。YOLOv6模型的吞吐量仍在模仿其对等机。

Table 18: YOLO-series comparison of latency and through- put on a T4 GPU with a higher version of TensorRT (8.2).
表18:T4 GPU上使用更高版本的TensorRT的YOLO系列延迟和吞吐量比较(8.2)。

#### A.3. V100 GPU Latency Table
See Table 19. The speed advantage of YOLOv6 is largely maintained.

见表19。YOLOv6的速度优势基本保持不变。

#### A.4. CPU Latency
We evaluate the performance of our models and other competitors on a 2.6 GHz Intel Core i7 CPU using OpenCV Deep Neural Network (DNN), as shown in Table 20.

我们使用OpenCV Deep Neural Network(DNN)评估了我们的模型和其他竞争对手在2.6 GHz Intel Core i7 CPU上的性能，如表20所示。

Table 19: YOLO-series comparison of latency and through- put on a V100 GPU. We measure all models at FP16- precision with the input size 640×640 in the exact same environment.

表19:V100 GPU上的YOLO系列延迟和吞吐量比较。我们在完全相同的环境中以FP16精度测量所有模型，输入大小为640×640。

### B. Quantization Details 量化详情
#### B.1. Feature Distribution Comparison 特征分布比较
We illustrate the feature distribution of more layers that are much alleviated after trained with RepOptimizer, see Fig. 6.

我们展示了使用RepOptimizer训练后大大减轻的更多层的特征分布，见图6。

#### B.2. Sensitivity Analysis Results 敏感性分析结果
See Fig. 7, we observe that SNR and Cosine similar- ity gives highly correlated results. However, directly eval- uating AP produces a different panorama. Nevertheless, in terms of final quantization performance, MSE is the closest to direct AP evaluation, see Table 21.

见图7，我们观察到信噪比和余弦相似性给出了高度相关的结果。然而，直接评估AP会产生不同的全景。然而，就最终量化性能而言，MSE最接近直接AP评估，见表21。

Table 21: Partial post-training quantization performance w.r.t. difference sensitivity metrics.
表21：训练后部分量化性能w.r.t.差异敏感性指标。

### C. Analysis of Gray Border 灰色边界分析
To analyze the effect of the gray border, we further ex- plore different border settings with the loaded images re- sized to different sizes and padded to 640×640. For ex- ample, when the image size is 608, and the border size is set to 16, and so on. In addition, we alleviate a problem of the information misalignment between pre-processing and post-processing via a simple adjustment. Results are shown in Fig. 8. We can observe that each model achieves the best AP with a different border size. Additionally, compared with an input size of 640, our models get about 0.3% higher AP on average if the input image size locates in the range from 632 to 638.

为了分析灰色边框的效果，我们进一步扩展了不同的边框设置，将加载的图像重新调整为不同大小，并填充到640×640。例如，当图像大小为608，边框大小设置为16，等等。此外，我们通过简单的调整缓解了预处理和后处理之间的信息不一致问题。结果如图8所示。我们可以观察到，在不同的边界大小下，每个模型都获得了最佳AP。此外，与640的输入大小相比，如果输入图像大小在632到638之间，我们的模型平均获得约0.3%的AP。