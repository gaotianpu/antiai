# YOLOX: Exceeding YOLO Series in 2021
YOLOX：2021超越YOLO系列 2021-7-18 论文：https://arxiv.org/abs/2107.08430

## Abstract
In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector—YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLONano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported. Source code is at https://github.com/Megvii-BaseDetection/YOLOX .

在本报告中，我们介绍了对YOLO系列的一些有经验的改进，形成了一种新的高性能检测器YOLOX。我们将YOLO探测器切换到无锚方式，并采用其他先进的检测技术，即去耦磁头和领先的标签分配策略SimOTA，以在大范围的模型中实现最先进的结果：对于只有0.91M参数和1.08G FLOP的YOLONano，我们在COCO上获得25.3%的AP，超过NanoDet 1.8%的AP; 对于工业上使用最广泛的探测器之一YOLOv3，我们将其在COCO上提升至47.3%AP，比当前最佳实践高3.0%AP; 对于YOLOX-L，其参数量与YOLOv4CSP、YOLOv 5-L大致相同，我们在特斯拉V100上以68.9 FPS的速度在COCO上实现了50.0%的AP，超过YOLOv-L 1.8%的AP。此外，我们使用一个YOLOX-L模型获得了流感知挑战赛(2021 CVPR自动驾驶研讨会)的第一名。我们希望这份报告能够为实际场景中的开发人员和研究人员提供有用的经验，我们还提供支持ONNX、TensorRT、NCNN和Openvino的部署版本。源代码位于https://github.com/Megvii-BaseDetection/YOLOX .

## 1. Introduction
With the development of object detection, YOLO series [23, 24, 25, 1, 7] always pursuit the optimal speed and accuracy trade-off for real-time applications. They extract the most advanced detection technologies available at the time (e.g., anchors [26] for YOLOv2 [24], Residual Net [9] for YOLOv3 [25]) and optimize the implementation for best practice. Currently, YOLOv5 [7] holds the best trade-off performance with 48.2% AP on COCO at 13.7 ms.1

随着目标检测的发展，YOLO系列[23、24、25、1、7]始终追求实时应用的最佳速度和精度权衡。他们提取了当时可用的最先进的检测技术(例如，YOLOv2的锚[26][24]，YOLOv3的残差网[9][25])，并优化了最佳实践的实施。目前，YOLOv5[7]拥有最佳的权衡性能，COCO上的AP为48.2%，时间为13.7 ms.1

Nevertheless, over the past two years, the major advances in object detection academia have focused on anchor-free detectors [29, 40, 14], advanced label assignment strategies [37, 36, 12, 41, 22, 4], and end-to-end (NMS-free) detectors [2, 32, 39]. These have not been integrated into YOLO families yet, as YOLOv4 and YOLOv5 are still anchor-based detectors with hand-crafted assigning rules for training.

然而，在过去两年里，目标检测学术界的主要进展集中在无锚探测器[29、40、14]、高级标签分配策略[37、36、12、41、22、4]和端到端(无NMS)探测器[2、32、39]。这些还没有集成到YOLO系列中，因为YOLOv4和YOLOv 5仍然是基于锚点的探测器，具有手工制作的分配规则用于训练。

That’s what brings us here, delivering those recent advancements to YOLO series with experienced optimization. Considering YOLOv4 and YOLOv5 may be a little over-optimized for the anchor-based pipeline, we choose YOLOv3 [25] as our start point (we set YOLOv3-SPP as the default YOLOv3). Indeed, YOLOv3 is still one of the most widely used detectors in the industry due to the limited computation resources and the insufficient software support in various practical applications.

这就是我们来到这里的原因，通过经验丰富的优化为YOLO系列提供了这些最新的改进。考虑到YOLOv4和YOLOv 5对于基于锚点的管道可能过于优化，我们选择YOLOv3[25]作为起点(我们将YOLOv-3 SPP设置为默认的YOLOf3)。事实上，YOLOv3仍然是业界使用最广泛的探测器之一，因为在各种实际应用中，计算资源有限，软件支持不足。

As shown in Fig. 1, with the experienced updates of the above techniques, we boost the YOLOv3 to 47.3% AP (YOLOX-DarkNet53) on COCO with 640 × 640 resolution, surpassing the current best practice of YOLOv3 (44.3% AP, ultralytics version2) by a large margin. Moreover, when switching to the advanced YOLOv5 architecture that adopts an advanced CSPNet [31] backbone and an additional PAN [19] head, YOLOX-L achieves 50.0% AP on COCO with 640 × 640 resolution, outperforming the counterpart YOLOv5-L by 1.8% AP. We also test our design strategies on models of small size. YOLOX-Tiny and YOLOX-Nano (only 0.91M Parameters and 1.08G FLOPs) outperform the corresponding counterparts YOLOv4-Tiny and NanoDet3 by 10% AP and 1.8% AP, respectively.

如图1所示，通过对上述技术进行经验丰富的更新，我们将COCO上的YOLOv3提高到47.3%AP(YOLOX-DarkNet53)，分辨率为640×640，大大超过了YOLOv当前的最佳实践(44.3%AP，ultralytics版本2)。此外，当切换到采用先进CSPNet[31]主干和额外PAN[19]头的先进YOLOv5架构时，YOLOX-L在COCO上以640×640分辨率达到50.0%的AP，比对应的YOLOv 5-L高1.8%的AP。我们还在小尺寸模型上测试我们的设计策略。YOLOX Tiny和YOLOX Nano(仅0.91M参数和1.08G FLOP)的表现分别优于相应的YOLOv4 Tiny和NanoDet3 10%AP和1.8%AP。

We have released our code at https://github. com/Megvii-BaseDetection/YOLOX, with ONNX, TensorRT, NCNN and Openvino supported. One more thing worth mentioning, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model.

我们已在发布代码https://github.com/Megvii BaseDetection/YOLOX，支持ONNX、TensorRT、NCNN和Openvino。值得一提的是，我们使用一个YOLOX-L模型在流感知挑战赛(2021 CVPR自动驾驶研讨会)上获得了第一名。

## 2. YOLOX
### 2.1. YOLOX-DarkNet53
We choose YOLOv3 [25] with Darknet53 as our baseline. In the following part, we will walk through the whole system designs in YOLOX step by step.

我们选择YOLOv3[25]，以Darknet53作为基线。在接下来的部分中，我们将逐步介绍YOLOX中的整个系统设计。

Implementation details Our training settings are mostly consistent from the baseline to our final model. We train the models for a total of 300 epochs with 5 epochs warmup on COCO train2017 [17]. We use stochastic gradient descent (SGD) for training. We use a learning rate of lr×BatchSize/64 (linear scaling [8]), with a initial lr = 0.01 and the cosine lr schedule. The weight decay is 0.0005 and the SGD momentum is 0.9. The batch size is 128 by default to typical 8-GPU devices. Other batch sizes include single GPU training also work well. The input size is evenly drawn from 448 to 832 with 32 strides. FPS and latency in this report are all measured with FP16-precision and batch=1 on a single Tesla V100.

实施细节 我们的训练设置从基线到最终模型基本一致。我们在COCO train2017上对模型进行了总共300个周期的训练，并进行了5个周期的预热[17]。我们使用随机梯度下降(SGD)进行训练。我们使用学习速率lr×BatchSize/64(线性缩放[8])，初始lr=0.01，余弦lr调度。权重衰减为0.0005，SGD动量为0.9。对于典型的8-GPU设备，默认批量大小为128。其他批量大小包括单GPU训练也很有效。输入大小从448均匀绘制到832，步幅为32。本报告中的FPS和延迟均以FP16精度和批次=1在单个特斯拉V100上测量。

YOLOv3 baseline Our baseline adopts the architecture of DarkNet53 backbone and an SPP layer, referred to YOLOv3-SPP in some papers [1, 7]. We slightly change some training strategies compared to the original implementation [25], adding EMA weights updating, cosine lr schedule, IoU loss and IoU-aware branch. We use BCE Loss for training cls and obj branch, and IoU Loss for training reg branch. These general training tricks are orthogonal to the key improvement of YOLOX, we thus put them on the baseline. Moreover, we only conduct RandomHorizontalFlip, ColorJitter and multi-scale for data augmentation and discard the RandomResizedCrop strategy, because we found the RandomResizedCrop is kind of overlapped with the planned mosaic augmentation. With those enhancements, our baseline achieves 38.5% AP on COCO val, as shown in Tab. 2.

YOLOv3基线我们的基线采用DarkNet53主干和SPP层的架构，在一些论文中提到了YOLOv 3 SPP[1,7]。与最初的实施相比，我们略微改变了一些训练策略[25]，增加了EMA权重更新、余弦lr计划、IoU损失和IoU感知分支。我们将BCE损失用于训练cls和obj分支机构，将IoU损失用于训练reg分支机构。这些一般的训练技巧与YOLOX的关键改进是正交的，因此我们将它们放在了基线上。此外，我们只进行了RandomHorizontalFlip、ColorJitter和multi-scale数据增广，并放弃了RandomResizedCrop策略，因为我们发现RandomResisedCrop与计划的马赛克增强有点重叠。通过这些增强，我们的基线在COCO值上达到38.5%AP，如表2所示。

Decoupled head In object detection, the conflict between classification and regression tasks is a well-known problem [27, 34]. Thus the decoupled head for classification and localization is widely used in the most of one-stage and two-stage detectors [16, 29, 35, 34]. However, as YOLO series’ backbones and feature pyramids ( e.g., FPN [13], PAN [20].) continuously evolving, their detection heads remain coupled as shown in Fig. 2.

在目标检测中，分类和回归任务之间的冲突是一个众所周知的问题[27，34]。因此，用于分类和定位的解耦磁头广泛用于大多数一级和两级探测器[16，29，35，34]。然而，随着YOLO系列的主干和特征金字塔(如FPN[13]、PAN[20].)不断发展，其探测头仍保持耦合，如图2所示。

Our two analytical experiments indicate that the coupled detection head may harm the performance. 1). Replacing YOLO’s head with a decoupled one greatly improves the converging speed as shown in Fig. 3. 2). The decoupled head is essential to the end-to-end version of YOLO (will be described next). One can tell from Tab. 1, the end-toend property decreases by 4.2% AP with the coupled head, while the decreasing reduces to 0.8% AP for a decoupled head. We thus replace the YOLO detect head with a lite decoupled head as in Fig. 2. Concretely, it contains a 1 × 1 conv layer to reduce the channel dimension, followed by two parallel branches with two 3 × 3 conv layers respectively. We report the inference time with batch=1 on V100 in Tab. 2 and the lite decoupled head brings additional 1.1 ms (11.6 ms v.s. 10.5 ms).

我们的两个分析实验表明，耦合探测头可能会损害性能。1). 如图3.2所示，将YOLO的头部更换为解耦的头部可大大提高收敛速度。解耦头对于端到端版本的YOLO至关重要(将在下文中介绍)。从表1可以看出，耦合头的端到端特性降低4.2%AP，而解耦头的端-端特性降低到0.8%AP。因此，我们将YOLO探测头替换为一个如图2所示的轻度解耦的探测头。具体来说，它包含一个1×1 conv层以减小通道尺寸，然后是两个平行分支，分别具有两个3×3 conv层。我们在表2中报告了V100上批次=1的推断时间，并且lite去耦头带来额外的1.1 ms(11.6 ms vs.10.5 ms)。

Figure 2: Illustration of the difference between YOLOv3 head and the proposed decoupled head. For each level of FPN feature, we first adopt a 1 × 1 conv layer to reduce the feature channel to 256 and then add two parallel branches with two 10 0.203918 10 0.090225 0.177918 0.057225 Step Decoupled head Step Baseline 3 × 320co0n.2v729l9a3yers each for20cla0.s15s9i8fi61cation an0d.24r6e9g93re0s.1s2i6o8n61 tasks respectively. IoU branch is added on the regression branch.

图2:YOLOv3封头和提出的解耦封头之间的差异图解。对于每个级别的FPN特征，我们首先采用1×1 conv层将特征通道减少到256，然后添加两个并行分支，两个分支分别为10 0.203918 10 0.090225 0.177918 0.057225步解耦头步基线3×320co0n。2 v729 l9层，每个层用于cla0.s15s8 fi8阳离子和d.24 r6 g9 re0。分别完成1项或6项任务。IoU分支添加到回归分支上。

Figure 3: .

Strong data augmentation We add Mosaic and MixUp into our augmentation strategies to boost YOLOX’s performance. Mosaic is an efficient augmentation strategy proposed by ultralytics-YOLOv32. It is then widely used in YOLOv4 [1], YOLOv5 [7] and other detectors [3]. MixUp [10] is originally designed for image classification task but then modified in BoF [38] for object detection training. We adopt the MixUp and Mosaic implementation in our model and close it for the last 15 epochs, achieving 42.0% AP in Tab. 2. After using strong data augmentation, we found ImageNet pre-training is no more beneficial, we thus train all the following models from scratch. 

强大的数据增广我们将Mosaic和MixUp添加到增强策略中，以提高YOLOX的性能。Mosaic是ultralytics-YOLOv32提出的一种高效增强策略。它随后被广泛应用于YOLOv 4[1]、YOLOf 5[7]和其他检测器[3]。MixUp[10]最初是为图像分类任务设计的，但后来在BoF[38]中进行了修改，用于目标检测训练。我们在模型中采用了MixUp和Mosaic实现，并在最后15个时期将其关闭，在表2中实现了42.0%的AP。在使用了强大的数据增广后，我们发现ImageNet预训练没有什么好处，因此我们从头开始训练以下所有模型。

Anchor-free Both YOLOv4 [1] and YOLOv5 [7] follow the original anchor-based pipeline of YOLOv3 [25]. However, the anchor mechanism has many known problems. First, to achieve optimal detection performance, one needs to conduct clustering analysis to determine a set of optimal anchors before training. Those clustered anchors are domain-specific and less generalized. Second, anchor mechanism increases the complexity of detection heads, as well as the number of predictions for each image. On some edge AI systems, moving such large amount of predictions between devices (e.g., from NPU to CPU) may become a potential bottleneck in terms of the overall latency.

无锚点YOLOv4[1]和YOLOv 5[7]均遵循YOLOv3[25]原有的基于锚点的管道。然而，锚机制有许多已知问题。首先，为了获得最佳检测性能，需要在训练之前进行聚类分析以确定一组最佳锚。这些集群锚定是特定于域的，不太通用。其次，锚机制增加了检测头的复杂性，以及每个图像的预测数量。在某些边缘人工智能系统中，在设备之间移动如此大量的预测(例如，从NPU到CPU)可能会成为整体延迟方面的潜在瓶颈。

Anchor-free detectors [29, 40, 14] have developed rapidly in the past two year. These works have shown that the performance of anchor-free detectors can be on par with anchor-based detectors. Anchor-free mechanism significantly reduces the number of design parameters which need heuristic tuning and many tricks involved (e.g., Anchor Clustering [24], Grid Sensitive [11].) for good performance, making the detector, especially its training and decoding phase, considerably simpler [29].

无锚探测器[29、40、14]在过去两年中发展迅速。这些工作表明，无锚探测器的性能可以与基于锚的探测器相当。无锚机制大大减少了需要启发式调整的设计参数的数量，并涉及到许多技巧(例如，锚聚类[24]、网格敏感[11])，以获得良好的性能，使检测器，尤其是其训练和解码阶段，大大简化[29]。

Switching YOLO to an anchor-free manner is quite simple. We reduce the predictions for each location from 3 to 1 and make them directly predict four values, i.e., two offsets in terms of the left-top corner of the grid, and the height and width of the predicted box. We assign the center location of each object as the positive sample and pre-define a scale range, as done in [29], to designate the FPN level for each object. Such modification reduces the parameters and GFLOPs of the detector and makes it faster, but obtains better performance – 42.9% AP as shown in Tab. 2.

将YOLO切换到无锚点方式非常简单。我们将每个位置的预测值从3减少到1，并使其直接预测四个值，即网格左上角的两个偏移，以及预测框的高度和宽度。我们将每个对象的中心位置指定为正样本，并预先定义比例范围，如[29]所述，以指定每个对象的FPN级别。这种修改减少了探测器的参数和GFLOP，使其更快，但获得了更好的性能——42.9%AP，如表2所示。

Multi positives To be consistent with the assigning rule of YOLOv3, the above anchor-free version selects only ONE positive sample (the center location) for each object meanwhile ignores other high quality predictions. However, optimizing those high quality predictions may also bring beneficial gradients, which may alleviates the extreme imbalance of positive/negative sampling during training. We simply assigns the center 3×3 area as positives, also named “center sampling” in FCOS [29]. The performance of the detector improves to 45.0% AP as in Tab. 2, already surpassing the current best practice of ultralytics-YOLOv3 (44.3% AP2).

多阳性为了符合YOLOv3的赋值规则，上述无锚版本仅为每个对象选择一个阳性样本(中心位置)，同时忽略其他高质量预测。然而，优化这些高质量的预测也可能带来有利的梯度，这可能会缓解训练期间正/负采样的极端不平衡。我们简单地将中心3×3区域指定为正值，在FCOS中也称为“中心采样”[29]。如表2所示，检测器的性能提高到45.0%AP，已经超过目前ultralytics-YOLOv3的最佳实践(44.3%AP2)。

SimOTA Advanced label assignment is another important progress of object detection in recent years. Based on our own study OTA [4], we conclude four key insights for an advanced label assignment: 1). loss/quality aware, 2). center prior, 3). dynamic number of positive anchors4 for each ground-truth (abbreviated as dynamic top-k), 4). global view. OTA meets all four rules above, hence we choose it as a candidate label assigning strategy.

SimOTA Advanced标签分配是近年来目标检测的另一个重要进展。根据我们自己的研究OTA[4]，我们得出了高级标签分配的四个关键见解：1)。损失/质量意识，2)。居中优先，3)。每个地面实况的正主持人的动态数量4(缩写为动态top-k)，4)。全局视图。OTA满足以上四条规则，因此我们选择它作为候选标签分配策略。

Specifically, OTA [4] analyzes the label assignment from a global perspective and formulate the assigning procedure as an Optimal Transport (OT) problem, producing the SOTA performance among the current assigning strategies [12, 41, 36, 22, 37]. However, in practice we found solving OT problem via Sinkhorn-Knopp algorithm brings 25% extra training time, which is quite expensive for training 300 epochs. We thus simplify it to dynamic top-k strategy, named SimOTA, to get an approximate solution.

具体而言，OTA[4]从全局角度分析标签分配，并将分配过程制定为最优运输(OT)问题，从而在当前分配策略中产生SOTA性能[12，41，36，22，37]。然而，在实践中，我们发现通过Sinkhorn-Knopp算法解决OT问题会带来25%的额外训练时间，这对于训练300个学时来说相当昂贵。因此，我们将其简化为名为SimOTA的动态top-k策略，以获得近似解。

We briefly introduce SimOTA here. SimOTA first calculates pair-wise matching degree, represented by cost [4, 5, 12, 2] or quality [33] for each prediction-gt pair. For example, in SimOTA, the cost between gt gi and prediction pj is calculated as:

我们在此简要介绍SimOTA。SimOTA首先计算成对匹配度，用每个预测成对的成本[4、5、12、2]或质量[33]表示。例如，在SimOTA中，gt-gi和预测pj之间的成本计算如下：

where λ is a balancing coefficient. Lcls and Lreg are classficiation loss and regression loss between gt gi and prediction pj. Then, for gt gi, we select the top k predictions with the least cost within a fixed center region as its positive samples. Finally, the corresponding grids of those positive predictions are assigned as positives, while the rest grids are negatives. Noted that the value k varies for different ground-truth. Please refer to Dynamic k Estimation strategy in OTA [4] for more details.

式中，λ为平衡系数。Lcls和Lreg是gt-gi和预测pj之间的分类损失和回归损失。然后，对于gt-gi，我们选择固定中心区域内成本最低的前k个预测作为其正样本。最后，这些正预测的对应网格被指定为正，而其余网格则为负。注意，值k随不同的地面真实值而变化。有关更多详情，请参阅OTA[4]中的动态k估计策略。

SimOTA not only reduces the training time but also avoids additional solver hyperparameters in SinkhornKnopp algorithm. As shown in Tab. 2, SimOTA raises the detector from 45.0% AP to 47.3% AP, higher than the SOTA ultralytics-YOLOv3 by 3.0% AP, showing the power of the advanced assigning strategy.

SimOTA不仅减少了训练时间，而且避免了SinkhornKnopp算法中额外的解算器超参数。如表2所示，SimOTA将探测器从45.0%AP提升至47.3%AP，比SOTA ultralytics-YOLOv3高3.0%AP，显示了高级分配策略的威力。

End-to-end YOLO We follow [39] to add two additional conv layers, one-to-one label assignment, and stop gradient. These enable the detector to perform an end-to-end manner, but slightly decreasing the performance and the inference speed, as listed in Tab. 2. We thus leave it as an optional module which is not involved in our final models.

端到端YOLO我们按照[39]添加了两个额外的conv层、一对一标签分配和停止渐变。这些使检测器能够执行端到端的方式，但会略微降低性能和推理速度，如表2所示。因此，我们将其作为一个可选模块，不涉及最终模型。

### 2.2. Other Backbones
Besides DarkNet53, we also test YOLOX on other backbones with different sizes, where YOLOX achieves consistent improvements against all the corresponding counterparts.

除了DarkNet53之外，我们还在其他不同大小的主干上测试YOLOX，其中YOLOX相对于所有对应的主干实现了一致的改进。

Modified CSPNet in YOLOv5 To give a fair comparison, we adopt the exact YOLOv5’s backbone including modified CSPNet [31], SiLU activation, and the PAN [19] head. We also follow its scaling rule to product YOLOXS, YOLOX-M, YOLOX-L, and YOLOX-X models. Compared to YOLOv5 in Tab. 3, our models get consistent improvement by ∼3.0% to ∼1.0% AP, with only marginal time increasing (comes from the decoupled head).

YOLOv5中的改良CSPNet为了进行公平比较，我们采用了准确的YOLOv 5主干，包括改良CSPNet[31]、SiLU激活和PAN[19]头部。我们还遵循其缩放规则来生产YOLOXS、YOLOX-M、YOLO_X-L和YOLOX-X模型。与表3中的YOLOv5相比，我们的模型通过∼3.0%至∼1.0%AP，仅边际时间增加(来自去耦水头)。

Tiny and Nano detectors We further shrink our model as YOLOX-Tiny to compare with YOLOv4-Tiny [30]. For mobile devices, we adopt depth wise convolution to construct a YOLOX-Nano model, which has only 0.91M parameters and 1.08G FLOPs. As shown in Tab. 4, YOLOX performs well with even smaller model size than the counterparts.

微型和纳米探测器我们进一步缩小了YOLOX Tiny的模型，以与YOLOv4 Tiny相比[30]。对于移动设备，我们采用深度卷积来构建YOLOX Nano模型，该模型只有0.91M参数和1.08G FLOP。如表4所示，YOLOX在模型尺寸更小的情况下表现良好。

Model size and data augmentation In our experiments, all the models keep almost the same learning schedule and optimizing parameters as depicted in 2.1. However, we found that the suitable augmentation strategy varies across different size of models. As Tab. 5 shows, while applying MixUp for YOLOX-L can improve AP by 0.9%, it is better to weaken the augmentation for small models like YOLOX-Nano. Specifically, we remove the mix up augmentation and weaken the mosaic (reduce the scale range from [0.1, 2.0] to [0.5, 1.5]) when training small models, i.e., YOLOX-S, YOLOX-Tiny, and YOLOX-Nano. Such a modification improves YOLOX-Nano’s AP from 24.0% to 25.3%.

模型大小和数据增广在我们的实验中，所有模型都保持了几乎相同的学习计划和优化参数，如2.1所示。然而，我们发现，在不同大小的模型中，合适的增强策略有所不同。如表5所示，虽然为YOLOX-L应用MixUp可以将AP提高0.9%，但最好减弱对YOLOX Nano等小型模型的增强。具体而言，我们在训练小型模型，即YOLOX-S、YOLOX Tiny和YOLOX Nano时，消除了混合增强并削弱了镶嵌效果(将比例范围从[0.1，2.0]减小到[0.5，1.5])。这种改性将YOLOX Nano的AP从24.0%提高到25.3%。

For large models, we also found that stronger augmentation is more helpful. Indeed, our MixUp implementation is part of heavier than the original version in [38]. Inspired by Copypaste [6], we jittered both images by a random sampled scale factor before mixing up them. To understand the power of Mixup with scale jittering, we compare it with Copypaste on YOLOX-L. Noted that Copypaste requires extra instance mask annotations while MixUp does not. But as shown in Tab. 5, these two methods achieve competitive performance, indicating that MixUp with scale jittering is a qualified replacement for Copypaste when no instance mask annotation is available.

对于大型模型，我们还发现增强更有用。实际上，我们的MixUp实现比[38]中的原始版本更重。受Copypaste[6]的启发，我们在混合之前通过随机采样比例因子对两幅图像进行抖动。为了了解混音与缩放抖动的威力，我们将其与YOLOX-L上的复制粘贴进行了比较。注意，Copypaste需要额外的实例掩码注释，而MixUp不需要。但如表5所示，这两种方法取得了竞争性的性能，表明当没有实例掩码注释可用时，具有缩放抖动的MixUp是Copypaste的合格替代品。

## 3. Comparison with the SOTA
There is a tradition to show the SOTA comparing table as in Tab. 6. However, keep in mind that the inference speed of the models in this table is often uncontrolled, as speed varies with software and hardware. We thus use the same hardware and code base for all the YOLO series in Fig. 1, plotting the somewhat controlled speed/accuracy curve.

传统上会显示表6中的SOTA比较表。但是，请记住，此表中模型的推理速度通常不受控制，因为速度随软件和硬件而变化。因此，对于图1中的所有YOLO系列，我们都使用相同的硬件和代码库，绘制出稍微受控的速度/精度曲线。

We notice that there are some high performance YOLO series with larger model sizes like Scale-YOLOv4 [30] and YOLOv5-P6 [7]. And the current Transformer based detectors [21] push the accuracy-SOTA to ∼60 AP. Due to the time and resource limitation, we did not explore those important features in this report. However, they are already in our scope.

我们注意到，有一些高性能的YOLO系列具有较大的模型，如Scale-YOLOv4[30]和YOLOv 5-P6[7]。基于电流互感器的探测器[21]将准确度SOTA推至∼60 AP。由于时间和资源的限制，我们没有在本报告中探讨这些重要特征。然而，它们已经在我们的范围内。

## 4. 1st Place on Streaming Perception Challenge (WAD at CVPR 2021)
Streaming Perception Challenge on WAD 2021 is a joint evaluation of accuracy and latency through a recently proposed metric: streaming accuracy [15]. The key insight behind this metric is to jointly evaluate the output of perception stack at every time instant, forcing the stack to consider the amount of streaming data that should be ignored while computation is occurring [15]. We found that the best trade-off point for the metric on 30 FPS data stream is a powerful model with the inference time ≤ 33ms. So we adopt a YOLOX-L model with TensorRT to product our final model for the challenge to win the 1st place. Please refer to the challenge website5 for more details.

WAD 2021的流感知挑战是通过最近提出的指标：流准确性[15]对准确性和延迟进行联合评估。该指标背后的关键洞察力是联合评估感知堆栈在每个时刻的输出，迫使堆栈考虑在计算过程中应忽略的流数据量[15]。我们发现，30 FPS数据流上度量的最佳权衡点是一个具有推理时间的强大模型≤ 33毫秒。因此，我们采用了YOLOX-L模型和TensorRT来生产我们的最终模型，以应对赢得第一名的挑战。有关更多详情，请参阅挑战网站5。

## 5. Conclusion
In this report, we present some experienced updates to YOLO series, which forms a high-performance anchorfree detector called YOLOX. Equipped with some recent advanced detection techniques, i.e., decoupled head, anchor-free, and advanced label assigning strategy, YOLOX achieves a better trade-off between speed and accuracy than other counterparts across all model sizes. It is remarkable that we boost the architecture of YOLOv3, which is still one of the most widely used detectors in industry due to its broad compatibility, to 47.3% AP on COCO, surpassing the current best practice by 3.0% AP. We hope this report can help developers and researchers get better experience in practical scenes.

在本报告中，我们介绍了对YOLO系列的一些经验丰富的更新，它形成了一个名为YOLOX的高性能无锚检测器。YOLOX配备了一些最新的高级检测技术，即去耦头、无锚和高级标签分配策略，与所有模型的其他同类产品相比，它在速度和准确性之间取得了更好的平衡。值得注意的是，我们将YOLOv3的架构提升到了47.3%，超过了当前最佳实践3.0%的AP。YOLOv 3由于其广泛的兼容性，仍然是工业上使用最广泛的探测器之一。我们希望这份报告能够帮助开发者和研究人员在实际场景中获得更好的体验。

## Acknowledge
This research was supported by National Key R&D Program of China (No. 2017YFA0700800). It was also funded by China Postdoctoral Science Foundation (2021M690375) and Beijing Postdoctoral Research Foundation

本研究得到国家重点研发计划(编号：2017YFA0700800)的资助。该项目也由中国博士后科学基金会(2021M690375)和北京博士后研究基金会资助

## References
1. Alexey Bochkovskiy, Chien-Yao Wang, and HongYuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020. 1, 2, 3, 6
2. NicolasCarion,FranciscoMassa,GabrielSynnaeve,Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In ECCV, 2020. 1, 4
3. Qiang Chen, Yingming Wang, Tong Yang, Xiangyu Zhang, Jian Cheng, and Jian Sun. You only look one-level feature. In CVPR, 2021. 3
4. Zheng Ge, Songtao Liu, Zeming Li, Osamu Yoshie, and Jian Sun. Ota: Optimal transport assignment for object detection. In CVPR, 2021. 1, 4
5. Zheng Ge, Jianfeng Wang, Xin Huang, Songtao Liu, and Osamu Yoshie. Lla: Loss-aware label assignment for dense pedestrian detection. arXiv preprint arXiv:2101.04307, 2021. 4
6. Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, TsungYi Lin, Ekin D Cubuk, Quoc V Le, and Barret Zoph. Simple copy-paste is a strong data augmentation method for instance segmentation. In CVPR, 2021. 5
glenn jocher et al. yolov5. https://github.com/
ultralytics/yolov5, 2021. 1, 2, 3, 5, 6
8. Priya Goyal, Piotr Dolla ́r, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint
arXiv:1706.02677, 2017. 2
9. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 1
10. Zhang Hongyi, Cisse Moustapha, N. Dauphin Yann, and David Lopez-Paz. mixup: Beyond empirical risk minimization. ICLR, 2018. 3
11. Xin Huang, Xinxin Wang, Wenyu Lv, Xiaying Bai, Xiang
Long, Kaipeng Deng, Qingqing Dang, Shumin Han, Qiwen Liu, Xiaoguang Hu, et al. Pp-yolov2: A practical object detector. arXiv preprint arXiv:2104.10419, 2021. 3, 6
12. Kang Kim and Hee Seok Lee. Probabilistic anchor assignment with iou prediction for object detection. In ECCV, 2020. 1, 4
13. Seung-Wook Kim, Hyong-Keun Kook, Jee-Young Sun, Mun-Cheon Kang, and Sung-Jea Ko. Parallel feature pyramid network for object detection. In ECCV, 2018. 2
14. Hei Law and Jia Deng. Cornernet: Detecting objects as paired keypoints. In ECCV, 2018. 1, 3
15. Mengtian Li, Yuxiong Wang, and Deva Ramanan. Towards streaming perception. In ECCV, 2020. 5, 6
16. Tsung-YiLin,PriyaGoyal,RossGirshick,KaimingHe,and Piotr Dolla ́r. Focal loss for dense object detection. In ICCV, 2017. 2
17. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dolla ́r, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 2
18. Songtao Liu, Di Huang, and Yunhong Wang. Learning spatial fusion for single-shot object detection. arXiv preprint arXiv:1911.09516, 2019. 6
19. Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. Path aggregation network for instance segmentation. In CVPR, 2018. 2, 5
20. Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. Path aggregation network for instance segmentation. In CVPR, 2018. 2
21. Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030, 2021. 5
22. Yuchen Ma, Songtao Liu, Zeming Li, and Jian Sun. Iqdet: Instance-wise quality distribution sampling for object detection. In CVPR, 2021. 1, 4
23. Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In CVPR, 2016. 1
24. Joseph Redmon and Ali Farhadi. Yolo9000: Better, faster, stronger. In CVPR, 2017. 1, 3
25. Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767, 2018. 1, 2, 3
26. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In NeurIPS, 2015. 1
27. Guanglu Song, Yu Liu, and Xiaogang Wang. Revisiting the sibling head in object detector. In CVPR, 2020. 2
28. MingxingTan,RuomingPang,andQuocVLe.Efficientdet: Scalable and efficient object detection. In CVPR, 2020. 6
29. Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. Fcos: Fully convolutional one-stage object detection. In ICCV, 2019.1,2,3,4
30. Chien-Yao Wang, Alexey Bochkovskiy, and HongYuan Mark Liao. Scaled-yolov4: Scaling cross stage partial network. arXiv preprint arXiv:2011.08036, 2020. 1, 5, 6
31. Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. Cspnet: A new backbone that can enhance learning capability of cnn. In CVPR workshops, 2020. 2, 5
32. Jianfeng Wang, Lin Song, Zeming Li, Hongbin Sun, Jian Sun, and Nanning Zheng. End-to-end object detection with fully convolutional network. In CVPR, 2020. 1
33. Jianfeng Wang, Lin Song, Zeming Li, Hongbin Sun, Jian Sun, and Nanning Zheng. End-to-end object detection with fully convolutional network. In CVPR, 2021. 4
34. Yue Wu, Yinpeng Chen, Lu Yuan, Zicheng Liu, Lijuan Wang, Hongzhi Li, and Yun Fu. Rethinking classification and localization for object detection. In CVPR, 2020. 2
35. Yue Wu, Yinpeng Chen, Lu Yuan, Zicheng Liu, Lijuan Wang, Hongzhi Li, and Yun Fu. Rethinking classification and localization for object detection. In CVPR, 2020. 2
36. Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In CVPR, 2020. 1, 4
37. Xiaosong Zhang, Fang Wan, Chang Liu, Rongrong Ji, and Qixiang Ye. Freeanchor: Learning to match anchors for visual object detection. In NeurIPS, 2019. 1, 4
38. Zhi Zhang, Tong He, Hang Zhang, Zhongyuan Zhang, Junyuan Xie, and Mu Li. Bag of freebies for training object detection neural networks. arXiv preprint arXiv:1902.04103, 2019. 3, 5
39. Qiang Zhou, Chaohui Yu, Chunhua Shen, Zhibin Wang, and Hao Li. Object detection made simpler by eliminating heuristic nms. arXiv preprint arXiv:2101.11782, 2021. 1, 4
40. Xingyi Zhou, Dequan Wang, and Philipp Kra ̈henbu ̈hl. Objects as points. arXiv preprint arXiv:1904.07850, 2019. 1, 3
41. Benjin Zhu, Jianfeng Wang, Zhengkai Jiang, Fuhang Zong, Songtao Liu, Zeming Li, and Jian Sun. Autoassign: Differentiable label assignment for dense object detection. arXiv preprint arXiv:2007.03496, 2020. 1, 4