# Rethinking Efficient Lane Detection via Curve Modeling
重新思考通过曲线建模的有效车道检测 2022.3.4 https://arxiv.org/abs/2203.02431


## Abstract
This paper presents a novel parametric curve-based method for lane detection in RGB images. Unlike stateof-the-art segmentation-based and point detection-based methods that typically require heuristics to either decode predictions or formulate a large sum of anchors, the curvebased methods can learn holistic lane representations naturally. To handle the optimization difficulties of existing polynomial curve methods, we propose to exploit the parametric B´ezier curve due to its ease of computation, stability, and high freedom degrees of transformations. In addition, we propose the deformable convolution-based feature flip fusion, for exploiting the symmetry properties of lanes in driving scenes. The proposed method achieves a new state-ofthe-art performance on the popular LLAMAS benchmark. It also achieves favorable accuracy on the TuSimple and CULane datasets, while retaining both low latency (>150 FPS) and small model size (<10M). Our method can serve as a new baseline, to shed the light on the parametric curves modeling for lane detection. Codes of our model and PytorchAutoDrive: a unified framework for self-driving perception, are available at: https://github.com/voldemortX/pytorch-auto-drive .

本文提出了一种新的基于参数曲线的RGB图像车道检测方法。与通常需要启发式来解码预测或制定大量锚的基于分段和基于点检测的现有技术方法不同，基于曲线的方法可以自然地学习整体车道表示。为了解决现有多项式曲线方法的优化困难，我们建议利用参数B´ezier曲线，因为它易于计算、稳定性和高自由度的变换。此外，我们提出了基于可变形卷积的特征翻转融合，以利用驾驶场景中车道的对称性属性。所提出的方法在流行的LLAMAS基准上实现了最先进的性能。它在TuSimple和CULane数据集上也获得了良好的准确性，同时保持了低延迟(>150 FPS)和小模型大小(<10M)。我们的方法可以作为一个新的基线，为车道检测的参数曲线建模提供帮助。我们的模型代码和PytorchAutoDrive：自动驾驶感知的统一框架，可在以下网站获得：https://github.com/voldemortX/pytorch-auto-drive .

## 1. Introduction
Lane detection is a fundamental task in autonomous driving systems, which supports the decision-making of lanekeeping, centering and changing, etc. Previous lane detection methods [2, 12] typically rely on expensive sensors such as LIDAR. Advanced by the rapid development of deep learning techniques, many works [19, 21, 22, 33, 41] are proposed to detect lane lines from RGB inputs captured by commercial front-mounted cameras. 

车道检测是自动驾驶系统中的一项基本任务，它支持车道跟踪、居中和改变等决策。以前的车道检测方法[2，12]通常依赖于昂贵的传感器，如激光雷达。随着深度学习技术的快速发展，提出了许多工作[19，21，22，33，41]，以从商用前置摄像头捕获的RGB输入中检测车道线。

Figure 1. Lane detection strategies. Segmentation-based and point detection-based representations are local and indirect. The abstract coefficients (a, b, c, d) used in polynomial curve are hard to optimize. The cubic B´ezier curve is defined by 4 actually existing control points, which roughly fit line shape and wrap the lane line in its convex hull (dashed red lines). Best viewed in color.
图1。车道检测策略。基于分割和基于点检测的表示是局部和间接的。多项式曲线中使用的抽象系数(a，b，c，d)很难优化。三次B´ezier曲线由4个实际存在的控制点定义，这些控制点大致符合线形，并将车道线包裹在其凸状外壳中(红色虚线)。最佳颜色。

Deep lane detection methods can be classified into three categories, i.e., segmentation-based, point detection-based, and curve-based methods (Figure 1). Among them, by relying on classic segmentation [5] and object detection [28] networks, the segmentation-based and point detectionbased methods typically achieve state-of-the-art lane detection performance. The segmentation-based methods [21,22, 41] exploit the foreground texture cues to segment the lane pixels and decode these pixels into line instances via heuristics. The point detection-based methods [15, 33, 39] typically adopt the R-CNN framework [9, 28], and detect lane lines by detecting a dense series of points (e.g., every 10 pixels in the vertical axis). Both kinds of approaches represent lane lines via indirect proxies (i.e., segmentation maps and points). To handle the learning of holistic lane lines, under cases of occlusions or adverse weather/illumination conditions, they have to rely on low-efficiency designs, such as recurrent feature aggregation (too heavy for this realtime task) [22, 41], or a large number of heuristic anchors (> 1000, which may be biased to dataset statistics) [33].

深车道检测方法可分为三类，即基于分割的方法、基于点检测的方法和基于曲线的方法(图1)。其中，通过依赖经典分割[5]和对象检测[28]网络，基于分割和基于点检测的方法通常实现最先进的车道检测性能。基于分割的方法[21，22，41]利用前景纹理线索来分割车道像素，并通过启发式将这些像素解码为线实例。基于点检测的方法[15，33，39]通常采用R-CNN框架[9，28]，并通过检测密集的点系列(例如，垂直轴上的每10个像素)来检测车道线。这两种方法都通过间接智能体(即分割地图和点)表示车道线。为了处理整体车道线的学习，在闭塞或不利天气/照明条件下，它们必须依赖低效率设计，例如重复性特征聚合(对于该实时任务来说太重)[22，41]，或大量启发式锚(>1000，可能偏向于数据集统计)[33]。

On the other hand, there are only a few methods [19, 32] proposed to model the lane lines as holistic curves (typically the polynomial curves, e.g., x = ay3 + by2 + cy + d). While we expect the holistic curve to be a concise and elegant way to model the geometric properties of lane line, the abstract polynomial coefficients are difficult to learn. Previous studies show that their performance lag behind the well-designed segmentation-based and point detectionbased methods by a large margin (up to 8% gap to state-ofthe-art methods on the CULane [22] dataset). In this paper, we aim to answer the question of whether it is possible to build a state-of-the-art curve-based lane detector.

另一方面，只有少数方法[19，32]提出将车道线建模为整体曲线(通常为多项式曲线，例如，x=ay3+by2+cy+d)。虽然我们期望整体曲线是一种简洁而优雅的方式来建模车道线的几何属性，但抽象多项式系数很难学习。先前的研究表明，它们的性能在很大程度上落后于设计良好的基于分割和基于点检测的方法(与CULane[22]数据集上的最先进方法差距高达8%)。在本文中，我们的目的是回答是否有可能建立一个最先进的基于曲线的车道检测器的问题。

We observe that the classic cubic B´ezier curves, with sufficient freedom degrees of parameterizing the deformations of lane lines in driving scenes, remain low computation complexity and high stability. This inspires us to propose to model the thin and long geometric shape properties of lane lines via B´ezier curves. The ease of optimization from on-image B´ezier control points enables the network to be end-to-end learnable with the bipartite matching loss [38], using a sparse set of lane proposals from simple column-wise Pooling (e.g., 50 proposals on the CULane dataset [22]), without any post-processing steps such as the Non-Maximum Suppression (NMS), or hand-crafted heuristics such as anchors, hence leads to high speed and small model size. In addition, we observe that lane lines appear symmetrically from a front-mounted camera (e.g., between ego lane lines, or immediate left and right lanes). To model this global structure of driving scenes, we further propose the feature flip fusion, to aggregate the feature map with its horizontally flipped version, to strengthen such coexistences. We base our design of feature flip fusion on the deformable convolution [42], for aligning the imperfect symmetries caused by, e.g., rotated camera, changing lane, non-paired lines. We conduct extensive experiments to analyze the properties of our method and show that it performs favorably against state-of-the-art lane detectors on three popular benchmark datasets. 

我们观察到，经典的三次B´ezier曲线具有足够的自由度来参数化驾驶场景中车道线的变形，保持了低计算复杂性和高稳定性。这激励我们提出通过B´ezier曲线来模拟车道线的细长几何形状属性。基于图像B´ezier控制点的易于优化使得网络能够在二分匹配损失[38]的情况下进行端到端学习，使用来自简单逐列池的稀疏车道建议集(例如，CULane数据集上的50个建议[22])，而无需任何后处理步骤，如非最大抑制(NMS)，因此导致高速和小的模型尺寸。此外，我们观察到，从前置摄像头上看，车道线是对称的(例如，在自我车道线之间，或紧邻左右车道之间)。为了对驾驶场景的这种全局结构进行建模，我们进一步提出了特征翻转融合，将特征地图与其水平翻转版本进行聚合，以加强这种共存。我们将特征翻转融合的设计基于可变形卷积[42]，用于对齐由旋转相机、改变车道、非成对线等引起的不完美对称性。我们进行了大量的实验来分析我们的方法的属性，并表明在三个流行的基准数据集上，该方法与最先进的车道检测器相比表现良好。

Our main contributions are summarized as follows:
* We propose a novel B´ezier curve-based deep lane detector, which can model the geometric shapes of lane lines effectively, and be naturally robust to adverse driving conditions.
* We propose a novel deformable convolution-based feature flip fusion module, to exploit the symmetry property of lanes observed from front-view cameras.
* We show that our method is fast, light-weight, and accurate through extensive experiments on three popular lane detection datasets. Specifically, our method outperforms all existing methods on the LLAMAS benchmark [3], with the light-weight ResNet-34 backbone.

我们的主要贡献总结如下：
* 提出了一种新的基于B´ezier曲线的深车道检测器，它可以有效地对车道线的几何形状进行建模，并对不利的驾驶条件具有自然的稳健性。
* 提出了一种新的基于可变形卷积的特征翻转融合模块，以利用从前视图相机观察到的车道的对称性。
* 在三个流行的车道检测数据集上进行了大量实验，结果表明我们的方法快速、轻便、准确。具体来说，我们的方法优于LLAMAS基准测试[3]上的所有现有方法，具有轻量级的ResNet-34主干。

## 2. Related Work
### Segmentation-based Lane Detection. 
These methods represent lanes as per-pixel segmentation. SCNN [22] formulates lane detection as multi-class semantic segmentation and is the basis of the 1st-place solution in TuSimple challenge [1]. It’s core spatial CNN module recurrently aggregates spatial information to complete the discontinuous segmentation predictions, which then requires heuristic postprocessing to decode the segmentation map. Hence, it has a high latency, and only struggles to be real-time after an optimization of Zheng et al. [41]. Others explore knowledge distillation [13] or generative modeling [8], but their performance merely surpasses the seminal SCNN. Moreover, these methods typically assume a fixed number (e.g., 4) of lines. LaneNet [21] leverages an instance segmentation pipeline to deal with a variable number of lines, but it requires post-inference clustering to generate line instances. Some methods leverage row-wise classification [26, 40], which is a customized down-sampling of per-pixel segmentation so that they still require post-processing. Qin et al. [26] propose to trade performance for low latency, but their use of fully-connected layers results in large model size.

基于分割的车道检测。这些方法表示按像素分割的车道。SCNN[22]将车道检测表述为多类语义分割，是TuSimple挑战[1]中排名第一的解决方案的基础。它的核心空间CNN模块循环地聚集空间信息以完成不连续的分割预测，然后需要启发式后处理来解码分割图。因此，它具有很高的延迟，只有在Zhenget al 的优化后才努力实现实时性[41]。其他人则探索知识提炼[13]或生成建模[8]，但他们的表现仅仅超越了开创性的SCNN。此外，这些方法通常假定固定数量(例如，4)的线。LaneNet[21]利用实例分割管道来处理可变数量的行，但它需要后推断聚类来生成行实例。一些方法利用行分类[26，40]，这是对每个像素分割的定制下采样，因此它们仍然需要后处理。Qinet al [26]提出以低延迟换取性能，但他们使用完全连接的层会导致大模型大小。

In short, segmentation-based methods all require heavy post-processing due to the misalignment of representations. They also suffer from the locality of segmentation task, so that they tend to perform worse under occlusions or extreme lighting conditions.

简而言之，基于分割的方法都需要大量的后处理，这是由于表示的不对准。它们还受到分割任务的局部性的影响，因此在遮挡或极端光照条件下，它们往往表现得更差。

### Point Detection-based Lane Detection. 
The success of object detection methods drives researchers to formulate lane detection as to detect lanes as a series of points (e.g., every 10 pixels in the vertical axis). Line-CNN [15] adapts classic Faster R-CNN [28] as a one-stage lane line detector, but it has a low inference speed (<30 FPS). Later, LaneATT [33] adopts a more general one-stage detection approach that achieves superior performance.

基于点检测的车道检测。目标检测方法的成功促使研究人员将车道检测公式化为一系列点(例如，垂直轴上每10个像素)来检测车道。Line CNN[15]采用经典的Faster R-CNN[28]作为单级车道线检测器，但其推理速度较低(<30 FPS)。后来，LaneATT[33]采用了一种更通用的单阶段检测方法，实现了卓越的性能。

However, these methods have to design heuristic lane anchors, which highly depend on dataset statistics, and require the Non-Maximum Suppression (NMS) as post-processing. On the contrary, we represent lane lines as curves with a fully end-to-end pipeline (anchor-free, NMS-free).

然而，这些方法必须设计启发式车道锚，这高度依赖于数据集统计，并且需要非最大抑制(NMS)作为后处理。相反，我们将车道线表示为具有完全端到端管道(无锚、无NMS)的曲线。

### Curve-based Lane Detection. 
The pioneering work [37] proposes a differentiable least squares fitting module to fit a polynomial curve (e.g., x = ay3 + by2 + cy + d) to points predicted by a deep neural network. The PolyLaneNet [32] then directly learns to predict the polynomial coefficients with simple fully-connected layers. Recently, LSTR [19] uses transformer blocks to predict polynomials in an endto-end fasion based on the DETR [4].

基于曲线的车道检测。开创性工作[37]提出了一种可微最小二乘拟合模块，以将多项式曲线(例如，x=ay3+by2+cy+d)拟合到深度神经网络预测的点。PolyLaneNet[32]然后直接学习用简单的完全连接层预测多项式系数。最近，LSTR[19]基于DETR[4]使用变换块来预测端到端的多项式。

Curve is a holistic representation of lane line, which naturally eliminates occlusions, requires no post-processing, and can predict a variable number of lines. However, their performance on large and challenging datasets (e.g., CULane [22] and LLAMAS [3]) still lag behind methods of other categories. They also suffer from slow convergence (over 2000 training epochs on TuSimple), high latency architecture (e.g., LSTR [19] uses transformer blocks which are difficult to optimize for low latency). We attribute their failure to the difficult-to-optimize and abstract polynomial coefficients. We propose to use the parametric B´ezier curve, which is defined by actual control points on the image coordinate system1 , to address these problems.

曲线是车道线的整体表示，它自然消除了闭塞，不需要后期处理，并且可以预测可变数量的线。然而，它们在大型和具有挑战性的数据集(如CULane[22]和LLAMAS[3])上的性能仍然落后于其他类别的方法。它们还受到缓慢收敛(TuSimple上超过2000个训练周期)、高延迟架构(例如，LSTR[19]使用难以优化低延迟的转换器块)的影响。我们将其失败归因于难以优化和抽象的多项式系数。我们建议使用由图像坐标系1上的实际控制点定义的参数B´ezier曲线来解决这些问题。

Table 1. Comparison of n-order B´ezier curves and polynomials (x = P ni=0 aiyi ) on TuSimple [1] test set (lower is better). Since the official metrics are too lose to show any meaningful difference, we use the fine-grained LPD metric following [32]. 
表1。TuSimple[1]测试集上n阶B´ezier曲线和多项式(x=P ni=0 aiyi)的比较(越低越好)。由于官方度量太过丢失，无法显示任何有意义的差异，我们使用细粒度LPD度量[32]。

### B´ezier curve in Deep Learning. 
To our knowledge, the only known successful application of B´ezier curves in deep learning is the ABCNet [20], which uses cubic B´ezier curve for text spotting. However, their method cannot be directly used for our tasks. First, this method still uses NMS so that it cannot be end-to-end. We show in our work that NMS is not necessary so that our method can be an endto-end solution. Second, it calculates L1 loss directly on the sparse B´ezier control points, which results in difficulties of optimization. We address this problem in our work by leveraging a fine-grained sampling loss. In addition, we propose the feature flip fusion module, which is specifically designed for the lane detection task.

深度学习中的B´ezier曲线。据我们所知，B´ezier曲线在深度学习中唯一已知的成功应用是ABCNet[20]，它使用三次B´izier曲线进行文本定位。然而，他们的方法不能直接用于我们的任务。首先，该方法仍然使用NMS，因此它不能是端到端的。我们在工作中表明，NMS是不必要的，因此我们的方法可以成为端到端解决方案。其次，它直接在稀疏的B´ezier控制点上计算L1损耗，这导致了优化的困难。我们在工作中通过利用细粒度采样损失来解决这个问题。此外，我们还提出了专门为车道检测任务设计的特征翻转融合模块。

## 3. B´ezierLaneNet
### 3.1. Overview
Preliminaries on B´ezier Curve. The B´ezier curve’s formulation is shown in Equation (1), which is a parametric curve defined by n + 1 control points:

B(t) = nXi=0 bi,n(t)Pi, 0 ≤ t ≤ 1, (1) where Pi is the i−th control point, bi,n are Bernstein basis polynomials of degree n: bi,n = Cinti (1 − t)n−i , i = 0, ..., n. (2)

We use the classic cubic B´ezier curve (n = 3), which is empirically found sufficient for modeling lane lines. It shows better ground truth fitting ability than 3rd order polynomial (Table 1), which is the base function for previous curve-based methods [19, 32]. Higher-order curves do not bring substantial gains while the high degrees of freedom leads to instability. All coordinates for points discussed here are relative to the image size (i.e., mostly in range [0, 1]). 1Actually control points of B´ezier curves can be outside the image, but statistically that rarely happens in autonomous driving scenes. +

Figure 2. Pipeline. Feature from a typical encoder (e.g., ResNet) is strengthened by feature flip fusion, then pooled to 1D and two 1D convolution layers are applied. At last the network predicts B´ezier curves through a classification branch and a regression branch.

The Proposed Architecture. The overall model architecture is shown in Figure 2. Specifically, we use layer-3 feature of ResNets [11] as backbone following RESA [41], but we replace the dilation inside the backbone network by two dilated blocks outside with dilation rates [4, 8] [6]. This strikes a better speed-accuracy trade-off for our method, which leaves a 16× down-sampled feature map with a larger receptive field. We then add the feature flip fusion module (Section 3.2) to aggregate opposite lane features.

The enriched feature map (C × H 16 × W 16 ) is then pooled to (C × W 16 ) by average pooling, resulting in W 16 proposals (50 for CULane [22]). Two 1 × 3 1D convolutions are used to transform the pooled features, while also conveniently modeling interactions between nearby lane proposals, guiding the network to learn a substitute for the non-maximal suppression (NMS) function. Lastly, the final prediction is obtained by the classification and regression branches (each is only one 1 × 1 1D convolution). The outputs are W 16 × 8 for regression of 4 control points, and W 16 × 1 for existence of lane line object.

### 3.2. Feature Flip Fusion
By modeling lane lines as holistic curves, we focus on the geometric properties of individual lane lines (e.g., thin, long, and continuous). Now we consider the global structure of lanes from a front-mounted camera view in driving scenes. Roads have equally spaced lane lines, which appear symmetrical and this property is worth modeling. For instance, the existence of left ego lane line should very likely indicate its right counterpart, the structure of immediate left lane could help describe the immediate right lane, etc.

To exploit this property, we fuse the feature map with its horizontally flipped version (Figure 3). Specifically, two separate convolution and normalization layers transform each feature map, they are then added together before a ReLU activation. With this module, we expect the model to base its predictions on both feature maps.

To account for the slight misalignment of camera captured image (e.g., rotated, turning, non-paired), we apply 3 +

Figure 3. Feature flip fusion. Alignment is achieved by calculating deformable convolution offsets, conditioned on both the flipped and original feature map. Best viewed in color. deformable convolution [42] with kernel size 3 × 3 for the flipped feature map while learning the offsets conditioned on the original feature map for feature alignment.

We add an auxiliary binary segmentation branch (to segment lane line or non-lane line areas, which would be removed after training) to the ResNet backbone, and we expect it to enforce the learning of spatial details. Interestingly, we find this auxiliary branch improves the performance only when it works with the feature fusion. This is because the localization of the segmentation task may provide a more spatially-accurate feature map, which in turn supports accurate fusion between the flipped features.

Visualizations are shown in Figure 4, from which we can see that the flipped feature does correct the error caused by the asymmetry introduced by the car (Figure 4(a)).

### 3.3. End-to-end Fit of a B´ezier Curve
Distances Between B´ezier Curves. The key to learning

B´ezier curves is to define a good distance metric measuring the distances between the ground truth curve and prediction. Naively, one can directly calculate the mean L1 distance between B´ezier curve control points, as in ABCNet [20]. However, as shown in Figure 5(a), a large L1 error in curvature control points can demonstrate a very small visual distance between B´ezier curves, especially on small or medium curvatures (which is often the case for lane lines).

Since B´ezier curves are parameterized by t ∈ [0, 1], we propose the more reasonable sampling loss for B´ezier curves (Figure 5(b)), by sampling curves at a uniformly spaced set of t values (T), which means equal curve length between adjacent sample points. The t values can be further transformed by a re-parameterization function f(t). Specifically, given B´ezier curves B(t), Bˆ(t), the sampling loss Lreg is:

Lreg = 1n X t∈T ||B(f(t)) − Bˆ(f(t))||1, (3) where n is the total number of sampled points and is set to 100. We empirically find f(t) = t works well. This simple yet effective loss formulation makes our model easy to converge and less sensitive to hyper-parameters that typically involved in other curved-based or point detection- (a) (b)<br/>
Figure 4. Grad-CAM [31] visualization on the last layer of ResNet backbone. (a) Our model can infer existence of an ill-marked lane line, from clear markings and cars around the opposite line. Note that the car is deviated to the left, this scene was not captured with perfect symmetry. (b) When entire road lacks clear marking, both sides are used for a better prediction. Best viewed in color. based methods, e.g., loss weighting for endpoints loss [19] and line length loss [33] (see Figure 5(b,c)).

B´ezier Ground Truth Generation. Now we introduce the generation of B´ezier curve ground truth. Since lane datasets are currently annotated by on-line key points, we need the

B´ezier control points for the above sampling loss. Given the annotated points {(kxi , kyi )}mi=1 on one lane line, where (kxi , kyi ) denotes the 2D-coordinates of the i-th point. Our goal is to obtain control points {Pi(xi , yi)}ni=1. Similarly to [20], we use standard least squares fitting: P0 P1... Pn = kx0 ky0 kx1 ky1 ... ... kxm kym b0,n(t0) · · · bn,n(t0) b0,n(t1) · · · bn,n(t1) ... ... ... b0,n(tm) · · · bn,n(tm)T (4) {ti}mi=0 is uniformly sampled from 0 to 1. Different from [20], we do not restrict ground truth to have same endpoints as original annotations, which leads to better quality labels.

Label and Prediction Matching. After obtaining the ground truth, in training, we perform a one-to-one assignment between G labels and N predictions (G < N) using optimal bipartite matching, to attain a fully end-toend pipeline. Following Wang et al. [38], we find a Gpermutation of N predictions π ∈ ΠNG that formulates the best bipartite matching: πˆ = arg max π∈ΠNG GXi Qi,π(i), (5)

Qi,π(i) =  pˆπ(i) 1−α ·  1 − L1 bi, ˆbπ(i)  α, (6) 4 (a) (b) (c)<br/>
Figure 5. Lane loss functions. (a) The L1 distance of control points is not highly correlated with the actual distance between curves. (b) The proposed sampling loss is one unified distance metric by t-sampling. (c) Typical loss for polynomial regression [19], at least 3 separate losses are required: y-sampling loss, y start point loss, y end point loss. where Qi,π(i) ∈ [0, 1] represents matching quality of the ith label with the π(i)-th prediction, based on L1 distance between curves bi, ˆbπ(i) (sampling loss) and class score pˆπ(i). α is set to 0.8 by default. The above equations can be efficiently solved by the well-known Hungarian algorithm.

Wang et al. [38] also use a spatial prior that restricts the matched prediction to a spatial neighborhood of the label (object center distance, the centerness prior in FCOS [35]).

However, since lots of lanes are long lines with a large slope, this centerness prior is not useful. See Appendix E for more investigations on matching priors.

Overall Loss. Other than B´ezier curve sampling loss, there is also the classification loss Lc for the lane object classi- fication (existence) branch. Since the imbalance between positive and negative examples is not as severe in lane detection as in object detection, instead of the focal loss [16], we use the simple weighted binary cross-entropy loss:

Lcls = −(y log(p) + w(1 − y) log(1 − p)), (7) where w is the weighting for negative samples, which is set to 0.4 in all experiments. The loss Lseg for the binary segmentation branch (Section 3.2) takes the same format.

The overall loss is a weighted sum of all three losses:

L = λ1Lreg + λ2Lcls + λ3Lseg, (8) where λ1, λ2, λ3 are set to 1, 0.1, 0.75, respectively.

## 4. Experiments
### 4.1. Datasets
To evaluate the proposed method, we conduct experiments on three well-known datasets: TuSimple [1], CULane [22] and LLAMAS [3]. TuSimple dataset was collected on highways with high-quality images, under fair



Table 2. Details of datasets. *Number of lines in LLAMAS dataset is more than 4, but official metric only evaluates 4 lines. weather conditions. CULane dataset contains more complex urban driving scenarios, including shades, extreme illuminations, and road congestion. LLAMAS is a newly formed large-scale dataset, it is the only lane detection benchmark without public test set labels. Details of these datasets can be found in Table 2.

### 4.2. Evalutaion Metics
For CULane [22] and LLAMAS [3], the official metric is F1 score from [22]:

F1 = 2 · Precision · Recall

Precision + Recall , (9) where Precision = T P

T P +F P and Recall = T P

T P +F N . Lines are assumed to be 30 pixels wide, prediction and ground truth lines with pixel IoU over 0.5 are considered a match.

For TuSimple [1] dataset, the official metrics include

Accuracy, false positive rate (FPR), and false negative rate (FNR). Accuracy is computed as Npred

Ngt , where Npred is the number of correctly predicted on-line points and Ngt is the number of ground truth on-line points.

### 4.3. Implementation Details
Fair Comparison. To fairly compare among different stateof-the-art methods, we re-implement representative methods [19, 22, 41] in a unified PyTorch framework. We Also provide a semantic segmentation baseline [5] originally proposed in [22]. All our implementations do not use val set in training, and tune hyper-parameters only on val set. Some methods with reliable open-source codes are reported from their own codes [26, 32, 33]. For platform sensitive metric Frames-Per-Second (FPS), we re-evaluate all reported methods on the same RTX 2080 Ti platform. More details for implementations and FPS tests are in Appendices A to C.

Training. We train 400, 36, 20 epochs for TuSimple, CULane, and LLAMAS, respectively (training of our model takes only 12 GPU hours on a single RTX 2080 Ti), and the input resolution is 288×800 for CULane [22] and 360×640 for others, following common practice. Other than these, all hyper-parameters are tuned on CULane [22] val set and remain the same for our method across datasets. We use

Adam optimizer with learning rate 6 × 10−4 , weight decay 1 × 10−4 , batch size 20, Cosine Annealing learning rate schedule as in [33]. Data augmentation includes random affine transforms, random horizontal flip, and color jitter. 5

CULane [22] TuSimple [1]


Table 3. Results on test set of CULane [22] and TuSimple [1]. *reproduced results in our code framework, best performance from three random runs. **reported from reliable open-source codes from the authors.

Testing. No post-processing is required for curve methods.

Standard Gaussian blur and row selection post-processing is applied to segmentation methods. NMS is used for

LaneATT [33], while we remove its post-inference B-Spline interpolation in CULane [22], to align with our framework.

### 4.4. Comparisons
Overview. Experimental results are shown in Tables 3 and 4. TuSimple [1] is a small dataset that features clearweather highway scenes and has a relatively easy metric, most methods thrive in this dataset. Thus, we mainly focus on the other two large-scale datasets [3, 22], where there is still a rather clear difference between methods. For highperformance methods (> 70% F1 on CULane [22]), we also show efficiency metrics (FPS, Parameter count) in Table 5.

Comparison with Curve-based Methods. As shown in

Tables 3 and 4, in all datasets, B´ezierLaneNet outperforms previous curve-based methods [19, 32] by a clear margin, advances the state-of-the-art of curve-based methods by 6.85% on CULane [22] and 6.77% on LLAMAS [3]. Thanks to our fully convolutional and fully end-toend pipeline, B´ezierLaneNet runs over 2× faster than LSTR [19]. LSTR has a speed bottleneck from transformer architecture, the 1× and 2× model have FPS 98 and 97, respectively2 . While curves are difficult to learn, our method converges 4-5× faster than LSTR. For the first time, an elegant curve-based method can challenge well-designed segmentation methods or point detection methods on these datasets 2The original 420 FPS report from LSTR paper [19], is throughput with batch size 16, detailed discussions in Appendix A. while showing a favorable trade-off, with an acceptable convergence time.

Comparison with Segmentation-based Methods. These methods tend to have a low speed due to recurrent feature aggregation [22, 41], and the use of high-resolution feature map [5, 22, 41]. B´ezierLaneNet outperforms them in both speed and accuracy. Our small models even compare favorably against RESA [41] and SCNN [22] with large ResNet- 101 backbone, surpassing them in CULane [22] with a clear margin (1 ∼ 2%). On LLAMAS [3], where the dataset restricts testing on 4 center lines, the segmentation approach shows strong performance (Table 4). Nevertheless, our

ResNet-34 model still outperforms SCNN by 0.92%.

UFLD [26] reformulates segmentation to row-wise classification on a down-sampled feature map to achieve fast speed, at the cost of accuracy. Compared to us, UFLD (ResNet-34) is 0.9% lower on CULane Normal, while 7.4%, 3.0%, 3.2% worse on Shadow, Crowd, Night, respectively. Overall, our method with the same backbones
 outperforms UFLD by 3 ∼ 5%, while being faster on

ResNet-34. Besides, UFLD uses large fully-connected layers to optimize latency, which causes a huge model size (the largest in Table 5).

A drawback for all segmentation methods is the weaker performance on Dazzle Light. Per-pixel (or per-pixel grid for UFLD [26]) segmentation methods may rely on information from local textures, which is destroyed by extreme exposure to light. While our method predicts lane lines as holistic curves, hence robust to changes in local textures.

Comparison with Point Detection-based Methods. Xu et al. [39] finds a series of point detection-based models with 6

LLAMAS [3]



Table 4. Results from LLAMAS [3] test server. neural architecture search techniques called CurveLanesNAS. Despite its complex pipeline and extensive architecture search for the best accuracy-FLOPs trade-off, our simple ResNet-34 backbone model (29.9 GFLOPs) still surpasses its large model (86.5 GFLOPs) by 0.8% on CULane.

CurveLanes-NAS also performs worse under occlusions, a similar drawback as the segmentation methods without recurrent feature fusion [5, 26]. As shown in Table 3, with similar model capacity compared to our ResNet-34 model, CurveLanes-NAS-M (35.7 GFLOPs) is 1.4% worse on Normal scenes, but the gap on Shadow and Crowd are 7.4% and 2.7%.

Recently, LaneATT [33] achieves higher performance with a point detection network. However, their design is not fully end-to-end (requires Non-Maximal Suppression (NMS)), based on heuristic anchors (>1000), which are calculated directly from the dataset’s statistics, thus may systematically pose difficulties in generalization. Still, with

ResNet-34, our method outperforms LaneATT on the LLAMAS [3] test server (1.43%), with a significantly higher recall (3.58%). We also achieve comparable performance to LaneATT on TuSimple [1] using only the train set, and only ∼ 1% worse on CULane. Our method performs significantly better in Dazzle Light (3.3% better), comparably in

Night (0.4% lower). It also has a lower False Positive (FP) rate on Crossroad scenes (Cross), even though LaneATT shows an extremely low-FP characteristic (large PrecisionRecall gap in Table 4). Methods that rely on heuristic anchors [33] or heuristic decoding process [22,26,39,41] tend to have more false predictions in this scene. Moreover, the

NMS is a sequential process that could have unstable runtime in real-world applications. Even when NMS was not evaluated on real inputs, our models are 29%, 28% faster, have 2.9×, 2.3× fewer parameters, compared to LaneATT on ResNet-18 and ResNet-34 backbones, respectively.

To summarize, previous curve-based methods (PolyLaneNet [32], LSTR [19]) have significantly worse performance. Fast methods trades either accuracy (UFLD [26]) or model size (UFLD [26], LaneATT [33]) for speed. 



Table 5. FPS (image/s) and model size. All FPS results are tested with 360 × 640 random inputs on the same platform. Here only shows models with > 70% CULane [22] F1 score. methods either discards the end-to-end pipeline (LaneATT [33]), or entirely fails the real-time requirement (SCNN [22], RESA [41]). While our B´ezierLaneNet is fully endto-end, fast (>150 FPS), light-weight (<10 million parameters) and maintains consistent high accuracy across datasets.

### 4.5. Analysis
Although we develop our method by tuning on the val set, we re-run ablation studies with ResNet-34 backbone (including our full method) and report performance on the

CULane test set for clear comparison.

Curve representation F1

Cubic B´ezier curve baseline 68.89 3rd Polynomial baseline 1.49

B´ezierLaneNet 75.41 3rd Polynomial from B´ezierLaneNet 5.01

Table 6. Curve representations. Baselines directly predict curve coefficients without feature flip fusion.

Importance of Parametric B´ezier Curve. We first replace the B´ezier curve prediction with a 3rd order polynomial, adding auxiliary losses for start and end points. As shown in Table 6, polynomials catastrophically fail to converge in our fully convolutional network, even when trained with 150 epochs (details in Appendix B.8). Then we consider modifying the LSTR [19] to predict cubic B´ezier curves, the performance is similar to predicting polynomials. We conclude that heavy MLP may be necessary to learn polynomials [19, 32], while predicting B´ezier control points from position-aware CNN is the best choice. The transformerbased LSTR decoder destroys the fine spatial information, suppresses the advancement of curve function.

Feature Flip Fusion Design. As shown in Table 7, feature flip fusion brings 4.07% improvement. We also find that the auxiliary segmentation loss can regularize and increase 7


Table 7. Ablations. CP: Control point loss [20]. SP: The proposed sampling loss. Flip: The feature flip fusion module. Deform:

Employ the deformable convolution in feature flip fusion. Seg:

Auxiliary segmentation loss. the performance further, by 2.45%. It is worth noting that auxiliary loss only works with feature fusion, it can lead to degenerated results when directly applied on the baseline (−3.07%). A standard 3 × 3 convolution performs worse than deformable convolution, by 2.68% and 1.44%, before and after adding the auxiliary segmentation loss, respectively. We attribute this to the effects of feature alignment.

B´ezier Curve Fitting Loss. As shown in Table 7, replacing the sampling loss by direct loss on control points lead to inferior performance (−5.15% in the baseline setup). Inspired by the success of IoU loss in object detection. We also implemented a IoU loss (formulas in Appendix D) for the convex hull of B´ezier control points. However, the convex hull of close-to-straight lane lines are too small, the IoU loss is numerically unstable, thus failing to facilitate the sampling loss.


Table 8. Augmentation ablations. Aug: Strong data augmentation.

Importance of Strong Data Augmentation. Strong data augmentation is defined by a series of affine transforms and color distortions, the exact policy may slightly vary for different methods. For instance, we use random affine transform, random horizontal flip, and color jitter. LSTR [19] also uses random lighting. Default augmentation includes only a small rotation (3 degrees). As shown in Table 8, strong augmentation is essential to avoid over-fitting for curve-based methods.

For segmentation-based methods [5, 22, 41], we fast validated strong augmentation on the smaller TuSimple [1] dataset. All shows a 1 ∼ 2% degradation. This suggests that they may be robust due to per-pixel prediction and heuristic post-processing. But they highly rely on learning the distribution of local features such as texture, which could become confusing by strong augmentation.

### 4.6. Limitations and Discussions
Curves are indeed a natural representation of lane lines.

However, their elegance in modeling inevitably brings a drawback. It is difficult for the curvature coefficients to generalize when the data distribution is highly biased (almost all lane lines are straight lines in CULane). Our B´ezier curve approach has already alleviated this problem to some extent and has achieved an acceptable performance (62.45) in CULane Curve. On datasets such as TuSimple and LLAMAS [1, 3], where the curvature distribution is fair enough for learning, our method achieves even better performance.

To handle broader corner cases, e.g., sharp turns, blockages and bad weather, datasets such as [30,34,39] may be useful.

The feature flip fusion is specifically designed for a front-mounted camera, which is the typical use case of deep lane detectors. Nevertheless, there is still a strong inductive bias by assuming scene symmetry. In future work, it would be interesting to find a replacement for this module, to achieve better generalization and to remove the deformable convolution operation, which poses difficulty for effective integration into edge devices such as Jetson.

More discussions in Appendix G.

## 5. Conclusions
In this paper, we have proposed B´ezierLaneNet: a novel fully end-to-end lane detector based on parametric B´ezier curves. The on-image B´ezier curves are easy to optimize and naturally model the continuous property of lane lines, without heavy designs such as recurrent feature aggregation or heuristic anchors. Besides, a feature flip fusion module is proposed. It efficiently models the symmetry property of the driving scene, while also being robust to slight asymmetries by using deformable convolution. The proposed model has achieved favorable performance on three datasets, defeating all existing methods on the popular LLAMAS benchmark. It is also both fast (>150 FPS) and light-weight (<10 million parameters).

## Acknowledgements. 
This work has been sponsored by National Key Research and Development Program of China (2019YFC1521104), National Natural Science

Foundation of China (61972157, 72192821), Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102), Shanghai Science and Technology

Commission (21511101200), Art major project of National

Social Science Fund (I8ZD22), and SenseTime Collaborative Research Grant. We thank Jiaping Qin for guidance on road design and geometry, Yuchen Gong and Pan Chen for helping with CAM visualizations, Zhijun Gong, Jiachen

Xu and Jingyu Gong for insightful discussions about math,

Fengqi Liu for providing GPUs, Lucas Tabelini for cooperation in evaluating [32, 33], and the CVPR reviewers for constructive comments. 8

## References
1. TuSimple benchmark. https : / / github . com / TuSimple/tusimple-benchmark, 2017. 2, 3, 5, 6, 7, 8, 12, 14, 15
2. Claudine Badue, Rˆanik Guidolini, Raphael Vivacqua Carneiro, Pedro Azevedo, Vinicius B Cardoso, Avelino Forechi, Luan Jesus, Rodrigo Berriel, Thiago M Paixao, Filipe Mutz, et al. Self-driving cars: A survey. Expert Systems with Applications, 2021. 1
3. Karsten Behrendt and Ryan Soussan. Unsupervised labeled lane markers using maps. In ICCV, 2019. 2, 5, 6, 7, 8, 14, 15
4. Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In ECCV, 2020. 2
5. Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille. Semantic image segmentation with deep convolutional nets and fully connected crfs. In ICLR, 2015. 1, 5, 6, 7, 8, 11
6. Qiang Chen, Yingming Wang, Tong Yang, Xiangyu Zhang, Jian Cheng, and Jian Sun. You only look one-level feature. In CVPR, 2021. 3
7. MOT Highway Department and Highway Engineering Committee under China Association for Engineering Construction Standardization. Technical Standard of Highway Engineering. 2004. 14
8. Mohsen Ghafoorian, Cedric Nugteren, N´ora Baka, Olaf Booij, and Michael Hofmann. El-gan: Embedding loss driven generative adversarial networks for lane detection. In ECCV Workshops, 2018. 2
9. Ross Girshick. Fast r-cnn. In ICCV, 2015. 1
10. Alexandru Gurghian, Tejaswi Koduri, Smita V Bailur, Kyle J Carey, and Vidya N Murali. Deeplanes: End-to-end lane position estimation using deep neural networksa. In CVPR Workshops, 2016. 14
11. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 3
12. Aharon Bar Hillel, Ronen Lerner, Dan Levi, and Guy Raz. Recent progress in road and lane detection: a survey. MVA, 2014. 1
13. Yuenan Hou, Zheng Ma, Chunxiao Liu, and Chen Change Loy. Learning lightweight lane detection cnns by self attention distillation. In ICCV, 2019. 2
14. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012. 12
15. Xiang Li, Jun Li, Xiaolin Hu, and Jian Yang. Line-cnn: End-to-end traffic line detection with line proposal unit. ITS, 2019. 1, 2
16. Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Doll´ar. Focal loss for dense object detection. In ICCV, 2017. 5
17. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV. Springer, 2014. 12
18. Lizhe Liu, Xiaohao Chen, Siyu Zhu, and Ping Tan. Condlanenet: a top-to-down lane detection framework based on conditional convolution. In ICCV, 2021. 14
19. Ruijin Liu, Zejian Yuan, Tie Liu, and Zhiliang Xiong. Endto-end lane shape prediction with transformers. In WACV, 2021. 1, 2, 3, 4, 5, 6, 7, 8, 11, 12
20. Yuliang Liu, Hao Chen, Chunhua Shen, Tong He, Lianwen Jin, and Liangwei Wang. Abcnet: Real-time scene text spotting with adaptive bezier-curve network. In CVPR, 2020. 3, 4, 8
21. Davy Neven, Bert De Brabandere, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool. Towards end-to-end lane detection: an instance segmentation approach. In IV, 2018. 1, 2
22. Xingang Pan, Jianping Shi, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Spatial as deep: Spatial cnn for traffic scene understanding. In AAAI, 2018. 1, 2, 3, 5, 6, 7, 8, 11, 14, 15
23. Tim A Pastva. Bezier curve fitting. Technical report, NAVAL POSTGRADUATE SCHOOL MONTEREY CA, 1998. 13
24. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019. 11, 14
25. Jonah Philion. Fastdraw: Addressing the long tail of lane detection by adapting a sequential prediction network. In CVPR, 2019. 6
26. Zequn Qin, Huanyu Wang, and Xi Li. Ultra fast structureaware deep lane detection. In ECCV, 2020. 2, 5, 6, 7, 11
27. Zhan Qu, Huan Jin, Yang Zhou, Zhen Yang, and Wei Zhang. Focus on local: Detecting lane marker from bottom up via key point. In CVPR, 2021. 14
28. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. NeurIPS, 2015. 1, 2
29. Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In CVPR, 2019. 14
30. Christos Sakaridis, Dengxin Dai, and Luc Van Gool. Semantic foggy scene understanding with synthetic data. IJCV, 2018. 8
31. Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In CVPR, 2017. 4
32. Lucas Tabelini, Rodrigo Berriel, Thiago M Paixao, Claudine Badue, Alberto F De Souza, and Thiago Oliveira-Santos. Polylanenet: Lane estimation via deep polynomial regression. In ICPR, 2020. 1, 2, 3, 5, 6, 7, 8, 12
33. Lucas Tabelini, Rodrigo Berriel, Thiago M Paixao, Claudine Badue, Alberto F De Souza, and Thiago Oliveira-Santos. Keep your eyes on the lane: Real-time attention-guided lane detection. In CVPR, 2021. 1, 2, 4, 5, 6, 7, 8, 11, 12
34. Xin Tan, Ke Xu, Ying Cao, Yiheng Zhang, Lizhuang Ma, and Rynson W. H. Lau. Night-time scene parsing with a large real dataset. TIP, 2021. 8 9
35. Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. Fcos: Fully convolutional one-stage object detection. In ICCV, 2019. 5
36. Federal Highway Administration under United States Department of Transportation. Standard Specifications for Construction of Roads and Bridges on Federal Highway Projects. 2014. 14
37. Wouter Van Gansbeke, Bert De Brabandere, Davy Neven, Marc Proesmans, and Luc Van Gool. End-to-end lane detection through differentiable least-squares fitting. In ICCV Workshops, 2019. 2
38. Jianfeng Wang, Lin Song, Zeming Li, Hongbin Sun, Jian Sun, and Nanning Zheng. End-to-end object detection with fully convolutional network. In CVPR, 2021. 2, 4, 5
39. Hang Xu, Shaoju Wang, Xinyue Cai, Wei Zhang, Xiaodan Liang, and Zhenguo Li. Curvelane-nas: Unifying lanesensitive architecture search and adaptive point blending. In ECCV, 2020. 1, 6, 7, 8
40. Seungwoo Yoo, Hee Seok Lee, Heesoo Myeong, Sungrack Yun, Hyoungwoo Park, Janghoon Cho, and Duck Hoon Kim. End-to-end lane marker detection via row-wise classification. In CVPR Workshops, 2020. 2
41. Tu Zheng, Hao Fang, Yi Zhang, Wenjian Tang, Zheng Yang, Haifeng Liu, and Deng Cai. Resa: Recurrent feature-shift aggregator for lane detection. In AAAI, 2021. 1, 2, 3, 5, 6, 7, 8, 11
42. Xizhou Zhu, Han Hu, Stephen Lin, and Jifeng Dai. Deformable convnets v2: More deformable, better results. In CVPR, 2019. 2, 4 10 

## Appendix Overview. 
The Appendix is organized as follows: Appendix A describes the FPS test protocol and environments; Appendix B introduces implementation details for each compared method (including ours in Appendix B.8); Appendix C provides implementation details for B´ezier curves, including sampling, ground truth generation and transforms; Appendix D formulates the IoU loss for B´ezier curves and discusses why it failed; Appendix E explores matching priors other than the centerness prior;

Appendix F shows extra ablation studies on datasets other than CULane [22], to verify the generalization of feature flip fusion; Appendix G discusses limitations and recognizes new progress in the field; Appendix H presents qualitative results from our method, visualized on three datasets.

### A. FPS Test Protocol

Let one Frames-Per-Second (FPS) test trial be the average runtime of 100 consecutive model inference with its PyTorch [24] implementation, without calculating gradients.

The input is a 3x360x640 random Tensor (some use all 1 [33], which does not have impact on speed). Note that all methods do not use optimization from packages like TensorRT. We wait for all CUDA kernels to finish before counting the whole runtime. We use Python time.perf counter() since it is more precise than time.time(). For all methods, the FPS is reported as the best result from 3 trials.

Before each test trial, at least 10 forward pass is conducted as warm-up of the device. For each new method to be tested, we keep running warm-up trials of a recorded method until the recorded FPS is reached again, so we can guarantee a similar peak machine condition as before.

Evaluation Environment. The evaluation platform is a 2080 Ti GPU (standard frequency), on a Intel Xeon-E3 CPU server, with CUDA 10.2, CuDNN 7.6.5, PyTorch 1.6.0.

FPS is a platform-sensitive metric, depending on GPU frequency, condition, bus bandwidth, software versions, etc.

Also using 2080 Ti, Tabelini et al. [33] can achieve a better peak performance for all methods. Thus we use the same platform for all FPS tests, to provide fair comparisons.

Remark. Note that FPS (image/s) is different from throughput (image/s). Since FPS restricts batch size to 1, which better simulates the real-time application scenario.

While throughput considers a batch size more than 1. LSTR [19] reported a 420 FPS for its fastest model, which is actually throughput with batch size 16. Our re-tested FPS is 98.

### B. Specifications for Compared Methods

#### B.1. Segmentation Baseline

The segmentation baseline is based on DeeplabV1 [5], originally proposed in SCNN [22]. It is essentially the original DeeplabV1 without CRF, lanes are considered as different classes, and a separate lane existence branch (a series of convolution, pooling and MLP) is used to facilitate lane post-processing. We optimized its training and testing scheme based on recent advances [41]. Re-implemented in our codebase, it attains higher performance than what recent papers usually report.

Post-processing. First, the existence of a lane is determined by the lane existence branch. Then, the predicted per-pixel probability map is interpolated to the input image size. After that, a 9 × 9 Gaussian blur is applied to smooth the predictions. Finally, for each existing lane class, the smoothed probability map is traversed by pre-defined Y coordinates (quantized), and corresponding X coordinates are recorded by the maximum probability position on the row (provided it passes a fixed threshold). Lanes with less than two quali- fied points are simply discarded.

Data Augmentation. We use a simple random rotation with small angles (3 degrees), then resize to input resolution.

#### B.2. SCNN

Our SCNN [22] is re-implemented from the Torch7 version of the official code. Advised by the authors, we added an initialization trick for the spatial CNN layers, and learning rate warm-up, to prevent gradient explosion caused by recurrent feature aggregation. Thus, we can safely adjust the learning rate. Our improved SCNN achieves signifi- cantly better performance than the original one.

Some may find reports of 96.53 accuracy of SCNN on

TuSimple. However, that was a competition entry trained with external data. We report SCNN with ResNet backbones, trained with the same data as other re-implemented methods in our codebase.

Post-processing. Same as Appendix B.1.

Data Augmentation. Same as Appendix B.1.

#### B.3. RESA

Our RESA [41] is implemented based on its published paper. A main difference to the official code release is we do not cutout no-lane areas (in each dataset, there is a certain height range for lane annotation). Because that trick is dataset specific and not generalizable, we do not use that for all compared methods. Other differences are all validated to have better performance than the official code, at least on the CULane val set.

Post-processing. Same as Appendix B.1.

Data Augmentation. Same as Appendix B.1. The original

RESA paper [41] also apply random horizontal flip, which was found ineffective in our re-implementation.

#### B.4. UFLD

Ultra Fast Lane Detection (UFLD) [26] is reported from their paper and open-source code. Since TuSimple FP and 11

FN information is not in the paper, and training from source code leads to very high FP rate (almost 20%), we did not report their performance on this dataset. We adjusted its profiling scripts to calculate number of parameters and FPS in our standard.

Post-processing. Since this method uses gridding cells (each cell is equivalent to several pixels in a segmentation probability map), each point’s X coordinate is calculated as the expectation of locations (cells from the same row), i.e. a weighted average by probability. Differently from segmentation post-processing, it is possible to be efficiently implemented.

Data Augmentation. Augmentations include random rotation and some form of random translation.

#### B.5. PolyLaneNet

PolyLaneNet [32] is reported from their paper and opensource code. We added a profiling script to calculate number of parameters and FPS in our standard.

Post-processing. This method requires no post-processing.

Data Augmentation. Augmentations include large random rotation (10 degrees), random horizontal flip and random crop. They are applied with a probability of 10 11 .

#### B.6. LaneATT

LaneATT [33] is reported from their paper and opensource code. We adjusted its profiling scripts to calculate parameters and FPS in our standard.

Post-processing. Non-Maximal Suppression (NMS) is implemented by a customized CUDA kernel. An extra interpolation of lanes by B-Spline is removed both in testing and profiling, since it is slowly executed on CPU and provides little improvement (∼ 0.2% on CULane).

Data Augmentation. LaneATT uses random affine transforms including scale, translation and rotation. While it also uses random horizontal flip.

Followup. We did not have time to validate the reimplementation of LaneATT in our codebase, prior the submission deadline. Therefore, the LaneATT performance is still reported from the official code. Our re-implementation indicates that all LaneATT results are reproducible except for the ResNet-34 backbone on CULane, which is slightly outside the standard deviation range, but still reasonable.

#### B.7. LSTR

LSTR [19] is re-implemented in our codebase. All

ResNet backbone methods start from ImageNet [14] pretraining. While LSTR [19] use 256 channels ResNet-18 for CULane (2×), 128 channels for other datasets (1×), which makes it impossible to use off-the-shelf pre-trained

ResNets. Although whether ImageNet pre-training helps lane detection is still an open question. Our reported performance of LSTR on CULane, is the first documented report of LSTR on this dataset. With tuning of hyper-parameters (learning rate, epochs, prediction threshold), bug fix (the original classification branch has 3 output channels, which should be 2), we achieve 4% better performance on CULane than the authors’ trial. Specifically, we use learning rate 2.5 × 10−4 with batch size 20. 150 and 2000 epochs,  0.95 and 0.5 prediction thresholds, for CULane and TuSimple. The lower threshold in TuSimple is due to the official
 test metric, which significantly favors a high recall. However, for real-world applications, a high recall leads to high

False Positive rate, which is undesired.

We divide the curve loss weighting by 10 with our

LSTR-Beizer ablation, since there were 100 sample points with both X and Y coordinates to fit, that is a loss scale about 10 times the original loss (LSTR loss takes summation of point L1 distances instead of average). This modulation achieves a similar loss landscape to original LSTR.

Post-processing. This method requires no post-processing.

Data Augmentation. Data augmentation includes PolyLaneNet’s (Appendix B.5), then appends random color distortions (brightness, contrast, saturation, hue) and random lighting by a light source calculated from the COCO dataset [17]. That is by far the most complex data augmentation pipeline in this research field, we have validated that all components of this pipeline helps LSTR training.

Remark. The polynomial coefficients of LSTR are unbounded, which leads to numerical instability (while the bipartite matching requires precision), and high failure rate of training. The failure rate of fp32 training on CULane is ∼ 30%. This is circumvented in B´ezierLaneNet, since our L1 loss can be bounded to [0, 1] without influence on learning (control points easily converges to on-image).

#### B.8. B´ezierLaneNet

B´ezierLaneNet is implemented in the same code framework where we re-implemented other methods. Same as

LSTR, the default prediction threshold is set to 0.95, while 0.5 is used for TuSimple [1].
Post-processing. This method requires no post-processing.

Data Augmentation. We use augmentations similar to

LSTR (Appendix B.7). Concretely, we remove the random lighting from LSTR (to strictly avoid using knowledge from external data), and replace the PolyLaneNet 10 11 chance augmentations with random affine transforms and random horizontal flip, like LaneATT (Appendix B.6). The random affine parameters are: rotation (10 degrees), translation (maximum 50 pixels on X, 20 on Y), scale (maximum 20%).

Polynomial Ablations. For the polynomial ablations (Table 7), we modified the network to predict 6 coefficients for 3rd order Polynomial (4 curve coefficients and start/end Y coordinates). Extra L1 losses are added for the start/end Y coordinates similar to LSTR [19]. With extensive tryouts (ad- 12 justing learning rate, loss weightings, number of epochs), even at the full B´ezierLaneNet setup, with 150 epochs on

CULane, the models still can not converge to a good enough solution. In other word, not precise enough to pass the

CULane metric. The sampling loss on polynomial curves can only get to 0.02, which means 0.02 × 1640pixels =  32.8pixels average X coordinate error on training set. CULane requires a 0.5 IoU between curves, which are enlarged to 30 pixels wide, thus at least around 10 pixels average error is needed to get meaningful results. By loosen up the

IoU requirement to 0.3, we can get F1 score 15.82 for “3rd

Polynomial from B´ezierLaneNet”. Although the reviewing committee suggested adding simple regularization for this ablation to converge, regretfully we failed to do this.

### C. B´ezier Curve Implementation Details

Fast Sampling. The sampling of B´ezier curves may seem tiresome due to the complex Bernstein basis polynomials.

To fast sample a B´ezier curve by a series of fixed t values, simply pre-compute the results from Bernstein basis polynomials, thus only one simple matrix multiplication is left.

Remarks on GT Generation. The ground truth of B´ezier curves are generated with least squares fitting, a common technique for polynomials. We use it for its simplicity and the fact that it already shows near-perfect lane line fitting ability (99.996 and 99.72 F1 score on CULane test and

LLAMAS val, respectively). However, it is not an ideal algorithm for parametric curves. There is a whole research field for fitting B´ezier curves better than least squares [23].

B´ezier Curve Transform. Another implementation diffi- culty on B´ezier curves is how to apply affine transform (for transforming ground truth curves in data augmentation).

Mathematically, affine transform on the control points is equivalent to affine transform on the entire curve. However, translation or rotation can move control points out of the image. In this case, a cutting of B´ezier curves is required.

The classical De Casteljau’s algorithm is used for cutting an on-image B´ezier curve segment. Assume a continuous onimage segment, valid sample points with minimum boundary t = t0, maximum boundary t = t1. The formula to cut a cubic B´ezier curve defined by control points P0,P1,P2,P3 to its on-image segment P00,P01,P02,P03 , is derived as:

P00 = u0u0u0P0 + (t0u0u0 + u0t0u0 + u0u0t0)P1 + (t0t0u0 + u0t0t0 + t0u0t0)P2 + t0t0t0P3, P01 = u0u0u1P0 + (t0u0u1 + u0t0u1 + u0u0t1)P1 + (t0t0u1 + u0t0t1 + t0u0t1)P2 + t0t0t1P3, P02 = u0u1u1P0 + (t0u1u1 + u0t1u1 + u0u1t1)P1 + (t0t1u1 + u0t1t1 + t0u1t1)P2 + t0t1t1P3, P03 = u1u1u1P0 + (t1u1u1 + u1t1u1 + u1u1t1)P1 + (t1t1u1 + u1t1t1 + t1u1t1)P2 + t1t1t1P3, (10) where u0 = 1 − t0, u1 = 1 − t1. This formula can be efficiently implemented by matrix multiplication. The possibility of noncontinuous cubic B´ezier segment on lane detection datasets is extremely low and thus ignored for simplicity. If it does happen, Equation (10) will not change the curve, while our network can also predict out-of-image control points, which still fit the on-image lane segments.

### D. IoU Loss for B´ezier Curves

Here we briefly introduce how we formulated the IoU loss between B´ezier curves. Before diving into the algorithm, there are two preliminaries.
* Polar sort: By anchoring on an arbitrary point inside the N-sided polygon with vertices ci(xi , yi)Ni=1 (normally the mean coordinate between vertices c0 = ( 1N P Ni=1 xi, 1N P Ni=1 yi)), vertices are sorted by its atan2 angles. This will return a clockwise or counterclockwise polygon.
* Convex polygon area: A sorted convex polygon can be efficiently cut into consecutive triangles by simple indexing operations. The convex polygon area is the sum of these triangles. The area S of triangle ((x1, y1),(x2, y2),(x3, y3)) is: S = 12 |x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)|.

Assume we have two convex hulls from B´ezier curves (there are a lot of convex hull algorithms). Now the IoU between B´ezier curves are converted to IoU between convex polygons. Based on the simple fact that the intersection of convex polygons is still a convex polygon, after polar sorting all the convex hulls and determining the intersected polygon, we can easily formulate IoU calculations as a series of convex polygon area calculations. The dif- ficulty lies in how to efficiently determine the intersection between convex polygon pairs.

Consider two intersected convex polygons, their intersection includes two types of vertices:
* Intersections: intersection points between edges.
* Insiders: vertices inside/on both polygons.

For Intersections, we first represent every polygon edge as the general line equation: ax + by = c. Then, for line a1x + b1y = c1 and line a2x + b2y = c2, the intersection (x0 , y0 ) is calculated by: x0 = (b2c1 − b1c2)/det y0 = (a1c2 − a2c1)/det, (11) where det = a1b2 − a2b1. All (x0 , y0 ) that is on the respective line segments are Intersections.

For Insiders, there is a certain definition: 13

Def. 1 For a convex polygon, point P(x, y) on the same side of each edge is inside the polygon.

A sorted convex polygon is a series of edges (line segments defined by P0(x0, y0), P1(x1, y1)), the equation to decide which side a point is to a line segment is as follows: sign = (y − y0)(x1 − x0) − (x − x0)(y1 − y0). (12) sign > 0 means P is on the right side, sign < 0 is the left side, and sign = 0 means P is on the line segment.

Note that equality is not a stable operation for float computations. But there are simple ways to circumvent that in coding, which we will not elaborate here.

There are other ways to determine Intersections and Insiders, but the above formulas can be efficiently implemented with matrix operations and indexing, making it possible to quickly train networks with batched inputs.

Finally, after being able to compute convex polygon intersections and areas, the Generalized IoU loss (GIoU) is simply (as in [29]): input : Two arbitrary convex shapes: A, B ⊆ S ∈ Rn output: GIoU

1. For A and B, find the smallest enclosing convex object C, where C ⊆ S ∈ Rn
2. IoU = |A ∩ B| |A ∪ B|
3. GIoU = IoU − |C\(A ∪ B)| |C|

Union is computed as A ∪ B = A + B − A ∩ B. The enclosing convex object C can be computed as the convex hull of two convex polygons, or upper-bounded by a enclosing rectangle. We implement the IoU computation purely in PyTorch [24], the runtime for our implementation is only about 5× the runtime of rectangle IoU loss computation.

However, lane lines are mostly straight based on road design regulations [7, 36]. This leads to extremely small convex hull area for B´ezier curves, thus introduces numerical instabilities in optimization. Although succeeded in a toy polygon fitting experiment, we currently failed to observe the loss’s convergence to help learning on lane datasets.

### E. GT and Prediction Matching Prior

Figure 6. Logits activation statistics (1 × W 16 ) on CULane [22].

Instead of the centerness prior, we explore a local maximum prior, i.e., restricts matched prediction to have a local maximum classification logit. This prior can facilitate the model to understand the spatially sparse structure of lane lines. As shown in Figure 6, the learned feature activation for classification logits exhibits a similar structure as an actual driving scene.

### F. Extra Results

TuSimple [1] LLAMAS [3] B´ezier Baseline 93.36 95.27 + Feature Flip Fusion 95.26 (+1.90) 96.00 (+0.73)<br/>
Table 9. Ablation study on TuSimple (test set Accuracy) and LLAMAS (val set F1), before and after adding the Feature Flip Fusion module. Reported 3-times average with the ResNet-34 backbone, since ablations often are not stable enough on these datasets to exhibit a clear difference between methods.

### G. Discussions

There exists a primitive application of lane detectors from lateral views to estimate the distance to the border of the drivable area [10], which contradicts the use of feature flip fusion. In this case, possibly a lower order B´ezier curve baseline (with row-wise instead of column-wise pooling) would suffice. This is out of the focus of this paper.

Recent Progress. Recently, others have explored alternative lane representation or formulation methods that do not fully fit in the three categories (segmentation, point detection, curve). Instead of the popular top-down regime, [27] propose a bottom-up approach that focus on local details. [18] achieve state-of-the-art performance, but the complex conditional decoding of lane lines results in unstable runtime depending on the input image, which is not desirable for a real-time system.

### H. Qualitative Results

Qualitative results are shown in Figure 7, from our

ResNet-34 backbone models. For each dataset, 4 results are shown in two rows: first row shows qualitative successful predictions; second row shows typical failure cases.

TuSimple. As shown in Figure 7(a), our model fits highway curves well, only slight errors are seen on the far side where image details are destroyed by projection. Our typical failure case is a high FP rate, mostly attributed to the use of low threshold (Appendix B.8). However, in the bottomright wide road scene, our FP prediction is actually a meaningful lane line that is ignored in center line annotations.

CULane. As shown in Figure 7(b), most lanes in this dataset are straight. Our model can make accurate predictions under heavy congestion (top-left) and shadows (topright, shadow cast by trees). A typical failure case is inaccurate prediction under occlusion (second row), in these cases 14 (a) TuSimple [1]. (b) CULane [22]. (c) LLAMAS [3].

Figure 7. Qualitative results from B´ezierLaneNet (ResNet-34) on val sets. False Positives (FP) are marked by red, True Positives (TP) are marked by green, ground truth are drawn in blue. Blue lines that are barely visible are precisely covered by green lines. B´ezier curve control points are marked with solid circles. Images are slightly resized for alignment. Best viewed in color, in 2× scale. one often cannot visually tell which one is better (ground truth or our FP prediction).

LLAMAS. As shown in Figure 7(c), our method performs accurate for clear straight-lines (top-left), and also good for large curvatures in a challenging scene almost entirely covered by shadow. In bottom-left image, our model fails in a low-illumination, tainted road. While in the other lowillumination scene (bottom-right), the unsupervised annotation from LIDAR and HD-map is misled by the white arrow (see the zigzag shape of the right-most blue line). 15

