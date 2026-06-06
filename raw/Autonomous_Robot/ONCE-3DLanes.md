# ONCE-3DLanes: Building Monocular 3D Lane Detection
ONCE-3D车道：构建单目3D车道检测 https://arxiv.org/abs/2205.00301 https://once-3dlanes.github.io

## Abstract
We present ONCE-3DLanes, a real-world autonomous driving dataset with lane layout annotation in 3D space. Conventional 2D lane detection from a monocular image yields poor performance of following planning and control tasks in autonomous driving due to the case of uneven road. Predicting the 3D lane layout is thus necessary and enables effective and safe driving. However, existing 3D lane detection datasets are either unpublished or synthesized from a simulated environment, severely hampering the development of this field. In this paper, we take steps towards addressing these issues. By exploiting the explicit relationship between point clouds and image pixels, a dataset annotation pipeline is designed to automatically generate high-quality 3D lane locations from 2D lane annotations in 211K road scenes. In addition, we present an extrinsic-free, anchorfree method, called SALAD, regressing the 3D coordinates of lanes in image view without converting the feature map into the bird’s-eye view (BEV). To facilitate future research on 3D lane detection, we benchmark the dataset and provide a novel evaluation metric, performing extensive experiments of both existing approaches and our proposed method.

我们展示了ONCE-3DLanes，这是一个具有三维空间中车道布局标注的真实世界自动驾驶数据集。由于道路不平的情况，来自单眼图像的传统2D车道检测在自动驾驶中产生了跟随规划和控制任务的较差性能。因此，预测3D车道布局是必要的，能够实现有效和安全的驾驶。然而，现有的3D车道检测数据集要么是未发布的，要么是从模拟环境中合成的，这严重阻碍了该领域的发展。在本文中，我们将采取措施解决这些问题。通过利用点云和图像像素之间的显式关系，设计了数据集标注流水线，以从211K道路场景中的2D车道标注自动生成高质量的3D车道位置。此外，我们提出了一种称为SALAD的外部自由无锚方法，在图像视图中回归车道的三维坐标，而不将特征图转换为鸟瞰图(BEV)。为了促进未来对3D车道检测的研究，我们对数据集进行了基准测试，并提供了一种新的评估指标，对现有方法和我们提出的方法进行了广泛的实验。

The aim of our work is to revive the interest of 3D lane detection in a real-world scenario. We believe our work can lead to the expected and unexpected innovations in both academia and industry.

我们工作的目的是在真实世界场景中恢复3D车道检测的兴趣。我们相信，我们的工作可以在学术界和工业界带来预期和意想不到的创新。

## 1. Introduction
The perception of lane structure is one of the most fundamental and safety-critical tasks in autonomous driving system. It is developed with the desired purpose of preventing accidents, reducing emissions and improving the traffic efficiency [5]. It serves as a key roll of many applications, such as lane keeping, high-definition (HD) map modeling [15], trajectory planning etc. In light of its importance, there has been a recent surge of interest in monocular 3D lane detection [7–9, 12, 17]. However, existing 3D lane detection datasets are either unpublished, or synthesized in a simulated environment due to the difficulty of data acquisition and high labor costs of annotation. With the only synthesized data, the model inevitably lacks generalization ability in real-word scenarios. Although benefiting from the development of domain adaptation method [10], it still cannot completely alleviate the domain gap.

车道结构的感知是自动驾驶系统中最基本和最安全的任务之一。它的开发目的是预防事故、减少排放和提高交通效率[5]。它是许多应用的关键，如车道保持、高清(HD)地图建模[15]、路径规划等。鉴于其重要性，最近对单眼3D车道检测的兴趣激增[7-9,12,17]。然而，现有的3D车道检测数据集要么是未发布的，要么是在模拟环境中合成的，这是由于数据获取的困难和标注的高人工成本。在仅有合成数据的情况下，该模型不可避免地缺乏真实场景中的泛化能力。尽管受益于领域适应方法的发展[10]，但它仍然不能完全缓解领域差距。

Figure 1. Images and 3D lane examples of ONCE-3DLanes dataset. ONCE-3DLanes covers various locations, illumination conditions, weather conditions and with numerous slope scenes. 
图1。ONCE-3DLanes数据集的图像和3D车道样本。ONCE-3D车道涵盖了各种位置、照明条件、天气条件以及众多斜坡场景。

Most existing image-based lane detection methods have exclusively focused on formulating the lane detection problem as a 2D task [30, 35, 36], in which a typical pipeline is to firstly detect lanes in the image plane based on semantic segmentation or coordinate regression and then project the detected lanes in top view by assuming the ground is flat [29, 34]. With well-calibrated camera extrinsics, the inverse perspective mapping (IPM) is able to obtain an acceptable approximation for 3D lane in the flat ground plane. However, in real-world driving environment, roads are not always flat [9] and camera extrinsics are sensitive to vehicle body motion due to speed change or bumpy road, which will 1 lead to the incorrect perception of the 3D road structure and thus unexpected behavior may happen to the autonomous driving vehicle.

大多数现有的基于图像的车道检测方法只专注于将车道检测问题表述为2D任务[30，35，36]，其中典型的流水线是首先基于语义分割或坐标回归检测图像平面中的车道，然后通过假设地面是平的来在俯视图中投影检测到的车道[29，34]。利用经过良好校准的摄像机外部，逆透视映射(IPM)能够获得平坦地面中3D车道的可接受近似值。然而，在现实世界的驾驶环境中，道路并不总是平坦的[9]，由于速度变化或颠簸的道路，摄像头外部对车身运动很敏感，这将导致对3D道路结构的错误感知，因此自动驾驶车辆可能会发生意外行为。

To overcome above shortcomings associated with flatground assumption, 3D-LaneNet [9], directly predicts the 3D lane coordinates in an end-to-end manner, in which the camera extrinsics are predicted in a supervised manner to facilitate the projection from image view to top view. In addition, an anchor-based lane prediction head is proposed to produce the final 3D lane coordinates from the virtual top view. Despite the promising result exhibits the feasibility of this task, the virtual IPM projection is difficult to learn without the hard-to-get extrinsics and the model is trained under the assumption that zero degree of camera roll towards the ground plane. Once the assumption is challenged or the need of extrinsic parameters is not satisfied, this method can barely work.

为了克服与平地假设相关的上述缺点，3D LaneNet[9]以端到端的方式直接预测3D车道坐标，其中以监督的方式预测摄像机外部，以便于从图像视图到俯视图的投影。此外，提出了一种基于锚的车道预测头，以从虚拟俯视图生成最终的3D车道坐标。尽管有希望的结果显示了这项任务的可行性，但如果没有难以获得的外部性，虚拟IPM投影很难学习，并且模型是在假设相机朝向地面滚动零度的情况下训练的。一旦假设受到挑战或不满足外部参数的需求，该方法就几乎无法工作。

In this work, we take steps towards addressing above issues. For the first time, we present a real-world 3D lane detection dataset ONCE-3DLanes, consisting of 211K images with labeled 3D lane points. Compared with previous 3D lane datasets, our dataset is the largest real-world lane detection dataset published up to now, containing more complex road scenarios with various weather conditions, different lighting conditions as well as a variety of geographical locations. An automatic data annotation pipeline is designed to minimize the manual labeling effort. Comparing to the method [9] of using multi-sensor and expensive HD maps, ours is simpler and easier to be implemented. In addition, we introduce a spatial-aware lane detection method, dubbed SALAD, in an extrinsic-free and end-to-end manner. Given a monocular input image, SALAD directly predicts the 2D lane segmentation results and the spatial contextual information to reconstruct the 3D lanes without explicit or implicit IPM projection.

在这项工作中，我们采取步骤解决上述问题。我们首次展示了一个真实世界的三维车道检测数据集ONCE-3DLanes，由211K张带有标记的三维车道点的图像组成。与之前的3D车道数据集相比，我们的数据集是迄今为止发布的最大的真实世界车道检测数据集，包含了各种天气条件、不同照明条件以及各种地理位置的更复杂的道路场景。设计了一个自动数据标注管道，以最小化手动标记工作。与使用多传感器和昂贵的高清地图的方法[9]相比，我们的方法更简单，更容易实现。此外，我们以外部自由和端到端的方式引入了一种空间感知车道检测方法，称为SALAD。给定单眼输入图像，SALAD直接预测2D车道分割结果和空间上下文信息，以重建3D车道，而无需显式或隐式IPM投影。

The contributions of this work are summarized as follows: (i) For the first time, we present a largest 3D lane detection dataset ONCE-3DLanes, alongside a more generalized evaluation metric to revive the interest of such task in a real-world scenario; (ii) We propose a method, SALAD that directly produce 3D lane layout from a monocular image without explicit or implicit IPM projection.

这项工作的贡献总结如下：(i)我们首次提出了一个最大的3D车道检测数据集ONCE-3DLanes，以及一个更通用的评估指标，以在真实世界场景中恢复对该任务的兴趣; (ii)我们提出了一种方法SALAD，该方法直接从单眼图像生成3D车道布局，而无需显式或隐式IPM投影。

## 2. Related work
### 2.1. 2D lane detection 2D车道检测
There are various methods [1,16,24,30,32,36] proposed to tackle the problem of 2D lane detection. Segmentation-based methods [20, 26,29, 30] predict pixel-wise segmentation labels and then cluster the pixels belonging to the same label together to predict lane instances. Proposal-based methods [35, 36] first generate lane proposals either from the vanishing points [34] or from the edges of image [23], and then optimize the lane shape by regressing the lane offset. There are some other methods [14, 29, 34, 38] trying to project the image into the top view and using the properties that lanes are almost parallel and can be fitted by lower order polynomial in the top view to fit lanes. However, most methods are limited in the image view, lack the image-toworld step or suffer from the untenable flat-ground assumption. As a result, formulating lane detection as a 2D task may cause inappropriate behaviors for autonomous vehicle when encountering hilly or slope roads [17].

提出了各种方法[1，16，24，30，32，36]来解决2D车道检测问题。基于分段的方法[20，26，29，30]预测逐像素分段标签，然后将属于同一标签的像素聚类在一起，以预测车道实例。基于建议的方法[35，36]首先从消失点[34]或从图像[23]的边缘生成车道建议，然后通过回归车道偏移来优化车道形状。还有其他一些方法[14、29、34、38]试图将图像投影到俯视图中，并使用车道几乎平行的属性，可以通过俯视图的低阶多项式拟合车道。然而，大多数方法都局限于图像视图，缺乏图像到世界的步骤，或者受到不成立的平地假设的影响。因此，将车道检测公式化为2D任务可能会导致自动驾驶车辆在遇到丘陵或斜坡道路时的不当行为[17]。

### 2.2. 3D lane detection
LiDAR-based lane detection. Several methods [18,19,37] have been proposed using LiDAR to detect 3D lanes, [13] use the characteristic that the intensity values of different material are different to filter out the point clouds of the lanes by a certain intensity threshold and then cluster them to obtain 3D lanes. However, it’s hard to determine the specific intensity threshold since the material used in different countries or regions is distinct, and the intensity value varies much in various weather conditions, e.g., rainy or snowy.

基于激光雷达的车道检测。已经提出了使用LiDAR检测3D车道的几种方法[18，19，37]，[13]利用不同材料的强度值不同的特性，通过一定的强度阈值过滤出车道的点云，然后对其进行聚类以获得3D车道。然而，很难确定具体的强度阈值，因为不同国家或地区使用的材料是不同的，并且强度值在不同的天气条件下变化很大，例如下雨或下雪。

Multi-sensor lane detection. Other methods [3, 42] try to aggregate information from both camera and LiDAR sensors to tackle lane detection task. Specifically, [2] predicts the ground height from LiDAR points to project the image to the dense ground. It combines the image information with LiDAR information to produce lane boundary detection results. Nevertheless, it’s difficult to guarantee that the image and the point clouds appear in pairs in the real scenes, e.g., CULane dataset only contains images.

多传感器车道检测。其他方法[3，42]试图收集来自摄像机和激光雷达传感器的信息，以完成车道检测任务。具体而言，[2]预测LiDAR点的地面高度，以将图像投影到密集地面。它将图像信息与激光雷达信息相结合以产生车道边界检测结果。然而，很难保证图像和点云在真实场景中成对出现，例如，CULane数据集仅包含图像。

Monocular lane detection. Recently there are a few methods [7–9, 12, 17] trying to address this problem by directly predicting from a single monocular image. The pioneering work 3D LaneNet [9] predicts camera extrinsics in a supervised manner to learn the inverse perspective mapping(IPM) projection, by combining the image-view features with the top-view features. Gen-LaneNet [12] proposed a new geometry-guided lane anchor in the virtual top view. By decoupling the learning of image segmentation and 3D lane prediction, it achieves higher performance and are more generalizable to unobserved scenes. Instead of associating each lane with a predefined anchor, 3D-LaneNet+ [8] proposes an anchor-free, semi-local representation method to represent lanes. Although the ability to detect more lane topology structures shows the anchorfree method’s power. However, all the above methods need to learn a projection matrix in a supervised way to align the image-view features with top-view features, which may cause height information loss. While our proposed method directly regresses the 3D coordinates in the image view without considering camera extrinsics. 

单眼车道检测。最近有一些方法[7-9,12,17]试图通过直接从单个单眼图像进行预测来解决这个问题。开创性工作3D LaneNet[9]通过将图像视图特征与俯视图特征相结合，以受监督的方式预测相机外部，以学习逆透视映射(IPM)投影。Gen LaneNet[12]在虚拟俯视图中提出了一种新的几何导向车道锚。通过将图像分割和3D车道预测的学习解耦，它实现了更高的性能，并且更易于推广到未观察到的场景。3D LaneNet+[8]提出了一种无锚、半局部表示方法来表示车道，而不是将每条车道与预定义的锚相关联。尽管检测更多车道拓扑结构的能力显示了无锚方法的威力。然而，所有上述方法都需要以监督的方式学习投影矩阵，以将图像视图特征与俯视图特征对齐，这可能导致高度信息丢失。而我们提出的方法直接回归图像视图中的3D坐标，而不考虑相机的外部性。

Table 1. Comparison of different 3D lane detection datasets. ”-” means not mentioned. Ours is the first published real-world dataset covering different weather conditions and geographical locations.
表1。不同3D车道检测数据集的比较。“-”未提及的意思。我们的数据集是第一个发布的涵盖不同天气条件和地理位置的真实世界数据集。

### 2.3. Lane datasets
Existing 3D lane detection datasets are either unpublished or synthesized in simulated environment. GenLanenet [12] uses Unity game engine to build 3D worlds and releases a synthetic 3D lane dataset, Apollo-Sim-3D, containing 10.5K images. 3D-LaneNet [9] adopts a graphics engine to model terrains using a Mixture of Gaussians distribution. Lanes modeled by a 4 th degree polynomial in top view are placed on the terrains to generate the synthetic 3D lanes dataset synthetic-3D-lanes, containing 306K images with a resolution of 360 × 480. A real 3D lane dataset Real-3D-lanes with 85K images is also created using multiple sensors including camera, LiDAR scanner and IMU as well as the expensive HD maps in [9].

现有的3D车道检测数据集要么未发布，要么在模拟环境中合成。GenLanenet[12]使用Unity游戏引擎构建3D世界，并发布了一个合成3D车道数据集Apollo-Sim-3D，包含10.5K张图像。3D LaneNet[9]采用图形引擎，使用高斯分布的混合模型对地形进行建模。在俯视图中由4次多项式建模的车道被放置在地形上，以生成合成3D车道数据集合成3D车道，包含306K张分辨率为360×480的图像。还使用包括相机、激光雷达扫描仪和IMU在内的多个传感器以及[9]中昂贵的高清地图创建了具有85K图像的真实3D车道数据集real-3D车道。

In this paper, we publish the first real-world 3D lane dataset ONCE-3DLanes which contains 211K images and covers abundant scenes with various weather conditions, different lighting conditions as well as a variety of geographical locations. Comprehensive comparisons of 3D lane detection datasets are shown in Table 1.

在本文中，我们发布了第一个真实世界3D车道数据集ONCE-3DLanes，其中包含211K张图像，涵盖了各种天气条件、不同照明条件以及各种地理位置的丰富场景。3D车道检测数据集的综合比较如表1所示。

## 3. ONCE-3DLanes
### 3.1. Dataset introduction
Raw data. We construct our ONCE-3DLanes dataset based on the most recent large-scale autonomous driving dataset

ONCE (one million scenes) [27] considering its superior data quality and diversity. ONCE contains 1 million scenes and 7 million corresponding images, the 3D scenes are recorded with 144 driving hours covering different time periods including morning, noon, afternoon and night, various weather conditions including sunny, cloudy and rainy days, as well as a variety of regions including downtown, suburbs, highway, bridges and tunnels. Since the camera data is captured at a speed of two frames per second and most adjacent frames are very similar, we take one frame every five frames to build our dataset to reduce data redundancy. Also, the distortions are removed to enhance the image quality and improve the projection accuracy from LiDAR to camera.


Figure 2. An overview of slope scenes statistics is shown in (a).

The distribution of the height of lane points is shown in (b). The histogram of the average number of lanes per image and the time periods statistics are shown in (c) and (d) respectively.

Thus, by downsampling the ONCE five times, our dataset contains 211k images taken by a front-facing camera.

Lane representation. A lane Lk in 3D space is represented by a series of points  (xki , yik , zik)	 ni=1, which are recorded in the 3D camera coordinate system with unit meter. The camera coordinate system is placed at the optical center of the camera, with X-axis positive to the right, Y-axis downward and Z-axis forward.

Dataset analysis. The projection error from front-view to top-view mainly occurs in the situation of slope ground, so we focus on analyzing the slope statistics on ONCE- 3DLanes. The mean slope of lanes in each scene is utilized to represent the slope of this scene. The slope of a specific lane in forward direction which is considered to be the most important is calculated as follows: slope = (y_2 - y_1)/(z_2 - z_1) (1) where (x1, y1, z1) and (x2, y2, z2) are the start point and the end point of the lane respectively. The distribution of the slope conditions and the histogram of the number of lanes per image are shown in Figure 2. It shows that our dataset is full of complexity and contains enough various slope scenes with different illumination conditions.

Dataset splits. Follow the ONCE dataset, our benchmark contains the same 3K scenes for validation and 8K scenes for testing. To fully make use of the raw data, the training dataset not only contains the original 5K scenes, but also the unlabeled 200K scenes.

### 3.2. Annotation pipeline
Lanes are a series of points on the ground, which are hard to be identified in point clouds. Hence the high-quality 

Figure 3. Dataset Annotation Pipeline: With the paired image and Lidar point clouds as input, the 2D lanes on the image are firstly labeled and broadened to get the lane regions; secondly the ground points in the point clouds are filtered out through ground segmentation; thirdly the filtered ground points are projected to the image and collect the points which are contained in the lane regions, finally cluster the points to get the real lane points. annotations of 3D lanes are expensive to obtain, whilst it is much cheaper to annotate the lanes in 2D images. The paired LiDAR point clouds and image pixels are thoroughly investigated and used to construct our 3D Lane dataset. An overview of dataset construction pipeline is shown in Figure 3. The pipeline consists of five steps: Ground segmentation, Point cloud projection, Human labeling/Auto labeling,

Adaptive lanes blending and Point cloud recovery. These steps are described in detail below.

Ground segmentation. Lanes are painted on the ground, which is a strong prior to locate the precise coordinate in the 3D space. To make full use of human prior and avoid the reflection of point clouds aliasing between lanes and other objects, the ground segmentation algorithm is utilized to get the ground LiDAR points at first.

The ground segmentation is performed in a coarse-tofine manner. In the coarse way, since the height of the LiDAR points reflected by the ground always settles in certain intervals, a pre-defined threshold is adopted to filter out those points lying on the ground coarsely based on the height statistics of the LiDAR points among the whole dataset as seen in Figure 2(b). In the fine way, several points in front of the vehicle are sampled randomly as seeds and then the classic region growth method is applied to get the fine segmentation result.

Point cloud projection. In this step, the previous extracted ground LiDAR points are projected to the image plane with the help of calibrated LiDAR-to-camera extrinsics and camera intrinsics based on the classic homogeneous transformation, which reveals the explicit corresponding relationship between the 3D ground LiDAR points and the 2D ground pixels in the image.

Human labeling / Auto labeling. To obtain 2D lane labels in the images and alleviate the taggers’ burden, a robust 2D lane detector which is trained in million-level scenes is firstly used to automatically pre-annotate pseudo lane labels. At the same time, professional taggers are required to verify and correct the pseudo labels to ensure the annotation accuracy and quality.

Adaptative lanes blending. After getting the accurate 2D lane labels and ground points, in order to judge whether a ground point belongs to the lane marker or not, we broaden 2D lane labels with an appropriate and adaptive width to get lane regions. Due to the perspective principle, a lane is broadened with different widths according to the distance from camera. After the point clouds projection procedure, we consider the ground points which are contained in the lane regions as the lane point clouds.

Point cloud recovery. Finally, we select these lane point clouds out. And for a specific lane, the lane point clouds in the same beam are clustered to get the lane center points to represent this lane.

To ensure the accuracy of annotations, we do not interpolate between center points in data collection stage. While in training stage, we use cubic spline interpolation to generate dense supervised labels. we also compared our annotations with manually labeling results on a small portion of data, which shows the high quality of our annotations. The interpolation code will be made public along with our dataset.

## 4. SALAD
In this section, we introduce SALAD, a spatial-aware monocular lane detection method to perform 3D lane detection directly on monocular images. In contrast to previous 3D lane detection algorithms [8, 9, 12], which project the image to top view and adopt a set of predefined anchors to regress 3D coordinates, our method does not require human-crafting anchors and the supervision of extrinsic parameters. Inspired by SMOKE [25], SALAD consists of two branches: semantic awareness branch and spatial contextual branch. The overall structure of our model is illustrated in Figure 4. In addition, we also adopt a revised joint 3D lane augmentation strategy to improve the generalization ability. The details of our network architecture and augmentation methods are discussed in the following parts.

### 4.1. Backbone
We choose the Segformer [41] as our backbone to extract the global contextual features and to learn the slender structure of lanes. To be concrete, given an image I ∈ RH×W×3, the hierarchical transformer backbone encodes the image

I into multi-level features at {1/4, 1/8, 1/16, 1/32} of the input resolution. Then all multi-level features are upsampled to H4 × W4 and concatenated through a MLP-based decoder and a convolutional feature fusion layer to aggregate 



Figure 4. The architecture of SALAD. The backbone encodes an input image into deep features and two branches namely semantic awareness branch and spatial contextual contextual branch decode the features to get the lane’s spatial information and the segmentation mask. Then 3D reconstruction is performed by integrating these information and finally obtain the 3D lane positions in real-world scene. the multi-level features into F ∈ R H4 × W4 ×C . Specifically, we adopt Segformer-B2 as our feature extractor.

### 4.2. Semantic awareness branch
Traditional 3D lane detection methods [8, 9, 12] directly projecting feature map from front view to top view, are not reasonable and harmful to the performance of prediction as the feature map may not be organized following the perspective principle.

In order to directly regress 3D lane coordinates, we first design a semantic awareness branch to make full use of 2D semantic information, which provides 2D lane point proposals to aggregate 3D information together. It is a relatively easy task to extract 2D features of lane markers from images with rich semantic information. In addition, current mature experience on semantic segmentation task can also be used to enhance our semantic awareness branch.

We follow the method of [29], to encode 2D lane coordinates  (uki , vki )	 ni=1 into ground-truth segmentation map

Sgt ∈ RH×W×1.

During training time, image I ∈ RH×W×3 and groundtruth Sgt ∈ RH×W×1 pairs are utilized to train semantic awareness branch. During inference, given an image I ∈ RH×W×3 , we are able to locate the foreground lane points on the binary mask S ∈ RH×W×1.

According to [6], inverse projection from 2D image to 3D space is an underdetermined problem. Based on the segmentation map generated by semantic awareness branch, spatial information of each pixel is also required to transfer this segmentation map from 2D image plane to 3D space.

### 4.3. Spatial contextual branch
To reconstruct 3D lanes from 2D lane points generated by semantic awareness branch, we propose the spatial contextual branch to predict vital 3D offsets. To sum up, our spatial contextual branch predicts a regression result O = [δu, δv, δz]T ∈ R3×H×W . δu and δv denote the pixel location offsets of lane points predicted in segmentation branch (us, vs), and generate accurate 2D lane positions. The δz denotes the pixel-level prediction of depth.

Due to the downsampling and lacking of global information, the locations of the predicted lane points are not accurate enough. Our spatial contextual branch accepts feature

F and outputs a pixel-level offset map, which predicts the spatial location shift δu and δv of lane points along u and v axis on the image plane. With the predictions of pixel location offsets δu and δv, the rough estimation of lane point locations is modified by global spatial context: \begin {bmatrix} u \\ v \end {bmatrix} = \begin {bmatrix} u_s + \delta _{u} \\ v_s + \delta _{v} \end {bmatrix}. (2)

In order to recover 3D lane information, the spatial contextual branch also generates a dense depth map to regress on depth offset δz for each pixel of the lane markers. Considering the depth of the ground on image plane increases along rows, we assign each row of the depth map a predefined shift αr and scale βr, and perform regression in a residual way. The standard depth value z is recovered as following: z = \alpha _{r} + \beta _{r}\delta _{z}. (3)

The ground-truth depth map is generated by projecting 3D lane points  (xki , yki , zki )	 ni=1 on the image plane to get pixel coordinates  (uki , vki , zik)	 ni=1. Then at each pixel (uki , vik), its corresponding depth value is assigned to zik.

Following [22], we apply depth completion on the sparse depth map to get the dense depth map Dgt to provide sufficient training signals for our spatial contextual branch. 5

### 4.4. Spatial reconstruction
The spatial information predicted by our spatial contextual branch of our model plays a virtual role in 3D lane reconstruction. To map 2D lane coordinates back to 3D spatial location in camera coordinate system, the depth information is an indispensable element. To be concrete, given the camera intrinsic matrix K3×3, a 3D point (x, y, z) in camera coordinate system can be projected to a 2D image pixel (u, v) as: z\begin {bmatrix} u \\ v \\1 \end {bmatrix} = K_{3\times 3} \begin {bmatrix} x \\ y \\z \end {bmatrix} = \begin {pmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end {pmatrix} \begin {bmatrix} x \\ y \\z \end {bmatrix}, (4) where fx and fy represent the focal length of the camera, (cx, cy) is the principal point and s is the axis skew. Thus, given a 2D lane point in the image with pixel coordinates (u, v) along with its depth information d, noted that the depth denotes the distance to the camera plane, so the depth d is the same as the z in the camera coordinate system. Thus 3D lane point in camera coordinate system (x, y, z) can be restored as follows: \left \{ \begin {aligned} z & = d, \\ y & = \frac {z}{f_y} \cdot (v-c_y), \\ x & = \frac {z}{f_x} \cdot [(u-c_x) - \frac {s}{f_y}(v-c_y)]. \end {aligned} \right . (5)

Utilizing the fixed parameters of the camera intrinsic, we can project the 2D lane proposal points back to 3D locations to reconstruct our 3D lanes.

### 4.5. Loss function
Given an image and its corresponding ground-truth 3D lanes, the loss function between predicted lanes and groundtruth lanes are formulated as: \mathcal {L} = \mathcal {L}_{seg} + \lambda \mathcal {L}_{reg}. (6)

Lseg is for the binary segmentation branch, with a crossentropy loss in a pixel-wise manner on the segmentation map. For a specific pixel in segmentation map S, yi is the label and pi is the probability of being foreground pixels: \mathcal {L}_{seg} = - \frac {1}{N}\sum _{i=1}^{N}[y_ilog(p_i) + (1-y_i)log(1-p_i)]. (7)

Lreg is for the spatial contextual branch, which predicts the spatial offsets O = [δu, δv, δz]T . we choose smooth L1 loss to regress these spatial contextual information O: \begin {aligned} \mathcal {L}_{reg} = &\frac {1}{N}\sum _{i=1}^{N}[smooth_{L_1}(\hat {O}_i - O_i)]. \end {aligned} (8) λ denotes the penalty term for regression loss and is set to 1 in our experiments.

### 4.6. Data augmentation
Randomly horizontal flip and image scaling are common data augmentation methods to improve the generalization ability of 2D lane detection models. However, it is worth noting that the image shift and scale augmentation methods will cause 3D information inconsistent with the data augmentation [25]. We revise it by proposing a joint scale strategy.

To ensure we can restore the same size of the original image from the scaled image, we first crop the top of the image with size c. Then we scale the cropped image with the proportion s. According to the similar triangles theorems, it is proved that the relationship of 3D information of a specific pixel before scaling (x, y, z) and after scaling (ˆx, ˆy, zˆ) is: (\hat {x},\hat {y},\hat {z}) = (x,y,z \cdot s). (9)

Take scaling factor s which is less than one for example. If the image is scaled with factor s, in the camera coordinate system, it is like the camera is moving forward in the Z direction. As for a specific point, the x and y keep the same while the z become smaller and the new z equals z·s. Using this strategy, we can ensure that the 3D ground truth remain consistent during 2D image data augmentation.

## 5. Experiment
In this section, our experiments are presented as follows.

First we introduce our experimental setups, including evaluation metrics and implementation details. Then we evaluate our baseline method on our ONCE-3DLanes dataset and investigate the evaluation performance of different hyperparameter settings. Next we compare our proposed method with the prior state-of-the-art to prove the superiority of our proposed method. Finally, we conduct several ablations studies to show the significance of modules in our network.

### 5.1. Evaluation metric
Evaluation metric is set to measure the similarity between the predicted lanes and the ground-truth lanes. Previous evaluation metric [12], which set predefined y-position to regress the x and z coordinates of lane points, is not sophisticated. Due to the fixed anchor design, this metric essentially performs badly when the lanes are horizontal.

To tackle this problem, we propose a two-stage evaluation metric, which regards 3D lane evaluation problem as a point cloud matching problem combined with top-view constraints. Intuitively, two 3D lane pairs are well matched when achieving little difference in the z-x plane (top view) alongside a close height distribution. Matching in the top view constrains the predicted lanes in the correct forward direction, and close point clouds distance in 3D space ensures the accuracy of the predicted lanes in spatial height. 6

Our proposed metric first calculate the matching degree of two lanes on the z-x plane. To be concrete, lane is represented as Lk =  (xki , yik , zik)	 ni=1. To judge whether predicted lane Lp matches ground-truth lane Lg , the first matching process is done in the z-x plane, namely top-view, we use the traditional IoU method [30] to judge whether Lp matches Lg . If the IoU is bigger than the IoU threshold, further we use a unilateral Chamfer Distance (CD) to calculate the curves matching error in the camera coordinates.

The curve matching error CDp,g between Lp and Lg is calculated as follows: \left \{ \begin {aligned} & CD_{p,g} = \frac {1}{m} \sum _{i=1}^{m} || P_{g_i} - \hat {P}_{p_j}||_2, \\ &\hat {P}_{p_j} = \mathop {min}\limits _{P_{p_j}\in L^p}||P_{p_j}-P_{g_i}||_2, \\ \end {aligned} \right . (10) where Ppj = (xpj , ypj , zpj ) and Pgi = (xgi , ygi , zgi ) are point of Lp and Lg respectively, and Pˆpj is the nearest point to the specific point Pgi . m represents the number of points token at an equal distance from the ground-truth lane.

If the unilateral chamfer distance is less than the chamfer distance threshold, written as τCD. We consider Lp matches

Lg and accept Lp as a true positive. The legend to calculate chamfer distance error is shown in Figure 5.

Figure 5. Unilateral chamfer distance Given a point on the gound-truth lane, find the nearest point on the predicted lane to calculate the chamfer distance.

This evaluation metric is intuitive and strict, and more importantly, it applies to more lane topologies such as vertical lanes, so it is more generalized. At last, since we know how to judge a predicted lane is true positive or not, we use the precision, recall and F-score as the evaluation metric.

### 5.2. Implementation details
Our experiments are carried out on our proposed ONCE- 3DLanes benchmark. We use the Segformer [41] as our backbone with two branches. The Segformer encoder is pre-trained with Imagenet [21]. The input resolution is set to 320 × 800 with our augmentation strategy during training. The augmentation is turned off during testing. The

Adamw optimizer is adopted to train 20 epochs, with an initial learning rate at 3e-3 and applied with a poly scheduler by default. For evaluation, We set the IoU threshold as 0.3, the chamfer distance thresh τCD as 0.3m. We also test our model using the Mindspore [28].


Figure 6. Qualitative results and failure cases analysis under specific threshold. The ground-truth lanes are colored in red, while true positives of our predictions are in blue and false positives in cyan. The τCD of 0.5 is somehow too loose for discrimination.

### 5.3. Benchmark performance
### 5.3.1 Main results
We train our model with the whole training set of 200k images and report the detection performance on the test set of ONCE-3DLanes dataset. To verify the rationality of the hyper-parameters setting in our evaluation metric, we evaluate our model under different Chamfer Distance threshold τCD and report the testing results in Table 2. 

Table 2. Performance of SALAD under different τCD thresholds on our test set.

We report our performance in different settings of τCD, in order to fully investigate the impact of tightening or loosening the criteria on model performance. The illustrations of criteria adopting various τCD are also proposed in

Figure 6. It can be seen that under the threshold of 0.5, some predicted lanes relatively far from the ground-truth are judged as true positives. While under the τCD of 0.15, the criteria seems too harsh. The τCD of 0.3 is more reasonable and our SALAD achieves 64.07% F1-score based on this threshold. Besides, since the distance is calculated based on real scene so it is highly adaptable to real-world datasets. In the remaining parts, the experimental results are reported under the threshold of 0.3.

### 5.3.2 Results of 3D lane detection methods
In order to further verify the authenticity of our dataset and the superiority of our method, which is extrinsic-free, we also conduct some experiments of other 3D lane detection algorithms on our dataset.

It is worth noting that all existing 3D lane detection algorithms need to provide camera poses as supervision information and have strict assumptions that the camera is installed at zero degrees roll relative to the ground plane [9].

Table 3. Performance of 3D lane detection on ONCE-3DLanes. 

However, our method requires no external parameter information. To make comparison, we used the camera-pose parameters provided by ONCE [27] to provide supervision signals for counterpart methods and ultimately evaluated them on our test set. The performance of 3D lane detection algorithms are presented in Table 3.

The comparison shows our method outperforms other 3D lane detection method in ONCE-3DLanes dataset. The comparison result indicates that extrinsic-required methods under the assumption of fixed camera pose and zero-degree camera roll may suffer in real 3D scenarios.

### 5.3.3 Results of extended 2D lane detection methods
ONCE-3DLanes dataset is a newly released 3D lane detection dataset and no previous work addresses the problem of extrinsic-free 3D lane detection. In order to verify the validity of our dataset and the efficiency of our method, we extend the existing 2D lane detection model and evaluate their performance on our dataset.

Different from 3D lane detection methods, 2D lane detection algorithms can only detect the pixel coordinates of lanes in the image plane, but cannot recover the spatial information of lanes. To obtain 3D lane detection results, we use a pre-trained depth estimation model MonoDepth2 [11] (finetuned to the depth scale of our dataset) to estimate the pixel-level depth of the image. It is worth mentioning that the depth model is finetuned on full ONCE dataset in order to avoid under-fitting caused by sparse supervision provided by lane points, which also indicates that this pipeline is difficult to be extended on other 3D lane benchmarks.

Combined with the detection results of extended 2D lane detection model, the spatial position of 3D lanes are reconstructed and our evaluation metric is used for performance evaluation. The results are showed in Table 4.

Table 4. Performance of extended 2D lane detection methods on

ONCE-3DLanes test set.

Experimental results show that the extended 2D models are effective to perform 3D lane detection task on our ONCE-3DLanes dataset. It can also be found that our proposed method can reach 64.07% F1-score, outperforms the best of other methods 56.57% by 7.5%, which shows the superiority of our method.

### 5.4. Ablation study
To verify the efficiency of revised data augmentation methods, we conduct ablation experiments by gradually turning off data augmentation strategy. As shown in Table 5, the performance of our method is constantly improved with the gradual introduction of our data augmentation strategy.

The flip method brings a 1.06% improvement to our model and the 3D scale provides a further 1.83% improvement, which proves the validity of our augmentation strategy.


Table 5. Ablation study on data augmentation strategy.

## 6. Conclusion and limitations
In this paper, we have presented a largest real-world 3D lane detection benchmark ONCE-3DLanes. To revive the interest of 3D lane detection, we benchmark the dataset with a novel evaluation metric and propose an extrinsic-free and anchor-free method, dubbed SALAD, directly predicting the 3D lane from a single image in an end-to-end manner. We believe our work can lead to the expected and unexpected innovations in communities of both academia and industry.

As our dataset construction requires LiDAR to provide 3D information, occlusions would lead to short interruption and we have used interpolation to fix it. Missing points problem in the distance still exists due to the low resolution of LiDAR. Future work will focus on the ground point clouds completion to generate full information for 3D lanes.

## Acknowledgments 
This work was supported in part by National Natural Science Foundation of China (Grant No. 6210020439), Lingang Laboratory (Grant No. LGQS-202202-07), Natural Science Foundation of Shanghai (Grant No. 22ZR1407500), Shanghai Municipal Science and Technology Major Project (Grant No. 2018SHZDZX01 and 2021SHZDZX0103), Science and Technology Innovation 2030 - Brain Science and Brain-Inspired Intelligence Project (Grant No. 2021ZD0200204), MindSpore and CAAI-Huawei MindSpore Open Fund. 

## References
1. Hala Abualsaud, Sean Liu, David Lu, Kenny Situ, Akshay Rangesh, and Mohan M Trivedi. Laneaf: Robust multi-lane detection with affinity fields. arXiv preprint, 2021. 2, 8
2. Min Bai, Gellert Mattyus, Namdar Homayounfar, Shenlong Wang, Shrinidhi Kowshika Lakshmikanth, and Raquel Urtasun. Deep multi-sensor lane detection. In IROS, 2018. 2
3. Luca Caltagirone, Mauro Bellone, Lennart Svensson, and Mattias Wahde. Lidar–camera fusion for road detection using fully convolutional neural networks. Robotics and Autonomous Systems, 2019. 2
4. Zhenpeng Chen, Qianfei Liu, and Chenfan Lian. Pointlanenet: Efficient end-to-end cnns for accurate real-time lane detection. In IV, 2019. 8
5. Travis J. Crayton and Benjamin Mason Meier. Autonomous vehicles: Developing a public health research agenda to frame the future of transportation policy. Journal of Transport & Health, 2017. 1
6. Paul E Debevec, Camillo J Taylor, and Jitendra Malik. Modeling and rendering architecture from photographs: A hybrid geometry-and image-based approach. In SIGGRAPH, 1996. 5
7. Netalee Efrat, Max Bluvstein, Noa Garnett, Dan Levi, Shaul Oron, and Bat El Shlomo. Semi-local 3d lane detection and uncertainty estimation. arXiv preprint, 2020. 1, 2
8. Netalee Efrat, Max Bluvstein, Shaul Oron, Dan Levi, Noa Garnett, and Bat El Shlomo. 3d-lanenet+: Anchor free lane detection using a semi-local representation. arXiv preprint, 2020. 1, 2, 4, 5
9. Noa Garnett, Rafi Cohen, Tomer Pe’er, Roee Lahav, and Dan Levi. 3d-lanenet: End-to-end 3d multiple lane detection. In ICCV, 2019. 1, 2, 3, 4, 5, 8
10. Noa Garnett, Roy Uziel, Netalee Efrat, and Dan Levi. Synthetic-to-real domain adaptation for lane detection. In ACCV, 2020. 1
11. Cl´ement Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J Brostow. Digging into self-supervised monocular depth estimation. In ICCV, 2019. 8
12. Yuliang Guo, Guang Chen, Peitao Zhao, Weide Zhang, Jinghao Miao, Jingao Wang, and Tae Eun Choe. Gen-lanenet: A generalized and scalable approach for 3d lane detection. In ECCV, 2020. 1, 2, 3, 4, 5, 6, 8
13. Alberto Hata and Denis Wolf. Road marking detection using lidar reflective intensity data and its application to vehicle localization. 2014. 2
14. Bei He, Rui Ai, Yang Yan, and Xianpeng Lang. Accurate and robust lane detection based on dual-view convolutional neutral network. In IV, 2016. 2
15. Namdar Homayounfar, Wei-Chiu Ma, Shrinidhi Kowshika Lakshmikanth, and Raquel Urtasun. Hierarchical recurrent attention networks for structured online maps. arXiv preprint, 2020. 1
16. Yuenan Hou, Zheng Ma, Chunxiao Liu, and Chen Change Loy. Learning lightweight lane detection cnns by self attention distillation. In ICCV, 2019. 2
17. Yujie Jin, Xiangxuan Ren, Fengxiang Chen, and Weidong Zhang. Robust monocular 3d lane detection with dual attention. In ICIP, 2021. 1, 2
18. Jiyoung Jung and Sung-Ho Bae. Real-time road lane detection in urban areas using lidar data. Electronics, 2018. 2
19. Soren Kammel and Benjamin Pitzer. Lidar-based lane marker detection and mapping. In IVS, 2008. 2
20. Yeongmin Ko, Younkwan Lee, Shoaib Azam, Farzeen Munir, Moongu Jeon, and Witold Pedrycz. Key points estimation and point instance segmentation approach for lane detection. IEEE Transactions on Intelligent Transportation Systems, 2021. 2
21. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. 2012. 7
22. Jason Ku, Ali Harakeh, and Steven L Waslander. In defense of classical image processing: Fast depth completion on the cpu. In CRV, 2018. 5
23. Xiang Li, Jun Li, Xiaolin Hu, and Jian Yang. Line-cnn: Endto-end traffic line detection with line proposal unit. IEEE Transactions on Intelligent Transportation Systems, 2019. 2
24. Lizhe Liu, Xiaohao Chen, Siyu Zhu, and Ping Tan. Condlanenet: a top-to-down lane detection framework based on conditional convolution. arXiv preprint, 2021. 2
25. Zechen Liu, Zizhang Wu, and Roland T´oth. Smoke: Singlestage monocular 3d object detection via keypoint estimation. In CVPR workshops, 2020. 4, 6
26. Sheng Lu, Zhaojie Luo, Feng Gao, Mingjie Liu, KyungHi Chang, and Changhao Piao. A fast and robust lane detection method based on semantic segmentation and optical flow estimation. Sensors, 2021. 2
27. Jiageng Mao, Minzhe Niu, Chenhan Jiang, Hanxue Liang, Jingheng Chen, Xiaodan Liang, Yamin Li, Chaoqiang Ye, Wei Zhang, Zhenguo Li, et al. One million scenes for autonomous driving: Once dataset. arXiv preprint, 2021. 3, 8
28. Mindspore. https://www.mindspore.cn/, 2020. 7
29. Davy Neven, Bert De Brabandere, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool. Towards end-to-end lane detection: an instance segmentation approach. In IV, 2018. 1, 2, 5
30. Xingang Pan, Jianping Shi, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Spatial as deep: Spatial cnn for traffic scene understanding. In AAAI, 2018. 1, 2, 7
31. Zequn Qin, Huanyu Wang, and Xi Li. Ultra fast structureaware deep lane detection. In ECCV, 2020. 8
32. Zhan Qu, Huan Jin, Yang Zhou, Zhen Yang, and Wei Zhang. Focus on local: Detecting lane marker from bottom up via key point. In CVPR, 2021. 2
33. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 10
34. Jinming Su, Chao Chen, Ke Zhang, Junfeng Luo, Xiaoming Wei, and Xiaolin Wei. Structure guided lane detection. arXiv preprint, 2021. 1, 2
35. Lucas Tabelini, Rodrigo Berriel, Thiago M Paixao, Claudine Badue, Alberto F De Souza, and Thiago Oliveira-Santos. 9 Polylanenet: Lane estimation via deep polynomial regression. In ICPR, 2021. 1, 2
36. Lucas Tabelini, Rodrigo Berriel, Thiago M Paix˜ao, Claudine Badue, Alberto F De Souza, and Thiago Olivera-Santos. Keep your eyes on the lane: Attention-guided lane detection. arXiv preprint, 2020. 1, 2, 8
37. Michael Thuy and F Puente Le´on. Lane detection and tracking based on lidar data. Metrology and Measurement Systems, 2010. 2
38. Wouter Van Gansbeke, Bert De Brabandere, Davy Neven, Marc Proesmans, and Luc Van Gool. End-to-end lane detection through differentiable least-squares fitting. In ICCV, 2019. 2
39. Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 10
40. Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Standalone axial-attention for panoptic segmentation. In ECCV, 2020. 10
41. Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. arXiv preprint, 2021. 4, 7, 10
42. Xinyu Zhang, Zhiwei Li, Xin Gao, Dafeng Jin, and Jun Li. Channel attention in lidar-camera fusion for lane line segmentation. Pattern Recognition, 2021. 2
43. Tu Zheng, Hao Fang, Yi Zhang, Wenjian Tang, Zheng Yang, Haifeng Liu, and Deng Cai. Resa: Recurrent feature-shift aggregator for lane detection. In AAAI, 2021. 8 

## A. Appendix
### A.1. Further analysis
We conduct ablation studies to show the rationality of our experiment settings including loss function, backbone network and regression method.

Loss function As shown in Table 6, for the spatial contextual branch, we study different loss functions. Results show the smooth L1 loss outperforms L1 and L2 loss functions at all metrics.

Backbone network We compare the SegFormer [41] with Unet [33] for the backbone network. Moreover, different attention mechanisms [39, 40] are added to Unet to help learn the global information of lane structures. Table 7 shows that model with SegFormer beat the variants of Unet by a clear margin.

Regression method We also conduct an ablation study to evaluate the way to predict the depth information in Table 8.

Our method regress in a residual manner is referred as relative method, and the method directly regress the depth information without pre-defined shift and scale is called absolute method. Table 8 shows the relative outperforms absolute by a large margin.

Loss function F1(%) Precision(%) Recall(%) CD error(m)

L1 63.47 75.08 54.97 0.101

L2 62.91 74.55 54.41 0.103

Smooth L1 64.07 75.90 55.42 0.098

Table 6. Ablation studies on loss functions in the spatial contextual branch.

Backbone F1(%) Precision(%) Recall(%) CD error(m)

Unet [33] 61.12 73.47 52.32 0.105

Unet+self-att. [39] 62.71 74.41 54.19 0.101

Unet+axial-att. [40] 63.15 74.81 54.64 0.101

SegFormer-B2 [41] 64.07 75.90 55.42 0.098

Table 7. Ablation studies on backbone networks.

Offset option F1(%) Precision(%) Recall(%) CD error(m) absolute 62.37 74.07 53.86 0.104 relative 64.07 75.90 55.42 0.098

Table 8. Ablation studies on depth regression methods.

### A.2. More qualitative results
We present the qualitative results of SALAD lane prediction in Figure 7. 2D projections are shown in the left and 3D visualizations are presented in the right. 

Figure 7. Visualization of SALAD on ONCE-3DLanes test set. The ground-truth lanes are colored in red while the predicted lanes are colored in blue. 2D projections are shown in the left and 3D visualizations in the right. 12
