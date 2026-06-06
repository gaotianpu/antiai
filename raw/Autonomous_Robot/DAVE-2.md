# End to End Learning for Self-Driving Cars
端到端学习自动驾驶汽车 2016.4 https://arxiv.org/abs/1604.07316

## 阅读笔记
* action：转向、加速减速. 端到端学习是多目标的，加速多少，转向多少等等？

## Abstract
We trained a convolutional neural network (CNN) to map raw pixels from a single front-facing camera directly to steering commands. This end-to-end approach proved surprisingly powerful. With minimum training data from humans the system learns to drive in traffic on local roads with or without lane markings and on highways. It also operates in areas with unclear visual guidance such as in parking lots and on unpaved roads.

我们训练了卷积神经网络(CNN)，将单个前置摄像头的原始像素直接映射到转向命令。这种端到端的方法被证明非常强大。通过人类提供的最少训练数据，该系统可以学习在有或没有车道标注的地方道路和高速公路上驾驶车辆。它也在视觉引导不清晰的区域运行，如停车场和未铺路面。<!--训练数据?-->

The system automatically learns internal representations of the necessary processing steps such as detecting useful road features with only the human steering angle as the training signal. We never explicitly trained it to detect, for example, the outline of roads.

该系统自动学习必要处理步骤的内部表示，例如仅以人类转向角作为训练信号来检测有用的道路特征。例如，我们从未明确训练它检测道路轮廓。

Compared to explicit decomposition of the problem, such as lane marking detection, path planning, and control, our end-to-end system optimizes all processing steps simultaneously. We argue that this will eventually lead to better performance and smaller systems. Better performance will result because the internal components self-optimize to maximize overall system performance, instead of optimizing human-selected intermediate criteria, e. g., lane detection. Such criteria understandably are selected for ease of human interpretation which doesn’t automatically guarantee maximum system performance. Smaller networks are possible because the system learns to solve the problem with the minimal number of processing steps.

与问题的显式分解(如车道标注检测、路径规划和控制)相比，我们的端到端系统同时优化了所有处理步骤。我们认为，这将最终导致更好的性能和更小的系统。由于内部组件自我优化以最大化系统整体性能，而不是优化人为选择的中间标准(例如车道检测)，因此将产生更好的性能。可以理解的是，选择这样的标准是为了便于人工解释，这并不能自动保证最大的系统性能。较小的网络是可能的，因为系统学习以最少的处理步骤来解决问题。

We used an NVIDIA DevBox and Torch 7 for training and an NVIDIA DRIVETM PX self-driving car computer also running Torch 7 for determining where to drive. The system operates at 30 frames per second (FPS). 

我们使用NVIDIA DevBox和Torch 7进行训练，并使用NVIDIADRIVETM PX自动驾驶汽车电脑运行Torch 7确定驾驶位置。系统以每秒30帧(FPS)的速度运行。

## 1 Introduction
CNNs [1] have revolutionized pattern recognition [2]. Prior to the widespread adoption of CNNs, most pattern recognition tasks were performed using an initial stage of hand-crafted feature extraction followed by a classifier. The breakthrough of CNNs is that features are learned automatically from training examples. The CNN approach is especially powerful in image recognition tasks because the convolution operation captures the 2D nature of images. Also, by using the convolution kernels to scan an entire image, relatively few parameters need to be learned compared to the total number of operations.

CNN[1]彻底改变了模式识别[2]。在广泛采用神经网络之前，大多数模式识别任务都是使用手工特征提取的初始阶段，然后是分类器。神经网络的突破在于特征是从训练样本中自动学习的。CNN方法在图像识别任务中尤其强大，因为卷积运算捕获图像的2D性质。此外，通过使用卷积核扫描整个图像，与操作总数相比，需要学习的参数相对较少。

While CNNs with learned features have been in commercial use for over twenty years [3], their adoption has exploded in the last few years because of two recent developments. First, large, labeled data sets such as the Large Scale Visual Recognition Challenge (ILSVRC) [4] have become available for training and validation. Second, CNN learning algorithms have been implemented on the massively parallel graphics processing units (GPUs) which tremendously accelerate learning and inference.

虽然具有学习功能的CNN[3]已经在商业上使用了20多年，但由于最近的两项发展，它们的采用在过去几年中呈爆炸式增长。首先，大型标注数据集(如大规模视觉识别挑战(ILSVRC)[4])已可用于训练和验证。第二，CNN学习算法已经在大规模并行图形处理单元(GPU)上实现，这大大加快了学习和推理。

In this paper, we describe a CNN that goes beyond pattern recognition. It learns the entire processing pipeline needed to steer an automobile. The groundwork for this project was done over 10 years ago in a Defense Advanced Research Projects Agency (DARPA) seedling project known as DARPA Autonomous Vehicle (DAVE) [5] in which a sub-scale radio control (RC) car drove through a junk-filled alley way. DAVE was trained on hours of human driving in similar, but not identical environments. The training data included video from two cameras coupled with left and right steering commands from a human operator.

在本文中，我们描述了一种超越模式识别的CNN。它学习驾驶汽车所需的整个处理流程。10多年前，国防高级研究计划局(DARPA)的一个名为DARPA自主车辆(DAVE)[5]的幼苗项目为该项目打下了基础，在该项目中，一辆小型无线电控制(RC)汽车驶过一条堆满垃圾的小巷。DAVE接受了在相似但不相同的环境中驾驶数小时的训练。训练数据包括来自两个摄像头的视频以及来自操作员的左右转向命令。

In many ways, DAVE-2 was inspired by the pioneering work of Pomerleau [6] who in 1989 built the Autonomous Land Vehicle in a Neural Network (ALVINN) system. It demonstrated that an end-to-end trained neural network can indeed steer a car on public roads. Our work differs in that 25 years of advances let us apply far more data and computational power to the task. In addition, our experience with CNNs lets us make use of this powerful technology. (ALVINN used a fully-connected network which is tiny by today’s standard.)

在许多方面，DAVE-2受到了Pomereau[6]的开创性工作的启发，他于1989年在神经网络(ALVINN)系统中制造了自主陆地车辆。它证明了一个端到端训练的神经网络确实可以在公共道路上驾驶汽车。我们的工作不同之处在于，25年的进步使我们能够将更多的数据和计算能力应用于这项任务。此外，我们在CNNs方面的经验使我们能够利用这一强大的技术。(ALVINN使用了一个完全连接的网络，以今天的标准来看，它很小。)

While DAVE demonstrated the potential of end-to-end learning, and indeed was used to justify starting the DARPA Learning Applied to Ground Robots (LAGR) program [7], DAVE’s performance was not sufficiently reliable to provide a full alternative to more modular approaches to off-road driving. DAVE’s mean distance between crashes was about 20 meters in complex environments.

尽管DAVE展示了端到端学习的潜力，并确实被用来证明启动DARPA地面机器人学习(LAGR)计划的合理性[7]，但DAVE的性能并不足够可靠，无法为越野驾驶提供更多模块化方法的完整替代方案。在复杂环境中，DAVE的平均碰撞距离约为20米。<!--mean distance between crashes 平均碰撞距离 -->

Nine months ago, a new effort was started at NVIDIA that sought to build on DAVE and create a robust system for driving on public roads. The primary motivation for this work is to avoid the need to recognize specific human-designated features, such as lane markings, guard rails, or other cars, and to avoid having to create a collection of “if, then, else” rules, based on observation of these features. This paper describes preliminary results of this new effort. 

九个月前，NVIDIA开始了一项新的努力，试图在DAVE的基础上建立一个强大的公共道路驾驶系统。这项工作的主要动机是避免需要识别特定的人类指定特征，如车道标注、护栏或其他汽车，并避免基于对这些特征的观察创建一组“如果，那么，否则”规则。本文描述了这项新工作的初步结果。

## 2 Overview of the DAVE-2 System
Figure 1 shows a simplified block diagram of the collection system for training data for DAVE-2. Three cameras are mounted behind the windshield of the data-acquisition car. Time-stamped video from the cameras is captured simultaneously with the steering angle applied by the human driver. This steering command is obtained by tapping into the vehicle’s Controller Area Network (CAN) bus. In order to make our system independent of the car geometry, we represent the steering command as 1/r where r is the turning radius in meters. We use 1/r instead of r to prevent a singularity when driving straight (the turning radius for driving straight is infinity). 1/r smoothly transitions through zero from left turns (negative values) to right turns (positive values).

图1显示了DAVE-2训练数据收集系统的简化框图。三个摄像头安装在数据采集车的挡风玻璃后面。摄像机的时间戳视频与人类驾驶员施加的转向角同时捕获。该转向命令通过轻敲车辆的控制器局域网(CAN)总线获得。为了使我们的系统独立于汽车几何结构，我们将转向命令表示为1/r，其中r是以米为单位的转弯半径。我们使用1/r代替r来防止直线行驶时出现奇点(直线行驶的转弯半径为无穷大)。1/r从左转弯(负值)到右转弯(正值)平滑地从零过渡。<!-- 1/r -->

Figure 1: High-level view of the data collection system.
图1：数据收集系统的高级视图。

Training data contains single images sampled from the video, paired with the corresponding steering command (1/r). Training with data from only the human driver is not sufficient. The network must learn how to recover from mistakes. Otherwise the car will slowly drift off the road. The training data is therefore augmented with additional images that show the car in different shifts from the center of the lane and rotations from the direction of the road. 

训练数据包含从视频中采样的单个图像，与相应的转向命令(1/r)配对。仅使用人类驾驶员的数据进行训练是不够的。网络必须学会如何从错误中恢复。否则，汽车会慢慢偏离道路。因此，训练数据用额外的图像来增强，这些图像显示了汽车从车道中心的不同移动以及从道路方向的旋转。
<!--单帧图像和转向命令对，过去一段连续视频作为输入呢？-->

Images for two specific off-center shifts can be obtained from the left and the right camera. Additional shifts between the cameras and all rotations are simulated by viewpoint transformation of the image from the nearest camera. Precise viewpoint transformation requires 3D scene knowledge which we don’t have. We therefore approximate the transformation by assuming all points below the horizon are on flat ground and all points above the horizon are infinitely far away. This works fine for flat terrain but it introduces distortions for objects that stick above the ground, such as cars, poles, trees, and buildings. Fortunately these distortions don’t pose a big problem for network training. The steering label for transformed images is adjusted to one that would steer the vehicle back to the desired location and orientation in two seconds.

可以从左侧和右侧相机获得两个特定偏心偏移的图像。通过对来自最近相机的图像进行视点变换来模拟相机之间的额外移动和所有旋转。精确的视点转换需要我们不具备的3D场景知识。因此，我们通过假设地平线下的所有点都在平坦的地面上，而地平线上的所有点距离无限远来近似转换。这对于平坦的地形很好，但它会导致地面上的物体变形，例如汽车、电线杆、树木和建筑物。幸运的是，这些失真不会对网络训练造成大问题。变换图像的转向标签被调整为两秒钟内将车辆转向回所需位置和方向的标签。

A block diagram of our training system is shown in Figure 2. Images are fed into a CNN which then computes a proposed steering command. The proposed command is compared to the desired command for that image and the weights of the CNN are adjusted to bring the CNN output closer to the desired output. The weight adjustment is accomplished using back propagation as implemented in the Torch 7 machine learning package.

我们的训练系统框图如图2所示。图像被馈送到CNN，然后CNN计算提议的转向命令。将建议的命令与该图像的期望命令进行比较，并且调整CNN的权重以使CNN输出更接近期望输出。使用Torch 7机器学习包中实现的反向传播来完成权重调整。
 
Figure 2: Training the neural network.
图2：训练神经网络。

Figure 3: The trained network is used to generate steering commands from a single front-facing center camera. 
图3：经过训练的网络用于从单个前向中央摄像头生成转向命令。

## 3 Data Collection
Training data was collected by driving on a wide variety of roads and in a diverse set of lighting and weather conditions. Most road data was collected in central New Jersey, although highway data was also collected from Illinois, Michigan, Pennsylvania, and New York. Other road types include two-lane roads (with and without lane markings), residential roads with parked cars, tunnels, and unpaved roads. Data was collected in clear, cloudy, foggy, snowy, and rainy weather, both day and night. In some instances, the sun was low in the sky, resulting in glare reflecting from the road surface and scattering from the windshield.

训练数据是通过在各种道路上以及在各种照明和天气条件下驾驶收集的。大多数道路数据是在新泽西州中部收集的，尽管高速公路数据也从伊利诺伊州、密歇根州、宾夕法尼亚州和纽约州收集。其他道路类型包括双车道道路(有或无车道标线)、有停车场的住宅道路、隧道和未铺砌道路。数据是在晴朗、多云、多雾、下雪和下雨的天气中收集的，包括白天和晚上。在某些情况下，太阳在天空中很低，导致强光从路面反射并从挡风玻璃散射。

Data was acquired using either our drive-by-wire test vehicle, which is a 2016 Lincoln MKZ, or using a 2013 Ford Focus with cameras placed in similar positions to those in the Lincoln. The system has no dependencies on any particular vehicle make or model. Drivers were encouraged to maintain full attentiveness, but otherwise drive as they usually do. As of March 28, 2016, about 72 hours of driving data was collected. 

数据采集使用的是我们的线控测试车(2016款林肯MKZ)，或者使用2013款福特福克斯(Ford Focus)，相机放置在与林肯相似的位置。该系统不依赖于任何特定的车辆品牌或模型。我们鼓励驾驶员保持全神贯注，否则就照常驾驶。截至2016年3月28日，我们收集了大约72小时的驾驶数据。

## 4 Network Architecture
We train the weights of our network to minimize the mean squared error between the steering command output by the network and the command of either the human driver, or the adjusted steering command for off-center and rotated images (see Section 5.2). Our network architecture is shown in Figure 4. The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The input image is split into YUV planes and passed to the network.

我们训练网络的权重，以最小化网络输出的转向命令与人类驾驶员的命令之间的均方误差，或偏心和旋转图像的调整转向命令(见第5.2节)。我们的网络架构如图4所示。该网络由9个层组成，包括一个归一化层、5个卷积层和3个完全连接层。输入图像被分割成YUV平面并传递到网络。

The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture and to be accelerated via GPU processing.

网络的第一层执行图像归一化。归一化器是硬编码的，在学习过程中不进行调整。在网络中执行归一化允许归一化方案随网络架构而改变，并通过GPU处理来加速。

The convolutional layers were designed to perform feature extraction and were chosen empirically through a series of experiments that varied layer configurations. We use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.

卷积层被设计为执行特征提取，并通过一系列改变层配置的实验根据经验进行选择。我们在前三个卷积层中使用2×2步长和5×5内核的跨步卷积，在最后两个卷积层使用3×3内核大小的非跨步卷积。

We follow the five convolutional layers with three fully connected layers leading to an output control value which is the inverse turning radius. The fully connected layers are designed to function as a controller for steering, but we note that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor and which serve as controller. 

我们用三个完全连接的层跟随五个卷积层，得到一个输出控制值，即反向转弯半径。完全连接的层被设计为用作转向控制器，但我们注意到，通过对系统进行端到端训练，不可能在网络的哪些部分主要用作特征提取器和哪些部分用作控制器之间进行彻底的区分。

## 5 Training Details
### 5.1 Data Selection
The first step to training a neural network is selecting the frames to use. Our collected data is labeled with road type, weather condition, and the driver’s activity (staying in a lane, switching lanes, turning, and so forth). To train a CNN to do lane following we only select data where the driver was staying in a lane and discard the rest. We then sample that video at 10 FPS. A higher sampling rate would result in including images that are highly similar and thus not provide much useful information. 

训练神经网络的第一步是选择要使用的帧。 我们收集的数据标有道路类型、天气状况和驾驶员的活动(留在车道上、换道、转弯等)。 为了训练 CNN 进行车道跟随，我们只选择驾驶员停留在车道上的数据并丢弃其余数据。 然后我们以 10 FPS 的速度对该视频进行采样。 较高的采样率会导致包括高度相似的图像，因此不会提供太多有用的信息。

Figure 4: CNN architecture. The network has about 27 million connections and 250 thousand parameters.
图 4：CNN 架构。 该网络有大约 2700 万个连接和 25 万个参数。

To remove a bias towards driving straight the training data includes a higher proportion of frames that represent road curves.
为了消除对直线行驶的偏见，训练数据包括更高比例的代表道路曲线的帧。

### 5.2 Augmentation
After selecting the final set of frames we augment the data by adding artificial shifts and rotations to teach the network how to recover from a poor position or orientation. The magnitude of these perturbations is chosen randomly from a normal distribution. The distribution has zero mean, and the standard deviation is twice the standard deviation that we measured with human drivers. Artificially augmenting the data does add undesirable artifacts as the magnitude increases (see Section 2). 

选择最后一组帧后，我们通过添加人工移位和旋转来增强数据，以教会网络如何从不良位置或方向中恢复。 这些扰动的大小是从正态分布中随机选择的。 该分布的均值为零，标准差是我们用人类司机测量的标准差的两倍。 随着幅度的增加，人为地增加数据确实会增加不需要的伪影(见第 2 节)。

## 6 Simulation
Before road-testing a trained CNN, we first evaluate the networks performance in simulation. A simplified block diagram of the simulation system is shown in Figure 5.

在对训练有素的 CNN 进行道路测试之前，我们首先评估模拟中的网络性能。 仿真系统的简化框图如图 5 所示。

The simulator takes pre-recorded videos from a forward-facing on-board camera on a human-driven data-collection vehicle and generates images that approximate what would appear if the CNN were, instead, steering the vehicle. These test videos are time-synchronized with recorded steering commands generated by the human driver. 

该模拟器从人类驾驶的数据收集车辆上的前向车载摄像头获取预先录制的视频，并生成近似于 CNN 驾驶车辆时会出现的图像。 这些测试视频与人类驾驶员生成的记录转向命令在时间上同步。

Since human drivers might not be driving in the center of the lane all the time, we manually calibrate the lane center associated with each frame in the video used by the simulator. We call this position the “ground truth”.

由于人类司机可能不会一直在车道中央行驶，因此我们手动校准与模拟器使用的视频中每一帧相关的车道中央。 我们称这个位置为“ground truth”。

The simulator transforms the original images to account for departures from the ground truth. Note that this transformation also includes any discrepancy between the human driven path and the ground truth. The transformation is accomplished by the same methods described in Section 2.

模拟器转换原始图像以解决与地面实况的偏差。 请注意，此转换还包括人类驱动路径与地面实况之间的任何差异。 转换是通过第 2 节中描述的相同方法完成的。

The simulator accesses the recorded test video along with the synchronized steering commands that occurred when the video was captured. The simulator sends the first frame of the chosen test video, adjusted for any departures from the ground truth, to the input of the trained CNN. The CNN then returns a steering command for that frame. The CNN steering commands as well as the recorded human-driver commands are fed into the dynamic model [8] of the vehicle to update the position and orientation of the simulated vehicle.

模拟器访问录制的测试视频以及捕获视频时发生的同步转向命令。 模拟器将所选测试视频的第一帧发送到训练有素的 CNN 的输入，并根据与地面实况的任何偏差进行调整。 然后 CNN 返回该帧的转向命令。 CNN 转向命令以及记录的人类驾驶员命令被输入到车辆的动态模型 [8] 中，以更新模拟车辆的位置和方向。

The simulator then modifies the next frame in the test video so that the image appears as if the vehicle were at the position that resulted by following steering commands from the CNN. This new image is then fed to the CNN and the process repeats.

然后，模拟器会修改测试视频中的下一帧，使图像看起来好像车辆处于遵循 CNN 转向命令所产生的位置。 然后将这个新图像馈送到 CNN 并重复该过程。

The simulator records the off-center distance (distance from the car to the lane center), the yaw, and the distance traveled by the virtual car. When the off-center distance exceeds one meter, a virtual human intervention is triggered, and the virtual vehicle position and orientation is reset to match the ground truth of the corresponding frame of the original test video.

模拟器记录偏心距离(汽车到车道中心的距离)、偏航和虚拟汽车行驶的距离。 当偏离中心距离超过一米时，触发虚拟人干预，并重置虚拟车辆位置和方向以匹配原始测试视频相应帧的地面实况。

Figure 5: Block-diagram of the drive simulator. 
图 5：驱动模拟器的框图。

## 7 Evaluation
Evaluating our networks is done in two steps, first in simulation, and then in on-road tests.

评估我们的网络分两步完成，首先是模拟，然后是道路测试。

In simulation we have the networks provide steering commands in our simulator to an ensemble of prerecorded test routes that correspond to about a total of three hours and 100 miles of driving in Monmouth County, NJ. The test data was taken in diverse lighting and weather conditions and includes highways, local roads, and residential streets.

在模拟中，我们让网络在我们的模拟器中向一组预先记录的测试路线提供转向命令，这些路线相当于在新泽西州蒙茅斯县总共行驶约 3 小时 100 英里。 测试数据是在不同的光照和天气条件下获取的，包括高速公路、地方道路和住宅街道。

### 7.1 Simulation Tests
We estimate what percentage of the time the network could drive the car (autonomy). The metric is determined by counting simulated human interventions (see Section 6). These interventions occur when the simulated vehicle departs from the center line by more than one meter. We assume that in real life an actual intervention would require a total of six seconds: this is the time required for a human to retake control of the vehicle, re-center it, and then restart the self-steering mode.

我们估计网络可以驱动汽车(自治)的时间百分比。 该指标是通过计算模拟的人工干预来确定的(见第 6 节)。 当模拟车辆偏离中心线超过一米时，就会发生这些干预。 我们假设在现实生活中，实际干预总共需要六秒：这是人类重新控制车辆、重新居中、然后重新启动自转向模式所需的时间。

We calculate the percentage autonomy by counting the number of interventions, multiplying by 6 seconds, dividing by the elapsed time of the simulated test, and then subtracting the result from 1: 

我们通过计算干预次数，乘以 6 秒，除以模拟测试的耗用时间，然后从 1 中减去结果来计算自主百分比：

$autonomy = (1 − \frac{(number\ of\ interventions) · 6 seconds } {elapsed\ time\ [seconds] } ) · 100 $ (1) 

Figure 6: Screen shot of the simulator in interactive mode. See Section 7.1 for explanation of the performance metrics. The green area on the left is unknown because of the viewpoint transformation. The highlighted wide rectangle below the horizon is the area which is sent to the CNN.
图 6：交互模式下模拟器的屏幕截图。 有关性能指标的说明，请参见第 7.1 节。 由于视点变换，左侧的绿色区域是未知的。 地平线下方突出显示的宽矩形是发送到 CNN 的区域。

Thus, if we had 10 interventions in 600 seconds, we would have an autonomy value of 

因此，如果我们在 600 秒内进行 10 次干预，我们的自主权值为

$ (1 − \frac{10 · 6}{600} ) · 100 = 90\% $

### 7.2 On-road Tests
After a trained network has demonstrated good performance in the simulator, the network is loaded on the DRIVETM PX in our test car and taken out for a road test. For these tests we measure performance as the fraction of time during which the car performs autonomous steering. This time excludes lane changes and turns from one road to another. For a typical drive in Monmouth County NJ from our office in Holmdel to Atlantic Highlands, we are autonomous approximately 98% of the time. We also drove 10 miles on the Garden State Parkway (a multi-lane divided highway with on and off ramps) with zero intercepts.

经过训练的网络在模拟器中表现出良好的性能后，将网络加载到我们测试车的 DRIVETM PX 上，并进行道路测试。 对于这些测试，我们将性能衡量为汽车执行自主转向的时间分数。 这次不包括变道和从一条道路转向另一条道路。 对于在新泽西州蒙茅斯县从我们位于 Holmdel 的办公室到大西洋高地的典型驾车，我们大约 98% 的时间都是自主驾驶的。 我们还在 Garden State Parkway(一条带进出匝道的多车道分开的高速公路)上以零拦截行驶了 10 英里。

A video of our test car driving in diverse conditions can be seen in [9].

在 [9] 中可以看到我们的测试车在不同条件下行驶的视频。

### 7.3 Visualization of Internal CNN State 内部CNN状态的可视化
Figures 7 and 8 show the activations of the first two feature map layers for two different example inputs, an unpaved road and a forest. In case of the unpaved road, the feature map activations clearly show the outline of the road while in case of the forest the feature maps contain mostly noise, i. e., the CNN finds no useful information in this image.

图 7 和图 8 显示了前两个特征图层针对两个不同样本输入(一条未铺砌的道路和一片森林)的激活。 在未铺砌道路的情况下，特征图激活清楚地显示了道路的轮廓，而在森林的情况下，特征图主要包含噪声，i。 即，CNN 在此图像中找不到有用的信息。

This demonstrates that the CNN learned to detect useful road features on its own, i. e., with only the human steering angle as training signal. We never explicitly trained it to detect the outlines of roads, for example. 

这表明 CNN 学会了自己检测有用的道路特征，即。 e.，仅以人类转向角作为训练信号。 例如，我们从未明确训练过它来检测道路轮廓。

Figure 7: How the CNN “sees” an unpaved road. Top: subset of the camera image sent to the CNN. Bottom left: Activation of the first layer feature maps. Bottom right: Activation of the second layer feature maps. This demonstrates that the CNN learned to detect useful road features on its own, i. e., with only the human steering angle as training signal. We never explicitly trained it to detect the outlines of roads.
图 7：CNN 如何“看到”未铺砌的道路。 顶部：发送到 CNN 的相机图像的子集。 左下：激活第一层特征图。 右下：激活第二层特征图。 这表明 CNN 学会了自己检测有用的道路特征，即。 e.，仅以人类转向角作为训练信号。 我们从未明确训练过它来检测道路轮廓。

Figure 8: Example image with no road. The activations of the first two feature maps appear to contain mostly noise, i. e., the CNN doesn’t recognize any useful features in this image. 
图 8：没有道路的样本图像。 前两个特征图的激活似乎主要包含噪声，i。 例如，CNN 没有识别出这张图片中的任何有用特征。

## 8 Conclusions
We have empirically demonstrated that CNNs are able to learn the entire task of lane and road following without manual decomposition into road or lane marking detection, semantic abstraction, path planning, and control. A small amount of training data from less than a hundred hours of driving was sufficient to train the car to operate in diverse conditions, on highways, local and residential roads in sunny, cloudy, and rainy conditions. The CNN is able to learn meaningful road features from a very sparse training signal (steering alone).

我们已经凭经验证明，CNN 能够学习车道和道路跟随的整个任务，而无需手动分解为道路或车道标记检测、语义抽象、路径规划和控制。 来自不到一百小时驾驶的少量训练数据足以训练汽车在晴天、阴天和雨天的高速公路、地方和居民区道路等各种条件下运行。 CNN 能够从非常稀疏的训练信号(仅转向)中学习有意义的道路特征。

The system learns for example to detect the outline of a road without the need of explicit labels during training.

例如，系统学习在训练期间无需显式标签即可检测道路轮廓。

More work is needed to improve the robustness of the network, to find methods to verify the robustness, and to improve visualization of the network-internal processing steps.

需要做更多的工作来提高网络的稳健性，找到验证稳健性的方法，并改进网络内部处理步骤的可视化。

## References
1. Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1(4):541–551, Winter 1989. URL: http://yann.lecun.org/exdb/publis/pdf/lecun-89e.pdf.
2. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 1097–1105. Curran Associates, Inc., 2012. URL: http://papers.nips.cc/paper/ 4824-imagenet-classification-with-deep-convolutional-neural-networks. pdf.
3. L. D. Jackel, D. Sharman, Stenard C. E., Strom B. I., , and D Zuckert. Optical character recognition for self-service banking. AT&T Technical Journal, 74(1):16–24, 1995.
4. Large scale visual recognition challenge (ILSVRC). URL: http://www.image-net.org/ challenges/LSVRC/.
5. Net-Scale Technologies, Inc. Autonomous off-road vehicle control using end-to-end learning, July 2004. Final technical report. URL: http://net-scale.com/doc/net-scale-dave-report.pdf.
6. Dean A. Pomerleau. ALVINN, an autonomous land vehicle in a neural network. Technical report, Carnegie Mellon University, 1989. URL: http://repository.cmu.edu/cgi/viewcontent. cgi?article=2874&context=compsci.
7. Wikipedia.org. DARPA LAGR program. http://en.wikipedia.org/wiki/DARPA_LAGR_ Program.
8. Danwei Wang and Feng Qi. Trajectory planning for a four-wheel-steering vehicle. In Proceedings of the 2001 IEEE International Conference on Robotics & Automation, May 21–26 2001. URL: http: //www.ntu.edu.sg/home/edwwang/confpapers/wdwicar01.pdf.
9. DAVE 2 driving a lincoln. URL: https://drive.google.com/open?id= 0B9raQzOpizn1TkRIa241ZnBEcjQ. 9
