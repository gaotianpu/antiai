# H-Net: Unsupervised Attention-based Stereo Depth Estimation Leveraging
H-Net：基于注意力的无监督立体深度估计 https://arxiv.org/abs/2104.11288

## Abstract
Depth estimation from a stereo image pair has become one of the most explored applications in computer vision, with most of the previous methods relying on fully supervised learning settings. However, due to the difficulty in acquiring accurate and scalable ground truth data, the training of fully supervised methods is challenging. As an alternative, self-supervised methods are becoming more popular to mitigate this challenge. In this paper, we introduce the H-Net, a deep-learning framework for unsupervised stereo depth estimation that leverages epipolar geometry to refine stereo matching. For the first time, a Siamese autoencoder architecture is used for depth estimation which allows mutual information between the rectified stereo images to be extracted. To enforce the epipolar constraint, the mutual epipolar attention mechanism has been designed which gives more emphasis to correspondences of features which lie on the same epipolar line while learning mutual information between the input stereo pair. Stereo correspondences are further enhanced by incorporating semantic information to the proposed attention mechanism. More specifically, the optimal transport algorithm is used to suppress attention and eliminate outliers in areas not visible in both cameras. Extensive experiments on KITTI2015 and Cityscapes show that our method outperforms the state-ofthe-art unsupervised stereo depth estimation methods while closing the gap with the fully supervised approaches.

立体图像对的深度估计已成为计算机视觉中探索最多的应用之一，之前的大多数方法都依赖于完全监督的学习设置。 然而，由于难以获取准确且可扩展的基准实况数据，全监督方法的训练具有挑战性。 作为替代方案，自监督方法正变得越来越流行，以缓解这一挑战。 在本文中，我们介绍了 H-Net，这是一种用于无监督立体深度估计的深度学习框架，它利用对极几何来改进立体匹配。 孪生自动编码器架构首次用于深度估计，允许提取校正后的立体图像之间的互信息。 为了加强对极约束，设计了相互对极注意机制，在学习输入立体对之间的互信息时，它更加强调位于同一对极线上的特征的对应关系。 通过将语义信息合并到所提出的注意机制中，立体对应得到进一步增强。 更具体地说，最优传输算法用于抑制注意力并消除两个相机不可见区域中的异常值。 在 KITTI2015 和 Cityscapes 上进行的大量实验表明，我们的方法优于最先进的无监督立体深度估计方法，同时缩小了与完全监督方法的差距。

<!--
深度估计的输出：？
stereo image pair  立体图像对, 双目摄像头？
leverages epipolar geometry 对极几何?  相互对极注意机制
Epipolar line 对极线, 
optimal transport algorithm 最优传输算法

双目摄像头？

监督方式，也需要了解下？

深度图，图像、激光雷达等融合后中间态？

法线图和法线估计

多模融合的深度估计？

-->

## 1. Introduction
Humans are remarkably capable of inferring the 3D structure of a real world scene even over short timescales. For example, when navigating along a street, we are able to locate obstacles and vehicles in motion and avoid them with a fast response time. Years of substantial interest in geometric computer vision has not accomplished comparable modeling capabilities to humans for various real-world scenes where reflections, occlusions, non-rigidity and textureless areas exist. So what can human ability be attributed to? A central concept is that humans learn the regularities of the world while interacting with it, moving around, and observing vast quantities of scenes. Consequently, we develop a rich, consistent and structural understanding of the world, which is utilized when we perceive a new scene. Our binocular vision is a supporting feature, from which the brain can not only build disparity maps, but can also combine to obtain structural information. These two ideas prompt one of the fundamental problems in computer vision — depth estimation — whose quality has a direct influence on various application scenarios, such as autonomous driving, robotic navigation, augmented reality and 3D reconstruction.

即使在很短的时间内，人类也非常有能力推断真实世界场景的 3D 结构。 例如，在街道上行驶时，我们能够定位障碍物和行驶中的车辆，并以快速的响应时间避开它们。 多年来对几何计算机视觉的浓厚兴趣尚未实现对存在反射、遮挡、非刚性和无纹理区域的各种现实世界场景的与人类相当的建模能力。 那么人的能力可以归因于什么呢？ 一个核心概念是人类在与世界互动、四处走动和观察大量场景的同时学习世界的规律性。 因此，我们对世界形成了丰富的、一致的和结构化的理解，当我们感知一个新场景时，它会被利用。 我们的双目视觉是一个辅助特征，大脑不仅可以从中构建视差图，还可以组合以获得结构信息。 这两个想法引发了计算机视觉中的一个基本问题 —— 深度估计 —— 其质量直接影响到各种应用场景，例如自动驾驶、机器人导航、增强现实和 3D 重建。

Thanks to advanced deep learning techniques, the performance of depth estimation methods has improved significantly over the last few years. Most previous work relies on ground-truth depth data and considers deep architectures for generating depth maps in a supervised manner [5, 17, 26, 39]. However, collecting vast and varied training datasets with accurate per-pixel ground truth depth data for supervised learning is a formidable challenge. To overcome this limitation, some recent works have shown that self-supervised methods are instead able to effectively tackle the depth estimation task [10] [33]. We are particularly inspired by the approaches proposed in [11, 45, 8, 15] where they took view synthesis as supervisory signals to train the network and exploited differences between the original input, synthesized view, and left and right disparity results as penalties (i.e. a photometric image reconstruction cost, a left-right consistency cost and a disparity smoothness cost) to force the system to generate accurate disparity maps. However, although some works have tried to emphasize the complementary information in the stereo image pair and used sharing weights when extracted features from input images [33] [2], the contextual information between the multiple views — especially some strong feature matches — lie on the epipolar line, and this information has not been effectively explored and exploited.

得益于先进的深度学习技术，深度估计方法的性能在过去几年中有了显著提高。 大多数以前的工作都依赖于基准实况深度数据，并考虑以监督方式生成深度图的深度架构 [5、17、26、39]。 然而，为监督学习收集大量多样的训练数据集和准确的每像素地面真实深度数据是一项艰巨的挑战。 为了克服这一限制，最近的一些工作表明，自监督方法反而能够有效地解决深度估计任务 [10] [33]。 我们特别受到 [11、45、8、15] 中提出的方法的启发，他们将视图合成作为监督信号来训练网络，并利用原始输入、合成视图和左右视差结果之间的差异作为惩罚( 即光度图像重建成本、左右一致性成本和视差平滑成本)以迫使系统生成准确的视差图。 然而，尽管一些工作试图强调立体图像对中的互补信息，并在从输入图像中提取特征时使用共享权重 [33] [2]，但多个视图之间的上下文信息 —— 尤其是一些强特征匹配 —— 取决于对极线，并且此信息尚未得到有效探索和利用。

In this paper, we follow the unsupervised learning setting and introduce the H-Net, a novel end-to-end trainable network for depth estimation given rectified stereo image pairs. The proposed H-Net effectively fuses the information in the stereo pair and combines epipolar geometry with learningbased depth estimation approaches. In summary, our main contributions in this paper are: 
* We design a Siamese encoder-Siamese decoder network architecture, which fuses the complementary information in the stereo image pairs while enhancing the communication between them. To the best of our knowledge, this is the first time this architecture is used for depth estimation. 
* We propose a mutual epipolar attention module to enforce the epipolar constraint in feature matching and emphasise the strong relationship between the features located along the same epipolar lines in rectified stereo image pairs. 
* We further enhance the proposed attention module by using the optimal transport algorithm to incorporate in a novel fashion semantic information and filter out outlier feature correspondences.

在本文中，我们遵循无监督学习设置并介绍了 H-Net，这是一种新颖的端到端可训练网络，用于在给定校正立体图像对的情况下进行深度估计。 所提出的 H-Net 有效地融合了立体对中的信息，并将对极几何与基于学习的深度估计方法相结合。 总之，我们在本文中的主要贡献是：
* 我们设计了一个 孪生编码器-解码器 网络架构，它融合了立体图像对中的互补信息，同时增强了它们之间的通信。 据我们所知，这是该架构首次用于深度估计。
* 我们提出了一个相互对极注意模块来强制特征匹配中的对极约束，并强调在校正立体图像对中位于相同对极线上的特征之间的强关系。
* 我们通过使用最优传输算法以新的方式结合语义信息并过滤出离群特征对应，进一步增强了所提出的注意力模块。

We demonstrate the effectiveness of our approach on the challenging KITTI [9] and Cityscapes datasets [3]. Compared to previous approaches, the H-Net achieves state-ofthe-art results.

我们证明了我们的方法在具有挑战性的 KITTI [9] 和 Cityscapes 数据集 [3] 上的有效性。 与以前的方法相比，H-Net 取得了最先进的结果。

## 2. Related work
Estimating depth maps from stereo images has been explored for decades [1]. Accurate stereo depth estimation plays a critical role in perceiving the 3D geometric configuration of scenes and facilitating a variety of computer vision applications in the real world [16]. Recent work has shown that depth estimation from a stereo image pair can be effectively tackled by learning-based methods with convolutional neural networks (CNNs) [2].

从立体图像估计深度图已经探索了几十年 [1]。 准确的立体深度估计在感知场景的 3D 几何配置和促进现实世界中的各种计算机视觉应用方面起着关键作用 [16]。 最近的工作表明，通过使用卷积神经网络 (CNN) [2] 的基于学习的方法可以有效地解决立体图像对的深度估计问题。

### 2.1. Supervised Depth Estimation
A pyramid stereo matching network was proposed in [2], where spatial pyramid pooling and dilated convolution were adopted to enlarge the receptive fields, while a stacked hourglass CNN was designed to further boost the utilization of global context information. Duggal et al. [4] proposed a differentiable PatchMatch module to abandon most disparities without requiring full cost volume evaluation, and thus the specific range to prune for each pixel could be learned. Kusupati et al. [20] improved the depth quality by leveraging the predicted normal maps and a normal estimation model, and proposed a new consistency loss to refine the depths from depth/normal pairs.

[2] 提出了金字塔立体匹配网络，采用空间金字塔池化和扩张卷积来扩大感受野，同时设计了堆叠沙漏 CNN 以进一步提高全局上下文信息的利用率。 Duggal et al. [4] 提出了一个可区分的 PatchMatch 模块来放弃大多数差异而不需要完整的成本量评估，因此可以了解每个像素修剪的具体范围。 Kusupati et al. [20] 通过利用预测的法线图和法线估计模型提高了深度质量，并提出了一种新的一致性损失来细化深度/法线对的深度。

The above methods are fully supervised and rely on having large amounts of accurate ground truth depth for training. However, this is challenging to obtain data in various real-world settings [44]. Synthetic training data is a potential solution [7] [28], but still requires manual curation for every new application scenario.

上述方法是完全监督的，并且依赖于大量准确的基准实况深度进行训练。 然而，要在各种现实环境中获取数据具有挑战性 [44]。 合成训练数据是一种潜在的解决方案 [7] [28]，但仍需要针对每个新的应用场景进行手动管理。

### 2.2. Unsupervised Depth Estimation
Due to the lack of per-pixel ground truth depth data, selfsupervised depth estimation is an alternative, where image reconstruction is the supervisory signal during training [11]. The input for this type of model is usually a set of images, either as stereo pairs [33][32] or as monocular sequences [45] [15].

由于缺乏每像素基准实况深度数据，自监督深度估计是一种替代方法，其中图像重建是训练期间的监督信号 [11]。 这种类型的模型的输入通常是一组图像，可以是立体对 [33][32] 也可以是单目序列 [45] [15]。

Gard et al. [8] proposed an approach using a calibrated stereo camera pair setup for unsupervised monocular depth estimation, in which depth was generated as an intermediate output and the supervision signal came from the reconstruction combining the counterpart image in a stereo pair. Godard et al. [10] extended this work by using forward and backward reconstructions of different image views while adding an appearance matching loss and multi-scale loss to the model. Per-pixel minimum reprojection loss and automasking were explored in [11], which allowed the network to ignore objects moving at the same velocity as the camera or frames captured when the camera was static, with further improved results. Johnston et al. [15] introduced discrete disparity prediction and applied self-attention to the depth estimation framework, providing a more robust and sharper depth estimation map.

Gard et al. [8] 提出了一种使用校准的<strong>立体相机对</strong>设置进行无监督单目深度估计的方法，其中深度作为中间输出生成，监督信号来自结合立体对中的对应图像的重建。 Godard et al. [10] 通过使用不同图像视图的前向和后向重建来扩展这项工作，同时向模型添加外观匹配损失和多尺度损失。 在[11]中探索了每像素最小重投影损失和自动掩码，这允许网络忽略以与相机相同速度移动的物体或相机静止时捕获的帧，从而进一步改善结果。 Johnston et al. [15] 引入了离散视差预测并将自注意力应用于深度估计框架，提供了更稳健和更清晰的深度估计图。

It has been shown that training with an added binocular color image could help single image depth estimation by posing it as an image reconstruction problem without requiring ground truth [10] [11]. Andrea et al. [33] showed that the depth estimation results could be effectively improved within an adversarial learning framework, with a deep generative network that learned to predict the disparity map for a calibrated stereo camera using a wrapping operation.

已经表明使用添加的双目彩色图像进行训练可以通过将其作为图像重建问题来帮助单图像深度估计，而无需基准实况 [10] [11]。 Andrea et al. [33] 表明，深度估计结果可以在对抗性学习框架内得到有效改善，深度生成网络学习使用包装操作预测校准立体相机的视差图。

In the multi-view (stereo) depth estimation task, it is naturally to employ complementary features from different views to establish the geometric correspondences. Zhou et al. [44] presented a framework that learned stereo matching costs without human supervision, in which the network parameters were updated in an iterative manner and a left-right check was applied to guide the training procedure. Joung et al. [16] proposed a framework to compute matching cost in an unsupervised setting, where the putative positive samples in every training iteration were selected by exploiting the correspondence consistency between two stereo images.

在多视图(立体)深度估计任务中，自然会使用来自不同视图的互补特征来建立几何对应关系。 Zhou et al. [44] 提出了一个在没有人工监督的情况下学习立体匹配成本的框架，其中网络参数以迭代方式更新，并应用左右检查来指导训练过程。 Joung et al. [16] 提出了一个在无监督环境中计算匹配成本的框架，其中通过利用两个立体图像之间的对应一致性来选择每次训练迭代中的假定正样本。

Although these methods tried to explore the feature relationship between the stereo images, the concrete matching matrix have not been effectively exploited and been applied to the learning procedure, which leads to a cost of details and a waste of geometric information, especially the strong constraints on the epipolar line.

尽管这些方法试图探索立体图像之间的特征关系，但具体的匹配矩阵尚未被有效利用并应用于学习过程，这导致了细节成本和几何信息的浪费，尤其是对极线的强约束。

## 3. Method
Here, we describe the details of the process of depth prediction using the proposed H-Net.

在这里，我们描述了使用所提出的 H-Net 进行深度预测的过程的细节。

### 3.1. H-Net architecture
In this paper, the encoder-decoder structure Monodepth2 [11] was adopted as the fundamental backbone, based on the U-Net [35]. As shown in Fig. 1, the proposed architecture consisted of a double-branch encoder and a doublebranch decoder. To make the network compact, similar to [2] and [33], a Siamese Encoder - Siamese Decoder (SESD) structure was designed with shared weights between the two branches in both the encoder and the decoder. To our knowledge, this is the first time a SE-SD is used for stereo depth estimation enabling the extraction of mutual information from the pair of input images.

在本文中，基于 U-Net [35] 的编码器-解码器结构 Monodepth2 [11] 被用作基本骨干。 如图1 所示，所提出的架构由一个双分支编码器和一个双分支解码器组成。 为了使网络紧凑，类似于 [2] 和 [33]，设计了孪生编码器-解码器 (SESD) 结构，在编码器和解码器的两个分支之间共享权重。 据我们所知，这是第一次将 SE-SD 用于立体深度估计，从而能够从一对输入图像中提取互信息。

Figure 1. The H-Net architecture. 

The Siamese Encoder (SE) of H-net included two branches of Resnet18 [13] with shared trainable parameters. The left and right rectified images $I^l$,$I^r ∈ R^{3×h_0×w_0}$ were fed into each branch of the SE to extract common features from the input images, where $h_0$, $w_0$ denotes the image size. The outputs of the three deeper Residual-downsampling (Res-down) blocks in the SE were interconnected with a novel mutual attention block proposed in this work — the so-called Optimal Transport-based Mutual Epipolar Attention (OT-MEA) block, shown in Fig. 2 and explained in detail below.

H-net 的 孪生编码器(SE) 包括 Resnet18 [13] 的两个分支，它们具有共享的可训练参数。 左右校正后的图像 $I^l$,$I^r ∈ R^{3×h_0×w_0}$ 被送入 SE 的每个分支以从输入图像中提取共同特征，其中 $h_0$, $ w_0$ 表示图像大小。 SE 中三个更深的残差下采样 (Res-down) 块的输出与本工作中提出的一种新颖的相互注意块互连——所谓的基于最优传输的相互对极注意 (OT-MEA) 块，如图所示 在图 2 中，并在下面详细解释。

Figure 2. Optimal Transport based Mutual Epipolar Attention (OT-MEA) block combines OT retrieving (Eq. 4) into the MEA module (Eq. 1). 
图 2. 基于最佳传输的相互对极注意 (OT-MEA) 块将 OT 检索(方程式 4)组合到 MEA 模块(方程式 1)中。

The abstract latent features from the encoder were fused in the middle part by concatenating the feature maps extracted from each SE block between the two branches. Each concatenated map is then convolved by two separate convolution layers with different trainable parameters.

通过连接从两个分支之间的每个 SE 块提取的特征图，来自编码器的抽象潜在特征在中间部分融合。 然后，每个连接的映射由两个具有不同可训练参数的独立卷积层进行卷积。

The decoder took the fused latent features as inputs and generated sigmoid outputs for each input image similar to [10] and [11]. It was composed of the same number of Residual-up-sampling (Res-up) blocks as Res-down to recover the full resolution, as well as OT-MEA blocks inserted in the first three Res-up blocks. Each sigmoid output Ω of the decoder was transformed to scene depth as D = 1/(aΩ + b). The parameters a and b were selected to constrain depth D between 0.1 and 100 units.

解码器将融合的潜在特征作为输入，并为每个输入图像生成类似于 [10] 和 [11] 的 sigmoid 输出。 它由与 Res-down 相同数量的 Residual-up-sampling (Res-up) 块组成以恢复全分辨率，以及插入前三个 Res-up 块中的 OT-MEA 块。 解码器的每个 sigmoid 输出 Ω 被转换为场景深度 D = 1/(aΩ + b)。 选择参数 a 和 b 以将深度 D 限制在 0.1 到 100 个单位之间。

### 3.2. Mutual Epipolar Attention 相互对极注意
State-of-the-art deep learning methods for stereo depth estimation have not considered the epipolar constraint when estimating feature correspondences. In this work, we introduce a mutual attention mechanism to give more emphasis to features correspondences which lie on the same epipolar line.

用于立体深度估计的最先进的深度学习方法在估计特征对应时没有考虑极线约束。 在这项工作中，我们引入了一种相互关注机制，以更加强调位于同一对极线上的特征对应关系。

Recently, Wang et al. [38] proposed the Non-Local (NL) block which allowed them to exploit global-range attention in an image sequence. This was then extended with the introduction of the Mutual NL (MNL) block [43] to explore the mutual relationships between different inputs in multiview vision. However, global-range feature matching in the NL and MNL blocks suffers from the high number of parameters, memory requirement and training time. Furthermore, these blocks can be misled by repeated textures in the scenes.

最近，Wang et al. [38] 提出了非局部(NL)块，使他们能够利用图像序列中的全局范围注意力。 然后通过引入相互 NL (MNL) 块 [43] 来扩展，以探索多视图视觉中不同输入之间的相互关系。 然而，NL 和 MNL 块中的全局范围特征匹配受到大量参数、内存要求和训练时间的影响。 此外，这些块可能会被场景中重复的纹理所误导。



To overcome the above limitations, we designed the Mutual Epipolar Attention (MEA) module to constrain feature correspondences to the same epipolar line between a pair of rectified stereo images. Here, MEA was defined as: 

为了克服上述限制，我们设计了 Mutual Epipolar Attention (MEA) 模块来限制特征对应到一对校正立体图像之间的相同对极线。 在这里，MEA 被定义为：

$Y^{l→r} := Ψ(X^l) ⊗ Φ(X^l, X^r) $ 

$Y^{r→l} := Ψ(X^r) ⊗ Φ(X^r, X^l) $ (1)

where ⊗ denotes the batch matrix multiplication, $X^l$, $X^r ∈ R^{h×c×w}$ denote the transported and reshaped input signals from the two branches, $Y^{l→r}$, $Y^{l→l}$ ∈ Rh×c×w are the output signals from the MEA block. Φ : $R^{h×c×w}×R^{h×c×w} → R^{h×w×w},(X^1, X^2) → M^{1→2}$ is a pair-wise matching function, the so called retrieval function, which evaluates the compatibility between the two inputs. Ψ : $R^{h×c×w} → R^{h×c×w}, X → V$ is a unary function which maps vectors from one feature space to another which is essential for fusion.

其中 ⊗ 表示批量矩阵乘法，$X^l$，$X^r ∈ R^{h×c×w}$ 表示来自两个分支的传输和整形输入信号，$Y^{l→r}$ , $Y^{l→l}$ ∈ Rh×c×w 是 MEA 块的输出信号。 Φ : $R^{h×c×w}×R^{h×c×w} → R^{h×w×w},(X^1, X^2) → M^{1→2} $ 是一个成对匹配函数，即所谓的检索函数，它评估两个输入之间的兼容性。 Ψ : $R^{h×c×w} → R^{h×c×w}, X → V$ 是一元函数，它将向量从一个特征空间映射到另一个特征空间，这对于融合至关重要。

Following the settings in [38], the Embedded Gaussian (EG) similarity representation was used to define our matching function:

按照 [38] 中的设置，嵌入式高斯 (EG) 相似性表示用于定义我们的匹配函数：

$Φ_{EG}(X^1, X^2 ) := softmax(C_1(X^1)> ⊗ \ C_2(X^2 ))$ (2) 

where C is the 1 × 1 convolution, and was also used in the unary function for vector mapping:

其中 C 是 1 × 1 卷积，也用于矢量映射的一元函数：

Ψ := C (3)

In our experimental work, the EG-based MEA and MNL modules were compared and denoted as EG-MEA and EGMNL, respectively.

在我们的实验工作中，比较了基于 EG 的 MEA 和 MNL 模块，分别表示为 EG-MEA 和 EGMNL。

### 3.3. Optimal transport based mutual attention
In stereo vision, the input images have been captured from cameras at different positions and view angles making the field of view of the two images sightly different. This can cause outliers in depth estimation due to incorrect feature correspondences in the areas which are not visible to both cameras. To eliminate outliers in these areas, we further enhanced our proposed MEA module to suppress the contribution of correspondences in these occluded areas during feature matching. The EG similarity representation defined in Eq.(2) cannot achieve this because all the areas of the input signals are equally considered.

在立体视觉中，输入图像是从不同位置和视角的相机捕获的，这使得两个图像的视野略有不同。 由于两个相机都看不到的区域中的特征对应不正确，这可能会导致深度估计出现异常值。 为了消除这些区域中的异常值，我们进一步增强了我们提出的 MEA 模块，以抑制特征匹配期间这些遮挡区域中的对应关系的贡献。 等式（2）中定义的 EG 相似性表示无法实现这一点，因为输入信号的所有区域都被同等考虑。

For this purpose, we formulated the matching task in Eq.(1) as an optimal transport (OT) problem as it has already been proven that OT improves semantic correspondence [24]. Thus, a new OT-based retrieval function is further proposed, tailored to our stereo depth estimation problem:

为此，我们将等式 (1) 中的匹配任务表述为最佳传输 (OT) 问题，因为已经证明 OT 可以改善语义对应 [24]。 因此，进一步提出了一种新的基于 OT 的检索函数，专门针对我们的立体深度估计问题：

$Φ_{OT}(X^1, X^2 ) := arg min M k M  e1−C01(X1)> ⊗C02(X2)k 1 s.t. u ⊗ M = Θ(X2), u ⊗ M> = Θ(X^1) $ (4) 

where  denotes a Hadamard product, C0 is a sequence operation of convolution and channel-wise Euclidean normalization, u ∈ {1}h×1×w is a matrix with all elements equal to 1. Θ : Rh×c×w → Rh×1×w, X 7→ U is the sequence operation of convolution, ReLU activation and pixel-wise L1-normalization to generate the transported mass of pixels U. The matrix M is the variable to be optimised and represents the optimal matching matrix M1→2.

其中 表示 Hadamard 乘积，C0 是卷积和逐通道欧几里德归一化的序列运算，u ∈ {1}h×1×w 是一个所有元素都等于 1 的矩阵。 Θ : Rh×c×w → Rh× 1×w, X 7→ U 是卷积、ReLU 激活和像素级 L1 归一化的序列操作，以生成像素 U 的传输质量。矩阵 M 是要优化的变量，表示最佳匹配矩阵 M1→ 2.

Here, OT-based matching in Eq. (4) assigns to each pixel the sum of each column of the similarity weights in matching matrix M1→2 , which is constrained by the mass: 

在这里，方程式中基于 OT 的匹配。 (4) 为每个像素分配匹配矩阵 M1→2 中每列相似性权重的总和，受质量约束：

U1 ij = P k M1→2 ijk 

U2 ik = P j M1→2 ijk , 

∀i, j, k ∈ Z, i ≤ h, j, k ≤ w (5) 

where U1 ij , U2 ik and M1→2 ijk are the elements of the U1, U2 and M1→2 respectively indexed by i, j, k. In contrast to the equal consideration by EG-based matching in Eq. (2), varying weights are assigned to different correspondences in Eq. (5), determined by the latent semantic messages forwarded from the input signals. This enables the OT module to suppress the outliers and focus on correspondences with more mass which lie on the semantic areas.

其中U1 ij 、U2 ik 和M1→2 ijk 分别是U1、U2 和M1→2 中分别由i、j、k 索引的元素。 与等式中基于 EG 的匹配的同等考虑相反。 （2），不同的权重被分配给等式中的不同对应关系。 （5），由输入信号转发的潜在语义信息决定。 这使 OT 模块能够抑制异常值，并专注于语义区域上具有更多质量的对应关系。

In this paper, since Eq. (4) is a convex optimization problem, the Sinkhorn algorithm is used to obtain the numerical solution of this OT problem [24]. OT matching based MEA is denoted as OT-MEA and Fig. 2 illustrates the implementation sketch of the OT-MEA used in H-net. Both MEA and OT modules can be used separately or in combination and we present their impact with an ablation study in Section 5.2. OT-MEA was also compared in our experimental work to the OT matching based MNL, denoted as OT-MNL.

在本文中，由于方程式。 (4) 是一个凸优化问题，采用 Sinkhorn 算法求得该 OT 问题的数值解[24]。 基于OT匹配的MEA表示为OT-MEA，图2说明了在H-net中使用的OT-MEA的实现示意图。 MEA 和 OT 模块都可以单独使用或组合使用，我们在第 5.2 节中通过消融研究介绍了它们的影响。 在我们的实验工作中还将 OT-MEA 与基于 OT 匹配的 MNL 进行了比较，表示为 OT-MNL。

### 3.4. Self-Supervised Training
For the left and right input images Il,Ir ∈ R3×h0×w0 , the sigmoid outputs of the H-Net were transformed to depth maps Dl, Dr ∈ R1×h0×w0 as explained in Section 3.1. By combining one of the depth maps (e.g Dl ) and the countpart input image (Ir ), we were able to reconstruct the initial image (Il∗ ) using the re-projection sampler [14]. Here we used the left image Il as an example to present the supervisory signal and loss components. The final loss function included the loss terms for both left and right images. The similarity between the input image Il and the reconstructed image Il∗ provides our supervisory signal. Our photometric error function Ll ap was defined as the combination of L1- norm and structural similarity index (SSIM) [11]:

对于左右输入图像 Il,Ir ∈ R3×h0×w0，H-Net 的 sigmoid 输出被转换为深度图 Dl, Dr ∈ R1×h0×w0，如第 3.1 节所述。 通过组合深度图之一（例如 Dl）和对应输入图像（Ir），我们能够使用重投影采样器 [14] 重建初始图像（Il*）。 这里我们以左图 Il 为例来介绍监控信号和损耗组件。 最终的损失函数包括左右图像的损失项。 输入图像 Il 和重建图像 Il* 之间的相似性提供了我们的监督信号。 我们的光度误差函数 Ll ap 被定义为 L1 范数和结构相似性指数 (SSIM) [11] 的组合：

Ll ap = 1N X i,j γ2 (1−SSIM(Il ij , Il∗ ij ))+(1−γ)k Il ij − Il∗ ij k 1 (6) 

where, N denotes the number of pixels and γ is the weighting for L1-norm loss term. To improve the predictions around object boundaries, an edge-aware smoothness term Lds was applied [11, 15]:

其中，N 表示像素数，γ 是 L1 范数损失项的权重。 为了改进围绕对象边界的预测，应用了边缘感知平滑项 Lds [11, 15]：

Ll ds = 1N X i,j |∂x(dl∗ ij )|e −|∂xIl ij | + |∂y(dl∗ ij )|e −|∂yIl ij | (7) 

where dl∗ = dl√dl represents the mean-normalized inverse of depth (1/D) which aims at preventing shrinking of the depth prediction [37].

其中 dl* = dl√dl 表示深度的均值归一化倒数 (1/D)，旨在防止深度预测的收缩 [37]。

To overcome the gradient locality of the re-projection sampler, we adopted the multi-scale estimation method presented in [11], which first upsamples the low resolution depth maps (from the intermediate layers) to the input image resolution and then reprojects and resamples them. The errors were computed at the higher input resolution. Finally, the photometric loss and per-pixel smoothness loss were balanced by the smoothness term λ and the total loss was averaged over each scale (s), branch (left and right) and batch:

为了克服重投影采样器的梯度局部性，我们采用了[11]中提出的多尺度估计方法，该方法首先将低分辨率深度图（来自中间层）上采样到输入图像分辨率，然后再投影和重采样 他们。 误差是在较高的输入分辨率下计算的。 最后，光度损失和每像素平滑度损失由平滑度项 λ 平衡，总损失在每个尺度 (s)、分支（左和右）和批次上平均：

Ltotal = 12m mXs=1 (Lls + Lrs) = 

12m mXs=1  (Ll ap + λLl ds) + (Lr ap + λLr ds)) (8)

## 4. Experiments
We trained and evaluated the proposed H-Net on the KITTI2015 [9] with the full Eigen and Eigen split dataset [5]. For the full Eigen setting, there were 22600 pairs for training and 888 for validation while for Eigen split, there were 19905 training pairs and 2212 validation pairs. The same intrinsics were used for all images. The principal point of the camera was set to the image center and the focal length was the average of all the focal lengths in KITTI. All of the images were rectified and the transformation between the two stereo images was set to be a pure horizontal translation of fixed length. During the evaluation, only depths up to a fixed range of 80m were evaluated per standard practice [5, 8, 10, 11]. As our backbone model, we used Monodepth2 [11] and kept the original ResNet18 [13] as the encoder. Furthermore, we also trained and tested our H-Net on the Cityscapes dataset [3] to verify its generalisability.

我们在 KITTI2015 [9] 上使用完整的 Eigen 和 Eigen split 数据集 [5] 训练和评估了所提出的 H-Net。 对于完整的 Eigen 设置，有 22600 对用于训练，888 对用于验证，而对于 Eigen split，有 19905 对训练对和 2212 对验证。 所有图像都使用了相同的内在函数。 相机的主点设置为图像中心，焦距为 KITTI 中所有焦距的平均值。 所有图像都经过校正，两个立体图像之间的转换设置为固定长度的纯水平平移。 在评估过程中，根据标准做法 [5、8、10、11]，仅评估不超过 80 米固定范围的深度。 作为我们的主干模型，我们使用了 Monodepth2 [11] 并保留了原始的 ResNet18 [13] 作为编码器。 此外，我们还在 Cityscapes 数据集 [3] 上训练和测试了我们的 H-Net，以验证其通用性。

We compared our results with state-of-the-art supervised and self-supervised approaches and both qualitative and quantitative results were generated for comparison. To better understand how each component influenced the overall performance, we conducted an ablation study by turning various components of the model off, in turn.

我们将我们的结果与最先进的监督和自我监督方法进行了比较，并生成了定性和定量结果以进行比较。 为了更好地了解每个组件如何影响整体性能，我们通过依次关闭模型的各个组件进行了消融研究。

### 4.1. Implementation Details
Our H-Net model was trained using the PyTorch library [30], with an input/output resolution of 640 × 192 and a batch size of 8. The L1-norm loss term γ was set to 0.85 and the smoothness term λ was 0.001. The number of scales m was set to 4, which meant that totally we had 4 multi scales and there were 4 output scales as well with resolutions 210 , 211 , 212 and 213 of the input resolution. The model was trained for 20 epochs using the Adam optimizer [18] requiring approximately 14 hours on a single NVIDIA 2080Ti GPU. We set the learning rate to 10−4 for the first 15 epochs and dropped it to 10−5 for the remainder. As with previous papers [11], we also used a Resnet encoder with pre-trained weights on ImageNet [36], which proved able to improve the overall accuracy of the depth estimation and to reduce the training time [11, 15].

我们的 H-Net 模型使用 PyTorch 库 [30] 进行训练，输入/输出分辨率为 640 × 192，批量大小为 8。L1 范数损失项 γ 设置为 0.85，平滑度项 λ 为 0.001 . 尺度数 m 设置为 4，这意味着我们总共有 4 个多尺度，并且有 4 个输出尺度以及输入分辨率的分辨率 210 、 211 、 212 和 213 。 该模型使用 Adam 优化器 [18] 训练了 20 个时期，在单个 NVIDIA 2080Ti GPU 上需要大约 14 小时。 我们将前 15 个 epoch 的学习率设置为 10−4，然后将剩余的学习率降至 10−5。 与之前的论文 [11] 一样，我们还在 ImageNet [36] 上使用了具有预训练权重的 Resnet 编码器，这证明能够提高深度估计的整体精度并减少训练时间 [11, 15]。

Table 1. Quantitative results. Comparison of our proposed H-Net to existing methods on KITTI2015 [9] using the Eigen split unless marked with ‘Full Eigen’, which indicates the full Eigen dataset. The best result in each category are presented in bold while the second best results are underlined. All results here are shown without post-processing [10] unless marked with -+pp. The supervision mode for each method is indicated in the Train: D-Depth supervision, D*-Auxiliary depth supervision, M-Self-supervised mono supervision and S-self-supervised stereo supervision. Symbol † represents the new results from github. Metrics labeled by red mean lower is better while labeled by blue mean higher is better.
表 1. 定量结果。 我们提出的 H-Net 与 KITTI2015 [9] 上现有方法的比较使用特征分割，除非标有“完整特征”，这表示完整的特征数据集。 每个类别中的最佳结果以粗体显示，而第二好的结果则用下划线表示。 此处显示的所有结果均未进行后处理 [10]，除非标有 -+pp。 Train中指明了每种方法的监督模式：D-深度监督、D*-辅助深度监督、M-自监督单声道监督和S-自监督立体监督。 符号 † 代表来自 github 的新结果。 标有红色的指标意味着越低越好，而标有蓝色的指标意味着越高越好。


Table 2. Ablation Study. Results for different variants of our model (H-Net) on KITTI2015 [9] using full Eigen dataset with comparison to our backbone Monodepth2 [11]. We evaluate the impact of the Siamese encoder- Siamese decoder (SE-SD), mutual epipolar attention (MEA) and optimal transport (OT). Metrics labeled by red mean lower is better while labeled by blue mean higher is better
表 2. 消融研究。 我们的模型(H-Net)的不同变体在 KITTI2015 [9] 上的结果使用完整的 Eigen 数据集与我们的主干 Monodepth2 [11] 进行比较。 我们评估连体编码器-连体解码器 (SE-SD)、相互对极注意 (MEA) 和最优传输 (OT) 的影响。 红色标记的指标越低越好，蓝色标记的指标越高越好

## 5. Results and Discussion
### 5.1. KITTI Results
The qualitative results and quantitive results on the KITTI Eigen split are shown in Table 1 and Figure 4. In Table 1, it can be seen that the proposed H-Net outperforms all existing state-of-the-art self-supervised methods by a significant margin. Compared with other approaches that applied direct supervision signals (supervised methods), the model was still competitive. As our H-Net takes stereo image pairs as the input, in contrast with [11] and [15], we did not need to remove static frames. However, to make the comparison fair, we used both full Eigen and Eigen split dataset to make the dataset consistent with other methods. Among all the evaluation measures, the best and the second best ones were produced by our H-Net model, which indicates that our model can learn from the geometry constraints and benefits from the optimal transport solution, achieving state-of-the-art depth predictions. For the quantitative results, we can see that the depth maps generated by our model contained more details, i.e. the structural characteristics of buildings, protruding kerbs, bushes, and trees. Besides, our model could effectively distinguish different parts of every object, for example, the upper part of the tree is no longer uniform but is full of outlines and details.

KITTI Eigen split 的定性和定量结果如表 1 和图 4 所示。在表 1 中，可以看出所提出的 H-Net 优于所有现有的最先进的自监督方法 显著的保证金。 与应用直接监督信号(监督方法)的其他方法相比，该模型仍然具有竞争力。 由于我们的 H-Net 将立体图像对作为输入，与 [11] 和 [15] 相比，我们不需要删除静态帧。 然而，为了使比较公平，我们同时使用了完整的 Eigen 和 Eigen split数据集，以使数据集与其他方法保持一致。 在所有评估措施中，最好的和第二好的是由我们的 H-Net 模型产生的，这表明我们的模型可以从几何约束中学习并从最佳传输解决方案中获益，达到最先进的深度 预测。 对于定量结果，我们可以看到我们的模型生成的深度图包含更多细节，即建筑物的结构特征、突出的路缘、灌木和树木。 此外，我们的模型可以有效区分每个对象的不同部分，例如，树的上部不再是统一的，而是充满了轮廓和细节。

Figure 3. Qualitative results on the KITTI Eigen split. The depth prediction are all for the left input image. Our H-Net in the last row generates the depth maps with more details and performs better on distinguishing different parts in one object, i.e. buildings, kerbs bushes and trees, which reflects the superior quantitative results in Table 1.
图 3. KITTI Eigen split的定性结果。 深度预测都是针对左输入图像。 我们在最后一行的 H-Net 生成了具有更多细节的深度图，并且在区分一个对象的不同部分(即建筑物、路缘灌木和树木)方面表现更好，这反映了表 1 中出色的定量结果。

Table 3. Number of Parameters (M:million) for our models with different settings of the mutual attention module. 
表 3. 具有不同相互关注模块设置的模型的参数数量 (M:million)。

### 5.2. KITTI Ablation Study Results
The results of the ablation study on the KITTI dataset are shown in Table 2. We can see that the backbone Mon odepth2 model [11] performed the worst without any of our contributions but by changing the architecture to a Siamese encoder- Siamese decoder, the evaluation measures steadily improved. The reason might be that fusing the complementary information between the stereo image pair gave the framework higher chance to generate accurate predicted depth maps. Our MEA and OT modules were all incorporated in the SE-SD architecture. Row 4 shows that the addition of MEA benefits the depth estimation performance in all the evaluation measures, especially on metrics that are sensitive to large depth errors e.g. RMSE. The significantly large improvement of the SE-SD architecture with MEA, is likely due to the epipolar constraint, which allowed the network to learn strong correspondences limited on the same epipolar lines in the rectified stereo images. The impact of the OT-MNL is presented in Row 5, compared to the SE-SD we still can notice a dramatic increase in most of the evaluation metrics. The reason might be that the optimal transport algorithm further improved the MEA by increasing the correct correspondence weights, merging the semantic features while suppressing outliers. In the last row, by combining the backbone with all of our components, the effectiveness of the final framework was significantly improved, as expected, and state-of-the-art results were observed. Besides, although our OT-MEA module was inspired by the MNL, our results outperformed the same SE-SD architecture with MNL. Apart from the performance evaluation measures, we also estimated the number of parameters for each of the examined settings. While all of our proposed components contributed to the overall performance in the selfsupervised depth estimation task, the number of parameters was barely increased. We can see from Table 3 that our OTMEA module cost 0.6 million (2.0%) additional parameters compared with the pure SE-SD architecture.

KITTI 数据集的消融研究结果如表 2 所示。我们可以看到，主干 Mon odepth2 模型 [11] 在我们没有任何贡献的情况下表现最差，但通过将架构更改为 孪生编码器 - Siamese 解码器， 考核办法稳步完善。 原因可能是融合立体图像对之间的互补信息使框架有更高的机会生成准确的预测深度图。 我们的 MEA 和 OT 模块都包含在 SE-SD 架构中。 第 4 行表明，添加 MEA 有利于所有评估措施中的深度估计性能，尤其是对大深度误差敏感的指标，例如 均方根误差。 使用 MEA 的 SE-SD 架构的显著改进可能是由于极线约束，这使得网络能够学习限制在整流立体图像中相同极线上的强对应关系。 OT-MNL 的影响显示在第 5 行，与 SE-SD 相比，我们仍然可以注意到大多数评估指标的显著增加。 原因可能是最优传输算法通过增加正确对应权重、合并语义特征同时抑制异常值进一步改进了 MEA。 在最后一行，通过将主干与我们所有的组件相结合，最终框架的有效性如预期的那样得到了显著提高，并且观察到了最先进的结果。 此外，尽管我们的 OT-MEA 模块受到 MNL 的启发，但我们的结果优于具有 MNL 的相同 SE-SD 架构。 除了性能评估措施外，我们还估计了每个检查设置的参数数量。 虽然我们提出的所有组件都有助于自监督深度估计任务的整体性能，但参数数量几乎没有增加。 从表 3 可以看出，与纯 SE-SD 架构相比，我们的 OTMEA 模块额外花费了 60 万(2.0%)个参数。

Figure 4. Qualitative results on the Cityscapes dataset. Our H-Net generates very close predictions compared with the ground truth. 
图 4. Cityscapes 数据集的定性结果。 与基准实况相比，我们的 H-Net 生成非常接近的预测。

### 5.3. Cityscapes results
The performance of H-Net has been further evaluated on the Cityscape dataset. The results in Figure 4 show the accuracy of the depth estimated by H-Net compared to the ground truth, with detailed reconstructions of objects such as cars, human, and trees. More experimental results could be found in the supplementary material.

H-Net 的性能已在 Cityscape 数据集上得到进一步评估。 图 4 中的结果显示了 H-Net 估计的深度与基准实况相比的准确性，并详细重建了汽车、人和树木等物体。 更多实验结果可以在补充材料中找到。

## 6. Conclusion
In this paper we presented a novel network, the H-Net, for self-supervised depth estimation, achieving state-of-theart depth prediction. By designing the Siamese encoder - Siamese decoder architecture, exploiting the mutual epipolar attention, and formulating the optimal transport problem, both the global-range correspondence between stereo image pairs and strongly related feature correspondences satisfying the epipolar constraint in the rectified images were effectively explored and fused. We showed how this benefited the overall performance on public datasets and how together they gave a large improvement in evaluation measures, indicating that the model effectively tackled the limits of other self-supervised depth estimation methods and closed the gap with supervised approaches.

在本文中，我们提出了一种新颖的网络 H-Net，用于自监督深度估计，实现最先进的深度预测。 通过设计 Siamese encoder - Siamese decoder 架构，利用相互对极注意力，并制定最优传输问题，有效地探索和融合了立体图像对之间的全局范围对应关系和满足校正图像中对极约束的强相关特征对应关系。 我们展示了这如何有利于公共数据集的整体性能，以及它们如何共同大幅改进评估措施，表明该模型有效地解决了其他自监督深度估计方法的局限性，并缩小了与监督方法的差距。

## References
1. Stephen T Barnard and Martin A Fischler. Computational stereo. ACM Computing Surveys (CSUR), 14(4):553–572, 1982.
2. Jia-Ren Chang and Yong-Sheng Chen. Pyramid stereo matching network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5410– 5418, 2018.
3. Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.
4. Shivam Duggal, Shenlong Wang, Wei-Chiu Ma, Rui Hu, and Raquel Urtasun. Deeppruner: Learning efficient stereo matching via differentiable patchmatch. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4384–4393, 2019.
5. David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. arXiv preprint arXiv:1406.2283, 2014.
6. Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, and Dacheng Tao. Deep ordinal regression network for monocular depth estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2002–2011, 2018.
7. Adrien Gaidon, Qiao Wang, Yohann Cabon, and Eleonora Vig. Virtual worlds as proxy for multi-object tracking analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4340–4349, 2016.
8. Ravi Garg, Vijay Kumar Bg, Gustavo Carneiro, and Ian Reid. Unsupervised cnn for single view depth estimation: Geometry to the rescue. In European conference on computer vision, pages 740–756. Springer, 2016.
9. Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 3354–3361. IEEE, 2012.
10. Clement Godard, Oisin Mac Aodha, and Gabriel J. Brostow. Unsupervised monocular depth estimation with leftright consistency. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.
11. Cl´ement Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J Brostow. Digging into self-supervised monocular depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3828–3838, 2019.
12. Xiaoyang Guo, Hongsheng Li, Shuai Yi, Jimmy Ren, and Xiaogang Wang. Learning monocular depth by distilling cross-domain stereo networks. In Proceedings of the European Conference on Computer Vision (ECCV), pages 484– 500, 2018.
13. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
14. Max Jaderberg, Karen Simonyan, Andrew Zisserman, and Koray Kavukcuoglu. Spatial transformer networks. arXiv preprint arXiv:1506.02025, 2015.
15. Adrian Johnston and Gustavo Carneiro. Self-supervised monocular trained depth estimation using self-attention and discrete disparity volume. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4756–4765, 2020.
16. Sunghun Joung, Seungryong Kim, Kihong Park, and Kwanghoon Sohn. Unsupervised stereo matching using con- fidential correspondence consistency. IEEE Transactions on Intelligent Transportation Systems, 21(5):2190–2203, 2019.
17. Alex Kendall, Hayk Martirosyan, Saumitro Dasgupta, Peter Henry, Ryan Kennedy, Abraham Bachrach, and Adam Bry. End-to-end learning of geometry and context for deep stereo regression. In Proceedings of the IEEE International Conference on Computer Vision, pages 66–75, 2017.
18. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
19. Jogendra Nath Kundu, Phani Krishna Uppala, Anuj Pahuja, and R Venkatesh Babu. Adadepth: Unsupervised content congruent adaptation for depth estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2656–2665, 2018.
20. Uday Kusupati, Shuo Cheng, Rui Chen, and Hao Su. Normal assisted stereo depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2189–2199, 2020.
21. Yevhen Kuznietsov, Jorg Stuckler, and Bastian Leibe. Semisupervised deep learning for monocular depth map prediction. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6647–6655, 2017.
22. Ruihao Li, Sen Wang, Zhiqiang Long, and Dongbing Gu. Undeepvo: Monocular visual odometry through unsupervised deep learning. In 2018 IEEE international conference on robotics and automation (ICRA), pages 7286–7291. IEEE, 2018.
23. Fayao Liu, Chunhua Shen, Guosheng Lin, and Ian Reid. Learning depth from single monocular images using deep convolutional neural fields. IEEE transactions on pattern analysis and machine intelligence, 38(10):2024–2039, 2015.
24. Yanbin Liu, Linchao Zhu, Makoto Yamada, and Yi Yang. Semantic correspondence as an optimal transport problem. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4463–4472, 2020.
25. Chenxu Luo, Zhenheng Yang, Peng Wang, Yang Wang, Wei Xu, Ram Nevatia, and Alan Yuille. Every pixel counts++: Joint learning of geometry and motion with 3d holistic understanding. IEEE transactions on pattern analysis and machine intelligence, 42(10):2624–2641, 2019.
26. Wenjie Luo, Alexander G Schwing, and Raquel Urtasun. Ef- ficient deep learning for stereo matching. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5695–5703, 2016.
27. Yue Luo, Jimmy Ren, Mude Lin, Jiahao Pang, Wenxiu Sun, Hongsheng Li, and Liang Lin. Single view stereo matching. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 155–163, 2018.
28. Nikolaus Mayer, Eddy Ilg, Philipp Fischer, Caner Hazirbas, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. What makes good synthetic training data for learning disparity and optical flow estimation? International Journal of Computer Vision, 126(9):942–960, 2018.
29. Ishit Mehta, Parikshit Sakurikar, and PJ Narayanan. Structured adversarial training for unsupervised monocular depth estimation. In 2018 International Conference on 3D Vision (3DV), pages 314–323. IEEE, 2018.
30. Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.
31. Sudeep Pillai, Rares¸ Ambrus¸, and Adrien Gaidon. Superdepth: Self-supervised, super-resolved monocular depth estimation. In 2019 International Conference on Robotics and Automation (ICRA), pages 9250–9256. IEEE, 2019.
32. Andrea Pilzer, St´ephane Lathuili`ere, Dan Xu, Mihai Marian Puscas, Elisa Ricci, and Nicu Sebe. Progressive fusion for unsupervised binocular depth estimation using cycled networks. IEEE transactions on pattern analysis and machine intelligence, 42(10):2380–2395, 2019.
33. Andrea Pilzer, Dan Xu, Mihai Puscas, Elisa Ricci, and Nicu Sebe. Unsupervised adversarial depth estimation using cycled generative networks. In 2018 International Conference on 3D Vision (3DV), pages 587–595. IEEE, 2018.
34. Matteo Poggi, Fabio Tosi, and Stefano Mattoccia. Learning monocular depth estimation with unsupervised trinocular assumptions. In 2018 International conference on 3d vision (3DV), pages 324–333. IEEE, 2018.
35. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.
36. Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015.
37. Chaoyang Wang, Jos´e Miguel Buenaposada, Rui Zhu, and Simon Lucey. Learning depth from monocular videos using direct methods. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2022– 2030, 2018.
38. Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7794–7803, 2018.
39. Haofei Xu and Juyong Zhang. Aanet: Adaptive aggregation network for efficient stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1959–1968, 2020.
40. Nan Yang, Lukas von Stumberg, Rui Wang, and Daniel Cremers. D3vo: Deep depth, deep pose and deep uncertainty for monocular visual odometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1281–1292, 2020.
41. Nan Yang, Rui Wang, Jorg Stuckler, and Daniel Cremers. Deep virtual stereo odometry: Leveraging deep depth prediction for monocular direct sparse odometry. In Proceedings of the European Conference on Computer Vision (ECCV), pages 817–833, 2018.
42. Huangying Zhan, Ravi Garg, Chamara Saroj Weerasekera, Kejie Li, Harsh Agarwal, and Ian Reid. Unsupervised learning of monocular depth estimation and visual odometry with deep feature reconstruction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 340–349, 2018.
43. Jian-Qing Zheng, Ngee Han Lim, and Bartłomiej W Papie˙z. D-net: Siamese based network for arbitrarily oriented volume alignment. In International Workshop on Shape in Medical Imaging, pages 73–84. Springer, 2020.
44. Chao Zhou, Hong Zhang, Xiaoyong Shen, and Jiaya Jia. Unsupervised learning of stereo matching. In Proceedings of the IEEE International Conference on Computer Vision, pages 1567–1575, 2017.
45. Tinghui Zhou, Matthew Brown, Noah Snavely, and David G Lowe. Unsupervised learning of depth and ego-motion from video. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1851–1858, 2017.
