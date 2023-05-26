# Benchmarking Detection Transfer Learning with Vision Transformers
使用ViT进行基准检测迁移学习 2021.11.22 https://arxiv.org/abs/2111.11429

## 阅读笔记
* Reducing Memory and Time Complexity. 以ViTDet论文为主，但重点关注这节，因为ViTDet中这点没看明白。

## Abstract
Object detection is a central downstream task used to test if pre-trained network parameters confer benefits, such as improved accuracy or training speed. The complexity of object detection methods can make this benchmarking non-trivial when new architectures, such as Vision Transformer (ViT) models, arrive. These difficulties (e.g., architectural incompatibility, slow training, high memory consumption, unknown training formulae, etc.) have prevented recent studies from benchmarking detection transfer learning with standard ViT models. In this paper, we present training techniques that overcome these challenges, enabling the use of standard ViT models as the backbone of Mask R-CNN. These tools facilitate the primary goal of our study: we compare five ViT initializations, including recent state-of-the-art self-supervised learning methods, supervised initialization, and a strong random initialization baseline. Our results show that recent masking-based unsupervised learning methods may, for the first time, provide convincing transfer learning improvements on COCO, increasing APbox up to 4% (absolute) over supervised and prior self-supervised pre-training methods. Moreover, these masking-based initializations scale better, with the improvement growing as model size increases.

目标检测是一项中心下游任务，用于测试预训练的网络参数是否带来好处，例如提高准确性或训练速度。 当 Vision Transformer (ViT) 模型等新架构出现时，目标检测方法的复杂性会使这种基准测试变得非常重要。 这些困难(例如，架构不兼容、训练缓慢、内存消耗高、训练公式未知等)阻碍了最近的研究使用标准 ViT 模型对检测迁移学习进行基准测试。 在本文中，我们提出了克服这些挑战的训练技术，使标准 ViT 模型能够用作 Mask R-CNN 的主干。 这些工具促进了我们研究的主要目标：我们比较了五个 ViT 初始化，包括最近最先进的自监督学习方法、监督初始化和强大的随机初始化基线。 我们的结果表明，最近的基于掩码的无监督学习方法可能首次在 COCO 上提供令人信服的迁移学习改进，将 APbox 比监督和先前的自监督预训练方法提高 4%(绝对值)。 此外，这些基于掩码的初始化具有更好的扩展性，随着模型大小的增加而提高。

## 1. Introduction
Unsupervised/self-supervised deep learning is commonly used as a pre-training step that initializes model parameters before they are transferred to a downstream task, such as image classification or object detection, for finetuning. The utility of an unsupervised learning algorithm is judged by downstream task metrics (e.g. accuracy, convergence speed, etc.) in comparison to baselines, such as supervised pre-training or no pre-training at all, i.e., random initialization (often called training “from scratch”).

无监督/自监督深度学习通常用作预训练步骤，在将模型参数迁移到下游任务(例如图像分类或目标检测)以进行微调之前初始化模型参数。 无监督学习算法的效用是通过与基线相比的下游任务指标(例如准确性、收敛速度等)来判断的，例如有监督的预训练或根本没有预训练，即随机初始化(通常称为训练 “从头开始”)。

Unsupervised deep learning in computer vision typically uses standard convolutional network (CNN) models [25], such as ResNets [20]. Transferring these models is relatively straightforward because CNNs are in widespread use in most downstream tasks, and thus benchmarking protocols are easy to define and baselines are plentiful (e.g. [17]). In other words, unsupervised learning with CNNs produces a plug-and-play parameter initialization.

计算机视觉中的无监督深度学习通常使用标准卷积网络(CNN) 模型 [25]，例如 ResNets [20]。 迁移这些模型相对简单，因为 CNN 在大多数下游任务中得到广泛使用，因此基准测试协议很容易定义并且基线很丰富(例如 [17])。 换句话说，使用 CNN 进行无监督学习会产生即插即用的参数初始化。

We are now witnessing the growth of unsupervised learning with Vision Transformer (ViT) models [10], and while the high-level transfer learning methodology remains the same, the low-level details and baselines for some important downstream tasks have not been established. Notably, object detection, which has played a central role in the study of transfer learning over the last decade (e.g., [35, 14, 9, 17]), was not explored in the pioneering work on ViT training [10, 7, 5]—supervised or unsupervised—due to the challenges (described shortly) of integrating ViTs into common detection models, like Mask R-CNN [19].

我们现在正在见证使用 Vision Transformer (ViT) 模型 [10] 的无监督学习的增长，虽然高级迁移学习方法保持不变，但一些重要下游任务的低级细节和基线尚未建立。 值得注意的是，目标检测在过去十年的迁移学习研究中发挥了核心作用(例如，[35、14、9、17])，但在 ViT 训练的开创性工作中并未得到探索 [10、7、 5] —— 监督或非监督 —— 由于将 ViT 集成到常见检测模型(如 Mask R-CNN [19])中的挑战(稍后描述)。

To bridge this gap, this paper establishes a transfer learning protocol for evaluating ViT models on object detection and instance segmentation using the COCO dataset [28] and the Mask R-CNN framework. We focus on standard ViT models, with minimal modifications, as defined in the original ViT paper [10], because we expect this architecture will remain popular in unsupervised learning work over the next few years due to its simplicity and flexibility when exploring new techniques, e.g., masking-based methods [1, 16].

为了弥合这一差距，本文建立了一种迁移学习协议，用于使用 COCO 数据集 [28] 和 Mask R-CNN 框架评估目标检测和实例分割的 ViT 模型。 我们专注于标准 ViT 模型，修改最少，如原始 ViT 论文 [10] 中所定义，因为我们预计这种架构在未来几年将在无监督学习工作中继续流行，因为它在探索新技术时具有简单性和灵活性， 例如，基于掩码的方法 [1, 16]。

Establishing object detection baselines for ViT is challenging due to technical obstacles that include mitigating ViT’s large memory requirements when processing detection-sized inputs (e.g., ∼20× more patches than in pre-training), architectural incompatibilities (e.g., singlescale ViT vs. a multi-scale detector), and developing effective training formulae (i.e., learning schedules, regularization and data augmentation methods, etc.) for numerous pre-trained initializations, as well as random initialization. We overcome these obstacles and present strong ViT-based Mask R-CNN baselines on COCO when initializing ViT from-scratch [18], with pre-trained ImageNet [8] supervision, and with unsupervised pre-training using recent methods like MoCo v3 [7], BEiT [1], and MAE [16].

为 ViT 建立目标检测基线具有挑战性，因为技术障碍包括在处理检测大小的输入时减轻 ViT 的大内存需求(例如，比预训练多 20 倍的分块)、架构不兼容(例如，单尺度 ViT 与 多尺度检测器)，并为大量预训练初始化和随机初始化开发有效的训练公式(即学习计划、正则化和数据增广方法等)。 我们克服了这些障碍，并在从头开始 [18] 初始化 ViT 时，在 COCO 上提出了强大的基于 ViT 的 Mask R-CNN 基线，使用预训练的 ImageNet [8] 监督，以及使用最近的方法(如 MoCo v3[7]、BEiT [1] 和 MAE [16])进行无监督预训练 。

Looking beyond ViT, we hope our practices and observations will serve as a blueprint for future work comparing pre-training methods for more advanced ViT derivatives, like Swin [29] and MViT [12]. To facilitate community development we will release code in Detectron2 [40]. 

超越 ViT，我们希望我们的实践和观察将作为未来工作的蓝图，比较更高级的 ViT 衍生物(如 Swin [29] 和 MViT [12])的预训练方法。 为了促进社区发展，我们将在 Detectron2 [40] 中发布代码。

## 2. Approach
We select the Mask R-CNN [19] framework due to its ubiquitous presence in object detection and transfer learning research. Mask R-CNN is the foundation of higher complexity/higher performing systems, such as Cascade R-CNN [4] and HTC/HTC++ [6, 29], which may improve upon the results presented here at the cost of additional complexity that is orthogonal to the goal of benchmarking transfer learning. Our choice attempts to balance (relative) simplicity vs. complexity while providing compelling, even though not entirely state-of-the-art, results.

我们选择 Mask R-CNN [19] 框架是因为它在目标检测和迁移学习研究中无处不在。 Mask R-CNN 是更高复杂度/更高性能系统的基础，例如 Cascade R-CNN [4] 和 HTC/HTC++ [6, 29]，它们可能会以增加复杂性为代价改进此处呈现的结果 与基准迁移学习的目标正交。 我们的选择试图平衡(相对)简单性和复杂性，同时提供令人信服的(即使不完全是最先进的)结果。

We configure Mask R-CNN with a number of upgraded modules (described in §2.2) and training procedures (described in §2.3) relative to the original publication. These upgrades, developed primarily in [39, 18, 13], allow the model to be trained effectively from random initialization, thus enabling a meaningful from-scratch baseline. Next, we will discuss how the backbone, which would typically be a ResNet, can be replaced with a Vision Transformer.

我们为 Mask R-CNN 配置了一些相对于原始版本的升级模块(在 §2.2 中描述)和训练过程(在 §2.3 中描述)。 这些升级主要在 [39、18、13] 中开发，允许从随机初始化中有效地训练模型，从而实现有意义的从头开始的基线。 接下来，我们将讨论如何用 Vision Transformer 替换通常为 ResNet 的主干网。

### 2.1. ViT Backbone
In this section we address two technical obstacles when using ViT as the backbone in Mask R-CNN: (1) how to adapt it to work with a feature pyramid network (FPN) [27] and (2) how to reduce its memory footprint and runtime to make benchmarking large ViT backbones tractable.

在本节中，我们解决了在 Mask R-CNN 中使用 ViT 作为主干时的两个技术障碍：
1. 如何使其适应特征金字塔网络 (FPN) [27] 
2. 如何减少其内存占用 和运行时，使大型 ViT 主干的基准测试变得易于处理。

#### FPN Compatibility. 
Mask R-CNN can work with a backbone that either produces a single-scale feature map or feature maps at multiple scales that can be input into an FPN. Since FPN typically provides better detection results with minimal time and memory overhead, we adopt it.

FPN 兼容性。 Mask R-CNN 可以与生成单尺度特征图或多尺度特征图的主干一起工作，这些特征图可以输入到 FPN 中。 由于 FPN 通常以最少的时间和内存开销提供更好的检测结果，因此我们采用它。

However, using FPN presents a problem because ViT produces feature maps at a single scale (e.g., 1/16th), in contrast to the multi-scale feature maps produced by typical CNNs(1 We view the natural 2D spatial arrangement of intermediate ViT patch embeddings as a standard 2D feature map). To address this discrepancy, we employ a simple technique from [11] (used for the single-scale XCiT backbone) to either upsample or downsample intermediate ViT feature maps by placing four resolution-modifying modules at equally spaced intervals of d/4 transformer blocks, where d is the total number of blocks. See Figure 1 (green blocks).

然而，使用 FPN 存在一个问题，因为 ViT 以单一比例(例如，1/16)生成特征图，这与典型 CNN 生成的多比例特征图形成对比(1 我们查看中间 ViT 分块的自然 2D 空间排列 嵌入作为标准的 2D 特征图)。 为了解决这种差异，我们采用了 [11] 中的一种简单技术(用于单尺度 XCiT 主干)，通过将四个分辨率修改模块放置在 d/4 变换器块的等间距间隔处来上采样或下采样中间 ViT 特征图 ，其中 d 是块的总数。 参见图 1(绿色块)。

![Figure 1](../images/ViTDet_benchmarking/fig_1.png)<br/>
Figure 1. ViT-based Mask R-CNN. In §2 we describe how a standard ViT model can be used effectively as the backbone in Mask R-CNN. To save time and memory, we modify the ViT to use nonoverlapping windowed attention in all but four of its Transformer blocks, spaced at an interval of d/4, where d is the total number of blocks (blue) [26]. To adapt the single-scale ViT to the multiscale FPN (yellow), we make use of upsampling and downsampling modules (green) [11]. The rest of the system (light red) uses upgraded, but standard, Mask R-CNN components.
图1. 基于 ViT 的 Mask R-CNN。 在§2 中，我们描述了如何将标准 ViT 模型有效地用作 Mask R-CNN 的主干。 为了节省时间和内存，我们将 ViT 修改为在除四个 Transformer 块之外的所有块中使用非重叠窗口注意力，间隔为 d/4，其中 d 是块的总数(蓝色)[26]。 为了使单尺度 ViT 适应多尺度 FPN(黄色)，我们使用上采样和下采样模块(绿色)[11]。 系统的其余部分(浅红色)使用升级但标准的 Mask R-CNN 组件。
<!-- 预训练时图像大小224*224，每个patch的大小为14*14, patchs的数量为 16*16; 
目标检测时，图像大小1024*1024, 1024/16=64，patchs的数量为64*64，r=14 (64不重复的分14分？)
 -->

The first of these modules upsamples the feature map by a factor of 4 using a stride-two 2×2 transposed convolution, followed by group normalization [39] and GeLU [21], and finally another stride-two 2×2 transposed convolution. The next d/4th block’s output is upsampled by 2× using a single stride-two 2 × 2 transposed convolution (without normalization and non-linearity). The next d/4th block’s output is taken as is and the final ViT block’s output is downsampled by a factor of two using stride-two 2×2 max pooling. Each of these modules preserves the ViT’s embedding/channel dimension. Assuming a patch size of 16, these modules produce feature maps with strides of 4, 8, 16, and 32 pixels, w.r.t. the input image, that are ready to input into an FPN.

这些模块中的第一个使用 stride-two 2×2 转置卷积对特征图进行 4 倍上采样，然后是组归一化 [39] 和 GeLU [21]，最后是另一个 stride-two 2×2 转置卷积。 下一个 d/4 块的输出使用单步长 2 × 2 转置卷积(没有归一化和非线性)进行 2× 上采样。 下一个 d/4 块的输出按原样使用，最后一个 ViT 块的输出使用 stride-two 2×2 max pooling 下采样两倍。 这些模块中的每一个都保留了 ViT 的嵌入/通道维度。 假设分块大小为 16，这些模块生成的特征图的步幅为 4、8、16 和 32 像素，w.r.t。 准备好输入到 FPN 中的输入图像。

We note that recent work, such as Swin [29] and MViT [12], address the single vs. multi-scale feature map problem by modifying the core ViT architecture (in pretraining) so it is inherently multi-scale. This is an important direction, but it also complicates the simple ViT design and may impede the exploration of new unsupervised learning directions, such as methods that sparsely process unmasked patches [16]. Therefore, we focus on external additions to ViTs that allow them to integrate into multi-scale detection systems. We also note that Beal et al. [2] integrate standard ViT models with Faster R-CNN [34], but report substantially lower APbox compared to our results (>10 points lower), which suggests that our design is highly effective.

我们注意到最近的工作，例如 Swin [29] 和 MViT [12]，通过修改核心 ViT 架构(在预训练中)解决了单尺度与多尺度特征映射问题，因此它本质上是多尺度的。 这是一个重要的方向，但它也使简单的 ViT 设计复杂化，并可能阻碍探索新的无监督学习方向，例如稀疏处理未掩码分块的方法 [16]。 因此，我们专注于 ViT 的外部添加，使它们能够集成到多尺度检测系统中。 我们还注意到 Beal et al.  [2] 将标准 ViT 模型与 Faster R-CNN [34] 集成，但报告的 APbox 与我们的结果相比要低得多(> 10 个百分点)，这表明我们的设计非常有效。

#### Reducing Memory and Time Complexity. 
Using ViT as a backbone in Mask R-CNN introduces memory and runtime challenges. Each self-attention operation in ViT takes $O(h^2w^2)$ space and time for an image tiled (or “patchified”) into h × w non-overlapping patches [38].

降低内存和时间复杂度。 在 Mask R-CNN 中使用 ViT 作为主干引入了内存和运行时挑战。 ViT 中的每个自注意力操作都需要 $O(h^2w^2)$ 空间和时间来将图像平铺(或“分块”)成 h×w 个非重叠块 [38]。

During pre-training, this complexity is manageable as h = w = 14 is a typical setting (a 224 × 224 pixel image patchified into 16 × 16 pixel patches). In object detection, a standard image size is 1024 × 1024—approximately 21× more pixels and patches. This higher resolution is needed in order to detect relatively small objects as well as larger ones. Due to the quadratic complexity of self-attention, even the “base” size ViT-B may consume ∼20–30GB of GPU memory when used in Mask R-CNN with a single-image minibatch and half-precision floating point numbers. 

在预训练期间，这种复杂性是可控的，因为 h = w = 14 是一个典型的设置(将 224 × 224 像素图像分成 16 × 16 像素块)。 在目标检测中，标准图像大小为 1024 × 1024——大约多 21 倍的像素和块。 为了检测相对较小的物体以及较大的物体，需要这种更高的分辨率。 由于自注意力的二次复杂性，即使是“基本”大小的 ViT-B 在 Mask R-CNN 中使用单图像小批量和半精度浮点数时也可能消耗 ~20-30GB 的 GPU 内存。

To reduce space and time complexity we use restricted (or “windowed”) self-attention [38], which saves both space and time by replacing global computation with local computation. We partition the h × w patchified image into r × r patch non-overlapping windows and compute selfattention independently within each of these windows. This windowed self-attention has $O(r^2hw)$ space and time complexity (from $O(r^4)$ per-window complexity and h/r×w/r windows). We set r to the global self-attention size used in pre-training (e.g., r = 14 is typical).

为了降低空间和时间复杂度，我们使用受限(或“窗口化”)自注意力 [38]，它通过用局部计算代替全局计算来节省空间和时间。 我们将 h×w 分块图像划分为 r×r 分块非重叠窗口，并在每个窗口中独立计算自注意力。 这种窗口化自注意力具有 $O(r^2hw)$ 空间和时间复杂度(来自 $O(r^4)$ 每个窗口复杂度和 h/r×w/r 个窗口)。 我们将 r 设置为预训练中使用的全局自注意力大小(例如，r = 14 是典型值)。

A drawback of windowed self-attention is that the backbone does not integrate information across windows. Therefore we adopt the hybrid approach from [26] that includes four global self-attention blocks placed evenly at each d/4th block (these coincide with the up-/downsampling locations used for FPN integration; see Figure 1).

windowed self-attention 的一个缺点是 backbone 不会跨窗口整合信息。 因此，我们采用来自 [26] 的混合方法，包括四个全局自注意力块，均匀放置在每个 d/4 块(这些与用于 FPN 集成的上/下采样位置一致; 参见图 1)。

### 2.2. Upgraded Modules
Relative to the original Mask R-CNN in [19], we modernize several of its modules. Concisely, the modifications include: (1) following the convolutions in FPN with batch normalization (BN) [23], (2) using two convolutional layers in the region proposal network (RPN) [33] instead of one, (3) using four convolutional layers with BN followed by one linear layer for the region-of-interest (RoI) classification and box regression head [39] instead of a two-layer MLP without normalization, (4) and following the convolutions in the standard mask head with BN. Wherever BN is applied, we use synchronous BN across all GPUs. These upgrades are implemented in the Detectron2 model zoo.(2https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#new-baselines-using-large-scale-jitter-and-longer-training-schedule)

相对于 [19] 中的原始 Mask R-CNN，我们对其几个模块进行了现代化改造。 简而言之，修改包括：(1)在 FPN 中使用批量归一化(BN)[23] 进行卷积，(2)在区域提议网络(RPN)[33]中使用两个卷积层而不是一个，(3)使用 四个带 BN 的卷积层，后跟一个用于感兴趣区域 (RoI) 分类和框回归头的线性层 [39]，而不是没有归一化的两层 MLP，(4) 并跟随标准掩码头中的卷积 与 BN。 无论在何处应用 BN，我们都会在所有 GPU 上使用同步 BN。 这些升级是在 Detectron2 模型动物园中实现的。(2https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#new-baselines-using-large-scale-jitter-and-longer-training-schedule )

### 2.3. Training Formula
We adopt an upgraded training formula compared to the original Mask R-CNN. This formula was developed in [18], which demonstrated good from-scratch performance when training with normalization layers and for long enough, and [13], which demonstrated that a simple data augmentation method called large-scale jitter (LSJ) is effective at preventing overfitting and improves results when models are trained for very long schedules (e.g., 400 epochs).

与原始 Mask R-CNN 相比，我们采用了升级的训练公式。 这个公式是在 [18] 中开发的，它在使用归一化层进行足够长时间的训练时展示了良好的从头开始的性能，并且 [13] 证明了一种称为大规模抖动(LSJ)的简单数据增广方法在以下情况下是有效的 防止过度拟合并在模型训练时间很长(例如 400 个时期)时改善结果。

We aim to keep the number of hyperparameters low and therefore resist adopting additional data augmentation and regularization techniques. However, we found that drop path regularization [24, 22] is highly effective for ViT backbones and therefore we include it (e.g., it improves fromscratch training by up to 2 APbox).

我们的目标是保持超参数的数量较少，因此拒绝采用额外的数据增广和正则化技术。 然而，我们发现丢弃路径正则化 [24, 22] 对 ViT 主干非常有效，因此我们将其包括在内(例如，它将从头开始的训练提高了多达 2 个 APbox)。

In summary, we train all models with the same simple formula: LSJ (1024 × 1024 resolution, scale range [0.1, 2.0]), AdamW [30] (β1, β2 = 0.9, 0.999) with halfperiod cosine learning rate decay, linear warmup [15] for 0.25 epochs, and drop path regularization. When using a pre-trained initialization, we fine-tune Mask R-CNN for up to 100 epochs. When training from scratch, we consider schedules of up to 400 epochs since convergence is slower than when using pre-training. We distribute training over 32 or 64 GPUs (NVIDIA V100-32GB) and always use a minibatch size of 64 images. We use PyTorch’s automatic mixed precision. Additional hyperparameters are tuned by the consistent application of a protocol, describe next.

总之，我们使用相同的简单公式训练所有模型：LSJ(1024 × 1024 分辨率，尺度范围 [0.1, 2.0])，AdamW [30] (β1, β2 = 0.9, 0.999) 半周期余弦学习率衰减，线性 预热 [15] 0.25 个时期，并进行路径正则化。 当使用预训练初始化时，我们微调 Mask R-CNN 最多 100 个时期。 当从头开始训练时，我们考虑最多 400 个时期的时间表，因为收敛速度比使用预训练时慢。 我们将训练分布在 32 或 64 个 GPU (NVIDIA V100-32GB) 上，并始终使用 64 张图像的小批量大小。 我们使用 PyTorch 的自动混合精度。 其他超参数由协议的一致应用程序调整，接下来描述。

### 2.4. Hyperparameter Tuning Protocol
To adapt the training formula to each model, we tune three hyperparameters—learning rate (lr), weight decay (wd), and drop path rate (dp)—while keeping all others the same for all models. We conducted pilot experiments using ViT-B pre-trained with MoCo v3 to estimate reasonable hyperparameter ranges. Based on these estimates we established the following tuning protocol: 
1. For each initialization (from-scratch, supervised, etc.), we fix dp at 0.0 and perform a grid search over lr and wd using ViT-B and a 25 epoch schedule (or 100 epochs when initializing from scratch). We center a 3 × 3 grid at lr, wd = 1.6e−4, 0.1 and use doubled and halved values around the center. If a local optimum is not found (i.e. the best value is a boundary value), we expand the search. 
2. For ViT-B, we select dp from {0.0, 0.1, 0.2, 0.3} using a 50 epoch schedule for pre-trained initializations. The shorter 25 epoch schedule was unreliable and 100 epochs was deemed impractical. For random initialization we’re forced to use 100 epochs due to slow convergence. We found that dp = 0.1 is optimal for all initializations. 
3. For ViT-L, we adopt the optimal lr and wd from ViTB (searching with ViT-L is impractical) and find dp = 0.3 is best using the same procedure as for ViT-B.

为了使训练公式适应每个模型，我们调整了三个超参数——学习率 (lr)、权重衰减 (wd) 和下降路径率 (dp)——同时保持所有模型的所有其他参数相同。 我们使用经过 MoCo v3 预训练的 ViT-B 进行了试点实验，以估计合理的超参数范围。 基于这些估计，我们建立了以下调整协议：
1. 对于每次初始化(从头开始、监督等)，我们将 dp 固定为 0.0，并使用 ViT-B 和 25 个纪元计划(或从头开始初始化时为 100 个纪元)对 lr 和 wd 执行网格搜索。 我们在 lr, wd = 1.6e−4, 0.1 处以 3 × 3 网格为中心，并在中心周围使用双倍和减半的值。 如果未找到局部最优值(即最佳值是边界值)，我们将扩大搜索范围。
2. 对于 ViT-B，我们使用 50 个 epoch 计划从 {0.0, 0.1, 0.2, 0.3} 中选择 dp 进行预训练初始化。 较短的 25 个 epoch 时间表是不可靠的，100 个 epoch 被认为是不切实际的。 对于随机初始化，由于收敛速度慢，我们不得不使用 100 个 epoch。 我们发现 dp = 0.1 是所有初始化的最佳选择。
3. 对于 ViT-L，我们采用来自 ViTB 的最佳 lr 和 wd(使用 ViT-L 搜索是不切实际的)并且使用与 ViT-B 相同的程序发现 dp = 0.3 是最好的。

Limitations. The procedure above takes practical shortcuts to reduce the full hyperparameter tuning space. In particular, lr and wd are optimized separately from dp, thus the combination may be suboptimal. Further, we only tune lr and wd using ViT-B, therefore the choice may be suboptimal for ViT-L. We also tune lr and wd using a schedule that is 4× shorter than the longest schedule we eventually train at, which again may be suboptimal. Given these limitations we aim to avoid biasing results by applying the same tuning protocol to all initializations.

限制。 上述过程采用实用的捷径来减少完整的超参数调整空间。 特别是，lr 和 wd 与 dp 分开优化，因此组合可能不是最优的。 此外，我们仅使用 ViT-B 调整 lr 和 wd，因此选择对于 ViT-L 可能不是最优的。 我们还使用比我们最终训练的最长时间表短 4 倍的时间表调整 lr 和 wd，这同样可能是次优的。 鉴于这些限制，我们旨在通过对所有初始化应用相同的调整协议来避免产生偏差结果。

Finally, we note that we tune lr, wd, and dp on the COCO 2017 val split and report results on the same split. While technically not an ML best-practice, a multitude of comparisons on COCO val vs. test-dev results over many years demonstrate that overfitting in not a concern for this kind of low-degree-of-freedom hyperparameter tuning.(3 E.g., Table 2 in [29] (version 1) shows that text-dev APbox is systematically higher than val APbox in seven system-level comparisons.) 

最后，我们注意到我们在 COCO 2017 val 拆分上调整了 lr、wd 和 dp，并报告了同一拆分的结果。 虽然技术上不是 ML 最佳实践，但多年来对 COCO val 与 test-dev 结果的大量比较表明，过度拟合不是这种低自由度超参数调整的问题。(3 例如， [29](版本 1)中的表 2 显示 text-dev APbox 在七个系统级比较中系统地高于 val APbox。)

### 2.5. Additional Implementation Details
Images are padded during training and inference to form a 1024 × 1024 resolution input. During training, padding is necessary for batching. During (unbatched) inference, the input only needs to be a multiple of the ViT patch size on each side, which is possibly less than 1024 on one side. However, we found that such reduced padding performs worse (e.g., decrease of ∼0.5–1 APbox) than padding to the same resolution used during training, likely due to ViT’s use of positional information. Therefore, we use a 1024 × 1024 resolution input at inference time, even though the extra padding slows inference time by ∼30% on average.

图像在训练和推理过程中被填充以形成 1024 × 1024 分辨率的输入。 在训练期间，填充对于批处理是必要的。 在(非批处理)推理期间，输入只需要是每一侧 ViT 分块大小的倍数，一侧可能小于 1024。 然而，我们发现这种减少的填充比填充到训练期间使用的相同分辨率更差(例如，减少 ∼0.5-1 APbox)，这可能是由于 ViT 使用位置信息。 因此，我们在推理时使用 1024 × 1024 分辨率输入，即使额外的填充使推理时间平均减慢 ~30%。

## 3. Initialization Methods
We compare five initialization methods, which we briefly summarize below.

我们比较了五种初始化方法，我们在下面简要总结。

Random: All network weights are randomly initialized and no pre-training is used. The ViT backbone initialization follows the code of [1] and the Mask R-CNN initialization uses the defaults in Detectron2 [40].

Random：所有网络权值随机初始化，不使用预训练。 ViT 主干初始化遵循 [1] 的代码，Mask R-CNN 初始化使用 Detectron2 [40] 中的默认值。

Supervised: The ViT backbone is pre-trained for supervised classification using ImageNet-1k images and labels. We use the DeiT released weights [36] for ViT-B and the ViT-L weights from [16], which uses an even stronger training formula than DeiT to avoid overfitting (moreover, the DeiT release does not include ViT-L). ViT-B and ViT-L were pre-trained for 300 and 200 epochs, respectively.

监督：ViT 主干经过预训练，可以使用 ImageNet-1k 图像和标签进行监督分类。 我们使用 DeiT 发布的 ViT-B 权重 [36] 和 [16] 中的 ViT-L 权重，它使用比 DeiT 更强的训练公式来避免过度拟合(此外，DeiT 版本不包括 ViT-L)。 ViT-B 和 ViT-L 分别预训练了 300 和 200 个 epoch。

MoCo v3: We use the unsupervised ImageNet-1k pretrained ViT-B and ViT-L weights from the authors of [7] (ViT-B is public; ViT-L was provided via private communication). These models were pre-trained for 300 epochs.

MoCo v3：我们使用来自 [7] 作者的无监督 ImageNet-1k 预训练 ViT-B 和 ViT-L 权重(ViT-B 是公开的; ViT-L 是通过私人通信提供的)。 这些模型预训练了 300 个时期。

BEiT: Since ImageNet-1k pre-trained weights are not available, we use the official BEiT code release [1] to train ViT-B and ViT-L ourselves for 800 epochs (the default training length used in [1]) on unsupervised ImageNet-1k.

BEiT：由于 ImageNet-1k 预训练权重不可用，我们使用官方 BEiT 代码发布 [1] 在无监督上自行训练 ViT-B 和 ViT-L 800 个时期([1] 中使用的默认训练长度) ImageNet-1k。

MAE: We use the ViT-B and ViT-L weights pre-trained on unsupervised ImageNet-1k from the authors of [16]. These models were pre-trained for 1600 epochs using normalized pixels as the target.

MAE：我们使用 [16] 作者在无监督 ImageNet-1k 上预训练的 ViT-B 和 ViT-L 权重。 这些模型使用归一化像素作为目标进行了 1600 个 epoch 的预训练。

### 3.1. Nuisance Factors in Pre-training 预训练中的滋扰因素
We attempt to make comparisons as equally matched as possible, yet there are pre-training nuisance factors, listed below, that differ across methods. 
1. Different pre-training methods may use different numbers of epochs. We adopt the default number of pretraining epochs from the respective papers. While these values may not appear comparable, the reality is unclear: not all methods may benefit equally from longer training and not all methods have the same per-epoch training cost (e.g., BEiT uses roughly 3× more flops than MAE). 
2. BEiT uses learned relative position biases that are added to the self-attention logits [31] in each block, instead of the absolute position embeddings used by the other methods. To account for this, albeit imperfectly, we include both relative position biases and absolute position embeddings in all detection models regardless of their use in pretraining. For BEiT, we transfer the pre-trained biases and randomly initialize the absolute position embeddings. For all other methods, we zero-initialize the relative position biases and transfer the pre-trained absolute position embeddings. Relative position biases are shared across windowed attention blocks and (separately) shared across global attention blocks. When there is a spatial dimension mismatch between pre-training and fine-tuning, we resize the pre-trained parameters to the required fine-tuning resolution. 
3. BEiT makes use of layer scale [37] in pre-training, while the other methods do not. During fine-tuning, the BEiT-initialized model must also be parameterized to use layer scale with the pre-trained layer scaling parameters initialized from the pre-trained model. All other models do not use layer scale in pre-training or in fine-tuning. 
4. We try to standardize pre-training data to ImageNet1k, however BEiT uses the DALL·E [32] discrete VAE (dVAE), which was trained on ∼250 million proprietary and undisclosed images, as an image tokenizer. The impact of this additional training data is not fully understood.

我们尝试进行尽可能均等匹配的比较，但是下面列出的预训练滋扰因素因方法而异。
1. 不同的预训练方法可能使用不同数量的epochs。 我们采用各自论文中的默认预训练时期数。 虽然这些值可能看起来不具有可比性，但现实情况尚不清楚：并非所有方法都可以从更长的训练中同样受益，并且并非所有方法都具有相同的每轮训练成本(例如，BEiT 使用的触发器比 MAE 多大约 3 倍)。
2. BEiT 使用学习到的相对位置偏差，这些偏差被添加到每个块中的自我注意逻辑[31]，而不是其他方法使用的绝对位置嵌入。 为了解决这个问题，尽管不完美，我们在所有检测模型中都包含了相对位置偏差和绝对位置嵌入，而不管它们在预训练中的使用。 对于 BEiT，我们迁移预训练偏差并随机初始化绝对位置嵌入。 对于所有其他方法，我们零初始化相对位置偏差并迁移预训练的绝对位置嵌入。 相对位置偏差在窗口注意力块之间共享，并且(单独)在全局注意力块之间共享。 当预训练和微调之间存在空间维度不匹配时，我们将预训练参数调整为所需的微调分辨率。
3. BEiT在预训练中使用层尺度[37]，而其他方法则没有。 在微调期间，还必须对 BEiT 初始化的模型进行参数化，以使用层缩放以及从预训练模型初始化的预训练层缩放参数。 所有其他模型在预训练或微调中都不使用图层比例。
4. 我们尝试将预训练数据标准化为 ImageNet1k，但是 BEiT 使用 DALL·E [32] 离散 VAE (dVAE) 作为图像分词器，它在约 2.5 亿个专有和未公开的图像上进行了训练。 这种额外训练数据的影响尚不完全清楚。

## 4. Experiments and Analysis
### 4.1. Comparing Initializations
![Table 1](../images/ViTDet_benchmarking/tab_1.png)<br/>
Table 1. COCO object detection and instance segmentation using our ViT-based Mask R-CNN baseline. Results are reported on COCO 2017 val using the best schedule length (see Figure 2). Random initialization does not use any pre-training data, supervised initialization uses IN1k with labels, and all other initializations use IN1k without labels. Additionally, BEiT uses a dVAE trained on the proprietary DALL·E dataset of ∼250M images [32]. 
表 1. 使用我们基于 ViT 的 Mask R-CNN 基线的 COCO 目标检测和实例分割。 使用最佳时间表长度在 COCO 2017 val 上报告结果(见图 2)。 随机初始化不使用任何预训练数据，监督初始化使用带标签的 IN1k，所有其他初始化使用不带标签的 IN1k。 此外，BEiT 使用了一个 dVAE，该 dVAE 在大约 250M 图像的专有 DALL·E 数据集上训练 [32]。

Results. In Table 1, we compare COCO fine-tuning results using the pre-trained initializations and random initialization described in §3. We show results after maximizing APbox over the considered training lengths: 25, 50, or 100 epochs for pre-trained initializations, and 100, 200, or 400 epochs for random initialization. (We discuss convergence below.) 

结果。 在表 1 中，我们比较了使用 §3 中描述的预训练初始化和随机初始化的 COCO 微调结果。 我们在考虑的训练长度上最大化 APbox 后显示结果：预训练初始化为 25、50 或 100 个时期，随机初始化为 100、200 或 400 个时期。(我们在下面讨论收敛性。)

Next, we make several observations. 
1. Our updated Mask R-CNN trains smoothly with ViT-B and ViT-L backbones regardless of the initialization method. It does not exhibit instabilities nor does it require stabilizing techniques like gradient clipping.  
2. Training from scratch yields up to 1.4 higher APbox than fine-tuning from supervised IN1k pre-training (50.7 vs. 49.3). While the higher AP may sound surprising, the same trend is observed in [13]. Supervised pre-training is not always a stronger baseline than random initialization. 
3. The contrastive learning-based MoCo v3 underperforms random initialization’s AP and has similar results compared to supervised initialization. 
4. For ViT-B, BEiT and MAE outperform both random initialization by up to 1.4 APbox (50.3 vs. 48.9) and supervised initialization by up to 2.4 APbox (50.3 vs. 47.9). 
5. For ViT-L, the APbox gap increases, with BEiT and MAE substantially outperforming both random initialization by up to 2.6 APbox (53.3 vs. 50.7) and supervised initialization by up to 4.0 APbox (53.3 vs. 49.3).

接下来，我们进行几项观察。
1. 无论初始化方法如何，我们更新后的 Mask R-CNN 都能顺利地使用 ViT-B 和 ViT-L 骨干进行训练。 它没有表现出不稳定性，也不需要像梯度裁剪这样的稳定技术。
2. 从头训练产生的 APbox 比监督 IN1k 预训练的微调高出 1.4(50.7 对 49.3)。 虽然更高的 AP 听起来令人惊讶，但在 [13] 中观察到相同的趋势。 监督预训练并不总是比随机初始化更强的基线。
3. 基于对比学习的 MoCo v3 表现不及随机初始化的 AP，与监督初始化相比具有相似的结果。
4. 对于 ViT-B，BEiT 和 MAE 的性能优于随机初始化高达 1.4 个 APbox(50.3 对 48.9)和监督初始化高达 2.4 个 APbox(50.3 对 47.9)。
5. 对于 ViT-L，APbox 差距增加，BEiT 和 MAE 大大优于随机初始化高达 2.6 APbox(53.3 对 50.7)和监督初始化高达 4.0 APbox(53.3 对 49.3)。

![Figure 2](../images/ViTDet_benchmarking/fig_2.png)<br/>
Figure 2. Impact of fine-tuning epochs. Convergence plots for fine-tuning from 25 and 400 epochs on COCO. All pre-trained initializations converge much faster (∼4×) compared to random initialization, though they achieve varied peak APbox. The performance gap between the masking-based methods (MAE and BEiT) and all others is visually evident. When increasing model scale from ViT-B (top) to ViT-L (bottom), this gap also increases, suggesting that these methods may have superior scaling properties. 
图 2.微调时代的影响。 COCO 上 25 和 400 轮微调的收敛图。 与随机初始化相比，所有预训练的初始化收敛得更快(∼4×)，尽管它们实现了不同的峰值 APbox。 基于掩蔽的方法(MAE 和 BEiT)与所有其他方法之间的性能差距在视觉上是显而易见的。 当将模型比例从 ViT-B(顶部)增加到 ViT-L(底部)时，这种差距也会增加，这表明这些方法可能具有更好的缩放特性。

Convergence. In Figure 2 we show how pre-training impacts fine-tuning convergence. Given the tuned hyperparameters for each initialization method, we train models for 2× and 4× longer (and also 0.5× for random initialization). Generally, we find that all pre-trained initializations significantly accelerate convergence compared to random initialization, as observed in [18]. Most methods show signs of overfitting when the training schedule is made sufficiently long, typically by 100 epochs for pre-trained initializations and 400 epochs for random initialization. Based on this data, pre-training tends to accelerate training on COCO by roughly 4× compared to random initialization.

收敛。 在图 2 中，我们展示了预训练如何影响微调收敛。 给定每种初始化方法的调整超参数，我们训练模型的时间延长 2 倍和 4 倍(对于随机初始化也是 0.5 倍)。 一般来说，我们发现与随机初始化相比，所有预训练的初始化都显著加快了收敛速度，如 [18] 中所观察到的那样。 当训练计划足够长时，大多数方法都显示出过度拟合的迹象，通常预训练初始化为 100 轮，随机初始化为 400 轮。 基于这些数据，与随机初始化相比，预训练倾向于将 COCO 上的训练加速大约 4 倍。

We also note two caveats about these results: (i) The drop path rate should ideally be tuned for each training duration as we have observed that the optimal dp value may need to increase when models are trained for longer. (However, performing an exhaustive dp sweep for all initializations, model sizes, and training durations is likely computationally impractical.) (ii) Moreover, it may be possible to achieve better results in all cases by training for longer under a more complex training formula that employs heavier regularization and stronger data augmentation.

我们还注意到关于这些结果的两个警告：(i) 理想情况下，应该针对每个训练持续时间调整下降路径率，因为我们已经观察到，当模型训练时间更长时，最佳 dp 值可能需要增加。 (然而，对所有初始化、模型大小和训练持续时间执行详尽的 dp 扫描在计算上可能不切实际。)(ii)此外，通过在更复杂的训练公式下训练更长时间，可能在所有情况下都取得更好的结果 它采用更重的正则化和更强的数据增广。

Discussion. The COCO dataset is a challenging setting for transfer learning. Due to the large training set (∼118k images with ∼0.9M annotated objects), it is possible to achieve strong results when training from random initialization. We find that existing methods, like supervised IN1k or unsupervised MoCo v3 pre-training, actually underperform the AP of the random initialization baseline (though they yield faster convergence). Prior works reporting unsupervised transfer learning improvements on COCO (e.g., [17]) tend to show modest gains over supervised pre-training (e.g., ∼1 APbox) and do not include a strong random initialization baseline as we do here (because strong training formulae based on large-scale jitter had not yet been developed). Moreover, they use weaker models and report results that are overall much lower (e.g., ∼40 APbox) making it unclear how well the findings translate to state-of-the-art practices.

讨论。 COCO 数据集对于迁移学习来说是一个具有挑战性的设置。 由于训练集很大(~118k 图像和~0.9M 注释目标)，从随机初始化进行训练时可能会取得很好的结果。 我们发现现有方法，如有监督的 IN1k 或无监督的 MoCo v3 预训练，实际上低于随机初始化基线的 AP(尽管它们产生更快的收敛)。 先前的工作报告了 COCO 的无监督迁移学习改进(例如，[17])往往比有监督的预训练(例如，∼1 APbox)表现出适度的收益，并且不像我们在这里所做的那样包括强大的随机初始化基线(因为强大的训练 基于大规模抖动的公式尚未开发出来)。 此外，他们使用较弱的模型并报告总体上低得多的结果(例如，~40 APbox)，因此不清楚这些发现如何转化为最先进的实践。

We find that MAE and BEiT provide the first convincing results of substantial COCO AP improvements due to pretraining. Moreover, these masking-based methods show the potential to improve detection transfer learning as model size increases. We do not observe this important scaling trend with either supervised IN1k pre-training or unsupervised contrastive learning, as represented by MoCo v3.

我们发现 MAE 和 BEiT 提供了由于预训练而显著改善 COCO AP 的第一个令人信服的结果。 此外，这些基于掩码的方法显示出随着模型大小的增加而改进检测迁移学习的潜力。 我们没有通过有监督的 IN1k 预训练或无监督的对比学习观察到这种重要的缩放趋势，如 MoCo v3 所代表。

### 4.2. Ablations and Analysis
We ablate several factors involved in the system comparison, analyze model complexity, and report tuned hyperparameter values. For these experiments, we use MAE and 50 epoch fine-tuning by default.

我们消除了系统比较中涉及的几个因素，分析了模型的复杂性，并报告了调整后的超参数值。 对于这些实验，我们默认使用 MAE 和 50 个 epoch 微调。

Single-scale vs. Multi-scale. In Table 2 we compare our default FPN-based multi-scale detector to a single-scale variant. The single-scale variant simply applies RPN and RoIAlign [19] to the final 1/16th resolution feature map generated by the ViT backbone. The RoI heads and all other choices are the same between the systems (in particular, note that both use the same hybrid windowed/global attention). We observe that the multi-scale FPN design in creases APbox by ∼1.3-1.7 (e.g., 50.1 vs. 48.4), while increasing training and inference time by ∼5 and ∼10% relative, respectively. Multi-scale memory overhead is <1%.

单尺度与多尺度。 在表 2 中，我们将默认的基于 FPN 的多尺度检测器与单尺度变体进行了比较。 单尺度变体简单地将 RPN 和 RoIAlign [19] 应用于 ViT 主干生成的最终 1/16 分辨率特征图。 RoI 头和所有其他选择在系统之间是相同的(特别是，请注意，两者都使用相同的混合窗口/全局注意力)。 我们观察到多尺度 FPN 设计使 APbox 增加了 ~1.3-1.7(例如，50.1 对 48.4)，同时相对增加了 ~5% 和 ~10% 的训练和推理时间。 多尺度内存开销<1%。

![Table 2](../images/ViTDet_benchmarking/tab_2.png)<br/>
Table 2. Single-scale vs. multi-scale (FPN) ablation. FPN yields consistent improvements. Our default setting is marked in gray. 
表 2. 单尺度与多尺度 (FPN) 消融。 FPN 产生一致的改进。 我们的默认设置标记为灰色。

![Table 3](../images/ViTDet_benchmarking/tab_3.png)<br/>
Table 3. Memory and time reduction strategies. We compare methods for reducing memory and time when using ViT-L in Mask R-CNN. The strategies include: (1) replace all global selfattention with 14 × 14 non-overlapping windowed self-attention, (2) a hybrid that uses both windowed and global self-attention, or (3) all global attention with activation checkpointing. Without any of these strategies (row 4) an out-of-memory (OOM) error prevents training. We report APbox, peak GPU training memory, average per-iteration training time, and average per-image inference time using NVIDIA V100-32GB GPUs. The per-GPU batch size is 1. Our defaults (row 2) achieves a good balance between memory, time, and APbox metrics. In fact, our hybrid approach achieves comparable APbox to full global attention, while being much faster. 
表 3. 内存和时间减少策略。 我们比较了在 Mask R-CNN 中使用 ViT-L 时减少内存和时间的方法。 这些策略包括：(1)用 14 × 14 非重叠窗口化自我注意替换所有全局自我注意，(2)同时使用窗口化自我注意和全局自我注意的混合体，或(3)所有具有激活检查点的全局注意。 如果没有任何这些策略(第 4 行)，内存不足 (OOM) 错误会阻止训练。 我们使用 NVIDIA V100-32GB GPU 报告 APbox、峰值 GPU 训练内存、平均每次迭代训练时间和平均每张图像推理时间。 每个 GPU 的批大小为 1。我们的默认值(第 2 行)在内存、时间和 APbox 指标之间取得了良好的平衡。 事实上，我们的混合方法实现了与 APbox 相当的全球关注，同时速度更快。


Memory and Time Reduction. In Table 3 we compare several strategies for reducing memory and time complexity when using a standard ViT backbone in Mask R-CNN. Using a combination of 14 × 14 non-overlapping windowed self-attention together with four global attention blocks achieves a good balance between memory, training and inference time, and AP metrics. This finding motivates us to use this setting as our default. Somewhat surprisingly using only windowed attention is not catastrophic even though the backbone processes all windows entirely independently (APbox decreases from 53.3 to 50.7). This is likely due to cross-window computation introduced by convolutions and RoIAlign in the rest of the Mask R-CNN model.

内存和时间减少。 在表 3 中，我们比较了在 Mask R-CNN 中使用标准 ViT 主干时降低内存和时间复杂度的几种策略。 将 14 × 14 非重叠窗口化自注意力与四个全局注意力块结合使用，可以在内存、训练和推理时间以及 AP 指标之间实现良好的平衡。 这一发现促使我们将此设置用作默认设置。 有点令人惊讶的是，仅使用窗口注意力并不是灾难性的，即使主干完全独立地处理所有窗口(APbox 从 53.3 下降到 50.7)。 这可能是由于 Mask R-CNN 模型其余部分中的卷积和 RoIAlign 引入了跨窗口计算。

Positional Information. In the default BEiT code, the ViT is modified to use relative position biases [31] in each transformer block instead of adding absolute position embeddings to the patch embeddings. This choice is an orthogonal enhancement that is not used by the other pre-training methods (though it could be). In an attempt to make the comparison more equal, we include these biases (and absolute position embeddings) in all fine-tuning models by default, as discussed in §3.1.

位置信息。 在默认的 BEiT 代码中，ViT 被修改为在每个变换器块中使用相对位置偏差 [31]，而不是将绝对位置嵌入添加到分块嵌入中。 此选择是一种正交增广，其他预训练方法未使用(尽管可能)。 为了使比较更加平等，我们默认将这些偏差(和绝对位置嵌入)包含在所有微调模型中，如 §3.1 中所述。

In Table 4 we study the effect of relative position biases on fine-tuning performance. A detailed analysis is given in the caption. In summary, we observe that including relative position biases during fine-tuning may slightly improve APbox by ∼0.2–0.3 points (e.g., 53.0 to 53.3) for a model that was pre-trained with only absolute position embeddings. We also observe that pre-training relative position biases, as done by BEiT, may also have a slight positive effect of ∼0.1–0.3 points. Our practice of including both positional information types during fine-tuning appears to provide a reasonably fair comparison. We also note that using relative position biases introduces non-trivial overhead— it increases training and inference time by roughly 25% and 15% relative, respectively, increases memory by ∼15% (even with shared biases), and perhaps should be avoided. 

在表 4 中，我们研究了相对位置偏差对微调性能的影响。 标题中给出了详细的分析。 总之，我们观察到，对于仅使用绝对位置嵌入进行预训练的模型，在微调期间包括相对位置偏差可能会略微提高 APbox ~0.2-0.3 点(例如，53.0 到 53.3)。 我们还观察到，正如 BEiT 所做的那样，训练前的相对位置偏差也可能具有约 0.1-0.3 点的轻微积极影响。 我们在微调期间包括两种位置信息类型的做法似乎提供了一个相当公平的比较。 我们还注意到，使用相对位置偏差会带来不小的开销——它分别增加了大约 25% 和 15% 的训练和推理时间，增加了约 15% 的内存(即使有共享偏差)，也许应该避免。

![Table 4](../images/ViTDet_benchmarking/tab_4.png)<br/>
Table 4. Positional information ablation. In the BEiT code, the ViT is modified to use relative position biases (rel) instead of absolute position embeddings (abs). We study how these components impact results based on their use in pre-training (pt) and under various treatments in fine-tuning: (i) pt: initialized with pre-trained values; (ii) rand: random initialization; (iii) zero: initialized at zero; and (iv) no: this positional information is not used in the fine-tuned model. For BEiT† (row 3), we pre-train an additional model (ViT-L only) that, like MAE, uses absolute position embeddings instead of relative position biases. Our default settings are marked in gray. Comparing (1) and (2), we observe that pretrained relative position bias initialization provides a slight benefit over zero initialization. Comparing (1,2) to (3), we see that BEiT pre-trained with absolute position embeddings performs similarly (perhaps slightly worse) to pre-training with relative position biases. Comparing (4) and (5), we see that including relative position biases in addition to absolute position embeddings provides a small improvement. 
表 4. 位置信息消融。 在 BEiT 代码中，ViT 被修改为使用相对位置偏差 (rel) 而不是绝对位置嵌入 (abs)。 我们根据这些组件在预训练 (pt) 中的使用以及微调中的各种处理来研究这些组件如何影响结果：(i) pt：使用预训练值初始化;  (ii) rand：随机初始化;  (iii) 零：初始化为零;  (iv) 否：此位置信息未用于微调模型。 对于 BEiT†(第 3 行)，我们预训练了一个额外的模型(仅限 ViT-L)，它与 MAE 一样，使用绝对位置嵌入而不是相对位置偏差。 我们的默认设置标记为灰色。 比较 (1) 和 (2)，我们观察到预训练的相对位置偏差初始化比零初始化略有优势。 比较 (1,2) 和 (3)，我们看到使用绝对位置嵌入进行预训练的 BEiT 与使用相对位置偏差进行预训练的表现相似(可能稍差)。 比较 (4) 和 (5)，我们看到除了绝对位置嵌入之外还包括相对位置偏差提供了一个小的改进。


Pre-training Epochs. In Figure 3 we study the impact of MAE pre-training epochs on COCO APbox by sweeping pre-training epochs from 100 to 1600 (the default). The results show that pre-training duration has a significant impact on transfer learning performance with large increases in APbox continuing from 100 to 800 epochs. There is still a small improvement from 800 to 1600 epochs (+0.2 from 53.1 to 53.3), though the gradient has largely flattened.

预训练时代。 在图 3 中，我们通过将预训练时期从 100 扫到 1600(默认值)来研究 MAE 预训练时期对 COCO APbox 的影响。 结果表明，预训练持续时间对迁移学习性能有显著影响，APbox 的大幅增加持续从 100 到 800 个时期。 从 800 到 1600 个 epoch(从 53.1 到 53.3 +0.2)仍然有小幅改进，尽管梯度已经基本趋于平缓。

TIDE Error Type Analysis. In Figure 4 we show the error type analysis generated by the TIDE toolbox [3]. A detailed description and analysis is given in the caption. The analysis reveals more granular information about where MAE and BEiT improve overall AP relative to the other initializations. In summary, we observe that all initializations lead to roughly the same classification performance for correctly localized objects, however the MAE and BEiT initializations improve localization compared to the other initializations. We observe an even stronger effect when looking at missed detections: the masking-based initializations yield notably higher recall than the other initializations and thus leave fewer undetected objects. This higher recall creates a small increase in background errors, thus leading to better overall AP.

TIDE 错误类型分析。 在图 4 中，我们展示了 TIDE 工具箱 [3] 生成的错误类型分析。 标题中给出了详细的描述和分析。 该分析揭示了有关 MAE 和 BEiT 相对于其他初始化在哪些方面提高整体 AP 的更详情。 总之，我们观察到对于正确定位的目标，所有初始化都导致大致相同的分类性能，但是与其他初始化相比，MAE 和 BEiT 初始化改善了定位。 在查看漏检时，我们观察到更强烈的效果：基于掩码的初始化比其他初始化产生更高的召回率，因此留下更少的未检测到的目标。 这种更高的召回率会导致背景错误略有增加，从而导致更好的整体 AP。

![Figure 3](../images/ViTDet_benchmarking/fig_3.png)<br/>
Figure 3. Impact of pre-training epochs. Increasing MAE pretraining from 100 to 800 epochs confers large transfer learning gains. The improvements start to plateau after 800 epochs
图 3. 训练前时期的影响。 将 MAE 预训练从 100 增加到 800 轮可以带来巨大的迁移学习收益。 在 800 个 epoch 之后，改进开始趋于平稳

![Figure 4](../images/ViTDet_benchmarking/fig_4.png)<br/>
Figure 4. TIDE analysis. We plot the ∆APbox metric at an intersection-over-union (IoU) threshold of 0.5 as defined in [3]. Each bar shows how much AP can be added to the detector if an oracle fixes a certain error type. The error types are: cls: localized correctly (IoU ≥0.5), but classified incorrectly; loc: classified correctly, but localized incorrectly (IoU in [0.1, 0.5)); cls+loc: classified incorrectly and localized incorrectly; dup: detection would be correct if not for a higher scoring correct detection; bg: detection is in the background (IoU <0.1); miss: all undetected ground-truth objects not covered by other error types. (See [3] for more details and discussion.) We observe that the masking-based initializations (MAE and BEiT) make fewer localization errors than MoCo v3 and supervised initialization (random initialization is somewhere in-between) and, even more so, have fewer missed detections. The other error types are more similar across initializations. 
图 4.TIDE 分析。 我们在 [3] 中定义的 0.5 交并联合 (IoU) 阈值处绘制 ΔAPbox 度量。 每个条显示如果 oracle 修复了某种错误类型，可以将多少 AP 添加到检测器。 错误类型为：cls：定位正确(IoU≥0.5)，但分类错误;  loc：分类正确，但定位不正确(IoU in [0.1, 0.5));  cls+loc：分类错误，定位错误;  dup：如果没有更高评分的正确检测，检测将是正确的;  bg：检测处于后台(IoU <0.1);  miss：所有未被其他错误类型覆盖的未检测到的地面实况目标。 (有关更多详情和讨论，请参阅 [3]。)我们观察到基于掩码的初始化(MAE 和 BEiT)比 MoCo v3 和监督初始化(随机初始化介于两者之间)产生的定位错误更少，甚至更多， 漏检更少。 其他错误类型在初始化过程中更相似。


Model Complexity. Table 5 compares various complexity and wall-clock time measures of our specific Mask RCNN configuration. We also report these measures using a ResNet-101 backbone instead of ViT. When trained from scratch, both ResNet-101 and ViT-B backbones achieve 48.9 APbox. At inference time, the ResNet-101 backbone is much faster; however, during training ViT-B reaches peak performance at 200 epochs compared to 400 for ResNet101. ResNet-101 is not yet able to benefit from BEiT or MAE pre-training and therefore lags behind ViT-B in APbox (∼1 point) when those methods are used for initialization. 

模型复杂性。 表 5 比较了我们特定 Mask RCNN 配置的各种复杂性和挂钟时间度量。 我们还使用 ResNet-101 主干而不是 ViT 报告这些措施。 从头开始训练时，ResNet-101 和 ViT-B 主干都达到 48.9 APbox。 在推理时，ResNet-101 主干要快得多;  然而，在训练期间，ViT-B 在 200 个 epoch 时达到峰值性能，而 ResNet101 为 400 个。 ResNet-101 还不能从 BEiT 或 MAE 预训练中受益，因此当使用这些方法进行初始化时，在 APbox 中落后于 ViT-B(～1 分)。

![Table 5](../images/ViTDet_benchmarking/tab_5.png)<br/>
Table 5. Model complexity for inference with the specific Mask R-CNN configuration used in this report. For ViT, the image resolution is 1024 × 1024 (padded as necessary). The flop and activation counts are measured at runtime and vary based on the number of detected objects. We report the mean ± one standard deviation from 100 validation images. Results change very slightly when using different initializations. For reference, we report results using the ResNet-101 backbone, which can (and does) use non-square inputs at inference time (longest side is 1024); otherwise inference settings are the same. The ResNet-101 based Mask R-CNN achieves 48.9 APbox when trained from scratch for 400 epochs. We also report wall-clock speed in frames-per-second (fps) on an NVIDIA V100-32GB GPU.

表 5. 使用本报告中使用的特定 Mask R-CNN 配置进行推理的模型复杂性。 对于 ViT，图像分辨率为 1024 × 1024(根据需要进行填充)。 触发器和激活计数在运行时测量，并根据检测到的目标数量而变化。 我们报告了 100 张验证图像的平均值±一个标准差。 使用不同的初始化时，结果变化很小。 作为参考，我们使用 ResNet-101 主干报告结果，它可以(并且确实)在推理时使用非方形输入(最长边为 1024);  否则推理设置相同。 基于 ResNet-101 的 Mask R-CNN 在从头开始训练 400 个时期后达到 48.9 APbox。 我们还报告了 NVIDIA V100-32GB GPU 上以每秒帧数 (fps) 为单位的挂钟速度。

Hyperparameter Tuning. All pre-trained initializations preferred wd = 0.1 for fine-tuning. Random initialization benefitted from stronger regularization and selected a higher setting of 0.2. Most methods selected lr = 8.0e−5, except for random initialization and MoCo v3 initialization, which both preferred a higher setting of 1.6e−4. As described previously, the drop path rate could not be reliably tuned using shorter schedules. As a result, we tuned dp with 50 epoch training for pre-trained initializations and 100 epoch training for random initialization. Based on this tuning, all initializations selected dp = 0.1 when using ViT-B and 0.3 when using ViT-L.

超参数调整。 所有预训练的初始化首选 wd = 0.1 进行微调。 随机初始化受益于更强的正则化并选择了更高的设置 0.2。 大多数方法选择 lr = 8.0e-5，除了随机初始化和 MoCo v3 初始化，它们都更喜欢 1.6e-4 的更高设置。 如前所述，使用较短的时间表无法可靠地调整下降路径率。 因此，我们使用 50 个 epoch 训练进行预训练初始化和 100 个 epoch 随机初始化训练来调整 dp。 基于此调整，所有初始化在使用 ViT-B 时选择 dp = 0.1，在使用 ViT-L 时选择 0.3。

## 5. Conclusion
We have presented techniques that enable the practical use of standard ViT models as the backbone in Mask R-CNN. These methods yield acceptable training memory and time, while also achieving strong results on COCO without involving too many complex extensions. Using these techniques, we find effective training formulae that enable us to benchmark five different ViT initialization methods. We show that random initialization takes ∼4× longer than any of the pre-trained initializations, but achieves a meaningfully higher AP than ImageNet-1k supervised pre-training. We find that MoCo v3, a representative of contrastive unsupervised learning, performs nearly the same as supervised pre-training (and thus worse than random initialization). Importantly, we witness an exciting new result: masking-based methods (BEiT and MAE) show considerable gains over both supervised and random initialization and these gains increase as model size increases. This scaling behavior is not observed with either supervised or MoCo v3-based initialization. 

我们已经提出了一些技术，可以将标准 ViT 模型作为 Mask R-CNN 的主干实际使用。 这些方法产生了可接受的训练内存和时间，同时还在 COCO 上取得了很好的结果，而不涉及太多复杂的扩展。 使用这些技术，我们找到了有效的训练公式，使我们能够对五种不同的 ViT 初始化方法进行基准测试。 我们表明，随机初始化比任何预训练初始化花费的时间长约 4 倍，但实现了比 ImageNet-1k 监督预训练高得多的 AP。 我们发现 MoCo v3 作为对比无监督学习的代表，其性能几乎与有监督的预训练相同(因此比随机初始化差)。 重要的是，我们见证了一个令人兴奋的新结果：基于掩码的方法(BEiT 和 MAE)在监督和随机初始化方面都显示出相当大的收益，并且这些收益随着模型大小的增加而增加。 使用受监督或基于 MoCo v3 的初始化都不会观察到这种缩放行为。

## References
1. Hangbo Bao, Li Dong, and Furu Wei. BEiT: Bert pretraining of image transformers. arXiv:2106.08254, 2021.
2. Josh Beal, Eric Kim, Eric Tzeng, Dong Huk Park, Andrew Zhai, and Dmitry Kislyuk. Toward transformer-based object detection. arXiv preprint arXiv:2012.09958, 2020.
3. Daniel Bolya, Sean Foley, James Hays, and Judy Hoffman. TIDE: A general toolbox for identifying object detection errors. In ECCV, 2020.
4. Zhaowei Cai and Nuno Vasconcelos. Cascade R-CNN: Delving into high quality object detection. In CVPR, 2018.
5. Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, ´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.
6. Kai Chen, Jiangmiao Pang, Jiaqi Wang, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jianping Shi, Wanli Ouyang, et al. Hybrid task cascade for instance segmentation. In CVPR, 2019.
7. Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised Vision Transformers. In ICCV, 2021.
8. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.
9. Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015.
10. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
11. Alaaeldin El-Nouby, Hugo Touvron, Mathilde Caron, Piotr Bojanowski, Matthijs Douze, Armand Joulin, Ivan Laptev, Natalia Neverova, Gabriel Synnaeve, Jakob Verbeek, et al. XCiT: Cross-covariance image transformers. arXiv preprint arXiv:2106.09681, 2021.
12. Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. arXiv preprint arXiv:2104.11227, 2021.
13. Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, TsungYi Lin, Ekin D Cubuk, Quoc V Le, and Barret Zoph. Simple copy-paste is a strong data augmentation method for instance segmentation. In CVPR, 2021.
14. Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.
15. Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noord- ´ huis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677, 2017.
16. Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable ´ vision learners. arXiv preprint arXiv:2111.06377, 2021.
17. Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.
18. Kaiming He, Ross Girshick, and Piotr Dollar. Rethinking ´ ImageNet pre-training. In ICCV, 2019.
19. Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir- ´ shick. Mask R-CNN. In ICCV, 2017.
20. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.
21. Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv:1606.08415, 2016.
22. Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In ECCV, 2016.
23. Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.
24. Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. Fractalnet: Ultra-deep neural networks without residuals. ICLR, 2016.
25. Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.
26. Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Improved multiscale vision transformers for classification and detection. In preparation, 2021.
27. Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, ´ Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In CVPR, 2017.
28. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence ´ Zitnick. Microsoft COCO: Common objects in context. In ECCV. 2014.
29. Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030, 2021.
30. Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.
31. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.
32. Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. arXiv:2102.12092, 2021.
33. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NeurIPS, 2015.
34. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. TPAMI, 2017. 8
35. Pierre Sermanet, Koray Kavukcuoglu, Sandhya Chintala, and Yann LeCun. Pedestrian detection with unsupervised multi-stage feature learning. In CVPR, 2013.
36. Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J ´ egou. Training ´ data-efficient image transformers & distillation through attention. arXiv:2012.12877, 2020.
37. Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Herve J ´ egou. Going deeper with im- ´ age transformers. arXiv preprint arXiv:2103.17239, 2021.
38. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.
39. Yuxin Wu and Kaiming He. Group normalization. In ECCV, 2018.
40. Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github. com/facebookresearch/detectron2, 2019. 9
