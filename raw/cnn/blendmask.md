# BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation
BlendMask：自上而下满足自下而上的实例分割 2020-1-2 https://arxiv.org/abs/2001.00309

## Abstract
Instance segmentation is one of the fundamental vision tasks. Recently, fully convolutional instance segmentation methods have drawn much attention as they are often simpler and more efficient than two-stage approaches like Mask R-CNN. To date, almost all such approaches fall behind the two-stage Mask R-CNN method in mask precision when models have similar computation complexity, leaving great room for improvement. In this work, we achieve improved mask prediction by effectively combining instancelevel information with semantic information with lowerlevel fine-granularity. Our main contribution is a blender module which draws inspiration from both top-down and bottom-up instance segmentation approaches. The proposed BlendMask can effectively predict dense per-pixel position-sensitive instance features with very few channels, and learn attention maps for each instance with merely one convolution layer, thus being fast in inference. BlendMask can be easily incorporated with the state-of-the-art onestage detection frameworks and outperforms Mask R-CNN under the same training schedule while being 20% faster. A light-weight version of BlendMask achieves 34.2% mAP at 25 FPS evaluated on a single 1080Ti GPU card. Because of its simplicity and efficacy, we hope that our BlendMask could serve as a simple yet strong baseline for a wide range of instance-wise prediction tasks. Code is available at https://github.com/aim-uofa/AdelaiDet/

实例分割是基本的视觉任务之一。最近，全卷积实例分割方法引起了广泛注意，因为它们通常比掩码R-CNN这样的两阶段方法更简单、更有效。到目前为止，当模型具有相似的计算复杂度时，几乎所有这些方法在掩码精度方面都落后于两阶段掩码R-CNN方法，这留下了很大的改进空间。在这项工作中，我们通过有效地将实例级信息与具有较低粒度的语义信息相结合来实现改进的掩码预测。我们的主要贡献是一个blender模块，它从自上而下和自下而上的实例分割方法中获得灵感。提出的BlendMask可以用很少的通道有效地预测每像素位置敏感的密集实例特征，并且只需一个卷积层就可以学习每个实例的注意力图，因此推理速度很快。BlendMask可以很容易地与最先进的单阶段检测框架结合在一起，在相同的训练计划下，比Mask R-CNN更快20%。在单个1080Ti GPU卡上，BlendMask的轻量级版本在25 FPS下获得34.2%的mAP。由于它的简单性和有效性，我们希望我们的BlendMask可以作为一个简单而强大的基线，用于广泛的实例预测任务。代码位于https://github.com/aim-uofa/AdelaiDet/

## 1. Introduction
The top performing object detectors and segmenters often follow a two-stage paradigm. They consist of a fully convolutional network, region proposal network (RPN), to perform dense prediction of the most likely regions of interest (RoIs). A set of light-weight networks, a.k.a. heads, are applied to re-align the features of RoIs and generate predictions [24]. The quality and speed for mask generation is strongly tied to the structure of the mask heads. In addition, it is difficult for independent heads to share features with related tasks such as semantic segmentation which causes trouble for network architecture optimization.

性能最好的目标检测器和分割器通常遵循两个阶段的范例。它们由完全卷积网络、区域建议网络(RPN)组成，用于对最可能感兴趣区域(RoI)进行密集预测。一组轻量级网络，也称为头部，用于重新排列RoI的特征并生成预测[24]。掩码生成的质量和速度与掩码头的结构密切相关。此外，独立头很难与相关任务(如语义分割)共享特征，这给网络架构优化带来了麻烦。

Figure 1 – Blending process. We illustrate an example of the learned bases and attentions. Four bases and attention maps are shown in different colors. The first row are the bases, and the second row are the attentions. Here ⊗ represents element-wise product and ⊕ is element-wise sum. Each basis multiplies its attention and then is summed to output the final mask. 

图1–混合过程。我们举例说明了学习的基础和注意事项。四个基础和注意力图以不同的颜色显示。第一排是基础，第二排是注意点。在这里⊗ 表示元素的乘积⊕ 是逐元素求和。每个基础都会倍增其注意力，然后相加以输出最终掩码。

Recent advances in one-stage object detection prove that one-stage methods such as FCOS can outperform their twostage counterparts in accuracy [25]. Enabling such onestage detection frameworks to perform dense instance segmentation is highly desirable as 1) models consisting of only conventional operations are simpler and easier for cross-platform deployment; 2) a unified framework provides convenience and flexibility for multi-task network architecture optimization.

单阶段目标检测的最新进展证明，单阶段方法(如FCOS)在精度上优于两阶段方法[25]。使这种单阶段检测框架能够执行密集的实例分割是非常理想的，因为,
1. 仅由常规操作组成的模型对于跨平台部署更简单、容易; 
2. 统一的框架为多任务网络架构优化提供了便利和灵活性。

Dense instance segmenters can date back to DeepMask [23], a top-down approach which generates dense instance masks with a sliding window. The representation of mask is encoded into a one-dimensional vector at each spatial location. Albeit being simple in structure, it has several obstacles in training that prevent it from achieving superior performance: 1) local-coherence between features and masks is lost; 2) the feature representation is redundant because a mask is repeatedly encoded at each foreground feature; 3) position information is degraded after downsampling with strided convolutions.

密集实例分割器可以追溯到DeepMask[23]，这是一种自顶向下的方法，通过滑动窗口生成密集实例掩码。掩码的表示在每个空间位置被编码为一维矢量。尽管结构简单，但它在训练中存在一些障碍，阻碍了它取得优异的性能：
1. 特征和掩码之间的局部一致性丢失; 
2. 特征表示是冗余的，因为掩码在每个前景特征处被重复编码; 
3. 位置信息在用跨过卷积进行下采样之后劣化。

The first issue was studied by Dai et al. [8], who attempt to retain local-coherence by keeping multiple positionsensitive maps. This idea has been explored to its limits by Chen et al. [7], who proposes a dense aligned representation for each location of the target instance mask. However, this approach trades representation efficiency for alignment, making the second issue difficult to resolve. The third issue prevents heavily downsampled features to provide detailed instance information.

Daiet al [8]研究了第一个问题，他们试图通过保持多个位置敏感图来保持局部一致性。Chenet al [7]已经探索了这个想法的极限，他为目标实例掩码的每个位置提出了密集对齐表示。然而，这种方法以表现效率换取一致性，使得第二个问题难以解决。第三个问题阻止了大量的下采样功能来提供详细的实例信息。

Recognizing these difficulties, a line of research takes a bottom-up strategy [1, 21, 22]. These methods generate dense per-pixel embedding features and use some techniques to group them. Grouping strategies vary from simple clustering [4] to graph-based algorithms [21] depending on the embedding characteristics. By performing per-pixel predictions, the local-coherence and position information is well retained. The shortcomings for bottom-up approaches are: 1) heavy reliance on the dense prediction quality, leading to sub-par performance and fragmented/joint masks; 2) limited generalization ability to complex scenes with a large number of classes; 3) requirement for complex postprocessing techniques.

认识到这些困难，一系列研究采取了自下而上的策略[1，21，22]。这些方法生成每像素密集的嵌入特征，并使用一些技术对其进行分组。根据嵌入特性，分组策略从简单的聚类[4]到基于图的算法[21]不等。通过执行每像素预测，很好地保留了局部相干和位置信息。自下而上方法的缺点是：
1. 严重依赖密集的预测质量，导致性能低于标准，并且存在分段/联合掩码; 
2. 对具有大量类的复杂场景的泛化能力有限; 
3. 对复杂后处理技术的要求。

In this work, we consider hybridizing top-down and bottom-up approaches. We recognize two important predecessors, FCIS [18] and YOLACT [3]. They predict instance-level information such as bounding box locations and combine it with per-pixel predictions using cropping (FCIS) and weighted summation (YOLACT), respectively. We argue that these overly simplified assembling designs may not provide a good balance for the representation power of top- and bottom-level features.

在这项工作中，我们考虑混合自上而下和自下而上的方法。我们认识到两个重要的前辈，FCIS[18]和YOLACT[3]。它们预测实例级信息，例如边界框位置，并分别使用裁剪(FCIS)和加权求和(YOLACT)将其与每像素预测相结合。我们认为，这些过于简化的装配设计可能无法很好地平衡顶层和底层特征的表现力。

Higher-level features correspond to larger receptive field and can better capture overall information about instances such as poses, while lower-level features preserve better location information and can provide finer details. One of the focuses of our work is to investigate ways to better merging these two in fully convolutional instance segmentation. More specifically, we generalize the operations for proposal-based mask combination by enriching the instance-level information and performing more finegrained position-sensitive mask prediction. We carry out extensive ablation studies to discover the optimal dimensions, resolutions, alignment methods, and feature locations. 

较高级别的特征对应于较大的感受野，可以更好地捕捉有关姿势等实例的整体信息，而较低级别的特征保留更好的位置信息，可以提供更精细的细节。我们工作的重点之一是研究如何在完全卷积实例分割中更好地合并这两者。更具体地说，我们通过丰富实例级信息并执行更细粒度的位置敏感掩码预测来概括基于建议的掩码组合的操作。我们进行了广泛的消融研究，以发现最佳尺寸、分辨率、对准方法和特征位置。

Concretely, we are able to achieve the followings:
* We devise a flexible method for proposal-based instance mask generation called blender, which incorporate rich instance-level information with accurate dense pixel features. In head-to-head comparison, our blender surpasses the merging techniques in YOLACT [3] and FCIS [18] by 1.9 and 1.3 points in mAP on the COCO dataset respectively.
* We propose a simple architecture, BlendMask, which is closely tied to the state of the art one-stage object detector, FCOS [25], by adding moldiest computation overhead to the already simple framework.
* One obvious advantage of BlendMask is that its inference time does not increase with the number of predictions as conventional two-stage methods do, which makes it more robust in real-time scenarios.
* The performance of BlendMask achieves mAP of 37.0% with the ResNet-50 [15] backbone and 38.4% mAP with ResNet-101 on the COCO dataset, outperforming Mask R-CNN [13] in accuracy while being about 20% faster. We set new records for fully convolutional instance segmentation, surpassing TensorMask [7] by 1.1 points in mask mAP with only half training iterations and 1/5 inference time. To our knowledge, BlendMask may be the first algorithm that can outperform Mask R-CNN in both mask AP and inference efficiency.
* BlendMask can naturally solve panoptic segmentation without any modification (refer to Section 4.4), as the bottom module of BlendMask can segment ‘things and stuff’ simultaneously.
* Compared with Mask R-CNN’s mask head, which is typically of 28 × 28 resolution, BlendMask’s the bottom module is able to output masks of much higher resolution, due to its flexibility and the bottom module not being strictly tied to the FPN. Thus BlendMask is able to produce masks with more accurate edges, as shown in Figure 4. For applications such as graphics, this can be very important.
* The proposed BlendMask is general and flexible. With minimal modification, we can apply BlendMask to solve other instance-level recognition tasks such as keypoint detection.

具体而言，我们能够实现以下目标：
* 我们设计了一种灵活的基于提案的实例掩码生成方法，称为blender，它将丰富的实例级信息与精确的密集像素特征相结合。在面对面比较中，我们的blender在COCO数据集上的mAP分别超过YOLACT[3]和FCIS[18]中的合并技术1.9和1.3点。
* 我们提出了一个简单的架构BlendMask，它与最先进的一级目标检测器FCOS[25]紧密相连，在已经简单的框架中添加了最陈旧的计算开销。
* BlendMask的一个明显优点是，它的推理时间不会像传统的两阶段方法那样随着预测次数的增加而增加，这使得它在实时场景中更加健壮。
* BlendMask的性能在ResNet-50[15]主干上实现了37.0%的mAP，在COCO数据集上使用ResNet-101实现了38.4%的mAP。我们为完全卷积实例分割设置了新的记录，在掩码mAP中仅用一半的训练迭代和1/5的推理时间就超过了TensorMask[7]1.1个点。据我们所知，BlendMask可能是第一个在掩码AP和推理效率方面都优于Mask R-CNN的算法。
* BlendMask可以自然地解决全景分割，无需任何修改(请参阅第4.4节)，因为BlendMask的底部模块可以同时分割“东西”。
* 与通常分辨率为28×28的Mask R-CNN掩码头相比，BlendMask的底部模块能够输出分辨率更高的掩码，这是因为其灵活性以及底部模块不与FPN严格相连。因此，BlendMask能够生成具有更精确边缘的遮罩，如图4所示。对于图形等应用程序，这可能非常重要。
* 提出的BlendMask是通用且灵活的。只需少量修改，我们就可以应用BlendMask来解决其他实例级识别任务，例如关键点检测。

## 2. Related work
Anchor-free object detection. Recent advances in object detection unveil the possibilities of removing bounding box anchors [25], largely simplifying the detection pipeline. This much simpler design improves the box average precision (APbb) by 2.7% comparing to its anchor-based counterpart RetinaNet [19]. One possible reason responsible for the improvement is that without the restrictions of predefined anchor shapes, targets are freely matched to prediction features according to their effective receptive field. The hints for us are twofold. First, it is important to map target sizes with proper pyramid levels to fit the effective receptive field for the features. Second, removing anchors enables us to assign heavier duties to the top-level instance prediction module without introducing overall computation overhead. For example, inferring shape and pose information alongside the bounding box detection would take about eight times more computation for anchor-based frameworks than ours. This makes it intractable for anchor based detectors to balance the top vs. bottom workload (i.e., learning instance-aware maps1 vs. bases). We assume that this might be the reason why YOLACT can only learn one single scalar coefficient for each prototype/basis given an instance when computation complexity is taken into account. Only with the use of anchor-free bounding box detectors, this restriction is removed.

无锚目标检测. 目标检测的最新进展揭示了移除边界框锚的可能性[25]，大大简化了检测管道。与基于锚的RetinaNet相比，这种简单得多的设计将边框平均精度(APbb)提高了2.7%[19]。导致这种改进的一个可能原因是，在没有预定义锚形状的限制的情况下，目标可以根据其有效感受野自由地与预测特征匹配。对我们的提示是双重的。首先，重要的是用适当的金字塔级别映射目标大小，以适应特征的有效感受野。其次，移除锚使我们能够为顶级实例预测模块分配更重的任务，而不会引入总体计算开销。例如，在边界框检测的同时推断形状和姿势信息，对于基于锚的框架来说，所需的计算量大约是我们的八倍。这使得基于锚的检测器难以平衡顶部和底部的工作负载(即，学习实例感知映射1和基础)。我们假设这可能是YOLACT在考虑计算复杂性的情况下，对于给定的实例，每个原型/基只能学习一个标量系数的原因。只有使用无锚边界框检测器时，此限制才被删除。

Figure 2 – BlendMask pipeline. Our framework builds upon the state-of-the-art FCOS object detector [25] with minimal modification. The bottom module uses either backbone or FPN features to predict a set of bases. A single convolution layer is added on top of the detection towers to produce attention masks along with each bounding box prediction. For each predicted instance, the blender crops the bases with its bounding box and linearly combine them according the learned attention maps. Note that the Bottom Module can take features either from ‘C’, or ‘P’ as the input. 
图2– BlendMask管道. 我们的框架以最先进的FCOS目标检测器[25]为基础，只需进行最小的修改。底部模块使用主干或FPN特征来预测一组基础。单个卷积层被添加到检测塔的顶部，以产生注意掩码以及每个边界框预测。对于每个预测的实例，blender用其边界框裁剪基础，并根据学习到的注意力图线性组合它们。请注意，底部模块可以将“C”或“P”中的功能作为输入。

Detect-then-segment instance segmentation. The dominant instance segmentation paradigms take the two-stage methodology, first detecting the objects and then predicting the foreground masks on each of the proposals. The success of this framework partially is due to the alignment operation, RoIAlign [13], which provides local-coherence for the second-stage RoI heads missing in all one-stage top-down approaches. However, two issues exist in two-stage frameworks. For complicated scenarios with many instances, inference time for two-stage methods is proportional to the number of instances. Furthermore, the resolution for the RoI features and resulting mask is limited. We discuss the second issue in detail in Section 4.3.

检测后分割 实例分割. 主要的实例分割范式采用两阶段方法，首先检测对象，然后预测每个提案上的前景遮罩。该框架的成功部分归功于校准操作RoIAlign[13]，该操作为所有单阶段自顶向下方法中缺失的第二阶段RoI头提供了局部一致性。然而，在两阶段框架中存在两个问题。对于具有许多实例的复杂场景，两阶段方法的推理时间与实例数量成比例。此外，RoI特征和结果掩码的分辨率是有限的。我们在第4.3节中详细讨论了第二个问题。

These problems can be partly solved by replacing a RoI head with a simple crop-and-assemble module. In FCIS, Li et al. [18] add a bottom module to a detection network, for predicting position-sensitive score maps shared by all instances. This technique was first used in R-FCN [9] and later improved in MaskLab [5]. Each channel of the k 2 score maps corresponds to one crop of k × k evenly partitioned grid tiles of the proposal. Each score map represents the likelihood of the pixel belongs to a object and is at a certain relative position. Naturally, a higher resolution for location crops leads to more accurate predictions, but the computation cost also increases quadratically. Moreover, there are special cases where FCIS representation is not sufficient. When two instances share center positions (or any other relative positions), the score map representation on that crop is ambiguous, it is impossible to tell which instance this crop is describing.

这些问题可以通过用简单的裁剪和组装模块替换RoI头来部分解决。在FCIS中，Liet al [18]为检测网络添加了一个底部模块，用于预测所有实例共享的位置敏感得分图。该技术最初用于R-FCN[9]，后来在MaskLab[5]中得到改进。$k^2$得分图的每个通道对应于提案的k*k。每个得分图表示像素属于对象并且处于某个相对位置的可能性。自然，定位作物的分辨率越高，预测就越准确，但计算成本也会呈二次增长。此外，在某些特殊情况下，FCIS代表性不足。当两个实例共享中心位置(或任何其他相对位置)时，该裁剪上的得分图表示是不明确的，无法区分该裁剪描述的是哪个实例。

In YOLACT [3], an improved approach is used. Instead of using position-controlled tiles, a set of mask coefficients are learned alongside the box predictions. Then this set of coefficients guides the linear combination of cropped bottom mask bases to generate the final mask. Comparing to FCIS, the responsibility for predicting instance-level information is assigned to the top-level. We argue that using scalar coefficients to encode the instance information is suboptimal.

在YOLACT[3]中，使用了一种改进的方法。代替使用位置控制的块，一组掩码系数与框预测一起学习。然后，这组系数引导裁剪的底部掩码基的线性组合以生成最终掩码。与FCIS相比，预测实例级信息的责任分配给了顶层。我们认为使用标量系数来编码实例信息是次优的。

To break through these limitations, we propose a new proposal-based mask generation framework, termed BlendMask. The top- and bottom-level representation workloads are balanced by a blender module. Both levels are guaranteed to describe the instance information within their best capacities. As shown in our experiments in Section 4, our blender module improves the performance of bases combination methods comparing to YOLACT and FCIS by a large margin without increasing computation complexity.

为了突破这些限制，我们提出了一个新的基于提议的掩码生成框架，称为BlendMask。顶层和底层表示工作负载由blender模块平衡。这两个级别都保证在其最佳能力范围内描述实例信息。如第4节中的实验所示，与YOLACT和FCIS相比，我们的blender模块在不增加计算复杂度的情况下大幅提高了基础组合方法的性能。

Refining coarse masks with lower-level features. BlendMask merges top-level coarse instance information with lower-level fine-granularity. This idea resembles MaskLab [5] and Instance Mask Projection (IMP) [10], which concatenates mask predictions with lower layers of backbone features. The differences are clear. Our coarse mask acts like an attention map. The generation is extremely light-weight, without the need of using semantic or positional supervision, and is closely tied to the object generation. As shown in Section 3.4, our lower-level features have clear contextual meanings, even though not explicitly guided by bins or crops. Further, our blender does 3 not require a subnet on top of the merged features as in MaskLab [5] and IMP [10], which makes our method more efficient. In parallel to this work recent two single shot instance segmentation methods have shown good performance [26, 27].

使用较低级别特征细化粗糙遮罩。BlendMask将顶级粗实例信息与较低级别的细粒度合并。这个想法类似于MaskLab[5]和实例掩码投影(IMP)[10]，它们将掩码预测与较低层次的主干特征连接起来。差异是显而易见的。我们粗糙的面具就像注意力图。生成非常轻量级，不需要使用语义或位置监督，并且与对象生成密切相关。如第3.4节所示，我们的较低级别功能具有明确的上下文含义，即使没有明确的垃圾箱或作物指导。此外，我们的blender不需要像MaskLab[5]和IMP[10]中那样，在合并的功能之上有子网，这使得我们的方法更加高效。与此工作并行，最近两种单样本实例分割方法显示出良好的性能[26,27]。

## 3. Our BlendMask
### 3.1. Overall pipeline
BlendMask consists of a detector network and a mask branch. The mask branch has three parts, a bottom module to predict the score maps, a top layer to predict the instance attentions, and a blender module to merge the scores with attentions. The whole network is illustrated in Figure 2.

BlendMask由检测器网络和掩码分支组成。掩码分支有三个部分，一个底部模块用于预测分数图，一个顶层模块用于预测实例注意度，一个blender模块用于将分数与注意度合并。整个网络如图2所示。

Bottom module Similar to other proposal-based fully convolutional methods [3,18], we add a bottom module predicting score maps which we call bases, B. B has a shape of $N*K*(H/s)* (W/s)$ , where N is the batch size, K is the number of bases, H ×W is the input size and s is the score map output stride. We use the decoder of DeepLabV3+ in our experiments. Other dense prediction modules should also work without much difference. The input for the bottom module could be backbone features like conventional semantic segmentation networks [6], or the feature pyramids like YOLACT and Panoptic FPN [16].

底部模块类似于其他基于提议的完全卷积方法[3,18]，我们添加了一个底部模块预测得分图，我们称之为基数，B.B的形状为$N*K*(H/s)* (W/s)$，其中N是批次大小，K是基数，H×W是输入大小，s是得分图输出步长。我们在实验中使用了DeepLabV3+的解码器。其他密集预测模块也应该没有太大差异。底部模块的输入可以是主干特征，如传统语义分割网络[6]，或特征金字塔，如YOLACT和Panoptic FPN[16]。

Top layer We also append a single convolution layer on each of the detection towers to predict top-level attentions A. Unlike the mask coefficients in YOLACT, which for each pyramid with resolution $W_l*H_l$ takes the shape of $N × K × H_l × W_l$ , our A is a tensor at each location with shape N × (K · M · M) × Hl × Wl , where M × M is the attention resolution. With its 3D structure, our attention map can encode instance-level information, e.g. the coarse shape and pose of the object. M is typically smaller2 than the mask predictions in top-down methods since we only ask for a rough estimate. We predict it with a convolution with K · M · M output channels. Before sending them into the next module, we first apply FCOS [25] post-process to select the top D box predictions P = {pd ∈ R 4 ≥0 |d = 1 . . . D} and corresponding attentions A = {ad ∈ R K×M×M|d = 1 . . . D}.

顶层我们还在每个检测塔上附加一个卷积层，以预测顶级注意度A。与YOLACT中的掩码系数不同，对于每个分辨率为$W_l*H_l$的金字塔，其形状为$N × K × H_l × W_l$，我们的A是每个位置的张量，形状为$N x(K·M·M)×H_l x W_l$，其中M×M是注意度。通过其3D结构，我们的注意力图可以对实例级信息进行编码，例如对象的粗略形状和姿势。M通常比自顶向下方法中的掩码预测小2，因为我们只要求粗略估计。我们用K·M·M输出信道的卷积来预测它。在将它们发送到下一个模块之前，我们首先应用FCOS[25]后处理来选择顶部D框预测P={pd∈ 第4页≥0|d=1…d}和相应的注意事项$A={a_d ∈ R^{K x M x M} |d=1…d}$。

Blender module is the key part of our BlendMask. It combines position-sensitive bases according to the attentions to generate the final prediction. We discuss this module in detail in the next section.

Blender模块是我们的BlendMask的关键部分。它根据注意点组合位置敏感基以生成最终预测。我们将在下一节中详细讨论此模块。

### 3.2. Blender module
The inputs of the blender module are bottom-level bases B, the selected top-level attentions A and bounding box proposals P. First we use RoIPooler in Mask R-CNN [13] to crop bases with each proposal pd and then resize the region to a fixed size R × R feature map rd. 

混合器模块的输入是底层基础B、选定的顶层关注点A和边界框建议P。首先，我们在Mask R-CNN[13]中使用RoIPooler裁剪每个建议pd的基础，然后将区域调整为固定大小的R×R特征图rd。

rd = RoIPoolR×R(B, pd), ∀d ∈ {1 . . . D}. (1)

More specifically, we use sampling ratio 1 for RoIAlign, i.e. one bin for each sampling point. The performance of using nearest and bilinear poolers are compared in Table 6. During training, we simply use ground truth boxes as the proposals. During inference, we use FCOS prediction results.

更具体地说，我们对RoIAlign使用采样率1，即每个采样点使用一个bin。表6比较了使用最近和双线性池的性能。在训练期间，我们只使用地面真值框作为建议。在推断过程中，我们使用FCOS预测结果。

Our attention size M is smaller than R. We interpolate ad from M × M to R × R, into the shapes of R = {rd|d = 1 . . . D}. 

我们的注意力大小M小于R。我们将广告从M×。

a 0 d = interpolateM×M→R×R(ad), ∀d ∈ {1 . . . D}. (2)

Then a 0 d is normalize with softmax function along the K dimension to make it a set of score maps sd. 

然后用softmax函数沿K维对0d进行归一化，使其成为一组分数映射sd。

sd = softmax(a 0 d ), ∀d ∈ {1 . . . D}. (3)

Then we apply element-wise product between each entity rd, sd of the regions R and scores S, and sum along the K dimension to get our mask logit md: md = X K k=1 s k d ◦ r k d , ∀d ∈ {1 . . . D}, (4) where k is the index of the basis. We visualize the mask blending process with K = 4 in Figure 1.

然后，我们在区域R的每个实体rd、sd和得分S之间应用逐元素乘积，并沿着K维求和，得到掩码logit md:md=X K K=1 S K d◦ r k d，∀d∈ 其中k是基的索引。我们在图1中可视化了K＝4的掩码混合过程。

### 3.3. Configurations and baselines
We consider the following configurable hyperparameters for BlendMask:
* R, the bottom-level RoI resolution,
* M, the top-level prediction resolution,
* K, the number of bases,
* bottom module input features, it can either be features from the backbone or the FPN,
* sampling method for bottom bases, nearest-neighbour or bilinear pooling,
* interpolation method for top-level attentions, nearest neighbour or bilinear upsampling.

我们为BlendMask考虑以下可配置的超参数：
* R, 底层RoI分辨率，
* M, 顶层预测分辨率，
* K, 基数，
* 其可以是来自主干或FPN的特征，
* 底基采样方法、最近邻或双线性池，
* 用于顶级注意、最近邻或双线性上采样的插值方法。

We represent our models with abbreviation R K M. For example, 28 4 4 represents bottom-level region resolution of 28 × 28, 4 number of bases and 4 × 4 top-level instance attentions. By default, we use backbone features C3 and C5 to keep aligned with DeepLabv3+ [6]. Nearest neighbour interpolation is used in top-level interpolation, for a fair comparison with FCIS [18]. Bilinear sampling is used in the bottom level, consistent with RoIAlign [13]. 

我们用缩写R K M表示我们的模型。例如，28 4 4表示底层区域分辨率为28×。默认情况下，我们使用主干功能C3和C5与DeepLabv3+[6]保持一致。最近邻插值用于顶级插值，以便与FCIS进行公平比较[18]。双线性采样用于底层，与RoIAlign[13]一致。

### 3.4. Semantics encoded in learned bases and attentions 学习基础和注意中编码的语义
By examining the generated bases and attentions on val2017, we observe this pattern. On its bases, BlendMask encodes two types of local information, 1) whether the pixel is on an object (semantic masks), 2) whether the pixel is on certain part of the object (position-sensitive features).

通过检查2017年5月产生的基础和注意，我们观察到了这种模式。在此基础上，BlendMask编码两种类型的局部信息，
1. 像素是否在对象上(语义掩码)，
2. 像素是否位于对象的特定部分(位置敏感特征)。

Figure 3 – Detailed view of learned bases and attentions. The left four images are the bottom-level bases. The right image is the top-level attentions. Colors on each position of the attentions correspond to the weights of the bases, indicating from which part of which base is the mask assembled.
图3–学习基础和注意事项的详细视图。左边的四幅图像是最底层的基础。正确的形象是顶级注意点。注意的每个位置上的颜色都与底座的重量相对应，这表明面具是从底座的哪个部分组装的。

The complete bases and attentions projected onto the original image are illustrated in Figure 3. The first two bases (red and blue) detects points on the upper-right and bottom-left parts of the objects. The third (yellow) base activates on points more likely to be on an object. The fourth (green) base only activates on the borders of objects. Position-sensitive features help us separate overlapping instances, which enables BlendMask to represent all instances more efficiently than YOLACT [3]. The positive semantic mask makes our final prediction smoother than FCIS [18] and the negative one can further suppress out-of-instance activations. We compare our blender with YOLACT and FCIS counterparts in Table 1. BlendMask can learn more accurate features than YOLACT and FCIS with much fewer number of bases (4 vs. 32 vs. 49, see Section 4.2).

投影到原始图像上的完整基础和注意如图3所示。前两个基础(红色和蓝色)检测对象右上角和左下角的点。第三个(黄色)基础在更可能位于对象上的点上激活。第四个(绿色)基础仅在对象的边界上激活。位置敏感功能帮助我们分离重叠实例，这使得BlendMask比YOLACT更有效地表示所有实例[3]。正面语义掩码使我们的最终预测比FCIS[18]更平滑，负面语义掩码可以进一步抑制实例外激活。我们在表1中将我们的blender与YOLACT和FCIS的同类产品进行了比较。BlendMask可以比YOLACT和FCIS学习更准确的特征，但基数量要少得多(4对32对49，见第4.2节)。

## 4. Experiments
Our experiments are reported on the MSCOCO 2017 instance segmentation datatset [20]. It contains 123K images with 80-class instance labels. Our models are trained on the train2017 split (115K images) and the ablation study is carried out on the val2017 split (5K images). Final results are on test-dev. The evaluation metrics are COCO mask average precision (AP), AP at IoU 0.5 (AP50), 0.75 (AP75) and AP for objects at different sizes APS, APM, and APL.

我们的实验报告在MSCOCO 2017实例分段数据集[20]上。它包含具有80个类实例标签的123K图像。我们的模型在train2017分割(115K图像)上进行训练，并在val2017分割(5K图像)进行消融研究。最终结果在test-dev上。评估指标为COCO掩码平均精度(AP)、IoU 0.5(AP50)、0.75(AP75)的AP以及不同尺寸APS、APM和APL对象的AP。

Training details. Unless specified, ImageNet pretrained ResNet-50 [14] is used as our backbone network. DeepLabv3+ [6] with channel width 128 is used as our bottom module. For ablation study, all the networks are trained with the 1× schedule of FCOS [25], i.e., 90K iterations, batch size 16 on 4 GPUs, and base learning rate 0.01 with constant warm-up of 1k iterations. The learning rate is reduced by a factor of 10 at iteration 60K and 80K. Input images are resized to have shorter side 800 and longer side at maximum 1333. All hyperparameters are set to be the same with FCOS [25].

训练详情。除非另有规定，否则ImageNet预处理的ResNet-50[14]用作我们的主干网。通道宽度为128的DeepLabv3+[6]用作我们的底部模块。对于消融研究，所有网络均采用1×FCOS时间表[25]进行训练，即90K迭代，4个GPU上的批次大小为16，基本学习率为0.01，持续预热1k迭代。在迭代60K和80K处，学习率降低了10倍。调整输入图像的大小，使其具有最短边800和最长边1333。所有超参数都设置为与FCOS相同[25]。

Table 1 – Comparison of different strategies for merging top and bottom modules. Here the model used is 28 4 4. Weighted-sum is our analogy to YOLACT, reducing the top resolution to 1 × 1. Assembler is our analogy to FCIS, where the number of bases is increased to 16, matching each of the region crops without the need of top-level attentions. 
表1–合并顶部和底部模块的不同策略比较。这里所使用的模型是28 4 4。加权和是我们对YOLACT的类比，将最高分辨率降低到1×。

Testing details The unit for inference time is ‘ms’ in all our tables. For the ablation experiments, performance and time of our models are measured with one image per batch on one 1080Ti GPU.

测试细节在我们所有的表中，推理时间的单位是“ms”。对于消融实验，我们的模型的性能和时间是在一个1080Ti GPU上每批一幅图像测量的。

### 4.1. Ablation experiments
We investigate the effectiveness of our blender module by carrying out ablation experiments on the configurable hyperparameters in Section 3.3.

我们通过对第3.3节中的可配置超参数进行消融实验来研究我们的blender模块的有效性。

Merging methods: Blender vs. YOLACT vs. FCIS . Similar to our method, YOLACT [3] and FCIS [18] both merge proposal-based bottom regions to create mask prediction. YOLACT simply performs a weighted sum of the channels of the bottom regions; FCIS assembles crops of position-sensitive masks without modifications. Our blender can be regarded as a generalization where both YOLACT and FCIS merging are special cases: The blender with 1 × 1 top-level resolution degenerates to YOLACT; and FCIS is the case where we use fixed one-hot blending attentions and nearest neighbour top-level interpolation.

合并方法：Blender vs.YOLACT vs.FCIS。与我们的方法类似，YOLACT[3]和FCIS[18]都合并基于建议的底部区域以创建掩码预测。YOLACT简单地执行底部区域的通道的加权和; FCIS无需修改即可组装位置敏感口罩。我们的blender可以被看作是一个概括，其中YOLACT和FCIS合并都是特殊情况：顶级分辨率为1×; FCIS是我们使用固定的一个热混合注意点和最近邻顶级插值的情况。

Table 2 – Resolutions: Performance by varying top-/bottom-level resolutions, with the number of bases K = 4 for all models. Top-level attentions are interpolated with nearest neighbour. Bottom module uses backbone features C3, C5. The performance increases as the attention resolution grows, saturating at resolutions of near 1/4 of the region sizes. 

表2–分辨率：通过改变顶部/底部分辨率的性能，所有模型的底座数量K=4。顶级注意点与最近的邻居进行插值。底部模块使用主干功能C3、C5。随着注意力分辨率的增加，性能会提高，在接近1/4区域大小的分辨率下会达到饱和。

Results of these variations are shown in Table 1. Our blender surpasses the other alternatives by a large margin. We assume the reason is that other methods lack instanceaware guidance on the top. By contrast, our blender has a fine-grained top-level attention map, as illustrated in Figure 3.

这些变化的结果如表1所示。我们的blender远远超过其他替代产品。我们假设原因是其他方法在顶部缺少实例感知指导。相比之下，我们的blender有一个细粒度的顶级注意图，如图3所示。

Top and bottom resolutions: We measure the performances of our model with different top- and bottom-level resolutions, trying bottom pooler resolution R being 28 and 56, with R/M ratio from 14 to 4. As shown in Table 2, by increasing the attention resolution, we can incorporate more detailed instance-level information while keeping the running time roughly the same. Notice that the gain slows down at higher resolutions revealing limit of detailed information on the top-level. So we don’t include larger top settings with R/M ratio smaller than 4.

顶部和底部分辨率：我们使用不同的顶部和底部级别分辨率测量模型的性能，尝试底部池分辨率R为28和56，R/M比为14到4。如表2所示，通过增加注意力分辨率，我们可以在保持运行时间大致相同的情况下合并更详细的实例级别信息。请注意，在更高的分辨率下，增益会减慢，从而揭示顶层详情的限制。因此，我们不包括R/M比小于4的较大顶部设置。

Different from two-stage approaches, increasing the bottom-level bases pooling resolution does not introduce much computation overhead. Increasing it from 28 to 56 only increases the inference time within 0.2ms while mask AP increases by 1 point. In further ablation experiment, we set R = 56 and M = 7 for our baseline model if not specified.

与两阶段方法不同，增加底层基础池分辨率不会带来太多计算开销。将其从28增加到56只会增加0.2ms内的推断时间，而掩码AP增加1点。在进一步的消融实验中，如果未指定，我们为基线模型设置R=56和M=7。

Number of bases: YOLACT [3] uses 32 bases concerning the inference time. With our blender, the number of bases can be further reduced, to even just one. We report our models with number of bases varying from 1 to 8. Different from normal blender, the one-basis version uses sigmoid activation on both the base and the attention map. Results are shown in Table 3. Since instance-level information is better represented with the top-level attentions, we only need 4 bases to get the optimal accuracy. K = 4 is adopted by all subsequent experiments. 

基数：YOLACT[3]使用了32个关于推理时间的基。使用我们的blender，基的数量可以进一步减少，甚至可以减少到一个。我们报告了我们的模型，基的数量从1到8不等。与普通blender不同，单基版本在基和注意力图上都使用了S形激活。结果如表3所示。由于实例级信息用顶级注意度更好地表示，因此我们只需要4个基数即可获得最佳精度。K＝4被所有后续实验采用。

Table 3 – Number of bases: Performances of 56 K 7 models. For the configuration of one basis, we use sigmoid activation for both top and bottom features. Our model works with a small number of bases. 
表3–底座数量：56 K 7模型的性能。对于一个基的配置，我们对顶部和底部特征都使用S形激活。我们的模型只适用于少数基。

Table 4 – Bottom feature locations: Performance with bottom resolution 56 × 56, 4 bases and bilinear bottom interpolation. C3, C5 uses features from backbone. P3, P5 uses features from FPN.
表4–底部特征位置：底部分辨率为56×56、4个基点和双线性底部插值的性能。C3、C5使用来自主干的特征。P3、P5使用来自FPN的特征。

Bottom feature locations: backbone vs. FPN We compare our bottom module feature sampling locations. By using FPN features, we can improve the performance while reducing the running time (see Table 4). In later experiments, if not specified, we use P3 and P5 of FPN as our bottom module input.
底部特征位置：主干与FPN我们比较底部模块特征采样位置。通过使用FPN特性，我们可以提高性能，同时减少运行时间(见表4)。在以后的实验中，如果没有指定，我们使用FPN的P3和P5作为我们的底部模块输入。

Table 5 – Top interpolation: Performance with bottom resolution 56 × 56, 4 bases and bilinear bottom interpolation. Nearest represents nearestneighbour upsampling and bilinear is bilinear interpolation.
表5–顶部插值：底部分辨率为56×56、4个基点和双线性底部插值的性能。最近表示最近邻上采样，双线性是双线性插值。

Table 6 – Bottom Alignment: Performance with 4 bases and bilinear top interpolation. Nearest represents the original RoIPool in Fast R-CNN [11] and bilinear is the RoIAlign in Mask R-CNN [13].
表6–底部对齐：4个基点和双线性顶部插值的性能。Nearest表示Fast R-CNN[11]中的原始RoIPool，双线性表示Mask R-CNN[13]中的RoIAlign。

Interpolation method: nearest vs. bilinear In Mask R-CNN [13], RoIAlign plays a crucial role in aligning the pooled features to keep local-coherence. We investigate the effectiveness of bilinear interpolation for bottom RoI sampling and top-level attention re-scaling. As shown in Table 5, changing top interpolation from nearest to bilinear yields a marginal improvement of 0.2 AP.

插值方法：最近与双线性在掩码R-CNN[13]中，RoIAlign在对齐集合特征以保持局部一致性方面发挥了关键作用。我们研究了双线性插值在底层RoI采样和顶层注意力重缩放中的有效性。如表5所示，将顶部插值从最近值更改为双线性值会产生0.2 AP的边际改进。

The results of bottom sampling with RoIPool [11] (nearest) and RoIAlign [13] (bilinear) are shown in Table 6. For both resolutions, the aligned bilinear sampling could improve the performance by almost 2AP. Using aligned features for the bottom-level is more crucial, since it is where the detailed positions are predicted. Bilinear top and bottom interpolation are adopted for our final models.

使用RoIPool[11](最近)和RoIAlign[13](双线性)进行底部采样的结果如表6所示。对于这两种分辨率，对齐的双线性采样可以将性能提高近2AP。将对齐特征用于底层更为关键，因为它是预测详细位置的地方。我们的最终模型采用双线性顶部和底部插值。

Table 7 – Other improvements: We use 56 4 14x14 with bilinear interpolation for all models. ‘+semantic’ is the model with semantic supervision as auxiliary loss. ‘+128’ is the model with bottom module channel size being 256. ‘+s/4’ means using P2,P5 as the bottom input. Decoders in DeepLab V3+ and YOLACT (Proto) are compared. ‘Proto-P3’ has channel width of 256 and ‘Proto-FPN’ of 128. Both are trained with ‘+semantic’ setting. 

表7–其他改进：我们对所有模型使用56 4 14x14双线性插值。”+semantic是以语义监督为辅助损失的模型+128’是底部模块通道尺寸为256的模型。“+s/4”表示使用P2、P5作为底部输入。对DeepLab V3+和YOLACT(Proto)中的解码器进行了比较“Proto-P3”的通道宽度为256，“Proto-FPN”为128。这两个通道都使用“+语义”设置进行训练。

Other improvements: We experiment on other tricks to improve the performance. First we add auxiliary semantic segmentation supervision on P3 similar to YOLACT [3]. Then we increase the width of our bottom module from 128 to 256. Finally, we reduce the bases output stride from 8 to 4, to produce higher-quality bases. We achieve this by using P2 and P5 as the bottom module input. Table 7 shows the results. By adding semantic loss, detection and segmentation results are both improved. This is an interesting effect since the instance segmentation task itself does not improve the box AP. Although all tricks contribute to the improvements, we decide to not use larger basis resolution because it slows down the model by 10ms per image.

其他改进：我们尝试其他技巧来提高性能。首先，我们在P3上添加了类似于YOLACT[3]的辅助语义分段监督。然后我们将底部模块的宽度从128增加到256。最后，我们将基础输出步幅从8减少到4，以产生更高质量的基础。我们通过使用P2和P5作为底部模块输入来实现这一点。表7显示了结果。通过添加语义损失，检测和分割结果都得到了改善。这是一个有趣的效果，因为实例分割任务本身并没有改善边框子AP。尽管所有技巧都有助于改进，但我们决定不使用更大的基础分辨率，因为它会使模型每幅图像减慢10ms。

We also implement the protonet module in YOLACT [3] for comparison. We include a P3 version and an FPN version. The P3 version is identical to the one used in YOLACT. For the FPN version, we first change the channel width of P3, P4, and P5 to 128 with a 3×3 convolution. Then upsample all features to s/8 and sum them up. Following are the same as P2 version except that we reduce convolution layers by one. Auxiliary semantic loss is applied to both versions. As shown in Table 7, changing the bottom module from DeepLabv3+ to protonet does not modify the speed and performance significantly.

我们还在YOLACT[3]中实现了协议网模块，以进行比较。我们包括P3版本和FPN版本。P3版本与YOLACT中使用的版本相同。对于FPN版本，我们首先使用3×3卷积将P3、P4和P5的信道宽度更改为128。然后将所有特征上采样到s/8，并对其进行汇总。以下与P2版本相同，只是我们将卷积层减少了一个。辅助语义损失适用于两个版本。如表7所示，将底层模块从DeepLabv3+更改为protonet不会显著改变速度和性能。

### 4.2. Main result
Quantitative results We compare BlendMask with Mask R-CNN [13] and TensorMask [7] on the COCO testdev dataset(To make fair comparison with TensorMask, the code base that we use for main result is maskrcnn benchmark. Recently released Detectron2 fixed several issues of maskrcnn benchmark (ROIAlign and paste mask) in the previous repository and the performance is further improved). We use 56 4 14 with bilinear top interpolation, the DeepLabV3+ decoder with channel width 256 and P3, P5 input. Since our ablation models are heavily under-fitted, we increase the training iterations to 270K. (3× schedule), tuning learning rate down at 180K and 240K. Following Chen et al.’s strategy [7], we use multiscale training with shorter side randomly sampled from [640, 800]. As shown in Table 8, our BlendMask outperforms both the modified Mask R-CNN with deeper FPN and TensorMask using only half of their training iterations.

定量结果我们将BlendMask与COCO testdev数据集上的Mask R-CNN[13]和TensorMask[7]进行了比较(为了与TensorMask进行公平比较，我们用于主要结果的代码基础是maskrcnn基准测试。最近发布的Detectron2修复了前一个存储库中maskrcn基准测试(ROIAlign和粘贴掩码)的几个问题，并进一步提高了性能)。我们使用带有双线性顶部插值的56 4 14、通道宽度为256的DeepLabV3+解码器和P3、P5输入。由于我们的消融模型严重不足，我们将训练迭代次数增加到270K。(3×时间表)，将学习速率调低至180K和240K。按照Chenet al 的策略[7]，我们使用多尺度训练，从[640800]中随机抽取短边。如表8所示，我们的BlendMask在使用深度FPN和TensorMask的训练迭代中只使用了一半的时间，其性能都优于改进的Mask R-CNN。

BlendMask is also more efficient. Measured on a V100 GPU, the best R-101 BlendMask runs at 0.07s/im, vs. TensorMask’s 0.38s/im, vs. Mask R-CNN’s 0.09s/im [7]. Furthermore, a typical running time of our blender module is merely 0.6ms, which makes the additional time for complex scenes nearly negligible On the contrary, for two-stage Mask R-CNN with more expensive head computation, the inference time increases by a lot if the number of predicted instances grows.

BlendMask也更有效。在V100 GPU上测量，最好的R-101 BlendMask以0.07s/im的速度运行，而TensorMask的0.38s/im，而Mask R-CNN的0.09s/im[7]。此外，我们的blender模块的典型运行时间仅为0.6ms，这使得复杂场景的额外时间几乎可以忽略。相反，对于具有更昂贵头部计算的两级掩码R-CNN，如果预测实例的数量增加，推理时间会增加很多。

Real-time setting We design a compact version of our model, BlendMask-RT, to compare with YOLACT [3], a real-time instance segmentation method: i) the number of convolution layers in the prediction head is reduced to three, ii) and we merge the classification tower and box tower into one by sharing their features. We use Proto-FPN with four convolution layers with width 128 as the bottom module. The top FPN output P7 is removed because it has little effect on the detecting smaller objects. We train both BlendMaskRT and Mask R-CNN with the ×3 schedule, with shorter side randomly sampled from [440, 550].

实时设置我们设计了一个紧凑版本的模型BlendMask RT，与YOLACT[3]进行比较，YOLACT[3]是一种实时实例分割方法：i)预测头中的卷积层数减少到三层，ii)通过共享其特征，我们将分类塔和边框塔合并为一个。我们使用具有宽度128的四个卷积层的Proto FPN作为底部模块。顶部FPN输出P7被移除，因为它对检测较小对象的影响很小。我们使用×3时间表训练BlendMaskRT和Mask R-CNN，从[440550]随机抽取较短的边。

There are still two differences in the implementation comparing to YOLACT. YOLACT resizes all images to square, changing the aspect ratios of inputs. Also, a paralleled NMS algorithm called Fast NMS is used in YOLACT. We do not adopt these two configurations because they are not conventionally used in instance segmentation researches. In YOLACT, a speedup of 12ms is reported by using Fast NMS. We instead use the Batched NMS in Detectron2, which could be slower than Fast NMS but does not sacrifice the accuracy. Results in Table 9 shows that BlendMask-RT is 7ms faster and 3.3 AP higher than YOLACT-700. Making our model also competitive under the real-time settings.

与YOLACT相比，在实现上仍然存在两个差异。YOLACT将所有图像调整为正方形，改变输入的纵横比。此外，YOLACT中还使用了一种称为快速NMS的并行NMS算法。我们不采用这两种配置，因为它们通常不用于实例分割研究。在YOLACT中，使用Fast NMS报告了12毫秒的加速。我们改为在Detectron2中使用Batched NMS，它可能比Fast NMS慢，但不会牺牲准确性。表9中的结果显示，BlendMask RT比YOLACT-700快7ms，高3.3AP。这使得我们的模型在实时设置下也具有竞争力。

Qualitative results. We compare our model with the best available official YOLACT and Mask R-CNN models with ResNet-101 backbone. Masks are illustrated in Figure 4. Our model yields higher quality masks than Mask R-CNN. The first reason is that we predicts 56 × 56 masks while Mask R-CNN uses 28 × 28 masks. Also our segmentation module mostly utilizes high resolution features that preserve the original aspect-ratio, where Mask R-CNN also uses 28 × 28 features.

定性结果。我们将我们的模型与最好的官方YOLACT和Mask R-CNN模型与ResNet-101主干进行比较。掩码如图4所示。我们的模型比掩码R-CNN产生更高质量的掩码。第一个原因是我们预测56×56个掩码，而Mask R-CNN使用28×28个掩码。此外，我们的分割模块主要使用保留原始纵横比的高分辨率特征，而Mask R-CNN也使用28×28特征。

Note that YOLACT has difficulties discriminating instances of the same class close to each other. BlendMask can avoid this typical leakage. This is because its top module provides more detailed instance-level information, guiding the bases to capture position-sensitive information and suppressing the outside regions.

请注意，YOLACT很难区分彼此接近的同一类实例。BlendMask可以避免这种典型的泄漏。这是因为它的顶部模块提供更详细的实例级信息，引导基地捕获位置敏感信息并抑制外部区域。

Table 8 – Quantitative results on COCO test-dev. We compare our BlendMask against Mask R-CNN and TensorMask. Mask R-CNN* is the modified Mask R-CNN with implementation details in TensorMask [7]. Models with ‘aug.’ uses multi-scale training with shorter side range [640, 800]. Speed for Mask R-CNN 1× and BlendMask are measured with maskrcnn benchmark on a single 1080Ti GPU. BlendMask* is implemented with Detectron2, the speed difference is caused by different measuring rules. ‘+deform convs (interval = 3)’ uses deformable convolution in the backbone with interval 3, following [2].

表8–COCO试验的定量结果。我们将BlendMask与Mask R-CNN和TensorMask进行比较。Mask R-CNN*是修改后的Mask R-CNN，其实现细节见TensorMask[7]。带有“aug.”的模型使用具有较短边范围的多尺度训练[640800]。在单个1080Ti GPU上，使用maskrcnn基准测量Mask R-CNN 1×和BlendMask的速度。BlendMask*是用Detectron2实现的，速度差异是由不同的测量规则引起的+deform convs(interval＝3)'在主干中使用可变形卷积，间隔为3，如下[2]。

Table 9 – Real-time setting comparison of speed and accuracy with other state-of-the-art methods on COCO val2017. Metrics for YOLACT are obtained using their official code and trained model. Mask R-CNN and BlendMask models are trained and measured using Detectron2. Resolution 550 × ∗ means using shorter side 550 in inference. Our fast version of BlendMask significantly outperforms YOLACT in accuracy with on par execution time. 

表9–2017年COCO val的速度和精度与其他最先进方法的实时设置比较。YOLACT的指标使用其官方代码和经过训练的模型获得。使用Detectron2.分辨率550×∗ 意味着在推断中使用较短边550。我们的快速版本BlendMask在准确度和执行时间方面明显优于YOLACT。

### 4.3. Discussions
Comparison with Mask R-CNN Similar to Mask RCNN, we use RoIPooler to locate instances and extract features. We reduce the running time by moving the computation of R-CNN heads before the RoI sampling to generate position-sensitive feature maps. Repeated mask representation and computation for overlapping proposals are avoided.We further simplify the global map representation by replacing the hard alignment in R-FCN [9] and FCIS [18] with our attention guided blender, which needs ten times less channels for the same resolution.

与Mask R-CNN类似，我们使用RoIPooler定位实例并提取特征。我们通过在RoI采样之前移动R-CNN头的计算来减少运行时间，以生成位置敏感特征图。避免了重复的掩码表示和重叠建议的计算。我们将R-FCN[9]和FCIS[18]中的硬对齐替换为我们的注意力引导blender，从而进一步简化了全局图表示，对于相同的分辨率，该blender需要10倍的通道。

Another advantage of BlendMask is that it can produce higher quality masks, since our output resolution is not restricted by the top-level sampling. Increasing the RoIPooler resolution of Mask R-CNN will introduce the following problem. The head computation increases quadratically with respect to the RoI size. Larger RoIs requires deeper head structures. Different from dense pixel predictions, RoI foreground predictor has to be aware of whole instancelevel information to distinguish foreground from other overlapping instances. Thus, the larger the feature sizes are, the deeper sub-networks is needed.

BlendMask的另一个优点是它可以生成更高质量的掩码，因为我们的输出分辨率不受顶级采样的限制。增加掩码R-CNN的RoIPooler分辨率将引入以下问题。头部计算相对于RoI大小呈二次增加。较大的RoI需要较深的头部结构。与密集像素预测不同，RoI前景预测器必须了解整个实例级别的信息，以区分前景和其他重叠实例。因此，特征尺寸越大，需要的子网络越深。

Furthermore, it is not very friendly to real-time applications that the inference time of Mask R-CNN is proportional to the number of detections. By contrast, our blender module is very efficient (0.6ms on 1080 Ti). The additional inference time required after increasing the number of detections can be neglected.

此外，Mask R-CNN的推理时间与检测次数成正比，这对实时应用程序不是很友好。相比之下，我们的blender模块非常高效(1080 Ti上0.6ms)。可以忽略增加检测次数后所需的额外推断时间。

Our blender module is very flexible. Because our toplevel instance attention prediction is just a single convolution layer, it can be an almost free add-on to most modern object detectors. With its accurate instance prediction, it can also be used to refine two-stage instance predictions.

我们的blender模块非常灵活。因为我们的顶级实例注意力预测仅仅是一个卷积层，它几乎可以成为大多数现代目标检测器的免费附加组件。凭借其准确的实例预测，它还可以用于细化两阶段实例预测。

### 4.4. Panoptic Segmentation 全景分割
We use the semantic segmentation branch of PanopticFPN [16] to extend BlendMask to the panoptic segmentation task. We use annotations of COCO 2018 panoptic segmentaiton task. All models are trained on train2017 subset and tested on val2017. We train our model with the default FCOS [25] 3× schedule with scale jitter (shorter image side in [640, 800]. To combine instance and semantic results, we use the same strategy as in Panoptic-FPN, with instance confidence threshhold 0.2 and overlap threshhold 0.4.

我们使用PanopticFPN[16]的语义分割分支将BlendMask扩展到全景分割任务。我们使用COCO2018全景分割任务的注释。所有模型都在train2017子集上进行了训练，并在val2017上进行了测试。我们使用默认的FCOS[25]3×调度和缩放抖动([640，800]中的较短图像侧)训练模型。为了结合实例和语义结果，我们使用与Panoptic FPN相同的策略，实例置信阈值为0.2，重叠阈值为0.4。

Figure 4 – Detailed comparison with other methods. The large image on the left side is the segmentation result of our method. We further zoom in our result and compare against YOLACT [3] (31.2% mAP) and Mask R-CNN [13] (36.1% mAP) on the right side. Our masks are overall of higher quality.
图4–与其他方法的详细比较。左侧的大图像是我们方法的分割结果。我们进一步放大我们的结果，并与右侧的YOLACT[3](31.2%mAP)和Mask R-CNN[13](36.1%mAP)进行比较。我们的遮罩整体质量更高。

Results are reported in Table 10. Our model is consistently better than its Mask R-CNN counterpart, PanopticFPN. We assume there are three reasons. First, our instance segmentation is more accurate, this helps with both thing and stuff panoptic quality because instance masks are overlaid on top of semantic masks. Second, our pixel-level instance prediction is also generated from a global feature map, which has the same scale as the semantic prediction, thus the two results are more consistent. Last but not least, since the our bottom module shares structure with the semantic segmentation branch, it is easier for the network to share features during the closely related multi-task learning.

结果如表10所示。我们的模型始终优于其Mask R-CNN对应物PanopticFPN。我们假设有三个原因。首先，我们的实例分割更准确，这有助于提高事物和事物的全景质量，因为实例掩码覆盖在语义掩码之上。其次，我们的像素级实例预测也是从全局特征图生成的，该图与语义预测具有相同的尺度，因此两个结果更加一致。最后但并非最不重要的是，由于我们的底层模块与语义分割分支共享结构，因此在密切相关的多任务学习过程中，网络更容易共享特征。

### 4.5. More Qualitative Results 更多定性结果
We visualize qualitative results of Mask R-CNN and BlendMask on the validation set in Fig. 5. Four sets of images are listed in rows. Within each set, the top row is the Mask R-CNN results and the bottom is BlendMask. Both models are based on the newly released Detectron2 with use R101-FPN backbone. Both are trained with the 3× schedule. The Mask R-CNN model achieves 38.6% AP and ours 39.5% AP.

我们将Mask R-CNN和BlendMask的定性结果显示在图5中的验证集上。四组图像按行列出。在每个集合中，顶行是Mask R-CNN结果，底行是BlendMask。这两种模型都基于新发布的Detectron2，使用R101-FPN主干。两人都按照3×。Mask R-CNN模型达到38.6%的AP，我们的达到39.5%的AP。

Since this version of Mask R-CNN is a very strong baseline, and both models achieve very high accuracy, it is very difficult to tell the differences. To demonstrate our advantage, we select some samples where Mask R-CNN has trouble dealing with. Those cases include:
* Large objects with complex shapes (Horse ears, human poses). Mask R-CNN fails to provide sharp borders.
* Objects in separated parts (tennis players occluded by nets, trains divided by poles). Mask R-CNN tends to include occlusions as false positive or segment targets into separate objects.
* Overlapping objects (riders, crowds, drivers). Mask R-CNN gets uncertain on the borders and leaves larger false negative regions. Sometimes, it assigns parts to the wrong objects, such as the last example in the first row.

由于这个版本的Mask R-CNN是一个非常强的基线，并且两个模型都达到了非常高的精度，所以很难区分差异。为了证明我们的优势，我们选择了一些Mask R-CNN难以处理的样本。这些情况包括：
* 形状复杂的大型物体(马耳、人类姿势)。面具R-CNN无法提供清晰的边界。
* 分开的物体(网球运动员被网遮住，火车被杆子隔开)。遮罩R-CNN倾向于将遮挡作为假阳性或将目标分割为单独的对象。
* 重叠对象(乘客、人群、驾驶员)。掩码R-CNN在边界上变得不确定，并留下较大的假阴性区域。有时，它会将零件分配给错误的对象，例如第一行中的最后一个样本。

Our BlendMask performs better on these cases. 1) Generally, BlendMask utilizes features with higher resolution. Even for the large objects, we use stride-8 features. Thus details are better preserved. 2) As shown in previous illustrations, our bottom module acts as a class agnostic instance segmenter which is very sensitive to borders. 3) Sharing features with the bounding box regressor, our top module is very good at recognizing individual instances. It can generate attentions with flexible shapes to merge the fine-grained segments of bottom module outputs.

我们的BlendMask在这些情况下表现更好。1) 通常，BlendMask使用分辨率更高的功能。即使对于大型对象，我们也使用了stride-8特征。因此，细节得以更好地保存。2) 如前面的插图所示，我们的底部模块充当了一个类无关的实例分割器，它对边界非常敏感。3) 与边界框回归器共享特性，我们的顶部模块非常擅长识别单个实例。它可以通过灵活的形状来合并底部模块输出的细粒度段，从而引起注意。

### 4.6. Evaluating on LVIS annotations 评估LVIS注释
To quantify the high quality masks generated by BlendMask, we compare our results with on the higher-quality LVIS annotations [12]. Our model is compared to the best high resolution model we are aware of, recent PointRend [17], which uses multiple subnets to refine the local features to get higher resolution mask predictions. The description of the evaluation metric can be found in [17]. Table 11 shows that the evaluation numbers will improve further given more accurate ground truth annotations. Our method can benefits from the accurate bottom features and surpasses the high-res PointRend results.

为了量化BlendMask生成的高质量掩码，我们将我们的结果与更高质量的LVIS注释进行比较[12]。我们的模型与我们所知的最好的高分辨率模型相比较，最近的PointRend[17]，它使用多个子网来细化局部特征，以获得更高分辨率的掩码预测。评估指标的描述见[17]。表11显示，如果提供更准确的地面实况说明，评估数字将进一步提高。我们的方法可以从精确的底部特征中受益，并超过高分辨率的PointRend结果。

Table 10 – Panoptic results on COCO val2017. Panoptic-FPN results are from the official Detectron2 implementation, which are improved upon the original published results in [16]. 
表10–2017年COCO val的全景结果。全景FPN结果来自官方Detectron2实施，在[16]中原始发布结果的基础上进行了改进。

## 5. Conclusion
We have devised a novel blender module for instancelevel dense prediction tasks which uses both high-level instance and low-level semantic information. It is efficient and easy to integrate with different main-stream detection networks.

我们为实例级密集预测任务设计了一个新的blender模块，它使用高级实例和低级语义信息。它高效且易于与不同的主流检测网络集成。

Our framework BlendMask outperforms the carefullyengineered Mask R-CNN without bells and whistles while being 20% faster. Furthermore, the real-time version BlendMask-RT achieves 34.2% mAP at 25 FPS evaluated on a single 1080Ti GPU card. We believe that our BlendMask is capable of serving as an alternative to Mask RCNN [13] for many other instance-level recognition tasks.

我们的框架BlendMask的性能优于精心设计的Mask R-CNN，无需吹嘘，速度快20%。此外，实时版本BlendMask RT在单个1080Ti GPU卡上以25 FPS的速度获得34.2%的mAP。我们相信，对于许多其他实例级识别任务，我们的BlendMask能够作为Mask RCNN[13]的替代品。

## Acknowledgements
The authors would like to thank Huawei Technologies for the donation of GPU cloud computing resources.

作者想感谢华为技术公司捐赠GPU云计算资源。

## References
1.  Anurag Arnab and Philip H. S. Torr. Bottom-up instance segmentation using deep higher-order CRFs. In Proc. British Conf. Machine Vis., 2016. 2
2.  Daniel Bolya, Chong Zhou, Fanyi Xiao, and Yong Jae Lee. YOLACT++: Better real-time instance segmentation. arXiv preprint arXiv:1912.06218, 2019. 8
3.  Daniel Bolya, Chong Zhou, Fanyi Xiao, and Yong Jae Lee. YOLACT: real-time instance segmentation. Proc. Int. Conf. Comp. Vis., abs/1904.02689, 2019. 2, 3, 4, 5, 6, 7, 9
4.  Bert De Brabandere, Davy Neven, and Luc Van Gool. Semantic instance segmentation with a discriminative loss function. arXiv Comput. Res. Repository, abs/1708.02551, 2017. 2
5.  Liang-Chieh Chen, Alexander Hermans, George Papandreou, Florian Schroff, Peng Wang, and Hartwig Adam. Masklab: Instance segmentation by refining object detection with semantic and direction features. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 4013–4022, 2018. 3, 4
6.  Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proc. Eur. Conf. Comp. Vis., pages 833–851, 2018. 4, 5
7.  Xinlei Chen, Ross B. Girshick, Kaiming He, and Piotr Dollar. Tensormask: A foundation for dense object segmen- ´ tation. Proc. Int. Conf. Comp. Vis., abs/1903.12174, 2019. 2, 7, 8
8.  Jifeng Dai, Kaiming He, Yi Li, Shaoqing Ren, and Jian Sun. Instance-sensitive fully convolutional networks. In Proc. Eur. Conf. Comp. Vis., pages 534–549, 2016. 1
9.  Jifeng Dai, Yi Li, Kaiming He, and Jian Sun. R-FCN: object detection via region-based fully convolutional networks. In Proc. Adv. Neural Inf. Process. Syst., pages 379–387, 2016. 3, 8
10.  Cheng-Yang Fu, Tamara L. Berg, and Alexander C. Berg. IMP: instance mask projection for high accuracy semantic segmentation of things. arXiv Comput. Res. Repository, abs/1906.06597, 2019. 3, 4
11.  Ross B. Girshick. Fast R-CNN. In Proc. Int. Conf. Comp. Vis., pages 1440–1448, 2015. 6
12.  Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019. 9
13.  Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross B. ´ Girshick. Mask R-CNN. In Proc. Int. Conf. Comp. Vis., pages 2980–2988, 2017. 2, 3, 4, 6, 7, 8, 9, 10
14.  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 770–778, 2016. 5
15.  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In Proc. Eur. Conf. Comp. Vis., pages 630–645, 2016. 2
16.  Alexander Kirillov, Ross B. Girshick, Kaiming He, and Piotr Dollar. Panoptic feature pyramid networks. In ´ Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 6399–6408, 2019. 4, 8, 10
17.  Alexander Kirillov, Yuxin Wu, Kaiming He, and Ross Girshick. Pointrend: Image segmentation as rendering. arXiv preprint arXiv:1912.08193, 2019. 9, 11
18.  Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, and Yichen Wei. Fully convolutional instance-aware semantic segmentation. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 4438– 4446, 2017. 2, 3, 4, 5, 8
19.  Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In ´ Proc. Int. Conf. Comp. Vis., pages 2999–3007, 2017. 2
20.  Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and ´ C. Lawrence Zitnick. Microsoft COCO: common objects 10 Method Backbone resolution COCO AP LVIS AP? Mask R-CNN X101-FPN 28 × 28 39.5 40.7 PointRend X101-FPN 224 × 224 40.9 43.4 BlendMask R-101+dcni3 56 × 56 41.1 44.1 Table 11 – Comparison with PointRend. Mask R-CNN and PointRend results are quoted from Table 5 of 17. . Our model is the last model in Table 8. Our model is 0.2 points higher on COCO and 0.7 points higher on LVIS annotations. Here LVIS AP? is COCO mask AP evaluated against the higher-quality LVIS annotations. in context. In Proc. Eur. Conf. Comp. Vis., pages 740–755, 2014. 5
21.  Yiding Liu, Siyu Yang, Bin Li, Wengang Zhou, Jizheng Xu, Houqiang Li, and Yan Lu. Affinity derivation and graph merge for instance segmentation. In Proc. Eur. Conf. Comp. Vis., pages 708–724, 2018. 2
22.  Davy Neven, Bert De Brabandere, Marc Proesmans, and Luc Van Gool. Instance segmentation by jointly optimizing spatial embeddings and clustering bandwidth. In Proc. IEEE Conf. Comp. Vis. Patt. Recogn., pages 8837–8845, 2019. 2
23.  Pedro H. O. Pinheiro, Ronan Collobert, and Piotr Dollar. ´ Learning to segment object candidates. In Proc. Adv. Neural Inf. Process. Syst., pages 1990–1998, 2015. 1
24.  Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster R-CNN: towards real-time object detection with region proposal networks. In Proc. Adv. Neural Inf. Process. Syst., pages 91–99, 2015. 1
25.  Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. FCOS: fully convolutional one-stage object detection. Proc. Int. Conf. Comp. Vis., abs/1904.01355, 2019. 1, 2, 3, 4, 5, 8
26.  Xinlong Wang, Tao Kong, Chunhua Shen, Yuning Jiang, and Lei Li. Solo: Segmenting objects by locations. arXiv preprint arXiv:1912.04488, 2019. 4, 8
27.  Enze Xie, Peize Sun, Xiaoge Song, Wenhai Wang, Xuebo Liu, Ding Liang, Chunhua Shen, and Ping Luo. PolarMask: Single shot instance segmentation with polar representation. arXiv Comput. Res. Repository, 2019. arxiv.org/abs/1909.13226. 4 11 Figure 5 – Selected results of Mask R-CNN (top) and BlendMask (bottom). Both models are based on Detectron2. The Mask R-CNN model is the official 3× R101 model with 38.6 AP. BlendMask model obtains 39.5 AP. Best viewed in digital format with zoom. 
