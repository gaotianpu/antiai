# A Review on Deep Learning Techniques for Video Prediction
视频预测深度学习技术综述 https://arxiv.org/abs/2004.05214 

## 阅读笔记


## Abstract
The ability to predict, anticipate and reason about future outcomes is a key component of intelligent decision-making systems. In light of the success of deep learning in computer vision, deep-learning-based video prediction emerged as a promising research direction. Defined as a self-supervised learning task, video prediction represents a suitable framework for representation learning, as it demonstrated potential capabilities for extracting meaningful representations of the underlying patterns in natural videos. Motivated by the increasing interest in this task, we provide a review on the deep learning methods for prediction in video sequences. We firstly define the video prediction fundamentals, as well as mandatory background concepts and the most used datasets. Next, we carefully analyze existing video prediction models organized according to a proposed taxonomy, highlighting their contributions and their significance in the field. The summary of the datasets and methods is accompanied with experimental results that facilitate the assessment of the state of the art on a quantitative basis. The paper is summarized by drawing some general conclusions, identifying open research challenges and by pointing out future research directions.

预测、预期和推理未来结果的能力是智能决策系统的关键组成部分。 鉴于深度学习在计算机视觉领域的成功，基于深度学习的视频预测成为一个有前途的研究方向。 视频预测被定义为一种自我监督的学习任务，它代表了一种适合表示学习的框架，因为它展示了提取自然视频中潜在模式的有意义表示的潜在能力。 由于对这项任务的兴趣越来越大，我们对用于视频序列预测的深度学习方法进行了综述。 我们首先定义了视频预测的基础知识，以及强制性的背景概念和最常用的数据集。 接下来，我们仔细分析了根据提议的分类法组织的现有视频预测模型，突出了它们在该领域的贡献和意义。 数据集和方法摘要附有实验结果，有助于在定量评估最新技术水平。 本文通过得出一些一般性结论、确定开放性研究挑战和指出未来研究方向来总结。

Index Terms: Video prediction, future frame prediction, deep learning, representation learning, self-supervised learning ✦

## 1 Introduction
WILL the car hit the pedestrian? That might be one of the questions that comes to our minds when we observe Figure 1. Answering this question might be in principle a hard task; however, if we take a careful look into the image sequence we may notice subtle clues that can help us predicting into the future, e.g., the person’s body indicates that he is running fast enough so he will be able to escape the car’s trajectory. This example is just one situation among many others in which predicting future frames in video is useful.

汽车会撞到行人吗？ 当我们观察图1 时，这可能是我们想到的问题之一。回答这个问题原则上可能是一项艰巨的任务;  然而，如果我们仔细观察图像序列，我们可能会注意到可以帮助我们预测未来的微妙线索，例如，这个人的身体表明他跑得足够快，因此他将能够逃离汽车的轨迹。 这个例子只是预测视频中未来帧有用的许多情况中的一种。

Fig. 1: A pedestrian appeared from behind the white car with the intention of crossing the street. The autonomous car must make a call: hit the emergency braking routine or not. This all comes down to predict the next frames ($\hat Y_{t+1}, . . . , \hat Y_{t+m}$) given a sequence of context frames ($X_{t−n}, . . . , X_t$), where n and m denote the number of context and predicted frames, respectively. From these predictions at a representation level (RGB, high-level semantics, etc.) a decision-making system would make the car avoid the collision. 
图1：一名行人从白色汽车后面出现，意图穿过街道。自动驾驶汽车必须发出呼叫：是否执行紧急制动程序。这一切都归结为预测给定一系列上下文 帧序列($X_{t−n}，…，X_t$)的情况下预测下一帧($\hat Y_{t+1}，……，\hat Y{t+m}$)，其中n和m分别表示上下文和预测帧的数量。根据这些表示级别(RGB、高级语义等)的预测，决策系统将使汽车避免碰撞。

In general terms, the prediction and anticipation of future events is a key component of intelligent decisionmaking systems. Despite the fact that we, humans, solve this problem quite easily and effortlessly, it is extremely challenging from a machine’s point of view. Some of the factors that contribute to such complexity are occlusions, camera movement, lighting conditions, clutter, or object deformations. Nevertheless, despite such challenging conditions, many predictive methods have been applied with a certain degree of success in a broad range of application domains such as autonomous driving, robot navigation and human-machine interaction. Some of the tasks in which future prediction has been applied successfully are: anticipating activities and events [1]–[4], long-term planning [5], future prediction of object locations [6], video interpolation [7], predicting instance/semantic segmentation maps [8]– [10], prediction of pedestrian trajectories in traffic [11], anomaly detection [12], precipitation nowcasting [13], [14], and autonomous driving [15].

一般而言，对未来事件的预测和预期是智能决策系统的关键组成部分。 尽管我们人类可以轻松轻松地解决这个问题，但从机器的角度来看，这是极具挑战性的。 造成这种复杂性的一些因素是遮挡、相机移动、照明条件、杂乱或对象变形。 尽管条件如此具有挑战性，但许多预测方法已在广泛的应用领域(如自动驾驶、机器人导航和人机交互)中得到一定程度的成功应用。 成功应用未来预测的一些任务是：预测活动和事件 [1]-[4]、长期规划 [5]、目标位置的未来预测 [6]、视频插值 [7]、预测 实例/语义分割图[8]-[10]、交通行人轨迹预测[11]、异常检测[12]、临近降水预报[13]、[14]和自动驾驶[15]。

The great strides made by deep learning algorithms in a variety of research fields such as semantic segmentation [16], human action recognition and prediction [17], object pose estimation [18] and registration [19] to name a few, motivated authors to explore deep representationlearning models for future video frame prediction. What made the deep architectures take a leap over the traditional approaches is their ability to learn adequate representations from high-dimensional data in an end-to-end fashion without hand-engineered features [20]. Deep learningbased models fit perfectly into the learning by prediction paradigm, enabling the extraction of meaningful spatiotemporal correlations from video data in a self-supervised fashion.

深度学习算法在语义分割 [16]、人类行为识别和预测 [17]、物体姿势估计 [18] 和配准 [19] 等多个研究领域取得的巨大进步，激励作者 探索用于未来视频帧预测的深度表征学习模型。 深度架构超越传统方法的原因在于它们能够以端到端的方式从高维数据中学习足够的表示，而无需手动设计特征 [20]。 基于深度学习的模型非常适合预测范式学习，能够以自我监督的方式从视频数据中提取有意义的时空相关性。

In this review, we put our focus on deep learning techniques and how they have been extended or applied to future video prediction. We limit this review to the future video prediction given the context of a sequence of previous frames, leaving aside methods that predict future from a static image. In this context, the terms video prediction, future frame prediction, next video frame prediction, future frame forecasting, and future frame generation are used interchangeably. To the best of our knowledge, this is the first review in the literature that focuses on video prediction using deep learning techniques.

在这篇报告中，我们将重点放在深度学习技术以及它们如何扩展或应用到未来的视频预测中。 我们将此报告限制在给定一系列先前帧的上下文的未来视频预测，而忽略从静态图像预测未来的方法。 在此上下文中，术语视频预测、未来帧预测、下一视频帧预测、未来帧预测和未来帧生成可互换使用。 据我们所知，这是文献中第一篇关注使用深度学习技术进行视频预测的报告。

This review is organized as follows. First, Sections 2 and 3 lay down the terminology and explain important background concepts that will be necessary throughout the rest of the paper. Next, Section 4 surveys the datasets used by the video prediction methods that are carefully reviewed in Section 5, providing a comprehensive description as well as an analysis of their strengths and weaknesses. Section 6 analyzes typical metrics and evaluation protocols for the aforementioned methods and provides quantitative results for them in the reviewed datasets. Section 7 presents a brief discussion on the presented proposals and enumerates potential future research directions. Finally, Section 8 summarizes the paper and draws conclusions about this work. 

本次报告组织如下。 首先，第 2 节和第 3 节规定了术语并解释了贯穿本文其余部分的重要背景概念。 接下来，第 4 节调查了第 5 节中仔细回顾了视频预测方法使用的数据集，提供了全面的描述以及对它们的优缺点的分析。 第 6 节分析了上述方法的典型指标和评估协议，并在综述的数据集中为它们提供了定量结果。 第 7 节简要讨论了所提出的建议，并列举了未来可能的研究方向。 最后，第 8 节总结了本文并对这项工作得出结论。

## 2 VIDEO PREDICTION
The ability to predict, anticipate and reason about future events is the essence of intelligence [21] and one of the main goals of decision-making systems. This idea has biological roots, and also draws inspiration from the predictive coding paradigm [22] borrowed from the cognitive neuroscience field [23]. From a neuroscience perspective, the human brain builds complex mental representations of the physical and causal rules that govern the world. This is primarily through observation and interaction [24]–[26]. The common sense we have about the world arises from the conceptual acquisition and the accumulation of background knowledge from early ages, e.g. biological motion and intuitive physics to name a few. But how can the brain check and refine the learned mental representations from its raw sensory input? The brain is continuously learning through prediction, and refines the already understood world models from the mismatch between its predictions and what actually happened [27]. This is the essence of the predictive coding paradigm that early works tried to implement as computational models [22], [28]–[30].

预测、预期和推理未来事件的能力是智能的本质[21]，也是决策系统的主要目标之一。 这个想法有生物学根源，也从认知神经科学领域 [23] 借用的预测编码范式 [22] 中汲取灵感。 从神经科学的角度来看，人脑构建了支配世界的物理和因果规则的复杂心理表征。 这主要是通过观察和互动 [24]-[26]。 我们对世界的常识来自于早年的概念习得和背景知识的积累，例如 生物运动和直觉物理学等等。 但是，大脑如何从其原始感官输入中检查和完善学习到的心理表征呢？ 大脑通过预测不断学习，并从其预测与实际发生的事情之间的不匹配中改进已经理解的世界模型 [27]。 这是预测编码范式的本质，早期作品试图将其实现为计算模型 [22]、[28]-[30]。

Video prediction task closely captures the fundamentals of the predictive coding paradigm and it is considered the intermediate step between raw video data and decision making. Its potential to extract meaningful representations of the underlying patterns in video data makes the video prediction task a promising avenue for self-supervised representation learning.

视频预测任务紧密地捕捉了预测编码范式的基础，它被认为是原始视频数据和决策制定之间的中间步骤。 它在视频数据中提取潜在模式的有意义表示的潜力使视频预测任务成为自监督表示学习的有前途的途径。

### 2.1 Problem Definition
We formally define the task of predicting future frames in videos, i.e. video prediction, as follows. Let $X_t ∈ R^{w×h×c}$ be the t-th frame in the video sequence $X = (X_{t−n}, . . . , X_{t−1}, X_t)$ with n frames, where w, h, and c denote width, height, and number of channels, respectively. The target is to predict the next frames $Y = (\hat Y_{t+1}, \hat Y_{t+2}, . . . , \hat Y_{t+m})$ from the input X.

我们正式定义预测视频中未来帧的任务，即视频预测，如下所示。 设 $X_t ∈ R^{w×h×c}$ 为视频序列 $X = (X_{t−n}, . . ., X_{t−1}, X_t)$ 中的第 t 帧，其中 n 帧，其中 w、h 和 c 分别表示宽度、高度和通道数。 目标是根据输入 X 预测下一帧 $Y = (\hat Y_{t+1}, \hat Y_{t+2}, . . ., \hat Y_{t+m})$。

Under the assumption that good predictions can only be the result of accurate representations, learning by prediction is a feasible approach to verify how accurately the system has learned the underlying patterns in the input data. In other words, it represents a suitable framework for representation learning [31], [32]. The essence of predictive learning paradigm is the prediction of plausible future outcomes from a set of historical inputs. On this basis, the task of video prediction is defined as: given a sequence of video frames as context, predict the subsequent frames –generation of continuing video given a sequence of previous frames. Different from video generation that is mostly unconditioned, video prediction is conditioned on a previously learned representation from a sequence of input frames. At a first glance, and in the context of learning paradigms, we can think about the future video frame prediction task as a supervised learning approach because the target frame acts as a label. However, as this information is already available in the input video sequence, no extra labels or human supervision is needed. Therefore, learning by prediction is a self-supervised task, filling the gap between supervised and unsupervised learning.

假设良好的预测只能是准确表示的结果，通过预测学习是一种可行的方法来验证系统在输入数据中学习潜在模式的准确性。 换句话说，它代表了一个合适的表示学习框架[31]，[32]。 预测学习范式的本质是从一组历史输入中预测可能的未来结果。 在此基础上，视频预测的任务被定义为：给定一个视频帧序列作为上下文，预测后续帧 —— 给定一个先前帧序列生成连续视频。 与大多数无条件的视频生成不同，视频预测以先前从一系列输入帧中学习到的表示为条件。 乍一看，在学习范式的背景下，我们可以将未来视频帧预测任务视为一种监督学习方法，因为目标帧充当标签。 但是，由于此信息已在输入视频序列中可用，因此不需要额外的标签或人工监督。 因此，预测学习是一种自监督的任务，填补了监督学习和无监督学习之间的空白。

### 2.2 Exploiting the Time Dimension of Videos 利用视频的时间维度
Unlike static images, videos provide complex transformations and motion patterns in the time dimension. At a fine granularity, if we focus on a small patch at the same spatial location across consecutive time steps, we could identify a wide range of local visually similar deformations due to the temporal coherence. In contrast, by looking at the big picture, consecutive frames would be visually different but semantically coherent. This variability in the visual appearance of a video at different scales is mainly due to, occlusions, changes in the lighting conditions, and camera motion, among other factors. From this source of temporally ordered visual cues, predictive models are able to extract representative spatio-temporal correlations depicting the dynamics in a video sequence. For instance, Agrawal et al. [33] established a direct link between vision and motion, attempting to reduce supervision efforts when training deep predictive models.

与静态图像不同，视频在时间维度上提供复杂的变换和运动模式。 在细粒度上，如果我们在连续时间步长上关注相同空间位置的小块，我们可以识别由于时间相干性而产生的广泛的局部视觉相似变形。 相反，通过查看大图，连续的帧在视觉上会有所不同，但在语义上是一致的。 视频在不同比例下视觉外观的这种可变性主要是由于遮挡、照明条件的变化和相机运动等因素造成的。 从这种按时间顺序排列的视觉线索来源，预测模型能够提取描述视频序列中动态的代表性时空相关性。 例如，Agrawal et al. [33] 在视觉和运动之间建立了直接联系，试图在训练深度预测模型时减少监督工作。

Recent works study how important is the time dimension for video understanding models [34]. The implicit temporal ordering in videos, also known as the arrow of time, indicates whether a video sequence is playing forward or backward. This temporal direction is also used in the literature as a supervisory signal [35]–[37]. This further encouraged predictive models to implicitly or explicitly model the spatio-temporal correlations of a video sequence to understand the dynamics of a scene. The time dimension of a video reduces the supervision effort and makes the prediction task self-supervised. 

最近的工作研究了视频理解模型的时间维度有多重要 [34]。 视频中的隐式时间顺序，也称为时间箭头，指示视频序列是向前播放还是向后播放。 这个时间方向也在文献中用作监督信号 [35]-[37]。 这进一步鼓励预测模型隐式或显式地模拟视频序列的时空相关性，以了解场景的动态。 视频的时间维度减少了监督工作，使预测任务可以自我监督。

Fig. 2: At top, a deterministic environment where a geometric object, e.g. a black square, starts moving following a random direction. At bottom, probabilistic outcome. Darker areas correspond to higher probability outcomes. As uncertainty is introduced, probabilities get blurry and averaged. Figure inspired by [38]. 
图 2：在顶部，确定性环境中的几何对象，例如 一个黑色方块，开始沿着随机方向移动。 归根结底，概率结果。 较暗的区域对应于较高概率的结果。 随着不确定性的引入，概率变得模糊和平均。 图灵感来自 [38]。


### 2.3 Dealing with Stochasticity 处理随机性
Predicting how a square is moving, could be extremely challenging even in a deterministic environment such as the one represented in Figure 2. The lack of contextual information and the multiple equally probable outcomes hinder the prediction task. But, what if we use two consecutive frames as context? Under this configuration and assuming a physically perfect environment, the square will be indefinitely moving in the same direction. This represents a deterministic outcome, an assumption that many authors made in order to deal with future uncertainty. Assuming a deterministic outcome would narrow the prediction space to a unique solution. However, this assumption is not suitable for natural videos. The future is by nature multimodal, since the probability distribution defining all the possible future outcomes in a context has multiple modes, i.e. there are multiple equally probable and valid outcomes. Furthermore, on the basis of a deterministic universe, we indirectly assume that all possible outcomes are reflected in the input data. These assumptions make the prediction under uncertainty an extremely challenging task.

即使在如图 2 所示的确定性环境中，预测正方形的移动方式也极具挑战性。缺乏上下文信息和多个等概率结果阻碍了预测任务。 但是，如果我们使用两个连续的帧作为上下文呢？ 在这种配置下，假设物理环境完美，正方形将无限期地朝同一方向移动。 这代表了一个确定性的结果，许多作者为了应对未来的不确定性而做出的假设。 假设一个确定性的结果会将预测空间缩小到一个唯一的解决方案。 然而，这种假设不适用于自然视频。 未来本质上是多模态的，因为定义上下文中所有可能的未来结果的概率分布具有多种模式，即存在多个同样可能且有效的结果。 此外，在确定性宇宙的基础上，我们间接假设所有可能的结果都反映在输入数据中。 这些假设使得不确定性下的预测成为一项极具挑战性的任务。

Most of the existing deep learning-based models in the literature are deterministic. Although the future is uncertain, a deterministic prediction would suffice some easily predictable situations. For instance, most of the movement of a car is largely deterministic, while only a small part is uncertain. However, when multiple predictions are equally probable, a deterministic model will learn to average between all the possible outcomes. This unpredictability is visually represented in the predictions as blurriness, especially on long time horizons. As deterministic models are unable to handle real-world settings characterized by chaotic dynamics, authors considered that incorporating uncertainty to the model is a crucial aspect. Probabilistic approaches dealing with these issues are discussed in Section 5.6.

文献中大多数现有的基于深度学习的模型都是确定性的。 尽管未来是不确定的，但确定性预测足以应对一些容易预测的情况。 例如，汽车的大部分运动在很大程度上是确定性的，只有一小部分是不确定的。 然而，当多个预测的可能性相同时，确定性模型将学习在所有可能的结果之间进行平均。 这种不可预测性在预测中直观地表示为模糊，尤其是在长期范围内。 由于确定性模型无法处理以混沌动力学为特征的现实世界设置，作者认为将不确定性纳入模型是一个关键方面。 处理这些问题的概率方法在第 5.6 节中讨论。

### 2.4 The Devil is in the Loss Function 魔鬼在损失函数中
The design and selection of the loss function for the video prediction task is of utmost importance. Pixel-wise losses, e.g. Cross Entropy (CE), l2, l1 and Mean-Squared Error (MSE), are widely used in both unstructured and structured predictions. Although leading to plausible predictions in deterministic scenarios, such as synthetic datasets and video games, they struggle with the inherent uncertainty of natural videos. In a probabilistic environment, with different equally probable outcomes, pixel-wise losses aim to accommodate uncertainty by blurring the prediction, as we can observe in Figure 2. In other words, the deterministic loss functions average out multiple equally plausible outcomes in a single, blurred prediction. In the pixel space, these losses are unstable to slight deformations and fail to capture discriminative representations to efficiently regress the broad range of possible outcomes. This makes difficult to draw predictions maintaining the consistency with our visual similarity notion. Besides video prediction, several studies analyzed the impact of different loss functions in image restoration [39], classification [40], camera pose regression [41] and structured prediction [42], among others. This fosters reasoning about the importance of the loss function, particularly when making long-term predictions in high-dimensional and multimodal natural videos.

视频预测任务的损失函数的设计和选择至关重要。像素损失，例如交叉熵(CE)、l2, l1和均方误差(MSE)，广泛用于非结构化和结构化预测。尽管在确定性场景(如合成数据集和视频游戏)中得出了合理的预测，但它们与自然视频固有的不确定性作斗争。在概率环境中，具有不同的等概率结果，像素损失旨在通过模糊预测来适应不确定性，如图2所示。换句话说，确定性损失函数在一个模糊的预测中平均了多个同样合理的结果。在像素空间中，这些损失对于轻微的变形是不稳定的，并且无法捕获区分表示以有效地回归广泛的可能结果。这使得很难做出与我们的视觉相似性概念保持一致的预测。除了视频预测，一些研究分析了不同损失函数在图像恢复[39]、分类[40]、相机姿态回归[41]和结构化预测[42]等方面的影响。这有助于推理损失函数的重要性，尤其是在高维和多模态自然视频中进行长期预测时。

Most of distance-based loss functions, such as based on $l_p$ norm, come from the assumption that data is drawn from a Gaussian distribution. But, how these loss functions address multimodal distributions? Assuming that a pixel is drawn from a bimodal distribution with two equally likely modes $M_{o_1}$ and $M_{o_2}$, the mean value $M_o = (M_{o_1} + M_{o_2})/2$ would minimize the $l_p$-based losses over the data, even if Mo has very low probability [43]. This suggests that the average of two equally probable outcomes would minimize distance-based losses such as, the MSE loss. However, this applies to a lesser extent when using $l_1$ norm as the pixel values would be the median of the two equally likely modes in the distribution. In contrast to the $l_2$ norm that emphasizes outliers with the squaring term, the $l_1$ promotes sparsity thus making it more suitable for prediction in high-dimensional data [43]. Based on the $l_2$ norm, the MSE is also commonly used in the training of video prediction models. However, it produces low reconstruction errors by merely averaging all the possible outcomes in a blurry prediction as uncertainty is introduced. In other words, the mean image would minimize the MSE error as it is the global optimum, thus avoiding finer details such as facial features and subtle movements as they are noise for the model. Most of the video prediction approaches rely on pixel-wise loss functions, obtaining roughly accurate predictions in easily predictable datasets.

大多数基于距离的损失函数，例如基于 $l_p$ 范数的损失函数，都源于数据来自高斯分布的假设。 但是，这些损失函数如何解决多峰分布？ 假设一个像素是从具有两个同样可能的模式 $M_{o_1}$ 和 $M_{o_2}$ 的双峰分布中提取的，则平均值 $M_o = (M_{o_1} + M_{o_2})/2$ 将 最小化基于 $l_p$ 的数据损失，即使 $M_o$ 的概率非常低 [43]。 这表明两个等概率结果的平均值将最小化基于距离的损失，例如 MSE 损失。 然而，当使用 $l_1$ 范数时，这在较小程度上适用，因为像素值将是分布中两个同样可能模式的中值。 与用平方项强调离群值的 $l_2$ 范数相反，$l_1$ 促进了稀疏性，从而使其更适合在高维数据中进行预测 [43]。 基于$l_2$范数的MSE也常用于视频预测模型的训练。 但是，由于引入了不确定性，它仅通过对模糊预测中的所有可能结果进行平均来产生较低的重建误差。 换句话说，平均图像会最小化 MSE 误差，因为它是全局最优的，因此避免了更精细的细节，例如面部特征和细微运动，因为它们是模型的噪声。 大多数视频预测方法都依赖于像素损失函数，在易于预测的数据集中获得大致准确的预测。
<!-- pixel-wise loss functions -->

One of the ultimate goals of many video prediction approaches is to palliate the blurry predictions when it comes to uncertainty. For this purpose, authors broadly focused on: directly improving the loss functions; exploring adversarial training; alleviating the training process by reformulating the problem in a higher-level space; or exploring probabilistic alternatives. Some promising results were reported by combining the loss functions with sophisticated regularization terms, e.g. the Gradient Difference Loss (GDL) to enhance prediction sharpness [43] and the Total Variation (TV) regularization to reduce visual artifacts and enforce coherence [7]. Perceptual losses were also used to further improve the visual quality of the predictions [44]– [48]. However, in light of the success of the Generative Adversarial Networks (GANs), adversarial training emerged as a promising alternative to disambiguate between multiple equally probable modes. It was widely used in conjunction with different distance-based losses such as: MSE [49], l2 [50]–[52], or a combination of them [43], [53]–[57]. To alleviate the training process, many authors reformulated the optimization process in a higher-level space (see Section 5.5). While great strides have been made to mitigate blurriness, most of the existing approaches still rely on distance-based loss functions. As a consequence, the regress-to-the-mean problem remains an open issue. This has further encouraged authors to reformulate existing deterministic models in a probabilistic fashion. 

许多视频预测方法的最终目标之一是在不确定性方面减轻模糊预测。 为此，作者广泛关注：直接改进损失函数;  探索对抗训练;  通过在更高层次的空间中重新表述问题来减轻训练过程;  或探索概率替代方案。 通过将损失函数与复杂的正则化项相结合，报告了一些有希望的结果，例如 梯度差异损失 (GDL) 以增强预测清晰度 [43] 和总变差 (TV) 正则化以减少视觉伪影并加强一致性 [7]。 感知损失也被用来进一步提高预测的视觉质量[44]-[48]。 然而，鉴于生成对抗网络 (GAN) 的成功，对抗训练成为消除多个等概率模式之间歧义的有前途的替代方案。 它被广泛与不同的基于距离的损失结合使用，例如：MSE [49]、`2 [50]-[52]，或它们的组合 [43]、[53]-[57]。 为了减轻训练过程，许多作者在更高级别的空间中重新制定了优化过程(参见第 5.5 节)。 虽然在减轻模糊度方面取得了很大进展，但大多数现有方法仍然依赖于基于距离的损失函数。 因此，均值回归问题仍然是一个悬而未决的问题。 这进一步鼓励作者以概率方式重新制定现有的确定性模型。

## 3 BACKBONE DEEP LEARNING ARCHITECTURES
In this section, we will briefly review the most common deep networks that are used as building blocks for the video prediction models discussed in this review: convolutional neural networks, recurrent networks, and generative models.

在本节中，我们将简要回顾最常见的深度网络，这些网络用作本文讨论的视频预测模型的构建块：卷积神经网络、递归网络和生成模型。

### 3.1 Convolutional Models
Convolutional layers are the basic building blocks of deep learning architectures designed for visual reasoning since the Convolutional Neural Networks (CNNs) efficiently model the spatial structure of images [58]. As we focus on the visual prediction, CNNs represent the foundation of predictive learning literature. However, their performance is limited by the intra-frame and inter-frame dependencies.

卷积层是为视觉推理而设计的深度学习架构的基本构建块，因为卷积神经网络 (CNN) 可以有效地模拟图像的空间结构 [58]。 当我们专注于视觉预测时，CNN 代表了预测学习文献的基础。 然而，它们的性能受到帧内和帧间依赖性的限制。

Convolutional operations account for short-range intraframe dependencies due to their limited receptive fields, determined by the kernel size. This is a well-addressed issue, that many authors circumvented by (1) stacking more convolutional layers [59], (2) increasing the kernel size (although it becomes prohibitively expensive), (3) by linearly combining multiple scales [43] as in the reconstruction process of a Laplacian pyramid [60], (4) using dilated convolutions to capture long-range spatial dependencies [61], (5) enlarging the receptive fields [62], [63], or subsampling, i.e. using pooling operations in exchange for losing resolution. The latter could be mitigated by using residual connections [64], [65] to preserve resolution while increasing the number of stacking convolutions. But even addressing these limitations, would CNNs be able to predict in a longer time horizon?

卷积操作考虑了短程帧内依赖性，因为它们的接受域有限，由内核大小决定。 这是一个很好解决的问题，许多作者通过(1)堆叠更多卷积层[59]，(2)增加内核大小(尽管它变得非常昂贵)，(3)通过线性组合多个尺度[43]来规避 如在拉普拉斯金字塔[60]的重建过程中，(4)使用扩张卷积来捕获长程空间依赖性[61]，(5)扩大感受野[62]，[63]或子采样，即使用 合并操作以换取失去分辨率。 后者可以通过使用残差连接 [64]、[65] 来减轻，以在增加堆叠卷积数量的同时保持分辨率。 但即使解决了这些限制，CNN 是否能够在更长的时间范围内进行预测？

Vanilla CNNs lack of explicit inter-frame modeling capabilities. To properly model inter-frame variability in a video sequence, 3D convolutions come into play as a promising alternative to recurrent modeling. Several video prediction approaches leveraged 3D convolutions to capture temporal consistency [66]–[70]. Also modeling time dimension, Amersfoort et al. [71] replicated a purely convolutional approach in time to address multi-scale predictions in the transformation space. In this case, the learned affine transforms at each time step play the role of a recurrent state.

Vanilla CNN 缺乏明确的帧间建模能力。 为了正确地模拟视频序列中的帧间可变性，3D 卷积作为递归建模的有前途的替代方案开始发挥作用。 几种视频预测方法利用 3D 卷积来捕捉时间一致性 [66]-[70]。 同样对时间维度进行建模，Amersfoort et al. [71] 及时复制了一种纯卷积方法来解决变换空间中的多尺度预测。 在这种情况下，每个时间步学习的仿射变换都扮演了循环状态的角色。

### 3.2 Recurrent Models
Recurrent models were specifically designed to model a spatio-temporal representation of sequential data such as video sequences. Among other sequence learning tasks, such as machine translation, speech recognition and video captioning, to name a few, Recurrent Neural Networks (RNNs) [72] demonstrated great success in the video prediction scenario [10], [13], [49], [50], [52], [53], [53], [70], [73]– [85]. Vanilla RNNs have some important limitations when dealing with long-term representations due to the vanishing and exploding gradient issues, making the Backpropagation through time (BPTT) cumbersome. By extending classical RNNs to more sophisticated recurrent models, such as Long Short-Term Memory (LSTM) [86] and Gated Recurrent Unit (GRU) [87], these problems were mitigated. Shi et al. extended the use of LSTM-based models to the image space [13]. While some authors explored multidimensional LSTM (MD-LSTM) [88], others stacked recurrent layers to capture abstract spatio-temporal correlations [49], [89]. Zhang et al. addressed the duplicated representations along the same recurrent paths [90].

循环模型专门设计用于对序列数据(例如视频序列)的时空表示进行建模。 在机器翻译、语音识别和视频字幕等其他序列学习任务中，递归神经网络 (RNN) [72] 在视频预测场景 [10]、[13]、[49] 中取得了巨大成功。 , [50], [52], [53], [53], [70], [73]– [85]。 由于梯度消失和爆炸问题，Vanilla RNN 在处理长期表征时有一些重要的局限性，这使得反向传播(BPTT)变得很麻烦。 通过将经典 RNN 扩展到更复杂的循环模型，例如长短期记忆 (LSTM) [86] 和门控循环单元 (GRU) [87]，这些问题得到了缓解。 施等。 将基于 LSTM 的模型的使用扩展到图像空间 [13]。 虽然一些作者探索了多维 LSTM (MD-LSTM) [88]，但其他作者堆叠循环层以捕获抽象的时空相关性 [49]、[89]。 张等。 解决了沿相同循环路径的重复表示 [90]。

### 3.3 Generative Models
Whilst discriminative models learn the decision boundaries between classes, generative models learn the underlying distribution of individual classes. More formally, discriminative models capture the conditional probability p(y|x), while generative models capture the joint probability p(x, y), or p(x) in the absence of labels y. The goal of generative models is the following: given some training data, generate new samples from the same distribution. Let input data ∼ pdata(x) and generated samples ∼ pmodel(x) where, pdata and pmodel are the underlying input data and model’s probability distribution respectively. The training process consists in learning a pmodel(x) similar to pdata(x). This is done by explicitly, e.g VAEs, or implicitly, e.g. GANs, estimating a density function from the input data. In the context of video prediction, generative models are mainly used to cope with future uncertainty by generating a wide spectrum of feasible predictions rather than a single eventual outcome.

判别模型学习类别之间的决策边界，而生成模型学习各个类别的潜在分布。 更正式地说，判别模型捕获条件概率 p(y|x)，而生成模型捕获联合概率 p(x, y)，或在没有标签 y 的情况下 p(x)。 生成模型的目标如下：给定一些训练数据，从相同的分布生成新样本。 让输入数据∼ pdata(x) 和生成的样本∼ pmodel(x) 其中，pdata 和 pmodel 分别是基础输入数据和模型的概率分布。 训练过程包括学习类似于 pdata(x) 的 pmodel(x)。 这是通过显式完成的，例如 VAE，或隐式完成，例如 GAN，根据输入数据估计密度函数。 在视频预测的背景下，生成模型主要用于通过生成广泛的可行预测而不是单一的最终结果来应对未来的不确定性。

#### 3.3.1 Explicit Density Modeling 显式密度建模
These models explicitly define and solve for pmodel(x).

这些模型明确定义和求解 pmodel(x)。

##### PixelRNNs and PixelCNNs [91]: 
These are a type of Fully Visible Belief Networks (FVBNs) [92], [93] that explicitly define a tractable density and estimate the joint distribution p(x) as a product of conditional distributions over the pixels. Informally, they turn pixel generation into a sequential modeling problem, where next pixel values are determined by previously generated ones. In PixelRNNs, this conditional dependency on previous pixels is modeled using two-dimensional (2d) LSTMs. On the other hand, dependencies are modeled using convolutional operations 5 over a context region, thus making training faster. In a nutshell, these methods are outputting a distribution over pixel values at each location in the image, aiming to maximize the likelihood of the training data being generated. Further improvements of the original architectures have been carried out to address different issues. The Gated PixelCNN [94] is computationally more efficient and improves the receptive fields of the original architecture. In the same work, authors also explored conditional modeling of natural images, where the joint probability distribution is conditioned on a latent vector —it represents a high-level image description. This further enabled the extension to video prediction [95].

PixelRNNs 和 PixelCNNs [91]：这些是一种完全可见的信念网络(FVBNs)[92]，[93]，它们明确定义了一个易处理的密度，并将联合分布 p(x) 估计为像素上条件分布的产物 . 非正式地，他们将像素生成转化为顺序建模问题，其中下一个像素值由先前生成的像素值决定。 在 PixelRNN 中，这种对先前像素的条件依赖性是使用二维 (2d) LSTM 建模的。 另一方面，依赖性是在上下文区域上使用卷积运算 5 建模的，从而使训练更快。 简而言之，这些方法输出图像中每个位置的像素值分布，旨在最大化生成训练数据的可能性。 已经对原始架构进行了进一步的改进，以解决不同的问题。 Gated PixelCNN [94] 的计算效率更高，并改善了原始架构的感受野。 在同一项工作中，作者还探索了自然图像的条件建模，其中联合概率分布以潜在向量为条件——它表示高级图像描述。 这进一步实现了对视频预测的扩展[95]。

##### Variational Autoencoders (VAEs): 
These models are an extension of Autoencoders (AEs) that encode and reconstruct its own input data x in order to capture a low-dimensional representation z containing the most meaningful factors of variation in x. Extending this architecture to generation, VAEs aim to sample new images from a prior over the underlying latent representation z. VAEs represent a probabilistic spin over the deterministic latent space in AEs. Instead of directly optimizing the density function, which is intractable, they derive and optimize a lower bound on the likelihood. Data is generated from the learned distribution by perturbing the latent variables. In the video prediction context, VAEs are the foundation of many probabilistic models dealing with future uncertainty [9], [38], [55], [81], [85], [96], [97]. Although these variational approaches are able to generate various plausible outcomes, the predictions are blurrier and of lower quality compared to state-of-theart GAN-based models. Many approaches were taken to leverage the advantages of variational inference: combined adversarial training with VAEs [55], and others incorporated latent probabilistic variables into deterministic models, such as Variational Recurrent Neural Networks (VRNNs) [97], [98] and Variational Encoder-Decoders (VEDs) [99].

变分自动编码器 (VAE)：这些模型是自动编码器 (AE) 的扩展，它们对自己的输入数据 x 进行编码和重构，以捕获包含 x 中最有意义的变化因素的低维表示 z。 将这种架构扩展到生成，VAE 旨在从先验的潜在表示 z 上采样新图像。 VAE 表示 AE 中确定性潜在空间的概率旋转。 他们没有直接优化难以处理的密度函数，而是推导并优化了可能性的下限。 通过扰动潜在变量从学习分布生成数据。 在视频预测环境中，VAE 是许多处理未来不确定性的概率模型的基础 [9]、[38]、[55]、[81]、[85]、[96]、[97]。 尽管这些变分方法能够产生各种看似合理的结果，但与最先进的基于 GAN 的模型相比，预测更加模糊且质量较低。 许多方法被用来利用变分推理的优势：将对抗训练与 VAE [55] 相结合，其他方法将潜在概率变量纳入确定性模型，例如变分递归神经网络 (VRNN) [97]、[98] 和变分编码器 -解码器(VED)[99]。

#### 3.3.2 Implicit Density Modeling
These models learn to sample from pmodel without explicitly defining it.

这些模型学习从 pmodel 中采样而不明确定义它。

GANs [100]: are the backbone of many video prediction approaches [43], [49]–[55], [57], [65], [67], [68], [78], [101]– [106]. Inspired on game theory, these networks consist of two models that are jointly trained as a minimax game to generate new fake samples that resemble the real data. On one hand, we have the discriminator model featuring a probability distribution function describing the real data. On the other hand, we have the generator which tries to generate new samples that fool the discriminator. In their original formulation, GANs are unconditioned –the generator samples new data from a random noise, e.g. Gaussian noise. Nevertheless, Mirza et al. [107] proposed the conditional Generative Adversarial Network (cGAN), a conditional version where the generator and discriminator are conditioned on some extra information, e.g. class labels, previous predictions, and multimodal data, among others. CGANs are suitable for video prediction, since the spatiotemporal coherence between the generated frames and the input sequence is guaranteed. The use of adversarial training for the video prediction task, represented a leap over the previous state-of-the-art methods in terms of prediction quality and sharpness. However, adversarial training is unstable. Without an explicit latent variable interpretation, GANs are prone to mode collapse —generator fails to cover the space of possible predictions by getting stuck into a single mode [99], [108]. Moreover, GANs often struggle to balance between the adversarial and reconstruction loss, thus getting blurry predictions. Among the dense literature on adversarial networks, we find some other interesting works addressing GANs limitations [109], [110]. 

GANs [100]：是许多视频预测方法的支柱 [43]、[49]–[55]、[57]、[65]、[67]、[68]、[78]、[101]– [ 106]。 受博弈论的启发，这些网络由两个模型组成，这两个模型被联合训练为极小极大游戏，以生成类似于真实数据的新假样本。 一方面，我们有鉴别器模型，其特征是描述真实数据的概率分布函数。 另一方面，我们有生成器试图生成欺骗鉴别器的新样本。 在其原始公式中，GAN 是无条件的——生成器从随机噪声中采样新数据，例如 高斯噪声。 尽管如此，Mirza et al. [107] 提出了条件生成对抗网络(cGAN)，这是一种条件版本，其中生成器和鉴别器以一些额外信息为条件，例如 类标签、先前的预测和多模态数据等。 CGAN 适用于视频预测，因为生成的帧和输入序列之间的时空一致性得到保证。 在视频预测任务中使用对抗训练，在预测质量和清晰度方面代表了对先前最先进方法的飞跃。 然而，对抗训练是不稳定的。 如果没有明确的潜在变量解释，GAN 很容易出现模式崩溃——生成器无法通过陷入单一模式来覆盖可能的预测空间 [99]，[108]。 此外，GAN 常常难以在对抗损失和重建损失之间取得平衡，从而导致预测变得模糊。 在关于对抗网络的大量文献中，我们发现了一些其他有趣的作品来解决 GAN 的局限性 [109]、[110]。

## 4 DATASETS
As video prediction models are mostly self-supervised, they need video sequences as input data. However, some video prediction methods rely on extra supervisory signals, e.g. segmentation maps, and human poses. This makes outof-domain video datasets perfectly suitable for video prediction. This section describes the most relevant datasets, discussing their pros and cons. Datasets were organized according to their main purpose and summarized in Table 1.

由于视频预测模型大多是自我监督的，因此它们需要视频序列作为输入数据。 然而，一些视频预测方法依赖于额外的监督信号，例如 分割图和人体姿势。 这使得域外视频数据集非常适合视频预测。 本节介绍最相关的数据集，讨论它们的优缺点。 数据集根据其主要目的进行组织，并在表 1 中进行了总结。

### 4.1 Action and Human Pose Recognition Datasets
#### KTH [111]: 
is an action recognition dataset which includes 2391 video sequences of 4 seconds mean duration, each of them containing an actor performing an action taken with a static camera, over homogeneous backgrounds, at 25 frames per second (fps) and with its resolution downsampled to 160 × 120 pixels. Just 6 different actions are performed, but it was the biggest dataset of this kind at its moment.

KTH [111]：是一个动作识别数据集，包括 2391 个平均时长为 4 秒的视频序列，每个视频序列都包含一个演员在同质背景下以每秒 25 帧 (fps) 的速度执行使用静态相机样本的动作 其分辨率下采样到 160 × 120 像素。 只执行了 6 个不同的动作，但它是当时最大的此类数据集。

#### Weizmann [112]: 
is also an action recognition dataset, created for modelling actions as space-time shapes. For this reason, it was recorded at a higher frame rate (50 fps). It just includes 90 video sequences, but performing 10 different actions. It uses a static-camera, homogeneous backgrounds and low resolution (180 × 144 pixels). KTH and Weizmann are usually used together due to their similarities in order to augment the amount of available data.

Weizmann [112]：也是一个动作识别数据集，创建用于将动作建模为时空形状。 因此，它以更高的帧速率 (50 fps) 录制。 它只包含 90 个视频序列，但执行 10 个不同的动作。 它使用静态相机、均匀背景和低分辨率(180 × 144 像素)。 KTH 和 Weizmann 由于它们的相似性通常一起使用以增加可用数据量。

#### HMDB-51 [113]: 
is a large-scale database for human motion recognition. It claims to represent the richness of human motion taking profit from the huge amount of video available online. It is composed by 6766 normalized videos (with mean duration of 3.15 seconds) where humans appear performing one of the 51 considered action categories. Moreover, a stabilized dataset version is provided, in which camera movement is disabled by detecting static backgrounds and displacing the action as a window. It also provides interesting data for each sequence such as body parts visible, point of view respect the human, and if there is camera movement or not. It also exists a joint-annotated version called J-HMBD [114] in which the key points of joints were mannually added for 21 of the HMDB actions.

HMDB-51 [113]：是一个用于人体运动识别的大型数据库。 它声称代表了人类运动的丰富性，从在线提供的大量视频中获利。 它由 6766 个标准化视频(平均持续时间为 3.15 秒)组成，其中人类似乎在执行 51 个考虑的动作类别之一。 此外，还提供了稳定的数据集版本，其中通过检测静态背景并将动作移动为窗口来禁用相机移动。 它还为每个序列提供有趣的数据，例如可见的身体部位、尊重人类的观点以及是否有相机移动。 它还存在一个名为 J-HMBD [114] 的联合注释版本，其中为 21 个 HMDB 动作手动添加了关节的关键点。

#### UCF101 [115]: 
is an action recognition dataset of realistic action videos, collected from YouTube. It has 101 different action categories, and it is an extension of UCF50, which has 50 action categories. All videos have a frame rate of 25 fps and a resolution of 320 × 240 pixels. Despite being the most used dataset among predictive models, a problem it 6 has is that only a few sequences really represent movement, i.e. they often show an action over a fixed background.

UCF101 [115]：是一个真实动作视频的动作识别数据集，从 YouTube 收集。 它有 101 个不同的动作类别，它是 UCF50 的扩展，UCF50 有 50 个动作类别。 所有视频的帧率为 25 fps，分辨率为 320 × 240 像素。 尽管它是预测模型中使用最多的数据集，但它 6 存在的一个问题是只有少数序列真正代表运动，即它们通常显示固定背景上的动作。

#### Penn Action Dataset [116]: 
is an action and human pose recognition dataset from the University of Pennsylvania. It contains 2326 video sequences of 15 different actions, and it also provides human joint and viewpoint (position of the camera respect the human) annotations for each sequence. Each action is balanced in terms of different viewpoints representation.

Penn Action Dataset [116]：是来自宾夕法尼亚大学的动作和人体姿势识别数据集。 它包含 15 个不同动作的 2326 个视频序列，并且还为每个序列提供人体关节和视点(摄像机的位置尊重人类)注释。 根据不同的观点表示，每个动作都是平衡的。

#### Human3.6M [117]: 
is a human pose dataset in which 11 actors with marker-based suits were recorded performing 15 different types of actions. It features RGB images, depth maps (time-of-flight range data), poses and scanned 3D surface meshes of all actors. Silhouette masks and 2D bounding boxes are also provided. Moreover, the dataset was extended by inserting high-quality 3D rigged human models (animated with the previously recorded actions) in real videos, to create a realistic and complex background.

Human3.6M [117]：是一个人体姿势数据集，其中记录了 11 名穿着基于标记的套装的演员执行 15 种不同类型的动作。 它具有所有演员的 RGB 图像、深度图(飞行时间范围数据)、姿势和扫描的 3D 表面网格。 还提供了剪影蒙版和 2D 边界框。 此外，通过在真实视频中插入高质量的 3D 操纵人体模型(使用先前记录的动作进行动画处理)来扩展数据集，以创建逼真的复杂背景。

#### THUMOS-15 [118]: 
is an action recognition challenge that was celebrated in 2015. It didn’t just focus on recognizing an action in a video, but also on determining the time span in which that action occurs. With that purpose, the challenge provided a dataset that extends UCF101 [115] (trimmed videos with one action) with 2100 untrimmed videos where one or more actions take place (with the correspondent temporal annotations) and almost 3000 relevant videos without any of the 101 proposed actions.

THUMOS-15 [118]：是一项在 2015庆祝的动作识别挑战。它不仅关注识别视频中的动作，还关注确定该动作发生的时间跨度。 为此，挑战赛提供了一个数据集，该数据集扩展了 UCF101 [115](带有一个动作的修剪视频)，其中包含 2100 个未修剪的视频，其中发生了一个或多个动作(具有相应的时间注释)和近 3000 个相关视频，没有 101 建议的行动。

### 4.2 Driving and Urban Scene Understanding Datasets 驾驶和城市场景理解数据集
#### CamVid [136]: 
the Cambridge-driving Labeled Video Database is a driving/urban scene understanding dataset which consists of 5 video sequences recorded with a 960 × 720 pixels resolution camera mounted on the dashboard of a car. Four of those sequences were sampled at 1 fps, and one at 15 fps, resulting in 701 frames which were manually per-pixel annotated for semantic segmentation (under 32 classes). It was the first video sequence dataset of this kind to incorporate semantic annotations.

CamVid [136]：剑桥驾驶标记视频数据库是一个驾驶/城市场景理解数据集，由安装在汽车仪表板上的 960×720 像素分辨率摄像头记录的 5 个视频序列组成。 其中四个序列以 1 fps 采样，一个以 15 fps 采样，产生 701 帧，这些帧是手动逐像素注释的，用于语义分割(在 32 个类别下)。 这是第一个包含语义注释的此类视频序列数据集。

#### CalTech Pedestrian Dataset [119]: 
is a driving dataset focused on detecting pedestrians, since its unique annotations are pedestrian bounding boxes. It is conformed of approximately 10 hours of 640×480 30fps video taken from a vehicle driving through regular traffic in an urban environment, making a total of 250 000 annotated frames distributed in 137 approximately minute-long segments. The total pedestrian bounding boxes is 350 000, identifying 2300 unique pedestrians. Temporal correspondence between bounding boxes and detailed occlusion labels are also provided.

加州理工学院行人数据集[119]：是一个专注于检测行人的驾驶数据集，因为其独特的注释是行人边界框。 它包含大约 10 小时的 640×480 30fps 视频，这些视频是从在城市环境中通过常规交通行驶的车辆样本的，总共有 250,000 个带注释的帧分布在 137 个大约一分钟长的片段中。 行人边界框总数为 350 000，识别出 2300 名独特的行人。 还提供了边界框和详细遮挡标签之间的时间对应关系。

#### Kitti [120]: 
is one of the most popular datasets for mobile robotics and autonomous driving, as well as a benchmark for computer vision algorithms. It is composed by hours of traffic scenarios recorded with a variety of sensor modalities, including high-resolution RGB, gray-scale stereo cameras, and a 3D laser scanner. Despite its popularity, the original dataset did not contain ground truth for semantic segmentation. However, after various researchers manually annotated parts of the dataset to fit their necessities, in 2015 Kitti dataset was updated with 200 annotated frames at pixel level for both semantic and instance segmentation, following the format proposed by the Cityscapes [121] dataset.

Kitti [120]：是移动机器人和自动驾驶最流行的数据集之一，也是计算机视觉算法的基准。 它由使用各种传感器模式记录的数小时交通场景组成，包括高分辨率 RGB、灰度立体相机和 3D 激光扫描仪。 尽管它很受欢迎，但原始数据集并不包含用于语义分割的基本事实。 然而，在各种研究人员手动注释数据集的一部分以满足他们的需要之后，在 2015，Kitti 数据集更新为像素级别的 200 个注释帧，用于语义和实例分割，遵循 Cityscapes [121] 数据集提出的格式。

#### Cityscapes [121]: 
is a large-scale database which focuses on semantic understanding of urban street scenes. It provides semantic, instance-wise, and dense pixel annotations for 30 classes grouped into 8 categories. The dataset consist of around 5000 fine annotated images (1 frame in 30) and 20 000 coarse annotated ones (one frame every 20 seconds or 20 meters run by the car). Data was captured in 50 cities during several months, daytimes, and good weather conditions. All frames are provided as stereo pairs, and the dataset also includes vehicle odometry obtained from invehicle sensors, outside temperature, and GPS tracks.

Cityscapes [121]：是一个大型数据库，专注于对城市街道场景的语义理解。 它为分为 8 个类别的 30 个类提供语义、实例和密集像素注释。 该数据集由大约 5000 张精细标注图像(30 帧中的 1 帧)和 20000 幅粗略标注图像(每 20 秒或汽车每行驶 20 米一帧)组成。 在几个月、白天和良好的天气条件下，在 50 个城市中捕获了数据。 所有帧都以立体对的形式提供，数据集还包括从车载传感器、外部温度和 GPS 轨迹获得的车辆里程计。

#### Comma.ai steering angle [137]: 
is a driving dataset composed by 7.25 hours of largely highway routes. It was recorded as 360 × 180 camera images at 20 fps (divided in 11 different clips), and steering angles, among other vehicle data (speed, GPS, etc.).

Comma.ai 转向角 [137]：是一个由 7.25 小时的主要公路路线组成的驾驶数据集。 它被记录为 20 fps 的 360 × 180 摄像机图像(分为 11 个不同的剪辑)、转向角以及其他车辆数据(速度、GPS 等)。

#### Apolloscape [122]: 
is a driving/urban scene understanding dataset that focuses on 3D semantic reconstruction of the environment. It provides highly precise information about location and 6D camera pose, as well as a much bigger amount of dense per-pixel annotations than other datasets. Along with that, depth information is retireved from a LIDAR sensor, that allows to semantically reconstruct the scene in 3D as a point cloud. It also provides RGB stereo pairs as video sequences recorded under various weather conditions and daytimes. This video sequences and their per-pixel instance annotations make this dataset very interesting for a wide variety of predictive models.

Apolloscape [122]：是一个驾驶/城市场景理解数据集，专注于环境的 3D 语义重建。 它提供了关于位置和 6D 相机姿势的高精度信息，以及比其他数据集更多的密集每像素注释。 与此同时，深度信息从 LIDAR 传感器中移除，允许将 3D 场景语义重建为点云。 它还提供 RGB 立体声对作为在各种天气条件和白天记录的视频序列。 该视频序列及其每像素实例注释使该数据集对于各种预测模型非常有趣。

### 4.3 Object and Video Classification Datasets
#### Sports1M [123]: 
is a video classification dataset that also consists of annotated YouTube videos. In this case, it is fully focused on sports: its 487 classes correspond to the sport label retrieved from the YouTube Topics API. Video resolution, duration and frame rate differ across all available videos, but they can be normalized when accessed from YouTube. It is much bigger than UCF101 (over 1 million videos), and movement is also much more frequent.

Sports1M [123]：是一个视频分类数据集，也包含带注释的 YouTube 视频。 在本例中，它完全专注于运动：它的 487 个类对应于从 YouTube Topics API 检索到的运动标签。 所有可用视频的视频分辨率、持续时间和帧速率都不同，但从 YouTube 访问时可以对其进行标准化。 它比 UCF101(超过 100 万个视频)大得多，移动也更加频繁。

#### Youtube-8M [124]: 
Sports1M [123] dataset is, since 2016, part of a bigger one called YouTube8M, which follows the same philosophy, but with all kind of videos, not just sports. Moreover, it has been updated in order to improve the quality and precision of their annotations. In 2019 YouTube-8M Segments was released with segment-level human-verified labels on about 237 000 video segments on 1000 different classes, which are collected from the validation set of the YouTube-8M dataset. Since YouTube is the biggest video source on the planet, having annotations for some of their videos at segment level is great for predictive models.

Youtube-8M [124]：自 2016以来，Sports1M [123] 数据集是一个名为 YouTube8M 的更大数据集的一部分，它遵循相同的理念，但包含各种视频，而不仅仅是体育。 此外，它已经更新，以提高其注释的质量和精度。 2019发布的 YouTube-8M Segments 在 1000 个不同类别的约 237,000 个视频片段上带有片段级人工验证标签，这些视频片段是从 YouTube-8M 数据集的验证集中收集的。 由于 YouTube 是地球上最大的视频源，因此在片段级别对其某些视频进行注释非常适合预测模型。

#### YFCC100M [125]: 
Yahoo Flickr Creative Commons 100 Million Dataset is a collection of 100 million images and videos uploaded to Flickr between 2004 and 2014. All those media files were published in Flickr under Creative Commons license, overcoming one of the biggest issues affecting existing multimedia datasets, licensing and volume. Although only 0.8% of the elements of the dataset are videos, it is still useful for predictive models due to the great variety of these, and therefore the challenge that it represents.

YFCC100M [125]：雅虎 Flickr Creative Commons 1 亿数据集是 2004至 2014间上传到 Flickr 的 1 亿张图像和视频的集合。所有这些媒体文件都是在 Creative Commons 许可下在 Flickr 上发布的，克服了影响现有的最大问题之一 多媒体数据集、许可和数量。 尽管数据集中只有 0.8% 的元素是视频，但由于视频种类繁多，因此它对预测模型仍然有用，因此它所代表的挑战也很大。

TABLE 1: Summary of the most widely used datasets for video prediction (S/R: Synthetic/Real, st: stereo, de: depth, ss: semantic segmentation, is: instance segmentation, sem: semantic, I/O: Indoor/Outdoor environment, bb: bounding box, Act: Action label, ann: annotated, env: environment, ToF: Time of Flight, vp: camera viewpoints respect human). 
表 1：最广泛使用的视频预测数据集总结(S/R：合成/真实，st：立体，de：深度，ss：语义分割，is：实例分割，sem：语义，I/O：室内/ 室外环境，bb：边界框，Act：动作标签，ann：注释，env：环境，ToF：飞行时间，vp：相机视点尊重人类)。

1 some dataset names have been abbreviated to enhance table’s readability. 
2 values estimated based on the framerate and the total number of frames or videos, as the original values are not provided by the authors. 
3 custom indicates that as many frames as needed can be generated. This is related to datasets generated from a game, algorithm or simulation, involving interaction or randomness. 

1 一些数据集名称已被缩写以增强表格的可读性。
2 根据帧率和帧数或视频总数估算的值，因为作者未提供原始值。 
3 custom 表示可以生成所需数量的帧。 这与游戏、算法或模拟生成的数据集有关，涉及交互或随机性。


### 4.4 Video Prediction Datasets
#### Standard bouncing balls dataset [126]: 
is a common test set for models that generate high dimensional sequences. It consists of simulations of three balls bouncing in a box. Its clips can be generated randomly with custom resolution but the common structure is composed by 4000 training videos, 200 testing videos and 200 more for validation. This kind of datasets are purely focused on video prediction.

Standard bouncing balls dataset [126]：是生成高维序列的模型的通用测试集。 它包括三个球在盒子中弹跳的模拟。 它的剪辑可以自定义分辨率随机生成，但通用结构由 4000 个训练视频、200 个测试视频和另外 200 个用于验证的视频组成。 这种数据集纯粹专注于视频预测。

#### Van Hateren Dataset of natural videos (version [127]): 
is a very small dataset of 56 videos, each 64 frames long, that has been widely used in unsupervised learning. Original images were taken and given for scientific use by the photographer Hans van Hateren, and they feature moving animals in grasslands along rivers and streams. Its frame size is 128 × 128 pixels. The version we are reviewing is the one provided along with the work of Cadieu and Olshausen [127].

Van Hateren 自然视频数据集(版本 [127])：是一个非常小的数据集，包含 56 个视频，每个视频长 64 帧，已广泛用于无监督学习。 原始图像由摄影师汉斯·范·哈特伦 (Hans van Hateren) 样本并提供给科学使用，其中描绘了沿河流和溪流在草原上移动的动物。 其帧大小为 128 × 128 像素。 我们正在综述的版本是与 Cadieu 和 Olshausen [127] 的工作一起提供的版本。

#### NORBvideos [128]: 
NORB (NYU Object Recognition Benchmark) dataset [138] is a compilation of static stereo pairs of 50 homogeneously colored objects from various points of view and 6 lightning conditions. Those images were processed to obtain their object masks and even their casted shadows, allowing them to augment the dataset introducing random backgrounds. Viewpoints are determined by rotating the camera through 9 elevations and 18 azimuths (every 20 degrees) around the object. NORBvideos dataset was built by sequencing all these frames for each object.

NORBvideos [128]：NORB(纽约大学对象识别基准)数据集 [138] 是从不同角度和 6 种闪电条件下收集的 50 个均匀彩色对象的静态立体对的汇编。 这些图像经过处理以获得它们的对象蒙版甚至投射的阴影，从而使它们能够增加引入随机背景的数据集。 视点是通过将相机围绕物体旋转 9 个仰角和 18 个方位角(每 20 度)来确定的。 NORBvideos 数据集是通过对每个对象的所有这些帧进行排序而构建的。

#### Moving MNIST [74] (M-MNIST): 
is a video prediction dataset built from the composition of 20-frame video sequences where two handwritten digits from the MNIST database are combined inside a 64 × 64 patch, and moved with some velocity and direction along frames, potentially overlapping between them. This dataset is almost infinite (as new sequences can be generated on the fly), and it also has interesting behaviours due to occlusions and the dynamics of digits bouncing off the walls of the patch. For these reasons, this dataset is widely used by many predictive models. A stochastic variant of this dataset is also available. In the original M-MNIST the digits move with constant velocity and bounce off the walls in a deterministic manner. In contrast, in SM-MNIST digits move with a constant velocity along a trajectory until they hit at wall at which point they bounce off with a random speed and direction. In this way, 8 moments of uncertainty (each time a digit hits a wall) are interspersed with deterministic motion.

移动 MNIST [74] (M-MNIST)：是一个视频预测数据集，由 20 帧视频序列组成，其中来自 MNIST 数据库的两个手写数字在一个 64 × 64 块内组合，并以一定的速度和方向移动 沿着帧，它们之间可能重叠。 这个数据集几乎是无限的(因为可以动态生成新序列)，并且由于遮挡和数字从分块壁上弹起的动态，它也有有趣的行为。 由于这些原因，该数据集被许多预测模型广泛使用。 该数据集的随机变体也可用。 在最初的 M-MNIST 中，数字以恒定的速度移动并以确定的方式从墙上弹开。 相比之下，在 SM-MNIST 中，数字沿着轨迹以恒定速度移动，直到它们撞到墙上，此时它们以随机速度和方向反弹。 这样，8 个不确定时刻(每次一个数字撞到墙上)穿插在确定性运动中。

#### Robotic Pushing Dataset [89]: 
is a dataset created for learning about physical object motion. It consist on 640×512 pixels image sequences of 10 different 7-degree-of-freedom robotic arms interacting with real-world physical objects. No additional labeling is given, the dataset was designed to model motion at pixel level through deep learning algorithms based on convolutional LSTM (ConvLSTM).

Robotic Pushing Dataset [89]：是为学习物理对象运动而创建的数据集。 它由 10 个不同的 7 自由度机械臂与真实世界物理对象交互的 640×512 像素图像序列组成。 没有给出额外的标签，该数据集旨在通过基于卷积 LSTM (ConvLSTM) 的深度学习算法在像素级别对运动进行建模。

#### BAIR Robot Pushing Dataset (used in [129]): 
BAIR (Berkeley Artificial Intelligence Research) group has been working on robots that can learn through unsupervised training (also known in this case as self-supervised), this is, learning the consequences that its actions (movement of the arm and grip) have over the data it can measure (images from two cameras). In this way, the robot assimilates physics of the objects and can predict the effects that its actions will generate on the environment, allowing it to plan strategies to achieve more general goals. This was improved by showing the robot how it can grab tools to interact with other objects. The dataset is composed by hours of this self-supervised learning with the robotic arm Sawyer.

BAIR Robot Pushing Dataset(在[129]中使用)：BAIR(伯克利人工智能研究)小组一直致力于研究可以通过无监督训练(在这种情况下也称为自我监督)进行学习的机器人，这就是学习结果 它的动作(手臂的移动和抓握)超过了它可以测量的数据(来自两个摄像头的图像)。 通过这种方式，机器人可以吸收物体的物理特性，并可以预测其动作将对环境产生的影响，从而制定策略以实现更普遍的目标。 通过向机器人展示它如何抓取工具与其他物体交互来改进这一点。 该数据集由使用机械臂 Sawyer 进行的这种自我监督学习的数小时组成。

#### RoboNet [130]: 
is a dataset composed by the aggregation of various self-supervised training sequences of seven robotic arms from four different research laboratories. The previously described BAIR group is one of them, along with Stanford AI Laboratory, Grasp Lab of the University of Pennsylvania and Google Brain Robotics. It was created with the goal of being a standard, in the same way as ImageNet is for images, but for robotic self-supervised learning. Several experiments have been performed studying how the transfer among robotic arms can be achieved.

RoboNet [130]：是一个数据集，由来自四个不同研究实验室的七个机械臂的各种自我监督训练序列的聚合组成。 之前描述的 BAIR 小组就是其中之一，还有斯坦福人工智能实验室、宾夕法尼亚大学的 Grasp 实验室和谷歌大脑机器人。 它的创建目标是成为一个标准，就像 ImageNet 用于图像一样，但用于机器人自我监督学习。 已经进行了几项实验来研究如何实现机械臂之间的转移。

### 4.5 Other-purpose and Multi-purpose Datasets
#### ViSOR [131]: 
ViSOR (Video Surveillance Online Repository) is a repository designed with the aim of establishing an open platform for collecting, annotating, retrieving, and sharing surveillance videos, as well as evaluating the performance of automatic surveillance systems. Its raw data could be very useful for video prediction due to its implicit static camera.

ViSOR [131]：ViSOR(视频监控在线存储库)是一个存储库，旨在建立一个开放平台，用于收集、注释、检索和共享监控视频，以及评估自动监控系统的性能。 由于其隐式静态相机，其原始数据对于视频预测非常有用。

#### PROST [132]: 
is a method for online tracking that used ten manually annotated videos to test its performance. Four of them were created by PROST authors, and they conform the dataset with the same name. The remaining six sequences were borrowed from other authors, who released their annotated clips to test their tracking methods. We will consider both 4-sequences PROST dataset and 10-sequences aggregated dataset when providing statistics. In each video, different challenges are presented for tracking methods: occlusions, 3D motion, varying illumination, heavy appearance/scale changes, moving camera, motion blur, among others. Provided annotations include bounding boxes for the object/element being tracked.

PROST [132]：是一种在线跟踪方法，使用十个手动注释的视频来测试其性能。 其中四个由 PROST 作者创建，并且符合同名数据集。 其余六个序列是从其他作者那里借来的，他们发布了他们的注释剪辑来测试他们的跟踪方法。 在提供统计数据时，我们将同时考虑 4 序列 PROST 数据集和 10 序列聚合数据集。 在每个视频中，针对跟踪方法提出了不同的挑战：遮挡、3D 运动、变化的照明、大量外观/比例变化、移动相机、运动模糊等。 提供的注释包括被跟踪对象/元素的边界框。

#### Arcade Learning Environment [133]: 
is a platform that enables machine learning algorithms to interact with the Atari 2600 open-source emulator Stella to play over 500 Atari games. The interface provides a single 2D frame of 210×160 pixels resolution at 60 fps in real-time, and up to 6000 fps when it is running at full speed. It also offers the possibility of saving and restoring the state of a game. Although its obvious main application is reinforcement learning, it could also be profitable as source of almost-infinite interactive video sequences from which prediction models can learn.

Arcade Learning Environment [133]：是一个使机器学习算法能够与 Atari 2600 开源模拟器 Stella 交互以玩 500 多种 Atari 游戏的平台。 该界面以 60 fps 的速度实时提供 210×160 像素分辨率的单个 2D 帧，全速运行时最高可达 6000 fps。 它还提供了保存和恢复游戏状态的可能性。 尽管其明显的主要应用是强化学习，但作为预测模型可以从中学习的几乎无限的交互式视频序列的来源，它也可能是有利可图的。

#### Inria 3DMovie Dataset v2 [134]: 
is a video dataset which extracted its data from the StreetDance 3D stereo movies. The dataset includes stereo pairs, and manually generated ground-truth for human segmentation, poses and bounding boxes. The second version of this dataset, used in [134], is composed by 27 clips, which represent 2476 frames, of which just a sparse subset of 235 were annotated.

Inria 3DMovie Dataset v2 [134]：是一个视频数据集，它从 StreetDance 3D 立体电影中提取数据。 该数据集包括立体对，以及为人体分割、姿势和边界框手动生成的基准实况。 [134] 中使用的该数据集的第二个版本由 27 个剪辑组成，代表 2476 帧，其中仅注释了 235 帧的稀疏子集。

#### RobotriX [16]: 
is a synthetic dataset designed for assistance robotics, that consist of sequences where a humanoid robot is moving through various indoor scenes and interacting with objects, recorded from multiple points of view, including robot-mounted cameras. It provides a huge variety of ground-truth data generated synthetically from highlyrealistic environments deployed on the cutting-edge game engine UnrealEngine, through the also available tool UnrealROX [139]. RGB frames are provided at 1920 × 1080 pixels resolution and at 60 fps, along with pixel-precise instance masks, depth and normal maps, and 6D poses of objects, skeletons and cameras. Moreover, UnrealROX is an open source tool for retrieving ground-truth data from any simulation running in UnrealEngine.

RobotriX [16]：是一个专为辅助机器人技术设计的合成数据集，由人形机器人在各种室内场景中移动并与物体交互的序列组成，从多个角度记录，包括安装在机器人上的摄像机。 它通过同样可用的工具 UnrealROX [139]，提供了从部署在尖端游戏引擎 UnrealEngine 上的高度逼真的环境综合生成的大量基准实况数据。 提供 1920 × 1080 像素分辨率和 60 fps 的 RGB 帧，以及像素精确的实例蒙版、深度和法线贴图，以及物体、骨架和相机的 6D 姿势。 此外，UnrealROX 是一个开源工具，用于从 UnrealEngine 中运行的任何模拟中检索基准实况数据。

#### UASOL [135]: 
is a large-scale dataset consisting of highresolution sequences of stereo pairs recorded outdoors at pedestrian (egocentric) point of view. Along with them, precise depth maps are provided, computed offline from stereo pairs by the same camera. This dataset is intended to be useful for depth estimation, both from single and stereo images, research fields where outdoor and pedestrian-pointof-view data is not abundant. Frames were taken at a resolution of 2280 × 1282 pixels at 15 fps. 

UASOL [135]：是一个大型数据集，由以行人(自我中心)视角在户外记录的高分辨率立体对序列组成。 与它们一起，提供精确的深度图，由同一台相机从立体对中离线计算。 该数据集旨在用于深度估计，包括单幅图像和立体图像，以及室外和行人视点数据不丰富的研究领域。 帧以 2280 × 1282 像素的分辨率以 15 fps 的速度样本。

## 5 VIDEO PREDICTION METHODS
In the video prediction literature we find a broad range of different methods and approaches. Early models focused on directly predicting raw pixel intensities, by implicitly modeling scene dynamics and low-level details (Section 5.1). However, extracting a meaningful and robust representation from raw videos is challenging, since the pixel space is highly dimensional and extremely variable. From this point, reducing the supervision effort and the representation dimensionality emerged as a natural evolution. On the one hand, the authors aimed to disentangle the factors of variation from the visual content, i.e. factorizing the prediction space. For this purpose, they: (1) formulated the prediction problem into an intermediate transformation space by explicitly modeling the source of variability as transformations between frames (Section 5.2); (2) separated motion from the visual content with a two-stream computation (Section 5.3). On the other hand, some models narrowed the output space by conditioning the predictions on extra variables (Section 5.4), or reformulating the problem in a higher-level space (Section 5.5). High-level representation spaces are increasingly more attractive, since intelligent systems rarely rely on raw pixel information for decision making. Besides simplifying the prediction task, some other works addressed the future uncertainty in predictions. As the vast majority of video prediction models are deterministic, they are unable to manage probabilistic environments. To address this issue, several authors proposed modeling future uncertainty with probabilistic models (Section 5.6).

在视频预测文献中，我们发现了范围广泛的不同方法和途径。 早期模型侧重于通过隐式建模场景动态和低级细节(第 5.1 节)直接预测原始像素强度。 然而，从原始视频中提取有意义且稳健的表示具有挑战性，因为像素空间是高维且极易变化的。 从这一点来看，减少监督工作和表示维度成为一种自然演变。 一方面，作者旨在从视觉内容中分离出变化因素，即分解预测空间。 为此，他们：(1) 通过将可变性源显式建模为帧之间的转换(第 5.2 节)，将预测问题表述为中间转换空间;  (2) 通过双流计算将运动与视觉内容分开(第 5.3 节)。 另一方面，一些模型通过对额外变量进行预测(第 5.4 节)或在更高级别的空间中重新表述问题(第 5.5 节)来缩小输出空间。 高级表示空间越来越有吸引力，因为智能系统很少依赖原始像素信息进行决策。 除了简化预测任务外，其他一些工作还解决了预测中未来的不确定性。 由于绝大多数视频预测模型都是确定性的，因此它们无法管理概率环境。 为了解决这个问题，几位作者提出用概率模型对未来的不确定性进行建模(第 5.6 节)。

* Video Prediction
    * Through Direct Pixel Synthesis
        * Implicit Modeling of Scene Dynamics
    * Factorizing the Prediction Space
        * Using Explicit Transformations
        * With Explicit Motion from Content Separation
    * Narrowing the Prediction Space
        * By Conditioning on Extra Variables
        * To High-level Feature Space
    * By Incorporating Uncertainty
        * Using Probabilistic Approaches
Fig. 3: Classification of video prediction models. 

* 视频预测
     * 通过直接像素合成
         * 场景动力学的隐式建模
     * 分解预测空间
         * 使用显式转换
         * 内容分离的显式运动
     * 缩小预测空间
         * 以额外变量为条件
         * 到高级特征空间
     * 通过结合不确定性
         * 使用概率方法
图 3：视频预测模型的分类。

So far in the literature, there is no specific taxonomy that classifies video prediction models. In this review, we have classified the existing methods according to the video prediction problem they addressed and following the classifi- cation illustrated in Figure 3. For simplicity, each subsection extends directly the last level in the taxonomy. Moreover, some methods in this review can be classified in more than one category since they addressed multiple problems. For instance, [9], [54], [85] are probabilistic models making predictions in a high-level space as they addressed both the future uncertainty and high dimensionality in videos. The category of these models were specified according to their main contribution. The most relevant methods, ordered in a chronological order, are summarized in Table 2 containing low-level details. Prediction is a widely discussed topic in different fields and at different levels of abstraction. For instance, the future prediction from a static image [3], [106], [140]–[143], vehicle behavior prediction [144] and human action prediction [17] are a different but inspiring research fields. Although related, the aforementioned topics are outside the scope of this particular review, as it focuses purely on the video prediction methods using a sequence of previous frames as context and is limited to 2D RGB data.

到目前为止，在文献中，还没有对视频预测模型进行分类的具体分类法。 在这篇综述中，我们根据现有方法解决的视频预测问题并遵循图3 中所示的分类对现有方法进行了分类。为简单起见，每个小节直接扩展分类法中的最后一层。 此外，本综述中的一些方法可以归为多个类别，因为它们解决了多个问题。 例如，[9]、[54]、[85] 是在高层次空间中进行预测的概率模型，因为它们解决了视频中的未来不确定性和高维度。 这些模型的类别是根据它们的主要贡献来指定的。 表2 总结了最相关的方法(按时间顺序排列)，其中包含低级详情。 预测是不同领域和不同抽象层次上广泛讨论的话题。 例如，静态图像的未来预测 [3]、[106]、[140]-[143]、车辆行为预测 [144] 和人类行为预测 [17] 是不同但鼓舞人心的研究领域。 尽管相关，但上述主题不在本次特别评论的范围内，因为它纯粹关注使用一系列先前帧作为上下文的视频预测方法，并且仅限于 2D RGB 数据。

### 5.1 Direct Pixel Synthesis 直接像素合成
Initial video prediction models attempted to directly predict future pixel intensities without any explicit modeling of the scene dynamics. Ranzato et al. [73] discretized video frames in patch clusters using k-means. They assumed that non-overlapping patches are equally different in a k-means discretized space, yet similarities can be found between patches. The method is a convolutional extension of a RNNbased model [145] making short-term predictions at the patch-level. As the full-resolution frame is a composition of the predicted patches, some tilling effect can be noticed.

初始视频预测模型试图直接预测未来的像素强度，而无需对场景动态进行任何显式建模。 兰扎托et al. [73] 使用 k-means 离散化分块集群中的视频帧。 他们假设非重叠的分块在 k 均值离散空间中同样不同，但可以在分块之间找到相似之处。 该方法是基于 RNN 的模型 [145] 的卷积扩展，可在分块级别进行短期预测。 由于全分辨率帧是预测块的组合，因此可以注意到一些倾斜效果。

Predictions of large and fast-moving objects are accurate, however, when it comes to small and slow-moving objects there is still room for improvement. These are common issues for most methods making predictions at the patch-level. Addressing longer-term predictions, Srivastava et al. [74] proposed different AE-based approaches incorporating LSTM units to model the temporal coherence. Using convolutional [146] and flow [147] percepts alongside RGB image patches, authors tested the models on multi-domain tasks and considered both unconditioned and conditioned decoder versions. The latter only marginally improved the prediction accuracy. Replacing the fully connected LSTMs with convolutional LSTMs, Shi et al. proposed an end-to-end model efficiently exploiting spatial correlations [13]. This enhanced prediction accuracy and reduced the number of parameters.

对大型和快速移动的物体的预测是准确的，但是，当涉及到小的和缓慢移动的物体时仍有改进的空间。 这些是大多数在分块级别进行预测的方法的常见问题。 针对长期预测，Srivastava et al. [74] 提出了不同的基于 AE 的方法，结合 LSTM 单元来模拟时间相干性。 使用卷积 [146] 和流 [147] 感知以及 RGB 图像块，作者在多域任务上测试了模型，并考虑了无条件和有条件的解码器版本。 后者仅略微提高了预测精度。 Shi et al. 用卷积 LSTM 替换完全连接的 LSTM。 提出了一种有效利用空间相关性的端到端模型[13]。 这提高了预测精度并减少了参数数量。

#### Inspired on adversarial training: 
Building on the recent success of the Laplacian Generative Adversarial Network (LAPGAN), Mathieu et al. proposed the first multi-scale architecture for video prediction that was trained in an adversarial fashion [43]. Their novel GDL regularization combined with l1-based reconstruction and adversarial training represented a leap over the previous state-of-the-art models [73], [74] in terms of prediction sharpness. However, it was outperformed by the Predictive Coding Network (PredNet) [75] which stacked several ConvLSTMs vertically connected by a bottom-up propagation of the local l1 error computed at each level. Previously to PredNet, the same authors proposed the Predictive Generative Network (PGN) [49], an end-to-end model trained with a weighted combination of adversarial loss and MSE on synthetic data. However, no tests on natural videos and comparison with state-of-the-art predictive models were carried out. Using a similar training strategy as [43], Zhou et al. used a convolutional AE to learn long-term dependencies from time-lapse videos [103]. Build on Progressively Growing GANs (PGGANs) [148], Aigner et al. proposed the FutureGAN [69], a three-dimensional (3d) convolutional Encoder-decoder (ED)-based model. They used the Wasserstein GAN with gradient penalty (WGANGP) loss [149] and conducted experiments on increasingly complex datasets. Extending [13], Zhang et al. proposed a novel LSTM-based architecture where hidden states are updated along a z-order curve [70]. Dealing with distortion and temporal inconsistency in predictions and inspired by the Human Visual System (HVS), Jin et al. [150] first incorporated multi-frequency analysis into the video prediction task to decompose images into low and high frequency bands. This allowed high-fidelity and temporally consistent predictions with the ground truth, as the model better lever- ages the spatial and temporal details. The proposed method outperformed previous state-of-the-art in all metrics except in the Learned Perceptual Image Patch Similarity (LPIPS), where probabilistic models achieved a better performance since their predictions are clearer and realistic but less consistent with the ground truth. Distortion and blurriness are further accentuated when it comes to predict under fast camera motions. To this end, Shouno [151] implemented a hierarchical residual network with top-down connections. Leveraging parallel prediction at multiple scales, authors reported finer details and textures under fast and large camera motion.

受到对抗训练的启发：基于拉普拉斯生成对抗网络 (LAPGAN) 最近取得的成功，Mathieu et al. 提出了第一个以对抗方式训练的视频预测多尺度架构[43]。 他们新颖的 GDL 正则化与基于 l1 的重建和对抗训练相结合，在预测清晰度方面代表了对先前最先进模型 [73]、[74] 的飞跃。 然而，它的表现优于预测编码网络 (PredNet) [75]，该网络堆叠了多个 ConvLSTM，通过在每个级别计算的局部 l1 误差的自下而上传播垂直连接。 在 PredNet 之前，相同的作者提出了预测生成网络 (PGN) [49]，这是一种端到端模型，在合成数据上使用对抗性损失和 MSE 的加权组合进行训练。 然而，没有对自然视频进行测试，也没有与最先进的预测模型进行比较。 Zhou et al. 使用与 [43] 类似的训练策略。 使用卷积 AE 从延时视频中学习长期依赖性 [103]。 以 Progressively Growing GANs (PGGANs) [148] 为基础，Aigner et al. 提出了 FutureGAN [69]，一种基于三维 (3d) 卷积编码器-解码器 (ED) 的模型。 他们使用具有梯度惩罚 (WGANGP) 损失的 Wasserstein GAN [149]，并在越来越复杂的数据集上进行了实验。 扩展 [13]，Zhang et al. 提出了一种新颖的基于 LSTM 的架构，其中隐藏状态沿着 z 顺序曲线更新 [70]。 处理预测中的失真和时间不一致并受到人类视觉系统 (HVS) 的启发，Jin et al. [150]首先将多频分析纳入视频预测任务，将图像分解为低频段和高频段。 这允许高保真和时间一致的预测与基准实况，因为模型更好地利用空间和时间细节 。 所提出的方法在所有指标上都优于以前的最新技术，除了学习感知图像分块相似性 (LPIPS)，其中概率模型取得了更好的性能，因为它们的预测更清晰、更现实，但与基本事实不太一致。 当涉及到在快速相机运动下进行预测时，失真和模糊会进一步加剧。 为此，Shouno [151] 实现了一个具有自上而下连接的分层残差网络。 利用多尺度的并行预测，作者报告了在快速和大的相机运动下更精细的细节和纹理。

#### Bidirectional flow: 
Under the assumption that video sequences are symmetric in time, Kwon et al. [101] explored a retrospective prediction scheme training a generator for both, forward and backward prediction (reversing the input sequence to predict the past). Their cycle GAN-based approach ensure the consistency of bidirectional prediction through retrospective cycle constraints. Similarly, Hu et al. [57] proposed a novel cycle-consistency loss used to train a GAN-based approach (VPGAN). Future frames are generated from a sequence of context frames and their variation in time, denoted as Z. Under the assumption that Z is symmetric in the encoding space, it is manipulated by the model manipulates to generate desirable moving directions. In the same spirit, other works focused on both, forward and backward predictions [37], [152]. Enabling state sharing between the encoder and decoder, Oliu et al. proposed the folded Recurrent Neural Network (fRNN) [153], a recurrent AE architecture featuring GRUs that implement a bidirectional flow of the information. The model demonstrated a stratified representation, which makes the topology more explainable, as well as efficient compared to regular AEs in terms or memory consumption and computational requirements.

双向流：假设视频序列在时间上是对称的，Kwon et al. [101] 探索了一种回顾性预测方案，训练生成器进行前向和后向预测(反转输入序列以预测过去)。 他们基于循环 GAN 的方法通过追溯循环约束确保双向预测的一致性。 同样，Hu et al. [57] 提出了一种新的循环一致性损失，用于训练基于 GAN 的方法(VPGAN)。 未来帧是从一系列上下文帧及其随时间变化生成的，表示为 Z。假设 Z 在编码空间中是对称的，它由模型操纵生成所需的移动方向。 本着同样的精神，其他作品同时关注前向和后向预测 [37]、[152]。 Oliu et al. 在编码器和解码器之间启用状态共享。 提出了折叠循环神经网络 (fRNN) [153]，这是一种循环 AE 架构，具有实现双向信息流的 GRU。 该模型展示了分层表示，这使得拓扑更易于解释，并且在内存消耗和计算要求方面与常规 AE 相比更加高效。

#### Exploiting 3D convolutions: 
for modeling short-term features, Wang et al. [66] integrated them into a recurrent network demonstrating promising results in both video prediction and early activity recognition. While 3D convolutions efficiently preserves local dynamics, RNNs enables longrange video reasoning. The eidetic 3d LSTM (E3d-LSTM) network, represented in Figure 4, features a gated-controlled self-attention module, i.e. eidetic 3D memory, that effectively manages historical memory records across multiple time steps. Outperforming previous works, Yu et al. proposed the Conditionally Reversible Network (CrevNet) [154] consisting of two modules, an invertible AE and a Reversible Predictive Model (RPM). While the bijective two-way AE ensures no information loss and reduces the memory consumption, the RPM extends the reversibility from spatial to temporal domain. Some other works used 3D convolutional operations to model the time dimension [69].

利用 3D 卷积：用于建模短期特征，Wang et al. [66] 将它们整合到一个循环网络中，展示了在视频预测和早期活动识别方面有希望的结果。 虽然 3D 卷积有效地保留了局部动态，但 RNN 支持远程视频推理。 eidetic 3d LSTM (E3d-LSTM) 网络，如图 4 所示，具有门控自注意力模块，即 eidetic 3D 记忆，可有效管理跨多个时间步长的历史记忆记录。 Yu et al. 的表现优于以前的作品。 提出了条件可逆网络 (CrevNet) [154]，它由两个模块组成，一个可逆 AE 和一个可逆预测模型 (RPM)。 双射双向 AE 确保没有信息丢失并减少内存消耗，而 RPM 将可逆性从空间域扩展到时间域。 其他一些作品使用 3D 卷积运算来对时间维度进行建模 [69]。

Analyzing the previous works, Byeon et al. [76] identified a lack of spatial-temporal context in the representations, leading to blurry results when it comes to the future uncertainty. Although authors addressed this contextual limitation with dilated convolutions and multi-scale architectures, the context representation progressively vanishes in longterm predictions. To address this issue, they proposed a context-aware model that efficiently aggregates per-pixel contextual information at each layer and in multiple directions. The core of their proposal is a context-aware layer consisting of two blocks, one aggregating the information from multiple directions and the other blending them into a unified context.

分析以前的作品，Byeon et al. [76] 发现表示中缺乏时空背景，导致未来不确定性的结果模糊。 尽管作者通过扩张卷积和多尺度架构解决了这种上下文限制，但上下文表示在长期预测中逐渐消失。 为了解决这个问题，他们提出了一种上下文感知模型，可以在每一层和多个方向上有效地聚合每个像素的上下文信息。 他们提议的核心是一个由两个块组成的上下文感知层，一个从多个方向聚合信息，另一个将它们混合到一个统一的上下文中。

Fig. 4: Representation of the 3D encoder-decoder architecture of E3d-LSTM [66]. After reducing T consecutive input frames to high-dimensional feature maps, these are directly fed into a novel eidetic module for modeling long-term spatiotemporal dependencies. Finally, stacked 3D CNN decoder outputs the predicted video frames. For classification tasks the hidden states can be directly used as the learned video representation. Figure extracted from [66].
图 4：E3d-LSTM [66] 的 3D 编码器-解码器架构的表示。 在将 T 个连续输入帧减少为高维特征图后，这些被直接输入到一个新的 eidetic 模块中，用于建模长期时空依赖性。 最后，堆叠式 3D CNN 解码器输出预测的视频帧。 对于分类任务，隐藏状态可以直接用作学习的视频表示。 图摘自[66]。

Fig. 5: Representation of transformation-based approaches. (a) Vector-based with a bilinear interpolation. (b) Applying transformations as a convolutional operation. Figure inspired by [155]. 
图 5：基于转换的方法的表示。 (a) 基于向量的双线性插值。 (b) 将变换应用为卷积运算。 图的灵感来自 [155]。

Extracting a robust representation from raw pixel values is an overly complicated task due to the high-dimensionality of the pixel space. The per-pixel variability between consecutive frames, causes an exponential growth in the prediction error on the long-term horizon.

由于像素空间的高维性，从原始像素值中提取稳健表示是一项过于复杂的任务。 连续帧之间的每像素可变性导致长期范围内的预测误差呈指数增长。

### 5.2 Using Explicit Transformations 使用显式转换
Let X = (Xt−n, . . . , Xt−1, Xt) be a video sequence of n frames, where t denotes time. Instead of learning the visual appearance, transformation-based approaches assume that visual information is already available in the input sequence. To deal with the strong similarity and pixel redundancy between successive frames, these methods explicitly model the transformations that takes a frame at time t to the frame at t+1. These models are formally defined as follows:

设 X = (Xt-n, . . , Xt-1, Xt) 为 n 帧的视频序列，其中 t 表示时间。 基于转换的方法不是学习视觉外观，而是假设视觉信息已经在输入序列中可用。 为了处理连续帧之间的强相似性和像素冗余，这些方法显式地模拟了从时间 t 的帧到 t+1 的帧的转换。 这些模型的正式定义如下：

Yt+1 = T (G (Xt−n:t), Xt−n:t), (1) 

where G is a learned function that outputs future transformation parameters, which applied to the last observed frame Xt using the function T , generates the future frame prediction Yt+1. According to the classification of Reda et al. [155], T function can be defined as a vector-based resampling such as bilinear sampling, or adaptive kernelbased resampling, e.g. using convolutional operations. For instance, a bilinear sampling operation is defined as:

其中 G 是一个学习函数，它输出未来的变换参数，使用函数 T 将其应用于最后观察到的帧 Xt，生成未来帧预测 Yt+1。 根据 Reda et al. [155]的分类。 ，T 函数可以定义为基于向量的重采样，例如双线性采样，或基于自适应内核的重采样，例如 使用卷积运算。 例如，双线性采样操作定义为：

Yt+1(x, y) = f (Xt(x + u, y + v)), (2) 

where f is a bilinear interpolator such as [7], [156], [157], (u, v) is a motion vector predicted by G, and Xt(x, y) is a pixel value at (x,y) in the last observed frame Xt. Approaches following this formulation are categorized as vector-based resampling operations and are depicted in Figure 5a.

其中f是双线性插值器如[7]、[156]、[157]，(u, v)是G预测的运动向量，Xt(x, y)是(x,y)处的像素值 在最后观察到的帧 Xt 中。遵循此公式的方法被归类为基于矢量的重采样操作，如图 5a 所示。

On the other side, in the kernel-based resampling, the G function predicts the kernel K(x, y) which is applied as a convolution operation using T , as depicted in Figure 5b and is mathematically represented as follows:

另一方面，在基于内核的重采样中，G 函数预测内核 K(x, y)，它使用 T 作为卷积运算应用，如图 5b 所示，数学表示如下：

Yt+1(x, y) = K(x, y) ∗ Pt(x, y), (3) 

where K(x, y) ∈ R NxN is the 2D kernel predicted by the function G and Pt(x, y) is an N ×N patch centered at (x, y).

其中 K(x, y) ∈ R NxN 是由函数 G 预测的二维内核，Pt(x, y) 是以 (x, y) 为中心的 N × N 块。

Combining kernel and vector-based resampling into a hybrid solution, Reda et al. [155] proposed the Spatially Displaced Convolution (SDC) module that synthesizes highresolution images applying a learned per-pixel motion vector and kernel at a displaced location in the source image. Their 3D CNN model trained on synthetic data and featuring the SDC modules, reported promising predictions of a high-fidelity.

Reda et al. 将内核和基于矢量的重采样结合到一个混合解决方案中。 [155] 提出了空间位移卷积 (SDC) 模块，它在源图像的位移位置应用学习到的每像素运动向量和核来合成高分辨率图像。 他们的 3D CNN 模型在合成数据上训练并具有 SDC 模块，报告了高保真度的有希望的预测。

#### 5.2.1 Vector-based Resampling 基于向量的重采样
Bilinear models use multiplicative interactions to extract transformations from pairs of observations in order to relate images, such as Gated Autoencoders (GAEs) [158]. Inspired by these models, Michalski et al. proposed the Predictive Gating Pyramid (PGP) [159] consisting on a recurrent pyramid of stacked GAEs. To the best of our knowledge, this was the first attempt to predict future frames in the affine transform space. Multiple GAEs are stacked to represent a hierarchy of transformations and capture higher-order dependencies. From the experiments on predicting frequency modulated sin-waves, authors stated that standard RNNs were outperformed in terms of accuracy. However, no performance comparison was conducted on videos.

双线性模型使用乘法交互从成对的观察中提取变换，以便关联图像，例如门控自动编码器 (GAE) [158]。 受这些模型的启发，Michalski et al. 。 提出了预测门控金字塔 (PGP) [159]，该金字塔由堆叠 GAE 的循环金字塔组成。 据我们所知，这是在仿射变换空间中预测未来帧的首次尝试。 多个 GAE 被堆叠起来以表示转换的层次结构并捕获更高阶的依赖关系。 从预测调频正弦波的实验中，作者表示标准 RNN 在准确性方面优于标准 RNN。 但是，没有对视频进行性能比较。

##### Based on the Spatial Transformer (ST) module [160]: 
To provide spatial transformation capabilities to existing CNNs, Jaderberg et al. [160] proposed the ST module represented in Figure 6. It regresses different affine transformation parameters for each input, to be applied as a single transformation to the whole feature map(s) or image(s).

基于空间变换器 (ST) 模块 [160]：为现有 CNN 提供空间变换功能，Jaderberg et al. 。 [160] 提出了图 6 中表示的 ST 模块。它为每个输入回归不同的仿射变换参数，作为单个变换应用于整个特征图或图像。

Fig. 6: A representation of the spatial transformer module proposed by [160]. First, the localization network regresses the transformation parameters, denoted as θ, from the input feature map U. Then, the grid generator creates a sampling grid from the predicted transformation parameters. Finally, the sampler produces the output map by sampling the input at the points defined in the sampling grid. Figure extracted from [160].

图 6：[160] 提出的空间变换器模块的表示。 首先，定位网络从输入特征图 U 中回归变换参数，表示为 θ。然后，网格生成器根据预测的变换参数创建采样网格。 最后，采样器通过在采样网格中定义的点对输入进行采样来生成输出映射。 图摘自[160]。

Moreover, it can be incorporated at any part of the CNNs and it is fully differentiable. The ST module is the essence of vector-based resampling approaches for video prediction. As an extension, Patraucean et al. [77] modified the grid generator to consider per-pixel transformations instead of a single dense transformation map for the entire image. They nested a LSTM-based temporal encoder into a spatial AE, proposing the AE-convLSTM-flow architecture. The prediction is generated by resampling the current frame with the flow-based predicted transformation. Using the components of the AE-convLSTM-flow architecture, Lu et al. [78] assembled an extrapolation module which is unfolded in time for multi-step prediction. Their Flexible Spatio-semporal Network (FSTN) features a novel loss function using the DeePSiM perceptual loss [44] in order to mitigate blurriness. An exhaustive experimentation and ablation study was carried out, testing multiple combinations of loss functions. Also inspired on the ST module for the volume sampling layer, Liu et al. proposed the Deep Voxel Flow (DVF) architecture [7]. It consists of a multi-scale flow-based ED model originally designed for the video frame interpolation task, but also evaluated on a predictive basis reporting sharp results. Liang et al. [55] use a flow-warping layer based on a bilinear interpolation. Finn et al. proposed the Spatial Transformer Predictor (STP) motion-based model [89] producing 2D affine transformations for bilinear sampling. Pursuing efficiency, Amersfoort et al. [71] proposed a CNN designed to predict local affine transformations of overlapping image patches. Unlike the ST module, authors estimated transformations of input frames off-line and at patch level. As the model is parameter-efficient, it was unfolded in time for multi-step prediction. This resembles RNNs as the parameters are shared over time and the local affine transforms play the role of recurrent states.

此外，它可以合并到 CNN 的任何部分并且是完全可微的。 ST 模块是用于视频预测的基于矢量的重采样方法的本质。 作为扩展，Patraucean et al. 。 [77] 修改了网格生成器以考虑每个像素的变换，而不是整个图像的单个密集变换图。 他们将基于 LSTM 的时间编码器嵌套到空间 AE 中，提出了 AE-convLSTM-flow 架构。 通过使用基于流的预测变换对当前帧进行重采样来生成预测。 使用 AE-convLSTM-flow 架构的组件，Lu et al. 。 [78]组装了一个外推模块，该模块及时展开以进行多步预测。 他们的灵活时空网络 (FSTN) 具有使用 DeePSiM 感知损失 [44] 的新型损失函数，以减轻模糊度。 进行了详尽的实验和消融研究，测试了损失函数的多种组合。 Liu et al. 也受到体积采样层 ST 模块的启发。 提出了深度体素流(DVF)架构[7]。 它由最初为视频帧插值任务设计的基于流的多尺度 ED 模型组成，但也在预测的基础上进行了评估，报告了清晰的结果。 梁等。 [55] 使用基于双线性插值的流扭曲层。 芬恩等。 提出了空间变换器预测器(STP)基于运动的模型[89]，为双线性采样产生二维仿射变换。 为了追求效率，Amersfoort et al. 。 [71] 提出了一种 CNN，旨在预测重叠图像块的局部仿射变换。 与 ST 模块不同，作者估计了离线和补丁级别的输入帧的转换。 由于该模型具有参数效率，因此可以及时展开以进行多步预测。 这类似于 RNN，因为参数随时间共享，局部仿射变换扮演循环状态的角色。

#### 5.2.2 Kernel-based Resampling 基于内核的重采样
As a promising alternative to the vector-based resampling, recent approaches synthesize pixels by convolving input patches with a predicted kernel. However, convolutional operations are limited in learning spatial invariant representations of complex transformations. Moreover, due to 12 their local receptive fields, global spatial information is not fully preserved. Using larger kernels would help to preserve global features, but in exchange for a higher memory consumption. Pooling layers are another alternative, but loosing spatial resolution. Preserving spatial resolution at a low computational cost is still an open challenge for future video frame prediction task. Transformation layers used in vector-based resampling [7], [77], [160] enabled CNNs to be spatially invariant and also inspired kernel-based architectures.

作为基于矢量的重采样的有前途的替代方案，最近的方法通过将输入补丁与预测内核进行卷积来合成像素。 然而，卷积运算在学习复杂变换的空间不变表示方面受到限制。 此外，由于它们有 12 个局部感受野，全局空间信息没有得到完整保留。 使用更大的内核将有助于保留全局特征，但以换取更高的内存消耗。 池化层是另一种选择，但会降低空间分辨率。 以低计算成本保持空间分辨率仍然是未来视频帧预测任务的一个开放挑战。 基于矢量的重采样 [7]、[77]、[160] 中使用的转换层使 CNN 具有空间不变性，并启发了基于内核的架构。

##### Inspired on the Convolutional Dynamic Neural Advection (CDNA) module [89]: 
In addition to the STP vectorbased model, Finn et al. [89] proposed two different kernelbased motion prediction modules outperforming previous approaches [43], [80], (1) the Dynamic Neural Advection (DNA) module predicting different distributions for each pixel and (2) the CDNA module that instead of predicting different distributions for each pixel, it predicts multiple discrete distributions applied convolutionally to the input. While, CDNA and STP mask out objects that are moving in consistent directions, the DNA module produces perpixel motion. These modules inspired several kernel-based approaches. Similar to the CDNA module, Klein et al. proposed the Dynamic Convolutional Layer (DCL) [161] for short-range weather prediction. Likewise, Brabandere et al. [162] proposed the Dynamic Filter Networks (DFN) generating sample (for each image) and position-specific (for each pixel) kernels. This enabled sophisticated and local filtering operations in comparison with the ST module, that is limited to global spatial transformations. Different to the CDNA model, the DFN uses a softmax layer to filter values of greater magnitude, thus obtaining sharper predictions. Moreover, temporal correlations are exploited using a parameter-efficient recurrent layer, much simpler than [13], [74]. Exploiting adversarial training, Vondrick et al. proposed a cGAN-based model [102] consisting of a discriminator similar to [67] and a CNN generator featuring a transformer module inspired on the CDNA model. Different from the CDNA model, transformations are not applied recurrently on a per-frame basis. To deal with inthe-wild videos and make predictions invariant to camera motion, authors stabilized the input videos. However, no performance comparison with previous works has been conducted.

受卷积动态神经平流 (CDNA) 模块 [89] 的启发：除了基于 STP 矢量的模型，Finn et al. 。 [89] 提出了两种不同的基于内核的运动预测模块，优于以前的方法 [43]、[80]，(1) 动态神经平流 (DNA) 模块预测每个像素的不同分布，(2) CDNA 模块而不是预测不同的分布 每个像素的分布，它预测卷积应用于输入的多个离散分布。 虽然 CDNA 和 STP 掩盖了沿一致方向移动的物体，但 DNA 模块产生每像素运动。 这些模块启发了几种基于内核的方法。 与 CDNA 模块类似，Klein et al. 。 提出了用于短期天气预报的动态卷积层(DCL)[161]。 同样，Brabandere et al. 。 [162] 提出了动态滤波器网络 (DFN) 生成样本(针对每个图像)和特定于位置的(针对每个像素)内核。 与仅限于全局空间转换的 ST 模块相比，这启用了复杂的局部过滤操作。 与 CDNA 模型不同，DFN 使用 softmax 层来过滤更大数量级的值，从而获得更清晰的预测。 此外，使用参数有效的循环层利用时间相关性，比 [13]、[74] 简单得多。 利用对抗性训练，Vondrick et al. 。 提出了一种基于 cGAN 的模型 [102]，该模型由类似于 [67] 的鉴别器和一个 CNN 生成器组成，该生成器具有受 CDNA 模型启发的转换器模块。 与 CDNA 模型不同，转换不会在每帧的基础上循环应用。 为了处理野外视频并做出不随摄像机运动变化的预测，作者稳定了输入视频。 但是，没有与以前的作品进行性能比较。

Relying on kernel-based transformations and improving [163], Luc et al. [164] proposed the Transformation-based & TrIple Video Discriminator GAN (TrIVD-GAN-FP) featuring a novel recurrent unit that computes the parameters of a transformation used to warp previous hidden states without any supervision. These Transformation-based Spatial Recurrent Units (TSRUs) are generic modules and can replace any traditional recurrent unit in currently existent video prediction approaches.

依靠基于内核的转换和改进 [163]，Luc et al. 。 [164] 提出了基于转换的三重视频鉴别器 GAN(TrIVD-GAN-FP)，它具有一个新颖的循环单元，该单元计算用于在没有任何监督的情况下扭曲先前隐藏状态的转换参数。 这些基于转换的空间循环单元 (TSRU) 是通用模块，可以替代当前存在的视频预测方法中的任何传统循环单元。

##### Object-centric representation: 
Instead of focusing on the whole input, Chen et al. [50] modeled individual motion of local objects, i.e. object-centered representations. Based on the ST module and a pyramid-like sampling [165], authors implemented an attention mechanism for object selection.

以对象为中心的表示：Chen et al. 没有关注整个输入。 [50] 对局部对象的个体运动进行建模，即以对象为中心的表示。 基于 ST 模块和类似金字塔的采样 [165]，作者实现了一种用于对象选择的注意机制。

Moreover, transformation kernels were generated dynamically as in the DFN, to then apply them to the last patch containing an object. Although object-centered predictions is novel, performance drops when dealing with multiple objects and occlusions as the attention module fails to distinguish them correctly.

此外，转换内核像在 DFN 中一样动态生成，然后将它们应用于包含对象的最后一个补丁。 尽管以对象为中心的预测很新颖，但在处理多个对象和遮挡时性能会下降，因为注意力模块无法正确区分它们。

### 5.3 Explicit Motion from Content Separation 内容分离的显式运动
Drawing inspiration from two-stream architectures for action recognition [166], video generation from a static image [67], and unconditioned video generation [68], authors decided to factorize the video into content and motion to process each on a separate pathway. By decomposing the high-dimensional videos, the prediction is performed on a lower-dimensional temporal dynamics separately from the spatial layout. Although this makes end-to-end training difficult, factorizing the prediction task into more tractable problems demonstrated good results.

从用于动作识别的双流架构 [166]、从静态图像生成视频 [67] 和无条件视频生成 [68] 中汲取灵感，作者决定将视频分解为内容和动作，以便在单独的路径上进行处理。 通过分解高维视频，预测是在与空间布局分开的低维时间动态上进行的。 尽管这使得端到端训练变得困难，但将预测任务分解为更易于处理的问题证明了良好的结果。

The Motion-content Network (MCnet) [65], represented in Figure 7 was the first end-to-end model that disentangled scene dynamics from the visual appearance. Authors performed an in-depth performance analysis ensuring the motion and content separation through generalization capabilities and stable long-term predictions compared to models that lack of explicit motion-content factorization [43], [74]. In a similar fashion, yet working in a higher-level pose space, Denton et al. proposed Disentangled-representation Net (DRNET) [79] using a novel adversarial loss —it isolates the scene dynamics from the visual content, considered as the discriminative component— to completely disentangle motion dynamics from content. Outperforming [43], [65], the DRNET demonstrated a clean motion from content separation by reporting plausible long-term predictions on both synthetic and natural videos. To improve prediction variability, Liang et al. [55] fused the future-frame and future-flow prediction into a unified architecture with a shared probabilistic motion encoder. Aiming to mitigate the ghosting effect in disoccluded regions, Gae et al. [167] 13 proposed a two-staged approach consisting of a separate computation of flow and pixel predictions. As they focused on inpainting occluded regions of the image using flow information, they improved results on disoccluded areas avoiding undesirable artifacts and enhancing sharpness. Separating the moving objects and the static background, Wu et al. [168] proposed a two-staged architecture that firstly predicts the static background to then, using this information, predict the moving objects in the foreground. Final results are generated through composition and by means of a video inpainting module. Reported predictions are quite accurate, yet performance was not contrasted with the latest video prediction models.

运动内容网络 (MCnet) [65]，如图7 所示，是第一个将场景动态与视觉外观分离的端到端模型。 作者进行了深入的性能分析，与缺乏显式运动内容分解的模型相比，通过泛化能力和稳定的长期预测确保运动和内容分离 [43]，[74]。 以类似的方式，但在更高级别的姿势空间中工作，Denton et al. 。 提出了分离表示网络(DRNET)[79]，使用一种新颖的对抗性损失——它将场景动态与视觉内容隔离开来，被认为是判别成分——以完全分离运动动态与内容。 优于 [43]、[65]，DRNET 通过报告对合成视频和自然视频的合理长期预测，展示了内容分离的干净动作。 为了提高预测可变性，Liang et al. 。 [55] 将未来帧和未来流预测融合到一个具有共享概率运动编码器的统一架构中。 为了减轻不被遮挡区域的重影效应，Gae et al. 。 [167] 13 提出了一种两阶段方法，包括单独计算流量和像素预测。 当他们专注于使用流信息修复图像的遮挡区域时，他们改进了遮挡区域的结果，避免了不需要的伪影并增强了清晰度。 Wu et al. 将移动物体和静态背景分开。 [168] 提出了一种两阶段架构，首先预测静态背景，然后使用此信息预测前景中的移动物体。 最终结果是通过合成和视频修复模块生成的。 报告的预测非常准确，但性能并未与最新的视频预测模型进行对比。

Fig. 7: MCnet with Multi-scale Motion-Content Residuals. While the motion encoder captures the temporal dynamics in a sequence of image differences, the content encoder extracts meaningful spatial features from the last observed RGB frame. After that, the network computes motioncontent features that are fed into the decoder to predict the next frame. Figure extracted from [65]. 
图 7：具有多尺度运动内容残差的 MCnet。 当运动编码器捕获一系列图像差异中的时间动态时，内容编码器从最后观察到的 RGB 帧中提取有意义的空间特征。 之后，网络计算送入解码器的运动内容特征以预测下一帧。 图摘自[65]。

Although previous approaches disentangled motion from content, they have not performed an explicit decomposition into low-dimensional components. Addressing this issue, Hsieh et al. proposed the Decompositional Disentangled Predictive Autoencoder (DDPAE) [169] that decomposes the high-dimensional video into components represented with low-dimensional temporal dynamics. On the Moving MNIST dataset, DDPAE first decomposes images into individual digits (components) to then factorize each digit into its visual appearance and spatial location, being the latter easier to predict. Although experiments were performed on synthetic data, this approach represents a promising baseline to disentangle and decompose natural videos. Moreover, it is applicable to other existing models to improve their predictions.

尽管以前的方法将运动从内容中分离出来，但它们并没有对低维组件进行显式分解。 针对这个问题，Hsieh et al. 。 提出了分解解缠结预测自动编码器(DDPAE)[169]，它将高维视频分解为用低维时间动态表示的组件。 在 Moving MNIST 数据集上，DDPAE 首先将图像分解为单个数字(组件)，然后将每个数字分解为其视觉外观和空间位置，后者更易于预测。 虽然实验是在合成数据上进行的，但这种方法代表了一个有前途的基线来解开和分解自然视频。 此外，它适用于其他现有模型以改进其预测。

### 5.4 Conditioned on Extra Variables 以额外变量为条件
Conditioning the prediction on extra variables such as vehicle odometry or robot state, among others, would narrow the prediction space. These variables have a direct influence on the dynamics of the scene, providing valuable information that facilitates the prediction task. For instance, the motion captured by a camera placed on the dashboard of an autonomous vehicle is directly influenced by the wheelsteering and acceleration. Without explicitly exploiting this information, we rely blindly on the model’s capabilities to correlate the wheel-steering and acceleration with the perceived motion. However, the explicit use of these variables would guide the prediction.

以车辆里程计或机器人状态等额外变量为条件进行预测会缩小预测空间。 这些变量对场景的动态有直接影响，提供有助于预测任务的有价值信息。 例如，放置在自动驾驶汽车仪表板上的摄像头捕捉到的运动直接受到车轮转向和加速度的影响。 在没有明确利用这些信息的情况下，我们盲目地依赖模型的能力将车轮转向和加速度与感知运动相关联。 然而，明确使用这些变量将指导预测。

Following this paradigm, Oh et al. first made longterm video predictions conditioned by control inputs from Atari games [80]. Although the proposed ED-based models reported very long-term predictions (+100), performance drops when dealing with small objects (e.g. bullets in Space Invaders) and while handling stochasticity due to the squared error. However, by simply minimizing l2 error can lead to accurate and long-term predictions for deterministic synthetic videos, such as those extracted from Atari video games. Building on [80], Chiappa et al. [170] proposed alternative architectures and training schemes alongside an in-depth performance analysis for both short and long-term prediction. Similar model-based control from visual inputs performed well in restricted scenarios [171], but was inadequate for unconstrained environments. These deterministic approaches are unable to deal with natural videos in the absence of control variables.

遵循这种范式，Oh et al. 。 首先根据 Atari 游戏的控制输入进行长期视频预测 [80]。 尽管提议的基于 ED 的模型报告了非常长期的预测(+100)，但在处理小物体(例如太空入侵者中的子弹)和处理随机性时由于平方误差导致性能下降。 然而，通过简单地最小化 l2 误差可以导致对确定性合成视频的准确和长期预测，例如从 Atari 视频游戏中提取的视频。 在 [80] 的基础上，Chiappa et al. 。 [170] 提出了替代架构和训练方案，同时对短期和长期预测进行了深入的性能分析。 来自视觉输入的类似基于模型的控制在受限场景中表现良好 [171]，但不适用于不受约束的环境。 在没有控制变量的情况下，这些确定性方法无法处理自然视频。

To address this limitation, the models proposed by Finn et al. [89] successfully made predictions on natural images, conditioned on the robot state and robot-object interactions performed in a controlled scenario. These models predict per-pixel transformations conditioned by the previous frame, to finally combine them using a composition mask. They outperformed [43], [80] on both conditioned and unconditioned predictions, however the quality of long-term predictions degrades over time because of the blurriness caused by the MSE loss function. Also, using high-dimensional sensory such as images, Dosovitskiy et al. [172] proposed a sensorimotor control model which enables interaction in complex and dynamic 3d environments. The approach is a reinforcement learning (RL)-based techniques, with the difference that instead of building upon a monolithic state and a scalar reward, the authors consider high-dimensional input streams, such as raw visual input, alongside a stream of measurements or player statistics. Although the outputs are future measurements instead of visual predictions, it was proven that using multivariate data benefits decision-making over conventional scalar reward approaches.

为了解决这个限制，Finn et al. 提出的模型。 [89] 成功地对自然图像进行了预测，条件是在受控场景中执行的机器人状态和机器人 - 对象交互。 这些模型预测以前一帧为条件的每像素变换，最终使用合成掩码将它们组合起来。 它们在有条件和无条件预测方面都优于 [43]、[80]，但是由于 MSE 损失函数造成的模糊性，长期预测的质量会随着时间的推移而降低。 此外，使用图像等高维感官，Dosovitskiy et al. 。 [172] 提出了一种感觉运动控制模型，可以在复杂和动态的 3d 环境中进行交互。 该方法是一种基于强化学习 (RL) 的技术，不同之处在于作者没有建立在单一状态和标量奖励之上，而是考虑了高维输入流，例如原始视觉输入，以及测量流或 玩家统计。 虽然输出是未来的测量而不是视觉预测，但事实证明，使用多变量数据比传统的标量奖励方法更有利于决策。

### 5.5 In the High-level Feature Space 在高级特征空间中
Despite the vast work on video prediction models, there is still room for improvement in natural video prediction. To deal with the curse of dimensionality, authors reduced the prediction space to high-level representations, such as semantic and instance segmentation, and human pose. Since the pixels are categorical, the semantic space greatly simplifies the prediction task, yet unexpected deformations in semantic maps and disocclusions, i.e. initially occluded scene entities become visible, induce uncertainty. However, high-level prediction spaces are more tractable and constitute good intermediate representations. By bypassing the prediction in the pixel space, models become able to report longer-term and more accurate predictions.

尽管在视频预测模型方面做了大量工作，但自然视频预测仍有改进的空间。 为了应对维数灾难，作者将预测空间缩减为高级表示，例如语义和实例分割以及人体姿势。 由于像素是分类的，语义空间极大地简化了预测任务，但语义映射和遮挡中的意外变形，即最初被遮挡的场景实体变得可见，会引起不确定性。 然而，高级预测空间更易于处理并构成良好的中间表示。 通过绕过像素空间中的预测，模型能够报告更长期和更准确的预测。

#### 5.5.1 Semantic Segmentation
In recent years, semantic and instance representations have gained increasing attention, emerging as a promising avenue for complete scene understanding. By decomposing the visual scene into semantic entities, such as pedestrians, vehicles and obstacles, the output space is narrowed to high-level scene properties. This intermediate representation represents a more tractable space as pixel values of a semantic map are categorical. In other words, scene dynamics are modeled at the semantic entity level instead of being modeled at the pixel level. This has encouraged authors to (1) leverage future prediction to improve parsing results [51] and (2) directly predict segmentation maps into the future [8], [56], [173].

近年来，语义和实例表示越来越受到关注，成为完整场景理解的有前途的途径。 通过将视觉场景分解为语义实体，例如行人、车辆和障碍物，输出空间被缩小为高级场景属性。 这种中间表示表示更易处理的空间，因为语义图的像素值是分类的。 换句话说，场景动态是在语义实体级别建模的，而不是在像素级别建模的。 这鼓励作者 (1) 利用未来预测来改进解析结果 [51] 和 (2) 直接预测未来的分割图 [8]、[56]、[173]。

Fig. 8: Two-staged method proposed by Chiu et al. [174]. In the upper half, the student network consists on an ED-based architecture featuring a 3D convolutional forecasting module. It performs the forecasting task guided by an additional loss generated by the teacher network (represented in the lower half). Figure extracted from [174]. 
图 8：Chiu et al. 提出的两阶段方法。 [174]。 在上半部分，学生网络由一个基于 ED 的架构组成，该架构具有一个 3D 卷积预测模块。 它执行由教师网络(下半部分表示)产生的额外损失指导的预测任务。 图摘自[174]。

Exploring the scene parsing in future frames, Jin et al. proposed the Parsing with prEdictive feAtuRe Learning (PEARL) framework [51] which was the first to explore the potential of a GAN-based frame prediction model to improve per-pixel segmentation. Specifically, this framework conducts two complementary predictive learning tasks. Firstly, it captures the temporal context from input data by using a single-frame prediction network. Then, these temporal features are embedded into a frame parsing network through a transform layer for generating per-pixel future segmentations. Although the predictive net was not compared with existing approaches, PEARL outperforms the traditional parsing methods by generating temporally consistent segmentations. In a similar fashion, Luc et al. [56] extended the msCNN model of [43] to the novel task of predicting semantic segmentations of future frames, using softmax pre-activations instead of raw pixels as input. The use of intermediate features or higher-level data as input is a common practice in the video prediction performed in the high-level feature space. Some authors refer to this type or input data as percepts. Luc et al. explored different combinations of loss functions, inputs (using RGB information alongside percepts), and outputs (autoregressive and batch models). Results on short, medium and long-term predictions are sound, however, the models are not endto-end and they do not capture explicitly the temporal continuity across frames. To address this limitation and extending [51], Jin et al. first proposed a model for jointly predicting motion flow and scene parsing [175]. Flow-based representations implicitly draw temporal correlations from the input data, thus producing temporally coherent perpixel segmentations. As in [56], the authors tested different network configurations, as using Res101-FCN percepts for the prediction of semantic maps, and also performed multistep prediction up to 10 time-steps into the future. Perpixel accuracy improved when segmenting small objects, e.g. pedestrians and traffic signs, which are more likely to vanish in long-term predictions. Similarly, except that time dimension is modeled with LSTMs instead of motion flow estimation, Nabavi et al. proposed a simple bidirectional EDLSTM [82] using segmentation masks as input. Although the literature on knowledge distillation [176], [177] stated that softmax pre-activations carry more information than class labels, this model outperforms [56], [175] on short-term predictions.

探索未来帧中的场景解析，Jin et al. 。 提出了 Parsing with prEdictive feAtuRe Learning (PEARL) 框架 [51]，这是第一个探索基于 GAN 的帧预测模型改进逐像素分割潜力的方法。 具体来说，该框架执行两个互补的预测学习任务。 首先，它通过使用单帧预测网络从输入数据中捕获时间上下文。 然后，这些时间特征通过转换层嵌入到帧解析网络中，以生成每个像素的未来分割。 尽管未将预测网络与现有方法进行比较，但 PEARL 通过生成时间一致的分段优于传统的解析方法。 以类似的方式，Luc et al. 。 [56] 将 [43] 的 msCNN 模型扩展到预测未来帧的语义分割的新任务，使用 softmax 预激活而不是原始像素作为输入。 在高级特征空间中执行的视频预测中，使用中间特征或高级数据作为输入是一种常见做法。 一些作者将这种类型或输入数据称为感知。 吕克et al. 。 探索了损失函数、输入(使用 RGB 信息和感知)和输出(自回归和批处理模型)的不同组合。 短期、中期和长期预测的结果是合理的，但是，这些模型不是端到端的，它们没有明确捕捉跨帧的时间连续性。 为了解决这个限制和扩展 [51]，Jin et al. 。 首先提出了一个联合预测运动流和场景解析的模型[175]。 基于流的表示隐式地从输入数据中提取时间相关性，从而产生时间上相干的每像素分割。 与 [56] 一样，作者测试了不同的网络配置，如使用 Res101-FCN 感知来预测语义图，并且还对未来进行了多达 10 个时间步长的多步预测。 分割小物体时提高了每像素精度，例如 行人和交通标志，它们在长期预测中更有可能消失。 类似地，除了时间维度是用 LSTMs 而不是运动流估计建模，Nabavi et al. 。 提出了一个简单的双向 EDLSTM [82]，使用分割掩码作为输入。 尽管关于知识蒸馏的文献 [176]、[177] 指出 softmax 预激活比类标签携带更多信息，但该模型在短期预测方面优于 [56]、[175]。

Another relevant idea is to use both motion flow estimation alongside LSTM-based temporal modeling. In this direction, Terwilliger et al. [10] proposed a novel method performing a LSTM-based feature-flow aggregation. Authors also tried to further simplify the semantic space by disentangling motion from semantic entities [65], achieving low overhead and efficiency. The prediction problem was decomposed into two subtasks, that is, current frame segmentation and future optical flow prediction, which are finally combined with a novel end-to-end warp layer. An improvement on short-term predictions were reported over previous works [56], [175], yet performing worse on midterm predictions.

另一个相关的想法是同时使用运动流估计和基于 LSTM 的时间建模。 在这个方向上，Terwilliger et al. 。 [10] 提出了一种执行基于 LSTM 的特征流聚合的新方法。 作者还尝试通过从语义实体中分离运动来进一步简化语义空间 [65]，从而实现低开销和高效率。 预测问题被分解为两个子任务，即当前帧分割和未来光流预测，它们最终与一个新颖的端到端扭曲层相结合。 据报道，短期预测比以前的工作 [56]、[175] 有所改进，但在中期预测方面表现更差。

A different approach was proposed by Vora et al. [83] which first incorporated structure information to predict future 3D segmented point clouds. Their geometry-based model consists of several derivable sub-modules: (1) the pixel-wise segmentation and depth estimation modules which are jointly used to generate the 3d segmented point cloud of the current RGB frame; and (2) an LSTM-based module trained to predict future camera ego-motion trajectories. The future 3d segmented point clouds are obtained by transforming the previous point clouds with the predicted ego-motion. Their short-term predictions improved the results of [56], however, the use of structure information for longer-term predictions is not clear.

Vora et al. 提出了一种不同的方法。 [83] 首先结合结构信息来预测未来的 3D 分割点云。 他们基于几何的模型由几个可导出的子模块组成：(1)像素级分割和深度估计模块，它们共同用于生成当前 RGB 帧的 3d 分割点云;  (2) 一个基于 LSTM 的模块，经过训练可以预测未来的相机自运动轨迹。 未来的 3d 分割点云是通过用预测的自我运动转换先前的点云而获得的。 他们的短期预测改进了 [56] 的结果，但是，使用结构信息进行长期预测尚不清楚。

The main disadvantage of two-staged, i.e. not end-toend, approaches [10], [56], [82], [83], [175] is that their performance is constrained by external supervisory signals, e.g. optical flow [178], segmentation [179] and intermediate features or percepts [61]. Breaking this trend, Chiu et al. [174] first solved jointly the semantic segmentation and forecasting problems in a single end-to-end trainable model by using raw pixels as input. This ED architecture is based on two networks, with one performing the forecasting task (student) and the other (teacher) guiding the student by means of a novel knowledge distillation loss. An in-depth ablation study was performed, validating the performance of the ED architectures as well as the 3D convolution used for capturing temporal scale instead of a LSTM or ConvLSTM, as in previous works.

两阶段的主要缺点，即不是端到端的方法 [10]、[56]、[82]、[83]、[175] 是它们的性能受到外部监督信号的限制，例如 光流[178]、分割[179]和中间特征或感知[61]。 打破这一趋势，Chiu et al. 。 [174]首先通过使用原始像素作为输入，共同解决了单个端到端可训练模型中的语义分割和预测问题。 这种 ED 架构基于两个网络，一个执行预测任务(学生)，另一个(教师)通过一种新的知识蒸馏损失来指导学生。 进行了深入的消融研究，验证了 ED 架构的性能以及用于捕获时间尺度的 3D 卷积，而不是 LSTM 或 ConvLSTM，如之前的作品。

Avoiding the flood of deterministic models, Bhattacharyya et al. proposed a Bayesian formulation of the ResNet model in a novel architecture to capture model and observation uncertainty [9]. As main contribution, their dropout-based Bayesian approach leverages synthetic likelihoods [180] to encourage prediction diversity and deal with multi-modal outcomes. Since Cityscapes sequences have been recorded in the frame of reference of a moving vehicle, authors conditioned the predictions on vehicle odometry.

为了避免确定性模型的泛滥，Bhattacharyya et al. 。 在一种新颖的架构中提出了 ResNet 模型的贝叶斯公式，以捕获模型和观察的不确定性 [9]。 作为主要贡献，他们基于 dropout 的贝叶斯方法利用合成似然 [180] 来鼓励预测多样性和处理多模态结果。 由于 Cityscapes 序列已记录在移动车辆的参考系中，作者将预测作为车辆里程计的条件。

#### 5.5.2 Instance Segmentation
While great strides have been made in predicting future segmentation maps, the authors attempted to make predic- 15 tions at a semantically richer level, i.e. future prediction of semantic instances. Predicting future instance-level segmentations is a challenging and weakly unexplored task. This is because instance labels are inconsistent and variable in number across the frames in a video sequence. Since the representation of semantic segmentation prediction models is of fixed-size, they cannot directly address semantics at the instance level.

虽然在预测未来分割图方面取得了很大进展，但作者试图在语义更丰富的层面上进行预测，即未来语义实例的预测。 预测未来的实例级分割是一项具有挑战性且尚未探索的任务。 这是因为实例标签在视频序列的帧中不一致且数量可变。 由于语义分割预测模型的表示是固定大小的，它们不能直接在实例级别处理语义。

To overcome this limitation and introducing the novel task of predicting instance segmentations, Luc et al. [8] predict fixed-sized feature pyramids, i.e. features at multiple scales, used by the Mask R-CNN [181] network. The combination of dilated convolutions and multi-scale, efficiently preserve high-resolution details improving the results over previous methods [56]. To further improve predictions, Sun et al. [84] focused on modeling not only the spatio-temporal correlations between the pyramids, but also the intrinsic relations among the feature layers inside them. By enriching the contextual information using the proposed Context Pyramid ConvLSTMs (CP-ConvLSTM), an improvement in the prediction was noticed. Although the authors have not shown any long-term predictions nor compared with semantic segmentation models, their approach is currently the state of the art in the task of predicting instance segmentations, outperforming [8].

为了克服这一限制并引入预测实例分割的新任务，Luc et al. 。 [8] 预测固定大小的特征金字塔，即多个尺度的特征，由 Mask R-CNN [181] 网络使用。 扩张卷积和多尺度的结合，有效地保留了高分辨率的细节，改进了以前方法的结果[56]。 为了进一步改进预测，Sun et al. 。 [84] 不仅关注金字塔之间的时空相关性建模，还关注金字塔内部特征层之间的内在关系。 通过使用所提出的上下文金字塔 ConvLSTM (CP-ConvLSTM) 丰富上下文信息，人们注意到预测的改进。 尽管作者没有展示任何长期预测，也没有与语义分割模型进行比较，但他们的方法目前是预测实例分割任务的最新技术，优于 [8]。

#### 5.5.3 Other High-level Spaces
Although semantic and instance segmentation spaces were the most used in video prediction, other high-level spaces such as human pose and keypoints represent a promising avenue.

尽管语义和实例分割空间在视频预测中使用最多，但其他高级空间(例如人体姿势和关键点)代表了一个有前途的途径。

Human Pose: As the human pose is a low-dimensional and interpretable structure, it represents a cheap supervisory signal for predictive models. This fostered pose-guided prediction methods, where future frame regression in the pixel space is conditioned by intermediate prediction of human poses. However, these methods are limited to videos with human presence. As this review focuses on video prediction, we will briefly review some of the most relevant methods predicting human poses as an intermediate representation.

人体姿势：由于人体姿势是一种低维且可解释的结构，因此它代表了预测模型的廉价监督信号。 这促进了姿势引导的预测方法，其中像素空间中的未来帧回归以人体姿势的中间预测为条件。 但是，这些方法仅限于有人在场的视频。 由于这篇评论侧重于视频预测，我们将简要回顾一些最相关的预测人体姿势的方法作为中间表示。

From a supervised prediction of human poses, Villegas et al. [53] regress future frames through analogy making [182]. Although background is not considered in the prediction, authors compared the model against [13], [43] reporting long-term results. To make the model unsupervised on the human pose, Wichers et al. [52] adopted different training strategies: end-to-end prediction minimizing the l2 loss, and through analogy making, constraining the predicted features to be close to the outputs of the future encoder. Different from [53], in this work the predictions are made in the feature space. As a probabilistic alternative, Walker et al. [54] fused a conditioned Variational Autoencoder (cVAE)- based probabilistic pose predictor with a GAN. While the probabilistic predictor enhances the diversity in the predicted poses, the adversarial network ensures prediction realism. As this model struggles with long-term predictions, Fushishita et al. [183] addressed long-term video prediction of multiple outcomes avoiding the error accumulation and vanishing gradients by using a unidimensional CNN trained in an adversarial fashion. To enable multiple predictions, they have used additional inputs ensuring trajectory and behavior variability at a human pose level. To better preserve the visual appearance in the predictions than [53], [65], [108], Tang et al. [184] firstly predict human poses using a LSTM-based model to then synthesize pose-conditioned future frames using a combination of different networks: a global GAN modeling the time-invariant background and a coarse human pose, a local GAN refining the coarsepredicted human pose, and a 3D-AE to ensure temporal consistency across frames.

根据对人体姿势的监督预测，Villegas et al. 。 [53] 通过类比 [182] 回归未来框架。 尽管预测中未考虑背景，但作者将该模型与报告长期结果的 [13]、[43] 进行了比较。 为了使模型不受人体姿势的监督，Wichers et al. 。 [52] 采用了不同的训练策略：端到端预测最小化 l2 损失，并通过类比，将预测的特征约束为接近未来编码器的输出。 与 [53] 不同，在这项工作中，预测是在特征空间中进行的。 作为概率替代方案，Walker et al. 。 [54] 将基于条件变分自动编码器 (cVAE) 的概率姿态预测器与 GAN 融合在一起。 虽然概率预测器增强了预测姿势的多样性，但对抗网络确保了预测的真实性。 由于该模型难以进行长期预测，Fushishita et al. 。 [183] 通过使用以对抗方式训练的一维 CNN，解决了多个结果的长期视频预测，避免了错误累积和梯度消失。 为了实现多重预测，他们使用了额外的输入来确保人体姿势水平的轨迹和行为可变性。 为了比 [53]、[65]、[108] 更好地保留预测中的视觉外观，Tang et al. 。 [184] 首先使用基于 LSTM 的模型预测人体姿势，然后使用不同网络的组合合成姿势条件的未来帧：全局 GAN 建模时不变背景和粗略的人体姿势，局部 GAN 改进粗略预测的人体 姿势和 3D-AE 以确保跨帧的时间一致性。

Keypoints-based representations: The keypoint coordinate space is a meaningful, tractable and structured representation for prediction, ensuring stable learning. It enforces model’s internal representation to contain object-level information. This leads to better results on tasks requiring objectlevel understanding such as, trajectory prediction, action recognition and reward prediction. As keypoints are a natural representation of dynamic objects, Minderer et al. [85] reformulated the prediction task in the keypoint coordinate space. They proposed an AE architecture with a keypointbased representational bottleneck, consisting of a VRNN that predicts dynamics in the keypoint space. Although this model qualitatively outperforms the Stochastic Video Generation (SVG) [81], Stochastic Adversarial Video Prediction (SAVP) [108] and EPVA [52] models, the quantitative evaluation reported similar results.

基于关键点的表示：关键点坐标空间是一种有意义的、易处理的和结构化的预测表示，确保稳定的学习。 它强制模型的内部表示包含对象级信息。 这会在需要对象级理解的任务上产生更好的结果，例如轨迹预测、动作识别和奖励预测。 由于关键点是动态对象的自然表示，Minderer et al. 。 [85] 重新制定了关键点坐标空间中的预测任务。 他们提出了一种具有基于关键点的表示瓶颈的 AE 架构，由预测关键点空间中的动态的 VRNN 组成。 尽管该模型在质量上优于随机视频生成 (SVG) [81]、随机对抗视频预测 (SAVP) [108] 和 EPVA [52] 模型，但定量评估报告了相似的结果。

### 5.6 Incorporating Uncertainty 合并不确定性
Although high-level representations significantly reduce the prediction space, the underlying distribution still has multiple modes. In other words, different plausible outcomes would be equally probable for the same input sequence. Addressing multimodal distributions is not straightforward for regression and classification approaches, as they regress to the mean and aim to discretize a continuous highdimensional space, respectively. To deal with the inherent unpredictability of natural videos, some works introduced latent variables into existing deterministic models or directly relied on generative models such as GANs and VAEs.

尽管高级表示显著减少了预测空间，但底层分布仍然具有多种模式。 换句话说，对于相同的输入序列，不同的似是而非的结果同样可能出现。 解决多峰分布对于回归和分类方法来说并不简单，因为它们分别回归到均值并旨在离散化连续的高维空间。 为了应对自然视频固有的不可预测性，一些工作将潜在变量引入现有的确定性模型或直接依赖于 GAN 和 VAE 等生成模型。

Inspired by the DVF, Xue et al. [202] proposed a cVAEbased [222], [223] multi-scale model featuring a novel cross convolutional layer trained to regress the difference image or Eulerian motion [224]. Background on natural videos is not uniform, however the model implicitly assumes that the difference image would accurately capture the movement in foreground objects. Introducing latent variables into a convolutional AE, Goroshin et al. [211] proposed a probabilistic model for learning linearized feature representations to linearly extrapolate the predicted frame in a feature space. Uncertainty is introduced to the loss by using a cosine distance as an explicit curvature penalty. Authors focused on evaluating the linearization properties, yet the model was not contrasted to previous works. Extending [141], [202], Fragkiadaki et al. [96] proposed several architectural changes and training schemes to handle marginalization over stochastic variables, such as sampling from the prior and variational inference. They proposed a stochastic ED architecture that predicts future optical flow, i.e., dense pixel motion field, used to spatially transform the current frame into the next frame prediction. To introduce uncertainty in predictions, the authors proposed the k-best-sample-loss (MCbest) that draws K outcomes penalizing those similar to the ground-truth.

受 DVF 的启发，Xue et al. 。 [202] 提出了一种基于 cVAE 的 [222]、[223] 多尺度模型，该模型具有经过训练以回归差异图像或欧拉运动 [224] 的新型交叉卷积层。 自然视频的背景并不均匀，但该模型隐含地假设差异图像将准确捕捉前景物体的运动。 Goroshin et al. 将潜在变量引入卷积 AE。 [211] 提出了一种概率模型，用于学习线性化特征表示，以线性外推特征空间中的预测帧。 通过使用余弦距离作为显式曲率惩罚，将不确定性引入损失。 作者专注于评估线性化属性，但该模型并未与之前的作品进行对比。 扩展 [141]、[202]、Fragkiadaki et al. 。 [96] 提出了几种架构变化和训练方案来处理随机变量的边缘化，例如从先验和变分推理中抽样。 他们提出了一种预测未来光流的随机 ED 架构，即密集像素运动场，用于将当前帧空间转换为下一帧预测。 为了在预测中引入不确定性，作者提出了 k-best-sample-loss (MCbest)，它绘制 K 个结果来惩罚那些与基本事实相似的结果。

TABLE 2: Summary of video prediction models (c: convolutional; r: recurrent; v: variational; ms: multi-scale; st: stacked; bi: bidirectional; P: Percepts; M: Motion; PL: Perceptual Loss; AL: Adversarial Loss; S/R: using Synthetic/Real datasets; SS: Semantic Segmentation; D: Depth; S: State; Po: Pose; O: Odometry; IS: Instance Segmentation; ms: multi-step prediction; pred-fr: number of predicted frames, ? 1-5 frames, ? ? 5-10 frames, ? ? ? 10-100 frames, ? ? ? ? over 100 frames; ood: indicates if model was tested on out-of-domain tasks). 
表 2：视频预测模型总结(c：卷积; r：循环; v：变分; ms：多尺度; st：堆叠; bi：双向; P：感知; M：运动; PL：感知损失; AL： Adversarial Loss; S/R：使用合成/真实数据集; SS：语义分割; D：深度; S：状态; Po：姿态; O：里程计; IS：实例分割; ms：多步预测; pred-fr： 预测帧数，? 1-5 帧，? ? 5-10 帧，? ? ? 10-100 帧，? ? ? ? 超过 100 帧; ood：表示模型是否在域外任务上进行了测试)。

Incorporating latent variables into the deterministic CDNA architecture for the first time, Babaeizadeh et al. proposed the Stochastic Variational Video Prediction (SV2P) [38] model handling natural videos. Their timeinvariant posterior distribution is approximated from the entire input video sequence. Authors demonstrated that, by explicitly modeling uncertainty with latent variables, the deterministic CDNA model is outperformed. By combining a standard deterministic architecture (LSTM-ED) with stochastic latent variables, Denton et al. proposed the SVG network [81]. Different from SV2P, the prior is sampled from a time-varying posterior distribution, i.e. it is a learned-prior instead of fixed-prior sampled from the same distribution. Most of the VAEs use a fixed Gaussian as a prior, sampling randomly at each time step. Exploiting the temporal dependencies, a learned-prior predicts high variance in uncertain situations, and a low variance when a deterministic prediction suffices. The SVG model is easier to train and reported sharper predictions in contrast to [38]. Built upon SVG, Villegas et al. [225] implemented a baseline to perform an in-depth empirical study on the importance of the inductive bias, stochasticity, and model’s capacity in the video prediction task. Different from previous approaches, Henaff et al. proposed the Error Encoding Network (EEN) [99] that incorporates uncertainty by feeding back the residual error —the difference between the ground truth and the deterministic prediction— encoded as a low-dimensional latent variable. In this way, the model implicitly separates the input video into deterministic and stochastic components.

Babaeizadeh et al. 首次将潜在变量纳入确定性 CDNA 架构。 提出了处理自然视频的随机变分视频预测(SV2P)[38]模型。 它们的时不变后验分布是从整个输入视频序列中近似得出的。 作者证明，通过使用潜在变量对不确定性进行显式建模，确定性 CDNA 模型的性能优于确定性 CDNA 模型。 通过将标准确定性架构 (LSTM-ED) 与随机潜在变量相结合，Denton et al. 。 提出了 SVG 网络 [81]。 与 SV2P 不同的是，先验是从时变后验分布中采样的，即它是从同一分布中学习到的先验而不是固定先验。 大多数 VAE 使用固定的高斯作为先验，在每个时间步随机采样。 利用时间依赖性，学习先验在不确定情况下预测高方差，在确定性预测足够时预测低方差。 与 [38] 相比，SVG 模型更容易训练并报告更准确的预测。 Villegas et al. 建立在 SVG 之上。 [225] 实施了一个基线，以对视频预测任务中归纳偏差、随机性和模型能力的重要性进行深入的实证研究。 与以前的方法不同，Henaff et al. 。 提出了误差编码网络 (EEN) [99]，它通过反馈残余误差(基本事实和确定性预测之间的差异)来结合不确定性，编码为低维潜在变量。 通过这种方式，该模型隐式地将输入视频分为确定性和随机性成分。

On the one hand, latent variable-based approaches cover the space of possible outcomes, yet predictions lack of realism. On the other hand, GANs struggle with uncertainty, but predictions are more realistic. Searching for a tradeoff between VAEs and GANs, Lee et al. [108] proposed the SAVP model, being the first to combine latent variable models with GANs to improve variability in video predictions, while maintaining realism. Under the assumption that blurry predictions of VAEs are a sign of underfitting, Castrejon et al. extended the VRNNs to leverage a hierarchy of latent variables and better approximate data likelihood [97]. Although the backpropagation through a hierarchy of conditioned latents is not straightforward, several techniques alleviated this issue such as, KL beta warm-up, dense connectivity pattern between inputs and latents, Ladder Variational Autoencoders (LVAEs) [226]. As most of the probabilistic approaches fail in approximating the true distribution of future frames, Pottorff et al. [227] reformulated the video prediction task without making any assumption about the data distribution. They proposed the Invertible Linear Embedding (ILE) enabling exact maximum likelihood learning of video sequences, by combining an invertible neural network [228], also known as reversible flows, and a linear time-invariant dynamic system. The ILE handles nonlinear motion in the pixel space and scales better to longer-term predictions compared to adversarial models [43].

一方面，基于潜在变量的方法涵盖了可能结果的空间，但预测缺乏现实性。 另一方面，GAN 与不确定性作斗争，但预测更为现实。 为了寻找 VAE 和 GAN 之间的权衡，Lee et al. 。 [108] 提出了 SAVP 模型，它是第一个将潜在变量模型与 GAN 结合起来以提高视频预测的可变性，同时保持真实性的模型。 假设 VAE 的模糊预测是欠拟合的标志，Castrejon et al. 。 扩展了 VRNN 以利用潜在变量的层次结构和更好的近似数据似然 [97]。 尽管通过条件潜变量的层次结构进行反向传播并不简单，但有几种技术缓解了这个问题，例如 KL beta 预热、输入和潜变量之间的密集连接模式、阶梯变分自编码器 (LVAE) [226]。 由于大多数概率方法无法逼近未来帧的真实分布，Pottorff et al. 。 [227] 在不对数据分布做出任何假设的情况下重新制定了视频预测任务。 他们提出了可逆线性嵌入 (ILE)，通过结合可逆神经网络 [228](也称为可逆流)和线性时不变动态系统，实现视频序列的精确最大似然学习。 与对抗模型 [43] 相比，ILE 处理像素空间中的非线性运动并更好地扩展到长期预测。

While previous variational approaches [81], [108] focused on predicting a single frame of low resolution in restricted, predictable or simulated datasets, Hu et al. [15] jointly predict full-frame ego-motion, static scene, and object dynamics on complex real-world urban driving. Featuring a novel spatio-temporal module, their five-component architecture learns rich representations that incorporate both local and global spatio-temporal context. Authors validated the model on predicting semantic segmentation, depth and optical flow, two seconds in the future outperforming existing spatio-temporal architectures. However, no performance comparison with [81], [108] has been carried out. 

虽然以前的变分方法 [81]、[108] 侧重于预测受限、可预测或模拟数据集中的低分辨率单帧，但 Hu et al. 。 [15] 在复杂的现实世界城市驾驶中联合预测全帧自我运动、静态场景和物体动力学。 他们的五组件架构具有新颖的时空模块，可学习结合本地和全球时空上下文的丰富表示。 作者在预测语义分割、深度和光流方面验证了该模型，未来两秒优于现有时空架构。 然而，没有与[81]、[108]进行性能比较。

## 6 PERFORMANCE EVALUATION
This section presents the results of the previously analyzed video prediction models on the most popular datasets on the basis of the metrics described below.

本节基于以下描述的指标，介绍了先前在最流行的数据集上分析的视频预测模型的结果。

### 6.1 Metrics and Evaluation Protocols
For a fair evaluation of video prediction systems, multiple aspects in the prediction have to be addressed such as whether the predicted sequences look realistic, are plausible and cover all possible outcomes. To the best of our knowledge, there are no evaluation protocols and metrics that evaluate the predictions by fulfilling simultaneously all these aspects.

为了对视频预测系统进行公平评估，必须解决预测中的多个方面，例如预测序列是否逼真、是否合理以及是否涵盖所有可能的结果。 据我们所知，没有评估协议和指标可以通过同时满足所有这些方面来评估预测。

The most widely used evaluation protocols for video prediction rely on image similarity-based metrics such as, Mean-Squared Error (MSE), Structural Similarity Index Measure (SSIM) [229], and Peak Signal to Noise Ratio (PSNR). However, evaluating a prediction according to the mismatch between its visual appearance and the ground truth is not always reliable. In practice, these metrics penalize all predictions that deviate from the ground truth. In other words, they prefer blurry predictions nearly accommodating the exact ground truth than sharper and plausible but imperfect generations [97], [108], [230]. Pixel-wise metrics do not always reflect how accurate a model captured video scene dynamics and their temporal variability. In addition, the success of a metric is influenced by the loss function used to train the model. For instance, the models trained with MSE loss function would obviously perform well on MSE metric, but also on PSNR metric as it is based on MSE. Suffering from similar problems, SSIM measures the similarity between two images, from −1 (very dissimilar) to +1 (the same image). As a difference, it measures similarities on image patches instead of performing pixelwise comparison. These metrics are easily fooled by learning to match the background in predictions. To address this issue, Mathieu et al. [43] evaluated the predictions only on the dynamic parts of the sequence, avoiding background influence.

最广泛使用的视频预测评估协议依赖于基于图像相似性的指标，例如均方误差 (MSE)、结构相似性指数度量 (SSIM) [229] 和峰值信噪比 (PSNR)。 然而，根据视觉外观与真实情况之间的不匹配来评估预测并不总是可靠的。 在实践中，这些指标会惩罚所有偏离真实情况的预测。 换句话说，他们更喜欢模糊的预测来接近准确的基本事实，而不是更清晰、合理但不完美的生成 [97]、[108]、[230]。 逐像素指标并不总是反映模型捕获视频场景动态及其时间变化的准确性。 此外，指标的成功受用于训练模型的损失函数的影响。 例如，用 MSE 损失函数训练的模型显然在 MSE 指标上表现良好，但在基于 MSE 的 PSNR 指标上也表现良好。 由于存在类似问题，SSIM 衡量两幅图像之间的相似度，从 -1(非常不相似)到 +1(相同图像)。 不同之处在于，它测量图像块的相似性，而不是执行像素比较。 通过学习匹配预测中的背景，这些指标很容易被愚弄。 为了解决这个问题，Mathieu et al. 。 [43] 仅对序列的动态部分评估预测，避免背景影响。

As the pixel space is multimodal and highlydimensional, it is challenging to evaluate how accurately a prediction sequence covers the full distribution of possible outcomes. Addressing this issue, some probabilistic approaches [81], [97], [108] adopted a different evaluation protocol to assess prediction coverage. Basically, they sample multiple random predictions and then they search for the best match with the ground truth sequence. Finally, they report the best match using common metrics. This represents the most common evaluation protocol for probabilistic video prediction. Other methods [97], [150], [151] also reported results using: LPIPS [230] as a perceptual metric comparing CNN features, or Frchet Video Distance (FVD) [231] to measure sample realism by comparing underlying distributions of predictions and ground truth. Moreover, Lee et al. [108] used the VGG Cosine Similarity metric that performs cosine similarity to the features extracted with the VGGnet [146] from the predictions.

由于像素空间是多模态和高维的，因此评估预测序列覆盖可能结果的完整分布的准确度具有挑战性。 为了解决这个问题，一些概率方法 [81]、[97]、[108] 采用了不同的评估协议来评估预测覆盖率。 基本上，他们对多个随机预测进行采样，然后搜索与地面实况序列的最佳匹配。 最后，他们使用通用指标报告最佳匹配。 这代表了最常见的概率视频预测评估协议。 其他方法 [97]、[150]、[151] 也报告了使用以下方法的结果：LPIPS [230] 作为比较 CNN 特征的感知指标，或 Frchet Video Distance (FVD) [231] 通过比较预测的潜在分布来衡量样本真实性 和真实情况。 此外，李et al. 。 [108] 使用 VGG 余弦相似度度量，该度量与 VGGnet [146] 从预测中提取的特征执行余弦相似度。

TABLE 3: Results on M-MNIST (Moving MNIST). Predicting the next y frames from x context frames (x → y). † results reported by Oliu et al. [153], ‡ results reported by Wang et al. [66], ∗ results reported by Wang et al. [197], / results reported by Wang et al. [235]. MSE represents per-pixel average MSE (10−3). 
表 3：M-MNIST(移动 MNIST)的结果。 从 x 上下文帧 (x → y) 预测下一个 y 帧。 † Oliu et al. 报告的结果。 [153]，‡ Wang et al. 报告的结果。 [66]， * Wang et al. 报告的结果。 [197]，/ Wang et al. 报告的结果。 [235]。 MSE 表示每像素平均 MSE (10−3)。

Some other alternative metrics include the inception score [232] introduced to deal with GANs mode collapse problem by measuring the diversity of generated samples; perceptual similarity metrics, such as DeePSiM [44]; measuring sharpness based on difference of gradients [43]; Parzen window [233], yet deficient for high-dimensional images; and the Laplacian of Gaussians (LoG) [60], [234] used in [101]. In the semantic segmentation space, authors used the popular Intersection over Union (IoU) metric. Inception score was also widely used to report results on different methods [54], [65], [67], [79]. Differently, on the basis of the EPVA model [52] a quantitative evaluation was performed, based on the confidence of an external method trained to identify whether the generated video contains a recognizable person. While some authors [10], [43], [56] evaluated the performance only on the dynamic parts of the image, other directly opted for visual human evaluation through Amazon Mechanical Turk (AMT) workers, without a direct quantitative evaluation.

其他一些替代指标包括引入初始分数 [232]，通过测量生成样本的多样性来处理 GAN 模式崩溃问题;  感知相似性度量，例如 DeePSiM [44];  基于梯度差异测量锐度[43];  Parzen 窗口 [233]，但对于高维图像存在缺陷;  以及[101]中使用的高斯拉普拉斯算子(LoG)[60]、[234]。 在语义分割空间中，作者使用了流行的 Intersection over Union (IoU) 度量。 Inception 分数也被广泛用于报告不同方法的结果 [54]、[65]、[67]、[79]。 不同的是，在 EPVA 模型 [52] 的基础上，基于外部方法的置信度进行了定量评估，该方法经过训练可以识别生成的视频是否包含可识别的人。 虽然一些作者 [10]、[43]、[56] 仅评估图像的动态部分的性能，但其他作者直接选择通过 Amazon Mechanical Turk (AMT) 工作人员进行视觉人类评估，而没有直接的定量评估。

### 6.2 Results
In this section we report the quantitative results of the most relevant methods reviewed in the previous sections. To achieve a wide comparison, we limited the quantitative results to the most common metrics and datasets. We have distributed the results in different tables, given the large variation in the evaluation protocols of the video prediction models.

在本节中，我们报告了前几节中综述的最相关方法的定量结果。 为了实现广泛的比较，我们将定量结果限制在最常见的指标和数据集。 鉴于视频预测模型的评估协议存在很大差异，我们将结果分布在不同的表格中。

TABLE 4: Results on KTH dataset. Predicting the next y frames from x context frames (x → y). † results reported by Oliu et al. [153], ‡ results reported by Wang et al. [66], ∗ results reported by Zhang et al. [70], / results reported by Jin et al. [150]. Per-pixel average MSE (10−3). Best results are represented in bold.
表 4：KTH 数据集的结果。 从 x 上下文帧 (x → y) 预测下一个 y 帧。 † Oliu et al. 报告的结果。 [153]，‡ Wang et al. 报告的结果。 [66]，*张et al. 报告的结果。 [70]，/ Jin et al. 报告的结果。 [150]。 每像素平均 MSE (10−3)。 最佳结果以粗体表示。

Many authors evaluated their methods on the Moving MNIST synthetic environment. Although it represents a restricted and quasi-deterministic scenario, long-term predictions are still challenging. The black and homogeneous background induce methods to accurately extrapolate black frames and vanish the predicted digits in the long-term horizon. Under this configuration, the CrevNet model demonstrated a leap over the previous state of the art. As the second best, the E3d-LSTM network reported stable errors in both short-term and longer-term predictions showing the advantages of their memory attention mechanism. It also reported the second best results on the KTH dataset, after [150] which achieved the best overall performance and demonstrated quality predictions on natural videos.

许多作者在 Moving MNIST 合成环境中评估了他们的方法。 虽然它代表了一个受限和准确定性的场景，但长期预测仍然具有挑战性。 黑色和均匀的背景导致了准确推断黑色帧并在长期范围内消除预测数字的方法。 在这种配置下，CrevNet 模型展示了对先前技术水平的飞跃。 作为第二好的，E3d-LSTM 网络在短期和长期预测中报告了稳定的错误，显示了其记忆注意机制的优势。 它还报告了 KTH 数据集上的第二好的结果，仅次于 [150]，后者实现了最佳的整体性能并证明了对自然视频的质量预测。

Performing short-term predictions in the KTH dataset, the Recurrent Ladder Network (RLN) outperformed MCnet and fRNN by a slight margin. The RLN architecture draws similarities with fRNN, except that the former uses bridge connections and the latter, state sharing that improves memory consumption. On the Moving MNIST and UCF101 datasets, fRNN outperformed RLN. Other interesting methods to highlight are PredRNN and PredRNN++, both providing close results to E3d-LSTM. State-of-the-art results using different metrics were reported on Caltech Pedestrian by Kwon et al. [101], CrevNet [154], and Jin et al. [150]. The former, by taking advantage of its retrospective prediction scheme, was also the overall winner on the UCF- 101 dataset meanwhile the latter outperformed previous methods on the BAIR Push dataset.

在 KTH 数据集中进行短期预测时，递归阶梯网络 (RLN) 的表现略优于 MCnet 和 fRNN。 RLN 架构与 fRNN 有相似之处，不同之处在于前者使用桥接，而后者使用状态共享来改善内存消耗。 在 Moving MNIST 和 UCF101 数据集上，fRNN 优于 RLN。 其他值得强调的有趣方法是 PredRNN 和 PredRNN++，它们都提供与 E3d-LSTM 接近的结果。 Kwon et al. 在 Caltech Pedestrian 上报告了使用不同指标的最先进结果。 [101]、CrevNet [154] 和 Jin et al. 。 [150]。 前者利用其回顾性预测方案，也是 UCF-101 数据集的总冠军，而后者在 BAIR Push 数据集上的表现优于之前的方法。

TABLE 5: Results on Caltech Pedestrian. Predicting the next y frames from x context frames (x → y). † reported by Kwon et al. [101], ‡ reported by Reda et al. [155], ∗ reported by Gao et al. [167], / reported by Jin et al. [150]. Per-pixel average MSE (10−3). Best results are represented in bold.
表 5：Caltech Pedestrian 的结果。 从 x 上下文帧 (x → y) 预测下一个 y 帧。 † 由 Kwon et al. 报道。 [101]，‡ 由 Reda et al. 报道。 [155]，*由 Gao et al. 报道。 [167]，/由 Jin et al. 报道。 [150]。 每像素平均 MSE (10−3)。 最佳结果以粗体表示。

On the one hand, some approaches have been evaluated on other datasets: SDC-Net [155] outperformed [43], [65] on YouTube8M, TrIVD-GAN-FP outperformed [163], [240] on Kinetics-600 test set [201], E3d-LSTM compared their method with [95], [153], [197], [235] on the TaxiBJ dataset [190], and CrevNet [154] on Traffic4cast [198]. On the other hand, some explored out-of-domain tasks [13], [66], [102], [154], [162] (see ood column in Table 2).

#### 6.2.1 Results on Probabilistic Approaches 概率方法的结果
Video prediction probabilistic methods have been mainly evaluated on the Stochastic Moving MNIST, Bair Push and Cityscapes datasets. Different from the original Moving MNIST dataset, the stochastic version includes uncertain digit trajectories, i.e. the digits bounce off the border with a random new direction. On this dataset, both versions of Castrejon et al. models (1L, without a hierarchy of latents, and 3L with a 3-level hierarchy of latents) outperform SVG by a large margin. On the Bair Push dataset, SAVP reported sharper and more realistic-looking predictions than SVG which suffer of blurriness. However, both models were outperformed by [97] as well on the Cityscapes dataset. The model based on a 3-level hierarchy of latents [97] outperform previous works on all three datasets, showing the advantages of the extra expressiveness of this model.

视频预测概率方法主要在 Stochastic Moving MNIST、Bair Push 和 Cityscapes 数据集上进行了评估。 与原始的 Moving MNIST 数据集不同，随机版本包括不确定的数字轨迹，即数字以随机的新方向弹离边界。 在这个数据集上，Castrejon et al. 的两个版本。 模型(1L，没有潜在层级，3L 有 3 层潜在层级)大大优于 SVG。 在 Bair Push 数据集上，SAVP 报告的预测比 SVG 更清晰、更逼真，后者存在模糊问题。 然而，这两个模型在 Cityscapes 数据集上的表现都优于 [97]。 基于 3 级潜在层次结构的模型 [97] 在所有三个数据集上的表现都优于以前的工作，显示了该模型额外表现力的优势。

TABLE 7: Results on SM-MNIST (Stochastic Moving MNIST), BAIR Push and Cityscapes datasets. † results reported by Castrejon et al. [97]. ‡ results reported by Jin et al. [150].
表 7：SM-MNIST(随机移动 MNIST)、BAIR Push 和 Cityscapes 数据集的结果。 † Castrejon et al. 报告的结果。 [97]。 ‡ Jin et al. 报告的结果。 [150]。

TABLE 8: Results on Cityscapes dataset. Predicting the next y time-steps of semantic segmented frames from 4 context frames (4 → y). ‡ IoU results on eight moving objects classes. † results reported by Chiu et al. [174]
表 8：Cityscapes 数据集的结果。 从 4 个上下文帧 (4 → y) 预测语义分段帧的下一个 y 时间步长。 ‡ 八个移动物体类的 IoU 结果。 † Chiu et al. 报告的结果。 [174]

#### 6.2.2 Results on the High-level Prediction Space 高级预测空间的结果
Most of the methods have chosen the semantic segmentation space to make predictions. Although they relied on different datasets for training, performance results were mostly reported on the Cityscapes dataset using the IoU metric. Authors explored short-term (next-frame prediction), midterm (+3 time steps in the future) and long-term (up to +10 time step in the future) predictions. On the semantic segmentation prediction space, Bayes-WD-SL [9], the model proposed by Terwilliger et al. [10], and Jin et al. [51] reported the best results. Among these methods, it is noteworthy 20 that Bayes-WD-SL was the only one to explore prediction diversity on the basis of a Bayesian formulation.

大多数方法都选择了语义分割空间来进行预测。 尽管他们依赖于不同的数据集进行训练，但性能结果主要是使用 IoU 指标在 Cityscapes 数据集上报告的。 作者探索了短期(下一帧预测)、中期(未来 +3 个时间步)和长期(未来最多 +10 个时间步)预测。 在语义分割预测空间上，Bayes-WD-SL [9]，由 Terwilliger et al. 提出的模型。 [10]，和金et al. 。 [51] 报告了最好的结果。 在这些方法中，值得注意的是 20 Bayes-WD-SL 是唯一一种基于贝叶斯公式探索预测多样性的方法。

In the instance segmentation space, the F2F pioneering method [8] was outperformed by Sun et al. [84] on short and mid-term predictions using the AP50 and AP evaluation metrics. On the other hand, in the keypoint coordinate space, the seminal model of Minderer et al. [85] qualitatively outperforms SVG [81], SAVP [108] and EPVA [52], yet pixelwise metrics reported similar results. In the human pose space, Tang et al. [184] by regressing future frames from human pose predictions outperformed SAVP [108], MCnet [65] and [53] on the basis of the PSNR and SSIM metrics on the Penn Action and J-HMDB [114] datasets. 

在实例分割空间中，F2F 开创性方法 [8] 的表现优于 Sun et al. 。 [84] 使用 AP50 和 AP 评估指标进行短期和中期预测。 另一方面，在关键点坐标空间中，Minderer et al. 的开创性模型。 [85] 在质量上优于 SVG [81]、SAVP [108] 和 EPVA [52]，但像素指标报告了相似的结果。 在人体姿势空间中，Tang et al. 。 [184] 基于 Penn Action 和 J-HMDB [114] 数据集上的 PSNR 和 SSIM 指标，通过回归人体姿态预测的未来帧优于 SAVP [108]、MCnet [65] 和 [53]。

## 7 DISCUSSION
The video prediction literature ranges from a direct synthesis of future pixel intensities, to complex probabilistic models addressing prediction uncertainty. The range between these approaches consists of methods that try to factorize or narrow the prediction space. Simplifying the prediction task has been a natural evolution of video prediction models, influenced by several open research challenges discussed below. Due to the curse of dimensionality and the inherent pixel variability, developing a robust prediction based on raw pixel intensities is overly-complicated. This often leads to the regression-to-the-mean problem, visually represented as blurriness. Making parametric models larger would improve the quality of predictions, yet this is currently incompatible with high-resolution predictions due to memory constraints. Transformation-based approaches propagate pixels from previous frames based on estimated flow maps. In this case, prediction quality is directly in- fluenced by the accuracy of the estimated flow. Similarly, the prediction in a high-level space is mostly conditioned by the quality of some extra supervisory signals such as semantic maps and human poses, to name a few. Erroneous supervision signals would harm prediction quality.

视频预测文献的范围从未来像素强度的直接合成到解决预测不确定性的复杂概率模型。 这些方法之间的范围包括尝试因式分解或缩小预测空间的方法。 简化预测任务一直是视频预测模型的自然演变，受到下面讨论的几个开放研究挑战的影响。 由于维数灾难和固有的像素可变性，基于原始像素强度开发稳健的预测过于复杂。 这通常会导致均值回归问题，在视觉上表现为模糊。 使参数化模型更大会提高预测质量，但由于内存限制，目前这与高分辨率预测不兼容。 基于变换的方法根据估计的流图传播来自先前帧的像素。 在这种情况下，预测质量直接受到估计流量准确性的影响。 同样，高级空间中的预测主要取决于一些额外的监督信号的质量，例如语义图和人体姿势等。 错误的监督信号会损害预测质量。

Analyzing the impact of the inductive bias on the performance of a video prediction model, Villegas et al. [225] demonstrated the maximization of the SVG model [81] performance with minimal inductive bias (e.g. segmentation or instance maps, optical flow, adversarial losses, etc.) by increasing progressively the scale of computation. A common assumption when addressing the prediction task in a highlevel feature space, is the direct improvement of long-term predictions as a result of simplifying the prediction space. Even if the complexity of the prediction space is reduced, it is still multimodal when dealing with natural videos. For instance, when it comes to long-term predictions in the semantic segmentation space, most of the models reported predictions only up to ten time steps into the future. This directly suggests that the choice of the prediction space is still an unsolved problem. Finding a trade-off between the complexity of the prediction space and the output quality is challenging. An overly-simplified representation could limit the prediction on complex data such as natural videos. Although abstract predictions suffice for many of the decisionmaking systems based on visual reasoning, prediction in pixel space is still being addressed.

Villegas et al. 分析了归纳偏差对视频预测模型性能的影响。 [225] 通过逐步增加计算规模，以最小的归纳偏差(例如分割或实例图、光流、对抗性损失等)证明了 SVG 模型 [81] 性能的最大化。 在高级特征空间中处理预测任务时，一个常见的假设是通过简化预测空间直接改进长期预测。 即使降低了预测空间的复杂度，在处理自然视频时仍然是多模态的。 例如，当谈到语义分割空间中的长期预测时，大多数模型只报告了未来十个时间步长的预测。 这直接表明预测空间的选择仍然是一个未解决的问题。 在预测空间的复杂性和输出质量之间找到权衡是具有挑战性的。 过度简化的表示可能会限制对自然视频等复杂数据的预测。 尽管抽象预测足以满足许多基于视觉推理的决策系统，但像素空间中的预测仍在研究中。

From the analysis performed in this review and in line with the conclusions extracted from [225] we state that: (1) including recurrent connections and stochasticity in a video prediction model generally lead to improved performance; (2) increasing model capacity while maintaining a low inductive bias also improves prediction performance; (3) multi-step predictions conditioned by previously generated outputs are prone to accumulate errors, diverging from the ground truth when addressing long-term horizons; (4) authors predicted further in the future without relying on high-level feature spaces; (5) combining pixelwise losses with adversarial training somewhat mitigates the regression-to-the-mean issue.

根据本次综述中进行的分析以及从 [225] 中提取的结论，我们指出：(1)在视频预测模型中包括循环连接和随机性通常会提高性能;  (2) 在保持低归纳偏差的同时增加模型容量也提高了预测性能;  (3) 以先前生成的输出为条件的多步预测容易累积错误，在处理长期视野时偏离基本事实;  (4) 作者在不依赖高级特征空间的情况下对未来进行了更进一步的预测;  (5) 将逐像素损失与对抗训练相结合，在一定程度上缓解了均值回归问题。

### 7.1 Research Challenges
Despite the wealth of currently existing video prediction approaches and the significant progress made in this field, there is still room to improve state-of-the-art algorithms. To foster progress, open research challenges must be clearly identified and disentangled. So far in this review, we have already discussed about: (1) the importance of spatiotemporal correlations as a self-supervisory signal for predictive models; (2) how to deal with future uncertainty and model the underlying multimodal distributions of natural videos; (3) the over-complicated task of learning meaningful representations and deal with the curse of dimensionality; (4) pixel-wise loss functions and blurry results when dealing with equally probable outcomes, i.e. probabilistic environments. These issues define the open research challenges in video prediction.

尽管目前存在丰富的视频预测方法并且在该领域取得了重大进展，但仍有改进最先进算法的空间。 为了促进进步，必须清楚地识别和理清开放的研究挑战。 到目前为止，在这篇评论中，我们已经讨论了：(1)时空相关性作为预测模型的自我监督信号的重要性;  (2) 如何处理未来的不确定性以及对自然视频的潜在多模态分布进行建模;  (3) 学习有意义的表示和处理维数灾难的任务过于复杂;  (4) 处理等概率结果时的逐像素损失函数和模糊结果，即概率环境。 这些问题定义了视频预测中的开放研究挑战。

Currently existing methods are limited to short-term horizons. While frames in the immediate future are extrapolated with high accuracy, in the long term horizon the prediction problem becomes multimodal by nature. Initial solutions consisted on conditioning the prediction on previously predicted frames. However, these autoregressive models tend to accumulate prediction errors that progressively diverge the generated prediction from the expected outcome. On the other hand, due to memory issues, there is a lack of resolution in predictions. Authors tried to address this issue by composing the full-resolution image from small predicted patches. However, as the results are not convincing because of the annoying tilling effect, most of the available models are still limited to low-resolution predictions. In addition to the lack of resolution and longterm predictions, models are still prone to the regress-to-themean problem that consists on averaging the output frame to accommodate multiple equally probable outcomes. This is directly related to the pixel-wise loss functions, that focus the learning process on the visual appearance. The choice of the loss function is an open research problem with a direct influence on the prediction quality. Finally, the lack of reliable and fair evaluation models makes the qualitative evaluation of video prediction challenging and represents another potential open problem.

目前现有的方法仅限于短期视野。 虽然在不久的将来以高精度推断出帧，但在长期范围内，预测问题本质上变成了多模态。 最初的解决方案包括在先前预测的帧上调节预测。 然而，这些自回归模型往往会累积预测误差，从而使生成的预测逐渐偏离预期结果。 另一方面，由于内存问题，预测缺乏分辨率。 作者试图通过从预测的小块中合成全分辨率图像来解决这个问题。 然而，由于烦人的耕作效应导致结果并不令人信服，大多数可用模型仍然限于低分辨率预测。 除了缺乏分辨率和长期预测之外，模型仍然容易出现均值回归问题，该问题包括对输出帧进行平均以适应多个同样可能的结果。 这与像素损失函数直接相关，它将学习过程集中在视觉外观上。 损失函数的选择是一个开放的研究问题，直接影响预测质量。 最后，缺乏可靠和公平的评估模型使得视频预测的定性评估具有挑战性，代表了另一个潜在的开放性问题。

### 7.2 Future Directions
Based on the reviewed research identifying the state-ofthe-art video prediction methods, we present some future promising research directions.

基于确定最先进的视频预测方法的综述研究，我们提出了一些未来有前途的研究方向。

Consider alternative loss functions: Pixel-wise loss functions are widely used in the video prediction task, causing blurry predictions when dealing with uncontrolled environments or long-term horizon. In this regard, great efforts have been made in the literature for identifying more suitable loss functions for the prediction task. However, despite the existing wide spectrum of loss functions, most models still blindly rely on deterministic loss functions.

考虑替代损失函数：像素级损失函数广泛用于视频预测任务，在处理不受控制的环境或长期视野时会导致预测模糊。 在这方面，文献中已经做出了很大的努力来为预测任务确定更合适的损失函数。 然而，尽管存在广泛的损失函数，但大多数模型仍然盲目地依赖确定性损失函数。

Alternatives to RNNs: Currently, RNNs are still widely used in this field to model temporal dependencies, and achieved state-of-the-art results on different benchmarks [66], [153], [197], [235]. Nevertheless, some methods also relied on 3D convolutions to further enhance video prediction [66], [174] representing a promising avenue.

RNN 的替代方案：目前，RNN 仍在该领域广泛用于对时间依赖性进行建模，并在不同的基准 [66]、[153]、[197]、[235] 上取得了最先进的结果。 尽管如此，一些方法还依赖于 3D 卷积来进一步增强视频预测 [66]、[174]，这是一条很有前途的途径。

Use synthetically generated videos: Simplifying the prediction is a current trend in the video prediction literature. A vast amount of video prediction models explored higherlevel features spaces to reformulate the prediction task into a more tractable problem. However, this mostly conditions the prediction to the accuracy of an external source of supervision such as optical flow, human pose, pre-activations (percepts) extracted from supervised networks, and more. However, this issue could be alleviated by taking advantage of existing fully-annotated and photorealistic synthetic datasets or by using data generation tools. Video prediction in photorealistic synthetic scenarios has not been explored in the literature.

使用综合生成的视频：简化预测是视频预测文献中的当前趋势。 大量视频预测模型探索了更高层次的特征空间，将预测任务重新表述为更易于处理的问题。 然而，这主要限制了对外部监督来源的准确性的预测，例如光流、人体姿势、从监督网络中提取的预激活(感知)等。 然而，这个问题可以通过利用现有的完全注释和逼真的合成数据集或使用数据生成工具来缓解。 文献中尚未探索逼真的合成场景中的视频预测。

Evaluation metrics: Since the most widely used evaluation protocols for video prediction rely on image similaritybased metrics, the need for fairer evaluation metrics is imminent. A fair metric should not penalize predictions that deviate from the ground truth at the pixel level, if their content represents a plausible future prediction in a higher level, i.e., the dynamics of the scene correspond to the reality of the labels. In this regard, some methods evaluate the similarity between distributions or at a higher-level. However, there is still room for improvement in the evaluation protocols for video prediction and generation [241]. 

评估指标：由于最广泛使用的视频预测评估协议依赖于基于图像相似性的指标，因此迫切需要更公平的评估指标。 一个公平的指标不应该惩罚在像素级别偏离真实情况的预测，如果它们的内容代表了更高级别的合理未来预测，即场景的动态对应于标签的现实。 在这方面，一些方法评估分布之间或更高级别的相似性。 然而，视频预测和生成的评估协议仍有改进的空间[241]。

## 8 CONCLUSION
In this review, after reformulating the predictive learning paradigm in the context of video prediction, we have closely reviewed the fundamentals on which it is based: exploiting the time dimension of videos, dealing with stochasticity, and the importance of the loss functions in the learning process. Moreover, an analysis of the backbone deep learningbased architectures for this task was performed in order to provide the reader the necessary background knowledge. The core of this study encompasses the analysis and classification of more than 50 methods and the datasets they have used. Methods were analyzed from three perspectives: method description, contribution over the previous works and performance results. They have also been classified according to a proposed taxonomy based on their main contribution. In addition, we have presented a comparative summary of the datasets and methods in tabular form so as the reader, at a glance, could identify low-level details. In the end, we have discussed the performance results on the most popular datasets and metrics to finally provide useful insight in shape of future research directions and open problems. In conclusion, video prediction is a promising avenue for the self-supervised learning of rich spatiotemporal correlations to provide prediction capabilities to existing intelligent decision-making systems. While great strides have been made, there is still room for improvement in video prediction using deep learning techniques.

在这篇综述中，在视频预测的背景下重新制定预测学习范式之后，我们仔细回顾了它所基于的基本原理：利用视频的时间维度，处理随机性，以及损失函数在学习中的重要性 过程。 此外，为了向读者提供必要的背景知识，还对该任务的基于深度学习的主干架构进行了分析。 这项研究的核心包括对 50 多种方法及其使用的数据集的分析和分类。 从三个角度分析方法：方法描述，对先前工作的贡献和性能结果。 它们还根据基于它们的主要贡献的拟议分类法进行了分类。 此外，我们还以表格形式对数据集和方法进行了比较总结，以便读者一眼就能识别出底层细节。 最后，我们讨论了最流行的数据集和指标的性能结果，最终为未来的研究方向和未解决的问题提供有用的见解。 总之，视频预测是一种很有前途的途径，可以通过丰富的时空相关性进行自我监督学习，从而为现有的智能决策系统提供预测能力。 尽管已经取得了长足的进步，但使用深度学习技术进行视频预测仍有改进的空间。

## Acknowledgments
This work has been funded by the Spanish Government PID2019-104818RB-I00 grant for the MoDeaAS project. This work has also been supported by two Spanish national grants for PhD studies, FPU17/00166, and ACIF/2018/197 respectively. Experiments were made possible by a generous hardware donation from NVIDIA.

这项工作由西班牙政府为 MoDeaAS 项目提供的 PID2019-104818RB-I00 赠款资助。 这项工作还得到了两项西班牙国家博士研究资助，分别为 FPU17/00166 和 ACIF/2018/197。 NVIDIA 慷慨捐赠的硬件使实验成为可能。

## References
1. M. H. Nguyen and F. D. la Torre, “Max-margin early event detectors,” in CVPR, 2012.
2. K. M. Kitani, B. D. Ziebart, J. A. Bagnell, and M. Hebert, “Activity Forecasting,” in ECCV, 2012.
3. C. Vondrick, H. Pirsiavash, and A. Torralba, “Anticipating Visual Representations from Unlabeled Video,” in CVPR, 2016.
4. K. Zeng, W. B. Shen, D. Huang, M. Sun, and J. C. Niebles, “Visual Forecasting by Imitating Dynamics in Natural Sequences,” in ICCV, 2017.
5. S. Shalev-Shwartz, N. Ben-Zrihem, A. Cohen, and A. Shashua, “Long-term planning by short-term prediction,” arXiv:1602.01580, 2016.
6. O. Makansi, E. Ilg, O. Cicek, and T. Brox, “Overcoming limitations of mixture density networks: A sampling and fitting framework for multimodal future prediction,” in CVPR, 2019.
7. Z. Liu, R. A. Yeh, X. Tang, Y. Liu, and A. Agarwala, “Video frame synthesis using deep voxel flow,” in ICCV, 2017.
8. P. Luc, C. Couprie, Y. LeCun, and J. Verbeek, “Predicting Future Instance Segmentation by Forecasting Convolutional Features,” in ECCV, 2018, pp. 593–608.
9. A. Bhattacharyya, M. Fritz, and B. Schiele, “Bayesian prediction of future street scenes using synthetic likelihoods,” in ICLR, 2019.
10. A. Terwilliger, G. Brazil, and X. Liu, “Recurrent flow-guided semantic forecasting,” in WACV, 2019.
11. A. Bhattacharyya, M. Fritz, and B. Schiele, “Long-Term On-Board Prediction of People in Traffic Scenes Under Uncertainty,” in CVPR, 2018.
12. W. Liu, W. Luo, D. Lian, and S. Gao, “Future frame prediction for anomaly detection - A new baseline,” in CVPR. IEEE, 2018.
13. X. Shi, Z. Chen, H. Wang, D. Yeung, W. Wong, and W. Woo, “Convolutional LSTM network: A machine learning approach for precipitation nowcasting,” in NeurIPS, 2015.
14. X. Shi, Z. Gao, L. Lausen, H. Wang, D.-Y. Yeung, W.-k. Wong, and W.-c. WOO, “Deep learning for precipitation nowcasting: A benchmark and a new model,” in NeurIPS, 2017.
15. A. Hu, F. Cotter, N. Mohan, C. Gurau, and A. Kendall, “Probabilistic future prediction for video scene understanding,” arXiv:2003.06409, 2020.
16. A. Garcia-Garcia, P. Martinez-Gonzalez, S. Oprea, J. A. CastroVargas, S. Orts-Escolano, J. Garcia-Rodriguez, and A. JoverAlvarez, “The robotrix: An extremely photorealistic and verylarge-scale indoor dataset of sequences with robot trajectories and interactions,” in IROS, 2018, pp. 6790–6797.
17. Y. Kong and Y. Fu, “Human action recognition and prediction: A survey,” arXiv:1806.11230, 2018.
18. C. Sahin, G. Garcia-Hernando, J. Sock, and T. Kim, “A review on object pose recovery: from 3d bounding box detectors to full 6d pose estimators,” arXiv:2001.10609, 2020.
19. V. Villena-Martinez, S. Oprea, M. Saval-Calvo, J. A. L´opez, A. F. Guill´o, and R. B. Fisher, “When deep learning meets data alignment: A review on deep registration networks (DRNs),” arXiv:2003.03167, 2020. 22
20. Y. LeCun, Y. Bengio, and G. E. Hinton, “Deep learning,” Nature, vol. 521, no. 7553, 2015.
21. J. Hawkins and S. Blakeslee, On Intelligence. Times Books, 2004.
22. R. P. N. Rao and D. H. Ballard, “Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects,” Nature Neuroscience, vol. 2, no. 1, 1999.
23. D. Mumford, “On the computational architecture of the neocortex,” Biological Cybernetics, vol. 66, no. 3, 1992.
24. A. Cleeremans and J. L. McClelland, “Learning the structure of event sequences.” Journal of Experimental Psychology: General, vol. 120, no. 3, 1991.
25. A. Cleeremans and J. Elman, Mechanisms of implicit learning: Connectionist models of sequence processing. MIT press, 1993.
26. R. Baker, M. Dexter, T. E. Hardwicke, A. Goldstone, and Z. Kourtzi, “Learning to predict: Exposure to temporal sequences facilitates prediction of future events,” Vision Research, vol. 99, 2014.
27. H. E. M. den Ouden, P. Kok, and F. P. de Lange, “How prediction errors shape perception, attention, and motivation,” in Front. Psychology, 2012.
28. W. R. Softky, “Unsupervised pixel-prediction,” in NeurIPS, 1995.
29. G. Deco and B. Sch¨urmann, “Predictive coding in the visual cortex by a recurrent network with gabor receptive fields,” Neural Processing Letters, vol. 14, no. 2, 2001.
30. A. Hollingworth, “Constructing visual representations of natural scenes: the roles of short- and long-term visual memory.” Journal of experimental psychology. Human perception and performance, vol. 30 3, 2004.
31. Y. Bengio, A. C. Courville, and P. Vincent, “Representation learning: A review and new perspectives,” Trans. on PAMI, vol. 35, no. 8, 2013.
32. X. Wang and A. Gupta, “Unsupervised Learning of Visual Representations Using Videos,” in ICCV, 2015.
33. P. Agrawal, J. Carreira, and J. Malik, “Learning to see by moving,” in ICCV, 2015.
34. D.-A. Huang, V. Ramanathan, D. Mahajan, L. Torresani, M. Paluri, L. Fei-Fei, and J. Carlos Niebles, “What makes a video a video: Analyzing temporal information in video understanding models and datasets,” in CVPR, June 2018.
35. L. C. Pickup, Z. Pan, D. Wei, Y. Shih, C. Zhang, A. Zisserman, B. Sch¨olkopf, and W. T. Freeman, “Seeing the arrow of time,” in CVPR, 2014.
36. D. Wei, J. J. Lim, A. Zisserman, and W. T. Freeman, “Learning and using the arrow of time,” in CVPR, 2018.
37. I. Misra, C. L. Zitnick, and M. Hebert, “Shuffle and learn: Unsupervised learning using temporal order verification,” in ECCV, 2016.
38. M. Babaeizadeh, C. Finn, D. Erhan, R. H. Campbell, and S. Levine, “Stochastic variational video prediction,” in ICLR, 2018.
39. H. Zhao, O. Gallo, I. Frosio, and J. Kautz, “Loss functions for image restoration with neural networks,” IEEE Trans. Computational Imaging, vol. 3, no. 1, 2017.
40. K. Janocha and W. M. Czarnecki, “On loss functions for deep neural networks in classification,” arXiv:1702.05659, 2017.
41. A. Kendall and R. Cipolla, “Geometric loss functions for camera pose regression with deep learning,” in CVPR, 2017.
42. J.-J. Hwang, T.-W. Ke, J. Shi, and S. X. Yu, “Adversarial structure matching for structured prediction tasks,” in CVPR, 2019.
43. M. Mathieu, C. Couprie, and Y. LeCun, “Deep multi-scale video prediction beyond mean square error,” in ICLR (Poster), 2016.
44. A. Dosovitskiy and T. Brox, “Generating images with perceptual similarity metrics based on deep networks,” in NIPS, 2016.
45. J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for realtime style transfer and super-resolution,” in ECCV, vol. 9906, 2016.
46. C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, A. P. Aitken, A. Tejani, J. Totz, Z. Wang, and W. Shi, “Photo-realistic single image super-resolution using a generative adversarial network,” in CVPR, 2017.
47. M. S. M. Sajjadi, B. Sch¨olkopf, and M. Hirsch, “Enhancenet: Single image super-resolution through automated texture synthesis,” in ICCV, 2017.
48. J. Zhu, P. Kr¨ahenb¨uhl, E. Shechtman, and A. A. Efros, “Generative visual manipulation on the natural image manifold,” in ECCV, ser. Lecture Notes in Computer Science, vol. 9909, 2016.
49. W. Lotter, G. Kreiman, and D. D. Cox, “Unsupervised learning of visual structure using predictive generative networks,” arXiv:1511.06380, 2015.
50. X. Chen, W. Wang, J. Wang, and W. Li, “Learning object-centric transformation for video prediction,” in ACM-MM, ser. MM ’17. New York, NY, USA: ACM, 2017.
51. X. Jin, X. Li, H. Xiao, X. Shen, Z. Lin, J. Yang, Y. Chen, J. Dong, L. Liu, Z. Jie, J. Feng, and S. Yan, “Video Scene Parsing with Predictive Feature Learning,” in ICCV, 2017.
52. N. Wichers, R. Villegas, D. Erhan, and H. Lee, “Hierarchical long-term video prediction without supervision,” in ICML, ser. Proceedings of Machine Learning Research, vol. 80, 2018.
53. R. Villegas, J. Yang, Y. Zou, S. Sohn, X. Lin, and H. Lee, “Learning to generate long-term future via hierarchical prediction,” in ICML, 2017.
54. J. Walker, K. Marino, A. Gupta, and M. Hebert, “The pose knows: Video forecasting by generating pose futures,” in ICCV, 2017.
55. X. Liang, L. Lee, W. Dai, and E. P. Xing, “Dual motion GAN for future-flow embedded video prediction,” in ICCV, 2017.
56. P. Luc, N. Neverova, C. Couprie, J. Verbeek, and Y. LeCun, “Predicting Deeper into the Future of Semantic Segmentation,” in ICCV, 2017.
57. Z. Hu and J. Wang, “A novel adversarial inference framework for video prediction with action control,” in ICCV Workshops, Oct 2019.
58. Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, Nov 1998.
59. V. Jain, J. F. Murray, F. Roth, S. C. Turaga, V. P. Zhigulin, K. L. Briggman, M. Helmstaedter, W. Denk, and H. S. Seung, “Supervised learning of image restoration with convolutional networks,” in ICCV, 2007.
60. E. L. Denton, S. Chintala, A. Szlam, and R. Fergus, “Deep generative image models using a laplacian pyramid of adversarial networks,” in NeurIPS, 2015.
61. F. Yu, V. Koltun, and T. A. Funkhouser, “Dilated residual networks,” in CVPR. IEEE, 2017.
62. L. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs,” TPAMI, vol. 40, no. 4, 2018.
63. W. Luo, Y. Li, R. Urtasun, and R. S. Zemel, “Understanding the Effective Receptive Field in Deep Convolutional Neural Networks,” in NeurIPS, 2016.
64. K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR, 2016.
65. R. Villegas, J. Yang, S. Hong, X. Lin, and H. Lee, “Decomposing motion and content for natural video sequence prediction,” in ICLR, 2017.
66. Y. Wang, L. Jiang, M.-H. Yang, L.-J. Li, M. Long, and L. Fei-Fei, “Eidetic 3d LSTM: A model for video prediction and beyond,” in ICLR, 2019.
67. C. Vondrick, H. Pirsiavash, and A. Torralba, “Generating Videos with Scene Dynamics,” in NeurIPS, 2016.
68. S. Tulyakov, M.-Y. Liu, X. Yang, and J. Kautz, “MoCoGAN: Decomposing motion and content for video generation,” in CVPR, June 2018.
69. S. Aigner and M. K¨orner, “Futuregan: Anticipating the future frames of video sequences using spatio-temporal 3d convolutions in progressively growing autoencoder gans,” arXiv:1810.01325, 2018.
70. J. Zhang, Y. Wang, M. Long, W. Jianmin, and P. S. Yu, “Z-order recurrent neural networks for video prediction,” in ICME, July 2019.
71. J. R. van Amersfoort, A. Kannan, M. Ranzato, A. Szlam, D. Tran, and S. Chintala, “Transformation-based models of video sequences,” arXiv:1701.08435, 2017.
72. D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by back-propagating errors,” Nature, vol. 323, no. 6088, 1986.
73. M. Ranzato, A. Szlam, J. Bruna, M. Mathieu, R. Collobert, and S. Chopra, “Video (language) modeling: a baseline for generative models of natural videos,” arXiv:1412.6604, 2014.
74. N. Srivastava, E. Mansimov, and R. Salakhutdinov, “Unsupervised Learning of Video Representations using LSTMs,” in ICML, 2015. 23
75. W. Lotter, G. Kreiman, and D. Cox, “Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning,” in ICLR (Poster), 2017.
76. W. Byeon, Q. Wang, R. K. Srivastava, and P. Koumoutsakos, “Contextvp: Fully context-aware video prediction,” in CVPR (Workshops), 2018.
77. V. Patraucean, A. Handa, and R. Cipolla, “Spatio-temporal video autoencoder with differentiable memory,” (ICLR) Workshop, 2015.
78. C. Lu, M. Hirsch, and B. Sch¨olkopf, “Flexible Spatio-Temporal Networks for Video Prediction,” in CVPR, 2017.
79. E. L. Denton and V. Birodkar, “Unsupervised learning of disentangled representations from video,” in NeurIPS, 2017.
80. J. Oh, X. Guo, H. Lee, R. L. Lewis, and S. P. Singh, “ActionConditional Video Prediction using Deep Networks in Atari Games,” in NeurIPS, 2015.
81. E. Denton and R. Fergus, “Stochastic video generation with a learned prior,” in ICML, ser. Proceedings of Machine Learning Research, J. G. Dy and A. Krause, Eds., vol. 80, 2018.
82. S. shahabeddin Nabavi, M. Rochan, and Y. Wang, “Future Semantic Segmentation with Convolutional LSTM,” in BMVC, 2018.
83. S. Vora, R. Mahjourian, S. Pirk, and A. Angelova, “Future segmentation using 3d structure,” arXiv:1811.11358, 2018.
84. J. Sun, J. Xie, J. Hu, Z. Lin, J. Lai, W. Zeng, and W. Zheng, “Predicting future instance segmentation with contextual pyramid convLSTMs,” in ACM Multimedia. ACM, 2019.
85. M. Minderer, C. Sun, R. Villegas, F. Cole, K. P. Murphy, and H. Lee, “Unsupervised learning of object structure and dynamics from videos,” in NeurIPS, 2019.
86. S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, 1997.
87. K. Cho, B. van Merrienboer, C¸ . G¨ulc¸ehre, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” EMNLP, pp. 1724–1734, 2014.
88. A. Graves, S. Fern´andez, and J. Schmidhuber, “Multidimensional recurrent neural networks,” in ICANN, vol. 4668, 2007.
89. C. Finn, I. J. Goodfellow, and S. Levine, “Unsupervised Learning for Physical Interaction through Video Prediction,” in NeurIPS, 2016.
90. E. Zhan, S. Zheng, Y. Yue, L. Sha, and P. Lucey, “Generating multi-agent trajectories using programmatic weak supervision,” in ICLR, 2019.
91. A. van den Oord, N. Kalchbrenner, and K. Kavukcuoglu, “Pixel Recurrent Neural Networks,” in ICML, 2016.
92. R. M. Neal, “Connectionist learning of belief networks,” Artif. Intell., vol. 56, no. 1, 1992.
93. Y. Bengio and S. Bengio, “Modeling high-dimensional discrete data with multi-layer neural networks,” in NeurIPS, 1999.
94. A. van den Oord, N. Kalchbrenner, L. Espeholt, K. Kavukcuoglu, O. Vinyals, and A. Graves, “Conditional image generation with pixelcnn decoders,” in NIPS, 2016.
95. N. Kalchbrenner, A. van den Oord, K. Simonyan, I. Danihelka, O. Vinyals, A. Graves, and K. Kavukcuoglu, “Video pixel networks,” in ICML, 2017, pp. 1771–1779.
96. K. Fragkiadaki, J. Huang, A. Alemi, S. Vijayanarasimhan, S. Ricco, and R. Sukthankar, “Motion prediction under multimodality with conditional stochastic networks,” arXiv:1705.02082, 2017.
97. L. Castrejon, N. Ballas, and A. Courville, “Improved conditional vrnns for video prediction,” in ICCV, 2019.
98. J. Chung, K. Kastner, L. Dinh, K. Goel, A. C. Courville, and Y. Bengio, “A recurrent latent variable model for sequential data,” in NIPS, 2015.
99. M. Henaff, J. J. Zhao, and Y. LeCun, “Prediction under uncertainty with error-encoding networks,” arXiv:1711.04994, 2017.
100. I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. WardeFarley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in NIPS, 2014, pp. 2672–2680.
101. Y.-H. Kwon and M.-G. Park, “Predicting future frames using retrospective cycle gan,” in CVPR, 2019.
102. C. Vondrick and A. Torralba, “Generating the Future with Adversarial Transformers,” in CVPR, 2017.
103. Y. Zhou and T. L. Berg, “Learning Temporal Transformations from Time-Lapse Videos,” in ECCV, 2016.
104. P. Bhattacharjee and S. Das, “Temporal coherency based criteria for predicting video frames using deep multi-stage generative adversarial networks,” in NIPS, 2017.
105. M. Saito, E. Matsumoto, and S. Saito, “Temporal generative adversarial nets with singular value clipping,” in ICCV, 2017.
106. B. Chen, W. Wang, and J. Wang, “Video imagination from a single image with transformation generation,” in ACM Multimedia, 2017.
107. M. Mirza and S. Osindero, “Conditional generative adversarial nets,” arXiv:1411.1784, 2014.
108. A. X. Lee, R. Zhang, F. Ebert, P. Abbeel, C. Finn, and S. Levine, “Stochastic adversarial video prediction,” arXiv:1804.01523, 2018.
109. A. Radford, L. Metz, and S. Chintala, “Unsupervised representation learning with deep convolutional generative adversarial networks,” in ICLR, 2016.
110. M. Arjovsky and L. Bottou, “Towards principled methods for training generative adversarial networks,” in ICLR, 2017.
111. C. Sch¨uldt, I. Laptev, and B. Caputo, “Recognizing human actions: A local SVM approach,” in ICPR. IEEE, 2004.
112. L. Gorelick, M. Blank, E. Shechtman, M. Irani, and R. Basri, “Actions as space-time shapes,” Trans. on PAMI, vol. 29, no. 12, 2007.
113. H. Kuehne, H. Jhuang, E. Garrote, T. A. Poggio, and T. Serre, “HMDB: A large video database for human motion recognition,” in ICCV, 2011.
114. H. Jhuang, J. Gall, S. Zuffi, C. Schmid, and M. J. Black, “Towards understanding action recognition,” in ICCV, 2013.
115. K. Soomro, A. R. Zamir, and M. Shah, “UCF101: A dataset of 101 human actions classes from videos in the wild,” arXiv:1212.0402, 2012.
116. W. Zhang, M. Zhu, and K. G. Derpanis, “From actemes to action: A strongly-supervised representation for detailed action understanding,” in ICCV, 2013.
117. C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu, “Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments,” Trans. on PAMI, vol. 36, no. 7, 2014.
118. H. Idrees, A. R. Zamir, Y. Jiang, A. Gorban, I. Laptev, R. Sukthankar, and M. Shah, “The THUMOS challenge on action recognition for videos ”in the wild”,” CVIU, vol. 155, 2017.
119. P. Doll´ar, C. Wojek, B. Schiele, and P. Perona, “Pedestrian detection: A benchmark,” in CVPR, 2009.
120. A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” IJRR, vol. 32, no. 11, 2013.
121. M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in CVPR, 2016.
122. X. Huang, X. Cheng, Q. Geng, B. Cao, D. Zhou, P. Wang, Y. Lin, and R. Yang, “The apolloscape dataset for autonomous driving,” arXiv: 1803.06184, 2018.
123. A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and F. Li, “Large-scale video classification with convolutional neural networks,” in CVPR, 2014.
124. S. Abu-El-Haija, N. Kothari, J. Lee, P. Natsev, G. Toderici, B. Varadarajan, and S. Vijayanarasimhan, “Youtube-8m: A largescale video classification benchmark,” arXiv:1609.08675, 2016.
125. B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland, D. Borth, and L. Li, “YFCC100M: the new data in multimedia research,” Commun. ACM, vol. 59, no. 2, 2016.
126. I. Sutskever, G. E. Hinton, and G. W. Taylor, “The recurrent temporal restricted boltzmann machine,” in NIPS, 2008.
127. C. F. Cadieu and B. A. Olshausen, “Learning intermediate-level representations of form and motion from natural movies,” Neural Computation, vol. 24, no. 4, 2012.
128. R. Memisevic and G. Exarchakis, “Learning invariant features by harnessing the aperture problem,” in ICML, vol. 28, 2013.
129. F. Ebert, C. Finn, A. X. Lee, and S. Levine, “Self-supervised visual planning with temporal skip connections,” in CoRL, ser. Proceedings of Machine Learning Research, vol. 78, 2017.
130. S. Dasari, F. Ebert, S. Tian, S. Nair, B. Bucher, K. Schmeckpeper, S. Singh, S. Levine, and C. Finn, “Robonet: Large-scale multirobot learning,” arXiv:1910.11215, 2019.
131. R. Vezzani and R. Cucchiara, “Video surveillance online repository (visor): an integrated framework,” Multimedia Tools Appl., vol. 50, no. 2, 2010.
132. J. Santner, C. Leistner, A. Saffari, T. Pock, and H. Bischof, “PROST: parallel robust online simple tracking,” in CVPR, 2010. 24
133. M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling, “The arcade learning environment: An evaluation platform for general agents,” J. Artif. Intell. Res., vol. 47, 2013.
134. G. Seguin, P. Bojanowski, R. Lajugie, and I. Laptev, “Instancelevel video segmentation from object tracks,” in CVPR, 2016.
135. Z. Bauer, F. Gomez-Donoso, E. Cruz, S. Orts-Escolano, and M. Cazorla, “UASOL, a large-scale high-resolution outdoor stereo dataset,” Scientific Data, vol. 6, no. 1, 2019.
136. G. J. Brostow, J. Shotton, J. Fauqueur, and R. Cipolla, “Segmentation and recognition using structure from motion point clouds,” in ECCV, vol. 5302, 2008.
137. E. Santana and G. Hotz, “Learning a driving simulator,” arXiv:1608.01230, 2016.
138. Y. LeCun, F. J. Huang, and L. Bottou, “Learning methods for generic object recognition with invariance to pose and lighting,” in CVPR, 2004.
139. P. Martinez-Gonzalez, S. Oprea, A. Garcia-Garcia, A. JoverAlvarez, S. Orts-Escolano, and J. Garcia-Rodriguez, “UnrealROX: An extremely photorealistic virtual reality environment for robotics simulations and synthetic data generation,” Virtual Reality, 2019.
140. D. Jayaraman and K. Grauman, “Look-ahead before you leap: End-to-end active recognition by forecasting the effect of motion,” in ECCV, vol. 9909, 2016.
141. J. Walker, C. Doersch, A. Gupta, and M. Hebert, “An Uncertain Future: Forecasting from Static Images Using Variational Autoencoders,” in ECCV, 2016.
142. Z. Hao, X. Huang, and S. J. Belongie, “Controllable video generation with sparse trajectories,” in CVPR, 2018.
143. Y. Ye, M. Singh, A. Gupta, and S. Tulsiani, “Compositional video prediction,” in ICCV, October 2019.
144. S. Mozaffari, O. Y. Al-Jarrah, M. Dianati, P. A. Jennings, and A. Mouzakitis, “Deep learning-based vehicle behaviour prediction for autonomous driving applications: A review,” arXiv:1912.11676, 2019.
145. T. Mikolov, M. Karafi´at, L. Burget, J. Cernock´y, and S. Khudanpur, “Recurrent neural network based language model,” in INTERSPEECH, 2010.
146. K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in ICLR, 2015.
147. T. Brox, A. Bruhn, N. Papenberg, and J. Weickert, “High accuracy optical flow estimation based on a theory for warping,” in ECCV, T. Pajdla and J. Matas, Eds., vol. 3024, 2004.
148. T. Karras, T. Aila, S. Laine, and J. Lehtinen, “Progressive growing of gans for improved quality, stability, and variation,” in ICLR, 2018.
149. I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, “Improved training of wasserstein gans,” in NIPS, 2017.
150. B. Jin, Y. Hu, Q. Tang, J. Niu, Z. Shi, Y. Han, and X. Li, “Exploring spatial-temporal multi-frequency analysis for high-fidelity and temporal-consistency video prediction,” arXiv:2002.09905, 2020.
151. O. Shouno, “Photo-realistic video prediction on natural videos of largely changing frames,” arXiv:2003.08635, 2020.
152. R. Hou, H. Chang, B. Ma, and X. Chen, “Video prediction with bidirectional constraint network,” in FG, May 2019.
153. M. Oliu, J. Selva, and S. Escalera, “Folded recurrent neural networks for future video prediction,” in ECCV, 2018.
154. W. Yu, Y. Lu, S. Easterbrook, and S. Fidler, “Efficient and information-preserving future frame prediction and beyond,” in ICLR, 2020.
155. F. A. Reda, G. Liu, K. J. Shih, R. Kirby, J. Barker, D. Tarjan, A. Tao, and B. Catanzaro, “SDC-Net: Video prediction using spatiallydisplaced convolution,” in ECCV, 2018.
156. R. Memisevic and G. E. Hinton, “Learning to represent spatial transformations with factored higher-order boltzmann machines,” Neural Computation, vol. 22, no. 6, 2010.
157. R. Memisevic, “Gradient-based learning of higher-order image features,” in ICCV, 2011.
158. ——, “Learning to relate images,” Trans. on PAMI, vol. 35, no. 8, 2013.
159. V. Michalski, R. Memisevic, and K. Konda, “Modeling deep temporal dependencies with recurrent grammar cells,” in NeurIPS, 2014.
160. M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu, “Spatial Transformer Networks,” in NeurIPS, 2015.
161. B. Klein, L. Wolf, and Y. Afek, “A dynamic convolutional layer for short rangeweather prediction,” in CVPR, 2015.
162. B. D. Brabandere, X. Jia, T. Tuytelaars, and L. V. Gool, “Dynamic filter networks,” in NeurIPS, 2016.
163. A. Clark, J. Donahue, and K. Simonyan, “Adversarial video generation on complex datasets,” 2019.
164. P. Luc, A. Clark, S. Dieleman, D. de Las Casas, Y. Doron, A. Cassirer, and K. Simonyan, “Transformation-based adversarial video prediction on large-scale data,” arXiv:2003.04035, 2020.
165. K. He, X. Zhang, S. Ren, and J. Sun, “Spatial pyramid pooling in deep convolutional networks for visual recognition,” Trans. on PAMI, vol. 37, no. 9, 2015.
166. K. Simonyan and A. Zisserman, “Two-stream convolutional networks for action recognition in videos,” in NeurIPS, Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, Eds., 2014.
167. H. Gao, H. Xu, Q. Cai, R. Wang, F. Yu, and T. Darrell, “Disentangling propagation and generation for video prediction,” in ICCV, 2019.
168. Y. Wu, R. Gao, J. Park, and Q. Chen, “Future video synthesis with object motion prediction,” 2020.
169. J. Hsieh, B. Liu, D. Huang, F. Li, and J. C. Niebles, “Learning to decompose and disentangle representations for video prediction,” in NeurIPS, 2018.
170. S. Chiappa, S. Racani`ere, D. Wierstra, and S. Mohamed, “Recurrent environment simulators,” in ICLR, 2017.
171. K. Fragkiadaki, P. Agrawal, S. Levine, and J. Malik, “Learning visual predictive models of physics for playing billiards,” in ICLR (Poster), 2016.
172. A. Dosovitskiy and V. Koltun, “Learning to Act by Predicting the Future,” in ICLR, 2017.
173. P. Luc, “Self-supervised learning of predictive segmentation models from video,” Theses, Universit´e Grenoble Alpes, Jun. 2019. [Online]. Available: https://tel.archives-ouvertes.fr/tel-0 2196890
174. H.-k. Chiu, E. Adeli, and J. C. Niebles, “Segmenting the future,” arXiv:1904.10666, 2019.
175. X. Jin, H. Xiao, X. Shen, J. Yang, Z. Lin, Y. Chen, Z. Jie, J. Feng, and S. Yan, “Predicting Scene Parsing and Motion Dynamics in the Future,” in NeurIPS, 2017.
176. J. Ba and R. Caruana, “Do deep nets really need to be deep?” in NIPS, 2014.
177. G. E. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015.
178. J. Revaud, P. Weinzaepfel, Z. Harchaoui, and C. Schmid, “EpicFlow: Edge-preserving interpolation of correspondences for optical flow,” in CVPR, 2015.
179. H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in CVPR, 2017.
180. M. Rosca, B. Lakshminarayanan, D. Warde-Farley, and S. Mohamed, “Variational approaches for auto-encoding generative adversarial networks,” arXiv:1706.04987, 2017.
181. K. He, G. Gkioxari, P. Doll´ar, and R. B. Girshick, “Mask R-CNN,” in ICCV, 2017.
182. S. E. Reed, Y. Zhang, Y. Zhang, and H. Lee, “Deep visual analogymaking,” in NIPS, 2015.
183. N. Fushishita, A. Tejero-de-Pablos, Y. Mukuta, and T. Harada, “Long-term video generation of multiple futures using human poses,” arXiv:1904.07538, 2019.
184. J. Tang, H. Hu, Q. Zhou, H. Shan, C. Tian, and T. Q. S. Quek, “Pose guided global and local gan for appearance preserving human video prediction,” in ICIP, Sep. 2019.
185. Y. Bengio, R. Ducharme, P. Vincent, and C. Janvin, “A neural probabilistic language model,” J. Mach. Learn. Res., vol. 3, no. null, p. 11371155, Mar. 2003.
186. I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with neural networks,” in NeurIPS, 2014.
187. A. Mahendran and A. Vedaldi, “Understanding deep image representations by inverting them,” in CVPR, 2015.
188. R. Chalasani and J. C. Pr´ıncipe, “Deep predictive coding networks,” in ICLR (Workshop Poster), 2013.
189. M. F. Stollenga, W. Byeon, M. Liwicki, and J. Schmidhuber, “Parallel multi-dimensional lstm, with application to fast biomedical volumetric image segmentation,” in NeurIPS, 2015.
190. J. Zhang, Y. Zheng, and D. Qi, “Deep spatio-temporal residual networks for citywide crowd flows prediction,” in AAAI, 2017. 25
191. R. Goyal, S. E. Kahou, V. Michalski, J. Materzynska, S. Westphal, H. Kim, V. Haenel, I. Fr¨und, P. Yianilos, M. Mueller-Freitag, F. Hoppe, C. Thurau, I. Bax, and R. Memisevic, “The ”something something” video database for learning and evaluating visual common sense,” in ICCV, 2017.
192. Z. Yi, H. R. Zhang, P. Tan, and M. Gong, “Dualgan: Unsupervised dual learning for image-to-image translation,” in ICCV, 2017.
193. J. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-toimage translation using cycle-consistent adversarial networks,” in ICCV, 2017.
194. W. Luo, W. Liu, and S. Gao, “A revisit of sparse coding based anomaly detection in stacked RNN framework,” in ICCV. IEEE, 2017.
195. M. Ravanbakhsh, M. Nabi, E. Sangineto, L. Marcenaro, C. S. Regazzoni, and N. Sebe, “Abnormal event detection in videos using generative adversarial nets,” in ICIP, 2017.
196. L. Dinh, D. Krueger, and Y. Bengio, “NICE: non-linear independent components estimation,” in ICLR (Workshop), 2015.
197. Y. Wang, M. Long, J. Wang, Z. Gao, and P. S. Yu, “Predrnn: Recurrent neural networks for predictive learning using spatiotemporal lstms,” in NeurIPS, 2017.
198. “Traffic4cast: Traffic map movie forecasting,” https://www.iara i.ac.at/traffic4cast/, accessed: 2020-04-14.
199. S. Niklaus, L. Mai, and F. Liu, “Video frame interpolation via adaptive separable convolution,” in ICCV. IEEE, 2017.
200. ——, “Video frame interpolation via adaptive convolution,” in CVPR. IEEE, 2017.
201. J. Carreira, E. Noland, A. Banki-Horvath, C. Hillier, and A. Zisserman, “A short note about kinetics-600,” arXiv:1808.01340, 2018.
202. T. Xue, J. Wu, K. L. Bouman, and B. Freeman, “Visual Dynamics: Probabilistic Future Frame Synthesis via Cross Convolutional Networks,” in NeurIPS, 2016.
203. S. Song, F. Yu, A. Zeng, A. X. Chang, M. Savva, and T. A. Funkhouser, “Semantic scene completion from a single depth image,” in CVPR. IEEE, 2017.
204. M. Menze and A. Geiger, “Object scene flow for autonomous vehicles,” in CVPR, 2015.
205. J. Janai, F. G¨uney, A. Ranjan, M. J. Black, and A. Geiger, “Unsupervised learning of multi-frame optical flow with occlusions,” in ECCV, vol. 11220, 2018.
206. S. E. Reed, Z. Akata, S. Mohan, S. Tenka, B. Schiele, and H. Lee, “Learning what and where to draw,” in NIPS, 2016.
207. A. Newell, K. Yang, and J. Deng, “Stacked hourglass networks for human pose estimation,” in ECCV, vol. 9912, 2016.
208. K. Fragkiadaki, S. Levine, P. Felsen, and J. Malik, “Recurrent Network Models for Human Dynamics,” in ICCV, 2015.
209. T. Jakab, A. Gupta, H. Bilen, and A. Vedaldi, “Conditional image generation for learning the structure of visual objects,” arXiv:1806.07823, 2018.
210. Y. Zhang, Y. Guo, Y. Jin, Y. Luo, Z. He, and H. Lee, “Unsupervised discovery of object landmarks as structural representations,” in CVPR, 2018.
211. R. Goroshin, M. Mathieu, and Y. LeCun, “Learning to linearize under uncertainty,” in NeurIPS, 2015.
212. G. E. Hinton, A. Krizhevsky, and S. D. Wang, “Transforming autoencoders,” in ICANN, vol. 6791. Springer, 2011.
213. R. Goroshin, J. Bruna, J. Tompson, D. Eigen, and Y. LeCun, “Unsupervised learning of spatiotemporally coherent metrics,” in ICCV, 2015.
214. T. Brox and J. Malik, “Object segmentation by long term analysis of point trajectories,” in ECCV, vol. 6315, 2010.
215. J. Schmidhuber, “Learning complex, extended sequences using the principle of history compression,” Neural Computation, vol. 4, no. 2, 1992.
216. P. Agrawal, A. Nair, P. Abbeel, J. Malik, and S. Levine, “Learning to poke by poking: Experiential learning of intuitive physics,” in NeurIPS, 2016, p. 50925100.
217. V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, “Asynchronous methods for deep reinforcement learning,” in ICML, vol. 48, 2016.
218. J. Zhang and K. Cho, “Query-efficient imitation learning for endto-end simulated driving,” in AAAI, 2017.
219. S. Kohl, B. Romera-Paredes, C. Meyer, J. De Fauw, J. R. Ledsam, K. Maier-Hein, S. A. Eslami, D. J. Rezende, and O. Ronneberger, “A probabilistic u-net for segmentation of ambiguous images,” in NeurIPS, 2018.
220. F. Yu, W. Xian, Y. Chen, F. Liu, M. Liao, V. Madhavan, and T. Darrell, “BDD100K: A diverse driving video database with scalable annotation tooling,” arXiv:1805.04687, 2018.
221. G. Neuhold, T. Ollmann, S. R. Bul`o, and P. Kontschieder, “The mapillary vistas dataset for semantic understanding of street scenes,” in ICCV, 2017.
222. D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in ICLR, 2014.
223. X. Yan, J. Yang, K. Sohn, and H. Lee, “Attribute2image: Conditional image generation from visual attributes,” in ECCV, vol. 9908, 2016.
224. H. Wu, M. Rubinstein, E. Shih, J. V. Guttag, F. Durand, and W. T. Freeman, “Eulerian video magnification for revealing subtle changes in the world,” ToG, vol. 31, no. 4, 2012.
225. R. Villegas, A. Pathak, H. Kannan, D. Erhan, Q. V. Le, and H. Lee, “High fidelity video prediction with large stochastic recurrent neural networks,” in NeurIPS, 2019, pp. 81–91.
226. C. K. Sønderby, T. Raiko, L. Maaløe, S. K. Sønderby, and O. Winther, “Ladder variational autoencoders,” in NIPS, 2016.
227. R. Pottorff, J. Nielsen, and D. Wingate, “Video extrapolation with an invertible linear embedding,” arXiv:1903.00133, 2019.
228. D. P. Kingma and P. Dhariwal, “Glow: Generative flow with invertible 1x1 convolutions,” in NeurIPS, 2018.
229. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Trans. Image Processing, vol. 13, no. 4, 2004.
230. R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018.
231. T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Towards accurate generative models of video: A new metric & challenges,” arXiv:1812.01717, 2018.
232. T. Salimans, I. J. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training gans,” in NIPS, 2016.
233. O. Breuleux, Y. Bengio, and P. Vincent, “Quickly generating representative samples from an rbm-derived process,” Neural Computation, vol. 23, no. 8, 2011.
234. E. Hildreth, “Theory of edge detection,” Proc. of Royal Society of London, vol. 207, no. 187-217, 1980.
235. Y. Wang, Z. Gao, M. Long, J. Wang, and P. S. Yu, “Predrnn++: Towards A resolution of the deep-in-time dilemma in spatiotemporal predictive learning,” in ICML, ser. Proceedings of Machine Learning Research, vol. 80, 2018.
236. F. Cricri, X. Ni, M. Honkala, E. Aksu, and M. Gabbouj, “Video ladder networks,” arXiv:1612.01756, 2016.
237. I. Pr´emont-Schwarz, A. Ilin, T. Hao, A. Rasmus, R. Boney, and H. Valpola, “Recurrent ladder networks,” in NIPS, 2017.
238. B. Jin, Y. Hu, Y. Zeng, Q. Tang, S. Liu, and J. Ye, “Varnet: Exploring variations for unsupervised video prediction,” in IROS, 2018.
239. J. Lee, J. Lee, S. Lee, and S. Yoon, “Mutual suppression network for video prediction using disentangled features,” arXiv:1804.04810, 2018.
240. D. Weissenborn, O. Tckstrm, and J. Uszkoreit, “Scaling autoregressive video models,” in ICLR, 2020.
241. L. Theis, A. van den Oord, and M. Bethge, “A note on the evaluation of generative models,” in ICLR, 2016. 

## Biograhpies
* Sergiu Oprea is a PhD student at the Department of Computer Technology (DTIC), University of Alicante. He received his MSc (Automation and Robotics) and BSc (Computer Science) from the same institution in 2017 and 2015 respectively. His main research interests include video prediction with deep learning, virtual reality, 3D computer vision, and parallel computing on GPUs. 

* Pablo Martinez Gonzalez is a PhD student at the Department of Computer Technology (DTIC), University of Alicante. He received his MSc (Computer Graphics, Games and Virtual Reality) and BSc (Computer Science) at the Rey Juan Carlos University and University of Alicante, in 2017 and 2015, respectively. His main research interests include deep learning, virtual reality and parallel computing on GPUs.

* Alberto Garcia Garcia is a Postdoctoral Researcher at the Institute of Space Sciences (ICECSIC, Barcelona) where he leads the efforts in code optimization, machine learning, and parallel computing on the MAGNESIA ERC Consolidator project. He received his PhD (Machine Learning and Computer Vision), MSc (Automation and Robotics) and BSc (Computer Science) from the same institution in 2019, 2016 and 2015 respectively. Previously he was an intern at NVIDIA Research/Engineering, Facebook Reality Labs, and Oculus Core Tech. His main research interests include deep learning (specially convolutional neural networks), virtual reality, 3D computer vision, and parallel computing on GPUs.

* John Alejandro Castro Vargas is a PhD student at the Department of Computer Technology (DTIC), University of Alicante. He received his MSc (Automation and Robotics) and BSc (Computer Science) from the same institution in 2017 and 2016 respectively. His main research interests include human behavior recognition with deep learning, virtual reality and parallel computing on GPUs.

* Sergio Orts-Escolano received a BSc, MSc and PhD in Computer Science from the University of Alicante in 2008, 2010 and 2014 respectively. His research interests include computer vision, assistive robotics, 3D sensors, GPU computing, virtual/augmented reality and deep learning. He has authored +50 publications in top journals and conferences like CVPR, SIGGRAPH, 3DV, BMVC, CVIU, IROS, UIST, RAS, etcetera. He is also a member of European Networks like HiPEAC and Eucog. He has experience as a professor in academia and industry, working as a research scientist for companies such as Google and Microsoft Research.

* Jose Garcia-Rodriguez received his Ph.D. degree, with specialization in Computer Vision and Neural Networks, from the University of Alicante (Spain). He is currently Full Professor at the Department of Computer Technology of the University of Alicante. His research areas of interest include: computer vision, computational intelligence, machine learning, pattern recognition, robotics, man-machine interfaces, ambient intelligence, computational chemistry, and parallel and multicore architectures.

* Antonis Argyros is a professor of computer science at the Computer Science Department, University of Crete and a researcher at the Institute of Computer Science, FORTH, in Heraklion, Crete, Greece. His research interests fall in the areas of computer vision and pattern recognition, with emphasis on the analysis of humans in images and videos, human pose analysis, recognition of human activities and gestures, 3D computer vision, as well as image motion and tracking. He is also interested in applications of computer vision in the fields of robotics and smart environments.
