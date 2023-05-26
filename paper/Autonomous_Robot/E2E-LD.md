# Towards Driving-Oriented Metric for Lane Detection Models
车道检测模型中面向驾驶的度量 https://arxiv.org/abs/2203.16851

## Abstract
After the 2017 TuSimple Lane Detection Challenge, its dataset and evaluation based on accuracy and F1 score have become the de facto standard to measure the performance of lane detection methods. While they have played a major role in improving the performance of lane detection methods, the validity of this evaluation method in downstream tasks has not been adequately researched. In this study, we design 2 new driving-oriented metrics for lane detection: End-to-End Lateral Deviation metric (E2E-LD) is directly formulated based on the requirements of autonomous driving, a core downstream task of lane detection; Per-frame Simulated Lateral Deviation metric (PSLD) is a lightweight surrogate metric of E2E-LD. To evaluate the validity of the metrics, we conduct a large-scale empirical study with 4 major types of lane detection approaches on the TuSimple dataset and our newly constructed dataset Comma2k19-LD. Our results show that the conventional metrics have strongly negative correlations (≤-0.55) with E2E-LD, meaning that some recent improvements purely targeting the conventional metrics may not have led to meaningful improvements in autonomous driving, but rather may actually have made it worse by overfitting to the conventional metrics. As autonomous driving is a security/safety-critical system, the underestimation of robustness hinders the sound development of practical lane detection models. We hope that our study will help the community achieve more downstream task-aware evaluations for lane detection.

2017年TuSimple车道检测挑战赛之后，其基于准确性和F1分数的数据集和评估已成为衡量车道检测方法性能的事实标准。虽然它们在提高车道检测方法的性能方面发挥了重要作用，但这种评估方法在下游任务中的有效性尚未得到充分研究。在这项研究中，我们设计了两种新的面向驾驶的车道检测度量：端到端横向偏差度量(E2E-LD)是基于自动驾驶的要求直接制定的，自动驾驶是车道检测的核心下游任务; 每帧模拟横向偏差度量(PSLD)是E2E-LD的轻量级替代度量。为了评估度量的有效性，我们在TuSimple数据集和我们新构建的数据集Comma2k19 LD上使用4种主要类型的车道检测方法进行了大规模的实证研究。我们的结果表明，传统指标与E2E-LD具有强烈的负相关(≤-0.55)，这意味着最近一些纯粹针对传统指标的改进可能没有导致自动驾驶方面的有意义的改进，反而可能通过过度拟合传统指标而使其变得更糟。由于自动驾驶是一个安全/安全关键系统，对稳健性的低估阻碍了实用车道检测模型的健全发展。我们希望，我们的研究将帮助社区实现车道检测的更多下游任务感知评估。

## 1. Introduction
Lane detection is one of the key technologies today for realizing autonomous driving. For lane detection, camera is the most frequently used sensor because it is a natural choice as lane lines are visual patterns [29]. Like most other computer vision areas, lane detection has been significantly benefited from the recent advances of deep neural networks (DNNs). In the 2017 TuSimple Lane Detection Challenge [9], DNN-based lane detection shows substantial performance as all top 3 teams opt for DNN-based lane detection. After this competition, its dataset and evaluation method based on accuracy and F1 score became the de facto standard in lane detection evaluation. These metrics are inherited by the subsequent datasets [15, 40].

车道检测是当今实现自动驾驶的关键技术之一。对于车道检测，摄像头是最常用的传感器，因为车道线是视觉模式，因此这是一种自然选择[29]。与大多数其他计算机视觉领域一样，车道检测已经从深度神经网络(DNN)的最新进展中得到了显著的好处。在2017年TuSimple车道检测挑战[9]中，基于DNN的车道检测显示出显著的性能，因为所有排名前三的团队都选择了基于DNN车道检测。在这场比赛之后，其基于准确性和F1分数的数据集和评估方法成为车道检测评估的事实标准。这些度量由后续数据集继承[15，40]。

Figure 1. Examples of lane detection results and the accuracy metric in benign and adversarial attack scenarios on TuSimple Challenge dataset [9]. As shown, the conventional accuracy metric does not necessarily indicate drivability if used in autonomous driving, the core downsteam task. For example, SCNN always has higher accuracy than PolyLaneNet, but its detection results are making it much harder to achieve lane centering (detailed in §4.2). 
图1。TuSimple Challenge数据集上良性和对抗性攻击场景中的车道检测结果和准确性度量样本[9]。如图所示，如果用于自动驾驶(核心下坡任务)，则常规精度度量不一定指示驾驶性能。例如，SCNN总是比PolyLaneNet具有更高的精度，但其检测结果使得实现车道居中变得更加困难(详见§4.2)。


However, the validity of this evaluation method in practical contexts, i.e., whether this is representative of practicality in real-world downstream applications, has not been adequately researched. Specifically, the main real-world applications of lane detection are for autonomous driving (AD), e.g., online detection for automated lane centering (for lower-level AD such as in Telsa AutoPilot [7]), and offline detection for high-definition map creation (for both low-level [6] and high-level AD [53]). With such an application domain as its main target, the robustness of lane detection is highly critical as errors from it could be fatal. Unfortunately, we find that the conventional evaluation metrics (i.e., accuracy and F1 score) have limitations to correctly reflect the performance of lane detection models in such main downstream application domain, especially in more challenging scenarios (e.g., when under adversarial attacks). Fig. 1 shows a few such examples that motivate this study. In the adversarial attack settings, the lane lines detected by SCNN [40] are largely disrupted, but their performance measured by the conventional accuracy metric is always higher than the one of PolyLaneNet [51], which are generally aligned with actual lane lines (and indeed lead to less lane center deviation than SCNN when used with driving models as quantified later in §4.2). In the benign settings, PolyLaneNet has the lowest accuracy and is underestimated, despite its seemingly perfect detection for humans. As lane detection has been evaluated using mainly relatively clean and homogeneous driver’s view images, it is not easy to identify such a great discrepancy at the metric level. Considering the criticality of robust lane detection to correct and safe AD, it is important to address such a metric-level limitation since (1) the cornerstone of realworld deployment and commercialization of AD today is exactly on the handling of those more challenging driving scenarios [24, 33, 57]; and (2) with increasingly more discoveries of physical-world adversarial attack on lane detection in AD context [34, 47], it is desired to have a more downstream task-aware performance metric when judging the model robustness (and its enhancement).

然而，这种评估方法在实际环境中的有效性，即这是否代表实际下游应用中的实用性，尚未得到充分研究。具体而言，车道检测的主要现实应用是自动驾驶(AD)，例如，自动车道居中的在线检测(用于较低级别的AD，如Telsa AutoPilot[7])，以及高清地图创建的离线检测(用于低级别的[6]和高级别的AD[53])。以这样的应用领域为主要目标，车道检测的稳健性非常关键，因为来自车道检测的错误可能是致命的。不幸的是，我们发现传统的评估度量(即准确度和F1分数)在正确反映此类主要下游应用领域中车道检测模型的性能方面存在局限性，尤其是在更具挑战性的场景中(例如，在对抗性攻击下)。图1显示了一些激励这项研究的例子。在对抗性攻击设置中，SCNN[40]检测到的车道线在很大程度上被破坏，但通过传统精度度量测量的它们的性能始终高于PolyLaneNet[51]，它们通常与实际车道线对齐(实际上，当与§4.2中稍后量化的驾驶模型一起使用时，会导致比SCNN更小的车道中心偏差)。在良性环境中，PolyLaneNet的准确度最低，而且被低估了，尽管它对人类的检测似乎很完美。由于车道检测主要使用相对干净和均匀的驾驶员视角图像进行评估，因此在度量级别上识别如此大的差异并不容易。考虑到稳健车道检测对纠正和安全AD的关键性，解决这样一个度量级别的限制很重要，因为(1)当今AD的现实部署和商业化的基石正是处理那些更具挑战性的驾驶场景[24，33，57]; (2)随着越来越多地发现AD上下文中对车道检测的物理世界对抗性攻击[34，47]，在判断模型稳健性(及其增强)时，需要有更下游的任务感知性能度量。

Motivated by such critical needs, we design 2 new driving-oriented metrics, End-to-End Lateral Deviation metric (E2E-LD) and Per-frame Simulated Lateral Deviation metric (PSLD), to measure the performance of lane detection models in AD, especially in Automated Lane Centring (ALC), a Level-2 driving automation that automatically steers a vehicle to keep it centered in the traffic lane [8]. E2E-LD is designed directly based on the requirements of driving automation by ALC. PSLD is a lightweight surrogate metric of E2E-LD that estimates the impact of lane detection results on driving from a single frame. This per-frame lightweight design allows the metric to be usable during upstream lane detection model training. To evaluate the validity of the metrics, we conduct a large-scale empirical study of the 4 major types of lane detection approaches on the TuSimple dataset and our newly constructed dataset, Comma2k19-LD, which contains both lane line annotation and driving information. To simulate corner-case but physically-realizable scenarios as in Fig. 1 for lane detection, we utilize and extend physical-world adversarial attacks on ALC [47]. We formulate attack objective functions to fairly generate adversarial attacks against the 4 major types of lane detection approaches. Throughout this study, we find that the conventional metrics have strongly negative correlations (r ≤-0.55) with E2E-LD in the benign scenarios, meaning that some recent improvements purely targeting the conventional metrics may not have led to meaningful improvements in AD, but rather may actually have made it worse by overfitting to the conventional metrics. In the attack scenarios, while we observe a slight positive correlation (r ≤0.08), it is not statistically significant. Consequently, we find that the conventional metrics tend to overestimate less robust models. On the contrary, our newly-designed PSLD metric is always strongly positively correlated with E2E-LD (r ≥0.38), and all correlations are statistically significant (p ≤ 0.001).

受这些关键需求的激励，我们设计了两个新的面向驾驶的度量，即端到端横向偏差度量(E2E-LD)和每帧模拟横向偏差度量值(PSLD)，以测量AD中车道检测模型的性能，尤其是自动车道居中(ALC)，这是一种2级驾驶自动化，可自动驾驶车辆，使其保持在行车道中心[8]。E2E-LD是根据ALC的驾驶自动化要求直接设计的。PSLD是E2E-LD的一个轻量级替代度量，用于估计车道检测结果对单帧驾驶的影响。这种每帧轻量化设计允许在上游车道检测模型训练期间使用度量。为了评估度量的有效性，我们在TuSimple数据集和我们新构建的数据集Comma2k19 LD上对4种主要类型的车道检测方法进行了大规模的实证研究，该数据集包含车道线注释和驾驶信息。为了模拟如图1所示的拐角情况但物理上可实现的场景，我们利用并扩展了ALC上的物理世界对抗性攻击[47]。我们制定了攻击目标函数，以公平地生成针对4种主要车道检测方法的对抗性攻击。在整个研究过程中，我们发现，在良性场景中，传统指标与E2E-LD具有强烈的负相关(r≤-0.55)，这意味着最近一些纯粹针对传统指标的改进可能没有导致AD的有意义的改善，反而可能通过过度拟合传统指标而使AD变得更糟。在攻击场景中，虽然我们观察到轻微的正相关(r≤0.08)，但没有统计学意义。因此，我们发现传统的度量往往高估了不太稳健的模型。相反，我们新设计的PSLD度量始终与E2E-LD强正相关(r≥0.38)，所有相关性均具有统计学意义(p≤0.001)。

While the TuSimple Challenge dataset and its evaluation metrics have played a substantial role in developing performant lane detection methods, the recent improvement on the conventional metrics does not lead to the improvement on the core downstream task AD. We thus want to inform the community of such limitations of the conventional evaluation and facilitate research to conduct more downstream task-aware evaluation for lane detection, as the gap between upstream evaluation metrics and downstream application performance may hinder the sound development of lane detection methods for real-world application scenarios.

虽然TuSimple Challenge数据集及其评估指标在开发高性能车道检测方法方面发挥了重要作用，但传统指标的最近改进并未导致核心下游任务AD的改进。因此，我们希望向社区通报传统评估的局限性，并促进研究，以便对车道检测进行更多的下游任务感知评估，因为上游评估指标和下游应用程序性能之间的差距可能会阻碍真实应用场景车道检测方法的健全发展。

In summary, our contributions are as follows: 
* We design 2 new driving-oriented metrics, E2E-LD and PSLD, that can more effectively measure the performance of lane detection models when used for AD, their core downstream task. 
* We design a methodology to fairly generate physicalworld adversarial attacks against the 4 major types of lane detection models. 
* We build a new dataset Comma2k19-LD that contains lane annotations and driving information. 
* We are the first to conduct a large-scale empirical study to measure the capability of 4 major types of lane detection models in supporting AD. 
* We highlight and discuss the critical limitations of the conventional evaluation and demonstrate the validity of our new downstream task-aware metrics.

总之，我们的贡献如下：
* 设计了两个新的面向驾驶的指标E2E-LD和PSLD，当用于AD(其核心下游任务)时，它们可以更有效地测量车道检测模型的性能。
* 设计了一种方法来公平地生成针对4种主要类型车道检测模型的物理世界对抗性攻击。
* 构建了一个新的数据集Comma2k19LD，其中包含车道标注和驾驶信息。
* 是第一个进行大规模实证研究，以衡量4种主要类型的车道检测模型在支持AD方面的能力。
* 强调并讨论了传统评估的关键局限性，并证明了我们新的下游任务感知度量的有效性。

Code and data release. All our codes and datasets are available in our project websites 1 .

代码和数据发布。我们的所有代码和数据集都可以在我们的项目网站1中找到。

## 2. Related Work
### 2.1. DNN-based Lane Detection
We taxonomize state-of-the-art DNN-based lane detection methods into 4 approaches. Similar taxonomy is also adopted in prior works [38, 50].

我们将最先进的基于DNN的车道检测方法分类为4种方法。在先前的工作中也采用了类似的分类[38，50]。

#### Segmentation approach. 
Segmentation approach handles lane detection as a segmentation task, which classifies whether each pixel is on a lane line or not. Since this approach achieved the state-of-the-art performance in the 2017 TuSimple Lane Detection Challenge [9] (all top3 winners adopt the segmentation approach [32, 39, 40]), it has been applied in many recent lane detection methods [31,58,59]. This segmentation approach is also used in the industry. A reverse-engineering study reveals that Tesla Model S adopts this segmentation-based approach [34]. The major drawback of this approach is its higher computational and memory cost than the other approaches. Due to the nature of the segmentation approach, it needs to predict the classification results for every pixel, the majority of which is just background. Additionally, this approach requires a postprocessing step to extract the lane line curves from the pixel-wise classification result.

分割方法将车道检测作为一项分割任务来处理，该任务对每个像素是否位于车道线上进行分类。由于该方法在2017年TuSimple车道检测挑战中取得了最先进的性能[9](所有前3名获奖者都采用了分割方法[32，39，40])，因此它已应用于许多最近的车道检测方法[31，58，59]。该细分方法也在行业中使用。一项逆向工程研究表明，特斯拉Model S采用了这种基于细分的方法[34]。这种方法的主要缺点是其比其他方法更高的计算和内存成本。由于分割方法的性质，它需要预测每个像素的分类结果，其中大部分只是背景。此外，该方法需要一个后处理步骤来从逐像素分类结果中提取车道线曲线。

1 https://github.com/ASGuard-UCI/ld-metric https://sites.google.com/view/cav-sec/ld-metric 

#### Row-wise classification approach. 
This approach [30, 38,42,56] leverages the domain-specific knowledge that the lane lines should locate the longitudinal direction of driving vehicles and should not be so curved to have more than 2 intersections in each row of the input image. Based on the assumption, this approach formulates the lane detection task as multiple row-wise classification tasks, i.e., only one pixel per row should have a lane line. Although it still needs to output classification results for every pixel similar to the segmentation approach, this divide-and-conquer strategy enables to reduce the model size and computation while keeping high accuracy. For example, UltraFast [42] reports that their method can work at more than 300 FPS with a comparable accuracy 95.87% on the TuSimple Challenge dataset [9]. On the other hand, SAD [31], a segmentation approach, works at 75 frames per second with 96.64% accuracy. This approach also requires a postprocessing step to extract the lane lines similar to the segmentation approach.

按行分类方法。该方法[30，38，42，56]利用了特定领域的知识，即车道线应定位行驶车辆的纵向方向，并且不应弯曲到在输入图像的每行中具有超过2个交叉点。基于该假设，该方法将车道检测任务表述为多行分类任务，即每行只有一个像素应该具有车道线。尽管与分割方法类似，它仍然需要输出每个像素的分类结果，但这种分而治之策略能够在保持高精度的同时减少模型大小和计算量。例如，UltraFast[42]报告称，他们的方法可以在TuSimple Challenge数据集上以超过300 FPS的速度工作，准确率达到95.87%[9]。另一方面，SAD[31]是一种分割方法，以每秒75帧的速度工作，准确率为96.64%。该方法还需要一个后处理步骤来提取车道线，类似于分割方法。

#### Curve-fitting approach. 
The curve-fitting approach [41, 51] fits the lane lines into parametric curves (e.g., polynomials and splines). This approach is applied in an open-source production driver assistance system, OpenPilot [5]. The main advantage of this approach is lightweight computation, allowing OpenPilot to run on a smartphone-like device without GPU. To achieve high efficiency, the accuracy is generally not high as other approaches. Additionally, prior work mentions that this approach is biased toward straight lines because the majority of lane lines in the training data are straight [51].

曲线拟合方法[41，51]将车道线拟合成参数曲线(例如，多项式和样条曲线)。这种方法应用于开源生产驱动程序辅助系统OpenPilot[5]。这种方法的主要优点是计算量轻，允许OpenPilot在没有GPU的智能手机设备上运行。为了实现高效率，精度通常不像其他方法那样高。此外，先前的工作提到，这种方法偏向于直线，因为训练数据中的大多数车道线都是直线[51]。

#### Anchor-based approach. 
Anchor-based approach [37, 43, 50] is inspired by region-based object detectors such as Faster R-CNN [45]. In this approach, each lane line is represented as a straight proposal line (anchor) and lateral offsets from the proposal line. Similar to the row-wise classification approach, this approach takes advantage of the domain-specific knowledge that the lane lines are generally straight. This design enables to achieve state-of-theart latency and performance. LaneATT [50] reports that it achieves a higher F1 score (96.77%) than the segmentation approaches (95.97%) [31, 40] on the TuSimple dataset.

基于锚点的方法[37，43，50]受到基于区域的物体检测器的启发，例如Faster R-CNN[45]。在这种方法中，每条车道线表示为直线提案线(锚点)和提案线的横向偏移。与按行分类方法类似，该方法利用了车道线通常是直线的领域特定知识。这种设计能够实现最先进的延迟和性能。LaneATT[50]报告说，它在TuSimple数据集上实现了比分割方法(95.97%)更高的F1分数(96.77%)[31，40]。

### 2.2. Evaluation Metrics for Lane Detection
All lane detection methods we discuss in §2.1 evaluate their performance on the accuracy and F1 score metrics used in the 2017 TuSimple Challenge [9]. The accuracy is calculated by P i∈H tpi |H| , where H is a set of sampled y-axis points in the driver’s view image and tpi is 1 if the difference of a predicted lane line point and the ground truth point at y = i is within α pixels; otherwise is 0. α is set to 20 pixels in the TuSimple Challenge. The detected lane line is associated with a ground truth line with the highest accuracy. In other datasets [15, 40], IoU (Intersection over Union) is also used instead of accuracy. However, the ground-truth area is only defined as a 30-pixel wide line based on lane points, and this metric is almost equivalent to accuracy. The F1 score is a common metric to measure the performance of binary classification tasks. This is the harmonic mean of precision and recall: 2 recall−1+precision−1 . In the TuSimple Challenge, the precision and recall are calculated at the lane line level: The precision is the true positive ratio of detected lane lines and the recall is the true positive ratio of ground truth lines. The true positive is defined if the accuracy of a pair of the ground truth line and detected line is ≥ β. β is set to 0.85 in the TuSimple Challenge. Although the accuracy and F1 score can measure the capability of lane detection at a certain level, these metrics do not fully represent the performance in the main real-world downstream application, AD [6, 7, 53], as concretely shown later in §4.2.

我们在§2.1中讨论的所有车道检测方法都会评估其在2017 TuSimple挑战赛中使用的准确性和F1分数指标方面的表现[9]。精度由P i∈H tpi|H|计算，其中H是驾驶员视图图像中的一组采样y轴点，如果预测车道线点和y=i处的地面真实点的差在α像素内，则tpi为1; 否则为0。在TuSimple挑战中，α设置为20像素。检测到的车道线与具有最高精度的地面真实线相关联。在其他数据集[15，40]中，也使用了IoU(联合上的交集)来代替精度。然而，地面真实区域仅定义为基于车道点的30像素宽的线，并且该度量几乎等同于精度。F1分数是衡量二进制分类任务性能的常用指标。这是精度和召回的调和平均值：2召回−1+精度−1。在TuSimple挑战中，精确性和召回率是在车道线水平上计算的：精确性是检测到的车道线的真正比率，召回率是地面真实线的真负比率。如果一对接地真线和检测线的精度≥β，则定义真正。β在TuSimple挑战中设置为0.85。尽管准确度和F1分数可以在一定程度上衡量车道检测的能力，但这些指标并不能完全代表主要真实世界下游应用程序AD[6，7，53]的性能，具体如§4.2所示。

Specifically, to reflect its performance if used in AD, or drivability, accuracy and F1 score metrics have 2 major limitations: (1) There is no justification of α = 20 pixels and β = 0.85 accuracy thresholds. For example, the ALC system can keep at the lane center even if the detection error is more than 20 pixels, as long as the detected lane lines are parallel with actual lane lines. Furthermore, the importance of detected lane line points should not be equal, i.e., closer points to the vehicle should be more important than the distanced points to control a vehicle. (2) The current metrics treat all lane lines in the driver’s view equally, e.g., detection errors for the ego lane’s left line are treated the same as the detection errors for the left lane’s left line. However, the former is much more important to ALC systems than the latter, as the former can directly impact the downstream calculation of the lane center. For example, if a model cannot detect the left lane’s left line but can still detect the ego lane’s left line, it won’t affect its use for ALC. However, if it cannot detect the latter but can detect the former, the accuracy metric remains the same but the downstream modules in ALC may consider the left lane’s left line as ego lane’s left line and thus mistakenly deviate to the left.

具体而言，为了反映其在AD或驾驶性能中的表现，准确度和F1分数指标有两个主要限制：(1)没有理由使用α=20像素和β=0.85准确度阈值。例如，只要检测到的车道线与实际车道线平行，即使检测误差超过20个像素，ALC系统也可以保持在车道中心。此外，检测到的车道线点的重要性不应相等，即离车辆较近的点应比控制车辆的距离点更重要。(2) 当前度量对驾驶员视野中的所有车道线一视同仁，例如，自我车道左线的检测误差与左车道左线检测误差相同。然而，前者对自动高度控制系统比后者重要得多，因为前者可以直接影响车道中心的下游计算。例如，如果一个模型不能检测到左车道的左线，但仍然可以检测到自我车道的左行，那么它不会影响它用于自动高度控制。然而，如果它不能检测到后者，但可以检测到前者，则精度度量保持不变，但ALC中的下游模块可能会将左侧车道的左侧线视为自我车道的左侧，从而错误地向左偏离。

### 2.3. Automated Lane Centering
Automated Lane Centering (ALC) is a Level-2 driving automation technology that automatically steers a vehicle to keep it centered in the traffic lane [8]. Recently, ALC is widely adopted in various vehicle models such as Tesla [7] and thus one of the most popular downstream applications of lane detection. Typical ALC systems [5, 11, 36] operate in 3 modules: lane detection, lateral control, and vehicle actuation. More details of ALC are in Appendix G. While there is a line of research that designs end-to-end DNNs for ALC or higher driving automation [14, 16, 19], the current industry-standard solutions adopt such a modular design to ensure accountability and safety. In the lateral control, ALC plans to follow the lane center as waypoints with Proportional-Integral-Derivative (PID) [23] or Model Predictive Control (MPC) [46].

自动车道居中(ALC)是一种2级驾驶自动化技术，可自动控制车辆，使其在行车道上居中[8]。最近，ALC被广泛应用于各种车型，如特斯拉[7]，因此是车道检测最受欢迎的下游应用之一。典型的自动高度控制系统[5、11、36]分为3个模块：车道检测、横向控制和车辆驱动。ALC的更多细节见附录G。虽然有一系列研究为ALC或更高的驾驶自动化设计端到端DNN[14，16，19]，但当前的行业标准解决方案采用了这种模块化设计，以确保可靠性和安全性。在横向控制中，ALC计划使用比例积分微分(PID)[23]或模型预测控制(MPC)[46]跟踪车道中心作为路线点。

Figure 2. Overview of our driving-oriented metrics for lane detection models: E2E-LD and PSLD. Xt are camera frames from driver’s view (lane detection model inputs). E2E-LD requires multiple (consecutive) camera frames, while PSLD only uses the current frame X0. 
图2:车道检测模型的驾驶导向指标概述：E2E-LD和PSLD。Xt是来自驾驶员视角的摄像机帧(车道检测模型输入)。E2E-LD需要多个(连续)相机帧，而PSLD仅使用当前帧X0。

Adversarial Attack on ALC. After researchers found DNN models generally vulnerable to adversarial attacks [27, 49], the following work further explored such attacks in the physical world [18, 26]. A recent study demonstrates that ALC systems are also vulnerable to physicalworld adversarial attacks [47]. Their attack, dubbed Dirty Road Patch (DRP) attack, targets industry-grade DNNbased ALC systems, and is designed to be robust to the vehicle position and heading changes caused by the attack in the earlier frames. We use the DRP attack to simulate challenging but realizable scenarios in our evaluations.

对ALC的对抗性攻击。在研究人员发现DNN模型通常容易受到对抗性攻击[27，49]之后，接下来的工作进一步探索了物理世界中的此类攻击[18，26]。最近的一项研究表明，ALC系统也容易受到物理世界对抗性攻击[47]。他们的攻击被称为“脏路分块”(DRP)攻击，目标是基于DNN的工业级自动高度控制系统，旨在对早期帧中的攻击导致的车辆位置和航向变化具有稳健性。我们在评估中使用DRP攻击来模拟具有挑战性但可实现的场景。

## 3. Methodology
In this section, we motivate the design of 2 new downstream task-aware metrics to measure the performance of lane detection models in ALC. To evaluate the validity of the metrics even in challenging scenarios, we formulate attack objective functions to fairly generate adversarial attacks against the 4 major types of lane detection methods.

### 3.1. End-to-End Lateral Deviation Metric
As the name of ALC indicates, the performance of ALC should be evaluated by how accurately it can drive in the lane center, i.e., the lateral (left or right) deviation from the lane center. In particular, the maximum lateral deviation from the lane center in continuous closed-loop perception and control is the ultimate downstream-task performance metric for lane detection. Such deviation is directly safetycritical as large lateral deviations can cause a fatal collision with other driving vehicles or roadside objects. We call it

End-to-End Lateral Deviation metric (E2E-LD), shown in

Fig. 2 (a). The E2E-LD at t = 0 is obtained as follows. \max _{t \leq T_E}(|L_t - C_t|) (1) , where Lt is the lateral (y-axis) coordinate of the vehicle at t. Ct is the lane center lateral (y-axis) coordinate corresponding to the vehicle position at t. We use the vehicle coordinate system at t = 0. TE is a hyperparameter to decide the time duration. If TE = 1 second, the E2ELD is the largest deviation within one second. To obtain

Lt, it requires a closed-loop mechanism to simulate a driving by ALC, such as AD simulators [3, 25]. Starting from t = 0, the vehicle position and heading at t = 1 is calculated based on the camera frame at t = 0 (X0): The lane detection model detects lane lines from the frame, the lateral control interprets it by a steering angle, and vehicle actuation operates the steering wheel. This procedure repeats until t = Te. Hence, multiple (consecutive) camera frames

X0,...,XTE are required and they are dynamically changed based on the lane detection results in the earlier frames.

However, such AD simulations are too computationally expensive for large-scale evaluations. Thus, we simulate vehicle trajectories by following prior work [47], which combines vehicle motion model [44] and perspective transformation [28, 52] to dynamically synthesize camera frames from existing frames according to a driving trajectory.

### 3.2. Per-Frame Simulated Lateral Deviation Metric
The E2E-LD metric is defined as the desired metric based on the requirements of downstream task ALC.

However, it is still too computationally intensive to be monitored during training of the upstream lane detection model. This overhead is mainly due to the camera frame inter-dependency that the camera frames are dynamically changed based on the lane detection results in the earlier frames. To address this limitation, we design the Per-Frame

Simulated Lateral Deviation metric (PSLD), which simulates E2E-LD only with a single camera input at the current frame (X0) and the geometry of the lane center.

The overview of PSLD is shown in Fig. 2 (b). The calculation consists of two stages: ⃝1 update the vehicle position with the current camera frame at t = 0 (X0) and its lane detection result, and ⃝2 apply the closed-loop simulation using the ground-truth lane center as waypoints from t = 1 to t = Tp. Note that we do not need camera frames in ⃝2 as the vehicle just tries to follow the ground-truth waypoints with lateral control, i.e., we bypass the lane detection assuming we know the ground-truth in t ≥ 1. We then take the maximum lateral deviation from the lane center as a metric as with E2E-LD. For convenience, we normalize the maximum lateral deviation by Tp to make it a per-frame metric. The definition of PSLD is as follows: \frac {1}{T_p}\max _{1 \leq t \leq T_p}(|\widetilde {L}_t - C_t|) 1

Tp max 1≤t≤Tp (|Let − Ct|) (2) , where the Let is the simulated lateral (y-axis) coordinate of the vehicle at t. For example, for Tp = 1, it is just a singlestep simulation with the current lane detection result. The longer Tp can simulate the tailing effect of the current frame result in the later frames, but it may suffer from accumulated errors. In §4.3, we explore which Tp achieves the best correlation between PSLD and E2E-LD. More details are in Appendix A.

### 3.3. Attack Generation
In this study, we utilize and extend physical-world adversarial attacks to evaluate the robustness of the lane detection system against challenging but realizable scenarios.

To fairly generate adversarial attacks for all 4 major types of lane detection methods, we design an attack objective that can be commonly applicable to them. We name it the expected road center, which averages all detected lane lines weighted with their probabilities. Intuitively, the average of all lane lines is expected to represent the road center. If the expected center locates at the center of the input image, its value will be 0.5 in the normalized image width. We maximize the expected road center to attack to the right and minimize it to attack to the left. Detailed calculation of the expected road center for each method is as follows.

Segmentation & row-wise classification approaches: \frac {1}{L\cdot H}\sum _{l = 1}^{L}\sum _{i = 1}^{W}\sum _{j = 1}^{H} i \cdot P^l_{ij} (3) , where H and W are the height and width of probability map, L is the number of probability maps (channels), and

P l ij is the lane line existence probability of the pixel in the (i, j) element of the probability map.

Curve-fitting approach: \frac {1}{L\cdot |\mathcal {H}|}\sum _{l = 1}^{L}\sum _{j \in \mathcal {H}} [j^d, j^{d-1}, \cdots , j, 1] p_l (4) , where L is the number of detected lane lines, d is the degrees of polynomial (d = 3 used in PolyLaneNet [51]), H is a set of sampled y-axis values, and pl ∈ R d+1 is the coefficient of detected lane line l.

Anchor-based approach: \sum _{l \in \mathcal {A}} \left [ \frac {1}{|\Delta ^l|} \sum _{j \in \Delta ^l} (a^l_j + \delta ^l_j) \right ] \cdot \pi ^l (5) , where A is a set of the anchor proposals, ∆l is an index set of y-axis value for anchor proposal l, π l is the probability of anchor proposal l, and a l j and δ l j are the x-axis value and its offset of anchor proposal l at y-axis index j respectively.

We incorporate this expected road center functions into

DRP attack [47] procedure to generate adversarial attacks

Table 1. Target lane detection methods. Acc. is the accuracy of the

TuSimple Challenge dataset [9] in the reference papers.

Approach Selected Method Acc.

Segmentation SCNN [40] 96.53%

Row-wise classification UltraFast (ResNet18) [42] 95.87%

Curve-fitting PolyLaneNet (b0) [51] 88.62%

Anchor-based LaneATT (ResNet34) [50] 95.63% that are effective for multiple frames.

## 4. Experiments
We conduct a large-scale empirical study to evaluate the validity of the conventional metrics and our PLSD by comparing them with the ultimate downstream-task performance metric E2E-LD. We evaluate the 4 major types of lane detection approaches. We select a representative model for each approach as shown in Table 1. The pretrained weights of all models are obtained from the authors’ or publicly available websites2 . All pretrained weights are trained on the TuSimple Challenge training dataset [9].

### 4.1. Conventional Evaluation on TuSimple Dataset
Evaluation Setup. We first evaluate the lane detection models with the conventional accuracy and F1 score metrics on the TuSimple dataset [9], which has 2,782 one-secondlong video clips as test data. Each clip consists of 20 frames, and only the last frame is annotated and used for evaluation. We randomly select 30 clips from the test data. For each clip, we consider two attack scenarios: attack to the left, and to the right. Thus, in total, we evaluate 60 different attack scenarios. In each scenario, we place 3.6 m x 36 m patches 7 m away from the vehicle as shown in Fig. 1. To know the world coordinate, we manually calibrate the camera matrix based on the size of lane width and lane marking.

To deal with the limitation (2) discussed in §2.2, we remove lane lines other than the ego-left and ego-right lane lines to evaluate the applicability to ALC systems more correctly.

More details of each attack implementation and parameters are Appendix D.

Results. Table 2 shows the accuracy and F1 score metrics in the benign and attacks scenarios. In the benign scenarios, LaneATT has the best accuracy (94%) and F1 score (88%). SCNN and UltraFast show also high accuracy and

F1 score while UltraFast has the lowest F1 score (8%) in the attack scenarios. PolyLaneNet has lower accuracy and

F1 score than the others in both benign and attack scenarios. These results are generally consistent with the reported performance as in Table 1. However, when we visually look into the detected lane lines under attack, we find quite some cases suggesting vastly different conclusions if used in AD as the downstream task. For example, as shown in Fig. 1, 2

LaneATT https://github.com/lucastabelini/LaneATT

SCNN https://github.com/harryhan618/SCNN Pytorch

UltraFast https://github.com/cfzd/Ultra-Fast-Lane-Detection

PolyLaneNet https://github.com/lucastabelini/PolyLaneNet 5

SCNN UltraFast PolyLaneNet

Benign Attack

LaneATT

Figure 3. Examples of the benign and attack-to-the-right scenarios on the Comma2k19-LD dataset. The red, blue, and green lines are the detected left and right lines and the ground-truth lines respectively.

Table 2. Accuracy and F1 scores for attack and benign cases on the

TuSimple Challenge dataset. The metrics are calculated only with ego left and right lanes. The bold and underlined letters mean the highest and lowest scores, respectively, among the 4 lane detection methods. The higher score means the higher performance. .

Accuracy F1 Score

Benign Attack Benign Attack

SCNN [40] 89% 58% 75% 28%

UltraFast [42] 87% 36% 77% 8%

PolyLaneNet [51] 72% 53% 50% 19%

LaneATT [50] 94% 51% 88% 29% even though SCNN has the highest accuracy in all three scenarios, its detected lane lines are heavily curved by the attack. In contrast, the detection of PolyLaneNet looks like the most robust among the 4 models, as the detected lane lines are generally parallel to the actual lane lines. However, its accuracy (63%) is smaller than the one of SCNN (51%) in the attack to the right scenario. In the benign scenario, PolyLaneNet has a lower accuracy (16% margin) than the others, but it is hard to find meaningful differences for humans as the detected lines are well-aligned with actual lane lines. We provide more examples in Appendix G.

Hence, the conventional accuracy and F1 score-based evaluation may not be well suitable to judge the performance of the lane detection model in representative downstream tasks such as AD.

### 4.2. Consistency of TuSimple Metrics with E2E-LD
To more systematically evaluate the consistency of the conventional accuracy and F1 score with the performance in AD as the downstream tasks, we conduct a large-scale empirical study on our newly-constructed dataset.

New Dataset: Comma2k19-LD. To evaluate both the conventional metrics and the downstream task-centric metrics E2E-LD and PSLD on the same dataset, we need both lane line annotations and driving information (e.g., position, steering angle, and velocity). Unfortunately, there is no existing dataset that satisfies the requirements to our best knowledge. Thus, we create a new dataset, coined

Comma2k19-LD, in which we manually annotate the left and right lane lines for 2,000 frames (100 scenarios of 1- second clips at 20 Hz). The selected scenarios are randomly selected from the scenarios with more than 30 mph (≈ 48 km/h) in the original Comma2k19 dataset [48]. Fig 3 shows the example frames of the Comma2k19-LD dataset.

These frames are the first frames of the scenario. The following 20 frames are also annotated and the same patch is used for each attack. More details are in Appendix C. The

Comma2k19-LD dataset is published on our website [12].

Evaluation Setup. We conduct the evaluation on the

Comma2k19-LD dataset. For the attack generation, we attack to the left in randomly selected 50 scenarios and attack to the right in the other 50 scenarios. For the lateral control, we use the implementation of MPC [46] in

OpenPilot v0.6.6, which is an open-source production ALC system. For the longitudinal control, we used the velocity in the original driving trace. For the motion model, we adopt the kinematic bicycle model [35], which is the most widely-used motion model for vehicles [2, 35, 55]. The vehicle parameters are from Toyota RAV4 2017 (e.g., wheelbase), which is used to collect the traces of the comma2k19 dataset. To make the model trained on the TuSimple dataset work on the Comma2k19-LD dataset, we manually adjust the input image size and field-of-view to be consistent with the TuSimple dataset. We place a 3.6 m x 36 m patch at 7 m away from the vehicle at the first frame. For the E2E-LD metric, we use TE = 20 frames (1 second). It follows the result that the average attack success time of the DRP attack is nearly 1 sec [47]. More setup details are in Appendix B, D, and G).

Results. Table 6 shows the evaluation results of conventional accuracy and F1 score and E2E-LD. We calculate the Pearson correlation coefficient r and its p value.

As shown, there are substantial inconsistencies between the downstream-task performance (from the heavy-weight E2ELD metric) and the conventional metrics. In the benign scenarios, SCNN has the highest accuracy (0.59) and F1 score (0.84) under the original parameters (α = 20, β = 0.85).

However, SCNN is one of the methods with the lowest

E2E-LD (0.21), and instead UltraFast has the highest E2ELD (0.18). In the attack scenarios, the inconsistency is more obvious: PolyLaneNet has the highest E2E-LD (0.38), but PolyLaneNet achieves the 2nd lowest accuracy (0.59) and the highest F1 score (0.13) with the original parameters. Hence, the E2E-LD draws quite different conclusions from the conventional metrics. If we adopt the conven6

Table 3. Evaluation results of the E2E-LD and the conventional metrics, accuracy and F1 in the benign and attack scenarios. For each metric, the corresponding Pearson correlation coefficient with E2E-LD in the bottom rows. The original parameters are the ones used in the TuSimple challenge. The best parameters are those that have the highest correlation between E2E-LD with respect to F1 score. The bold and underlined letters indicate the highest and lowest performance or correlation, respectively.

Benign Attack

Original Parameters (α = 20, β = 0.85)

Best Parameters (α = 5, β = 0.9)

Original Parameters (α = 20, β = 0.85)

Best Parameters (α = 50, β = 0.65)

E2E-LD [m] Accuracy F1 Accuracy F1 E2E-LD [m] Accuracy F1 Accuracy F1

Metric

SCNN [40] 0.21 0.93 0.84 0.59 0.03 0.48 0.68 0.31 0.83 0.76

UltraFast [42] 0.18 0.92 0.81 0.55 0.10 0.58 0.60 0.21 0.82 0.77

PolyLaneNet [51] 0.20 0.78 0.50 0.44 0.01 0.38 0.59 0.13 0.81 0.76

LaneATT [50] 0.21 0.89 0.75 0.54 0.06 0.72 0.51 0.14 0.66 0.48

Corr.

SCNN [40] - -0.65∗∗∗ -0.60∗∗∗ -0.33∗∗∗ -0.13ns - -0.13ns -0.06ns -0.14ns -0.06ns

UltraFast [42] - -0.58∗∗∗ -0.59∗∗∗ -0.38∗∗∗ -0.24∗ - -0.24∗ -0.14ns -0.20∗ -0.13ns

PolyLaneNet [51] - -0.60∗∗∗ -0.55∗∗∗ -0.46∗∗∗ 0.10ns - -0.27∗∗ -0.28∗∗ -0.06ns 0.01ns

LaneATT [50] - -0.57∗∗∗ -0.58∗∗∗ -0.34∗∗∗ -0.14ns - 0.08ns -0.09ns 0.11ns 0.12ns ns Not Significant (p > 0.05), ∗ p ≤ 0.05, ∗∗ p ≤ 0.01, ∗∗∗ p ≤ 0.001 tional metrics, SCNN should be preferred as the best performant model. This is consistent with the results in Table 1 and §4.1 since SCNN, UltraFast, and LaneATT show close performance in the conventional metrics (SCNN may have slight advantages in Comma2k19-LD). On the other hand, if we adopt E2E-LD, PolyLaneNet should be preferred since there is only a slight difference between the 4 lane detection methods in the benign scenarios and PolyLaneNet clearly outperforms the other methods in the attack scenarios.

The inconsistency between the E2E-LD and the conventional metrics can be more systematically quantified using

Pearson correlation coefficient r. Generally, the E2E-LD and the conventional metrics have strongly negative correlations (r ≤-0.55) with high statistical significance (p ≤ 0.001), meaning that some recent improvements in the conventional metrics may not have led to improvements in AD,
 but rather may have made it worse by overfitting to the metrics. SCNN, the segmentation approach, is the only one that does not use domain knowledge, e.g., lane lines are smooth lines (§2.1). This high degree of freedom in the model may lead to overfitting of the human annotations with noise.

Finally, we evaluate the parameters in the conventional metrics: α for the accuracy and β for F1 score. For α, we explore every 5 pixels from 5 pixels to 50 pixels. For β, we explore every 0.05 from 0.5 to 0.9. In the benign scenarios, (α = 20, β = 0.85) has the best correlation between the E2E-LD and F1 score. In the attack scenarios, (α = 50, β = 0.65) has the best correlation between the

E2E-LD and F1 score. However, the results are still similar to those using the original parameters: SCNN shows the highest accuracy; UltraFast has a higher F1 score than the others, but the correlation is still negative. Thus, such a naive parameter tuning does not resolve the limitations. of the conventional metrics.

### 4.3. Consistency of E2E-LD with PSLD
In this section, we evaluate the validity of PSLD as a per-frame surrogate metric of E2E-LD.

Evaluation Setup. We follow the same setup as in §4.2. 1 4 7 10 13 16 19



Figure 4. Pearson correlation coefficient r between E2E-LD and

PSLD when Tp is varied from 1 to 20 in the benign and attack scenarios. The red vertical lines are Tp with the largest average r. 1 4 7 10 13 16 19

Tp [frame]



Figure 5. PSLD for the 4 major lane detection models when Tp is varied from 1 to 20 frames in the benign and attack scenarios.

We generate the DRP attacks for 100 scenarios in the

Comma2k19-LD dataset with the same parameters. For the

PSLD, we obtain the ground truth waypoints by the following procedure. We generate a trajectory with the bicycle model and OpenPilot’s MPC by using the human driving trajectory as waypoints. We then use the generated trajectory as a ground-truth road center. While we can directly use the human driving trajectory as ground truth, human driving sometimes is not smooth and this approach can cancel the effect of motion models, which have differences from real vehicle dynamics. For the benign scenarios, we calculate the PSLD for each frame in the original human driving. For the attack scenarios, we use the frames synthesized by the method described in 3.1 instead of the original frames because the attacked trajectory and its camera frames are largely changed from the original human driving. For example, to obtain the PSLD at frame t = N, we simulate the trajectory until t = N − 1 and we then calculate the PSLD with the synthesized frame at t = N. 7

Table 4. Evaluation results of the E2E-LD and PSLD in the benign and attack scenarios. The format is the same as Table 6.

Benign Attack

E2E-LD [m] PSLD [m] E2E-LD [m] PSLD [m]

Metric

SCNN [40] 0.21 0.04 0.48 0.58

UltraFast [42] 0.18 0.03 0.58 0.62

PolyLaneNet [51] 0.20 0.03 0.38 0.42

LaneATT [50] 0.21 0.03 0.72 0.80

Corr.

SCNN [40] - 0.93∗∗∗ - 0.96∗∗∗

UltraFast [42] - 0.54∗∗∗ - 0.93∗∗∗

PolyLaneNet [51] - 0.49∗∗∗ - 0.97∗∗∗

LaneATT [50] - 0.38∗∗∗ - 0.95∗∗∗ ns Not Significant (p > 0.05), ∗ p ≤ 0.05, ∗∗ p ≤ 0.01, ∗∗∗ p ≤ 0.001

Results. Fig. 4 shows the Pearson correlation coefficient r between E2E-LD and PSLD when Tp is varied from 1 to 20 frames. As shown, the E2E-LD, PSLD has strong positive correlations in both benign and attack scenarios. In particular, there are significant correlations (>0.8) in the attack scenarios. This is because the direction of lateral deviation generally coincides with the attack direction. By contrast, in the benign scenarios, the vehicle drives around the road center with overshooting, and thus the direction of lateral deviation heavily depends on the initial states. Nevertheless, the PSLD has always high positive correlations with

E2E-LD (>0.2). In particular, SCNN has strong correlations (>0.8) with E2E-LD in all Tp. We consider that the high correlation can be due to the segmentation approach, which is the only method among the 4 methods that does not use the domain-specific knowledge the lane lines are generally smooth (§2.1). The detection of SCNN at the same location tends to be consistent across different frames, i.e.,

SCNN is less dependent on global information.

Finally, we explore the best Tp for PSLD to proxy E2ELD. As shown in Fig. 4, the average of the correlation coefficients of the 4 methods achieves the maximum at Tp = 10 in the benign scenarios and Tp = 5 in the attack scenarios respectively. We list the E2E-LD and PLSD with Tp = 10 and the corresponding r in Table 7. As shown, there are strong, statistically significant (p ≤ 0.001) positive correlations (≥ 0.38) between E2E-LD and PSLD in both cases.

The results strongly support the fact that PSLD can measure the performance of lane detection in ALCs based solely on the single camera frame and ground-truth road center geometry. We note that the PSLD is not so sensitive to the choice of Tp. As shown in Fig. 5, the magnitude relation of the 4 methods is generally consistent for all Tp.

## 5. Discussion
Alternative Metric Design. To improve the existing metrics, we explored other possible design choices. One of the most intuitive approaches is the L1 or L2 distance in the bird’s eye view. We evaluated the designs and confirmed that these metrics are still leading to erroneous judgment on downstream AD performance similar to the conventional metrics. Details are in Appendix F. We note that our metrics are specific to AD, the main downstream task of lane detection. For other downstream tasks, other metric designs can be more suitable.

Domain Shift. In this work, we use lane detection models pretrained on the TuSimple dataset and evaluate them on the Comma2k19-LD. To evaluate the impact of domain shift, we conduct further evaluation and confirm that our observations are generally consistent. Detailed results and discussions are in Appendix E.

Closed-loop Simulation. To obtain driving-oriented metrics, there are multiple parameters and design choices in the closed-loop simulation. In this study, we follow the parameters in the Comma2k19 datasets and select simple and popular designs, e.g., bicycle model and MPC. Meanwhile, we think that such design differences should only have minor effects on our observations because ALC, Level-2 driving automation, just follows the lane center line, which is designed to be smooth in normal roads.

Evaluation on Other Datasets. Our metrics are applicable to any dataset set that contains position data (e.g. GPS) and its camera frames, but ideally, velocity and ground-truth lane centers should be available. Such information is available in relatively new datasets such as [10, 20]. However, lane annotations are not directly available in the datasets and require considerable effort to obtain from map data and camera frames. To our knowledge, our Comma2k19-LD is so far the only dataset with both lane line annotation and driving information. We hope our work will facilitate further research to build datasets including them.

## 6. Conclusion
In this work, we design 2 new lane detection metrics, E2E-LD and PSLD, which can more faithfully reflect the performance of lane detection models in AD. Throughout a large-scale empirical study of the 4 major types of lane detection approaches on the TuSimple dataset and our new dataset Comma2k19-LD, we highlight critical limitations of the conventional metrics and demonstrate the high validity of our metrics to measure the performance in AD, the core downstream task of lane detection. In recent years, a wide variety of pretrained models have been used in many downstream application areas such as AD [1], natural language processing [22], and medical [21]. Reliable performance measurement is essential to facilitate the use of machine learning responsibly. We hope that our study will help the community make further progress in building a more downstream task-aware evaluation for lane detection.

## Acknowledgments
This research was supported in part by the NSF CNS1850533, CNS-1932464, CNS-1929771, CNS-2145493, and USDOT UTC Grant 69A3552047138. 8

## References
1. Baidu Apollo. https://github.com/ApolloAuto/ apollo. 8
2. Introduction to Self-Driving Cars. https://www.cour sera.org/learn/intro-self-driving-cars. 6
3. LGSVL Simulator: An Autonomous Vehicle Simulator. ht tps://github.com/lgsvl/simulator/. 4
4. Modeling a Vehicle Dynamics System. https://www. mathworks.com/help/ident/ug/modeling-avehicle-dynamics-system.html. 13
5. OpenPilot: Open Source Driving Agent. https://gith ub.com/commaai/openpilot. 3, 12
6. Super Cruise - Hands Free Driving — Cadillac Ownership. https://www.cadillac.com/world-of-cadill ac/innovation/super-cruise. 1, 3
7. Tesla Autopilot. https://www.tesla.com/autopi lot. 1, 3
8. Taxonomy and Definitions for Terms Related to Driving Automation Systems for On-Road Motor Vehicles. SAE International,(J3016), 2016. 2, 3
9. TuSimple Lane Detection Challenge. https://github .com/TuSimple/tusimple-benchmark/tree/m aster/doc/lane detection, 2017. 1, 2, 3, 5, 13, 14
10. Waymo Open Dataset: An Autonomous Driving Dataset. ht tps://www.waymo.com/open, 2019. 8
11. Lane Keeping Assist System Using Model Predictive Control. https://www.mathworks.com/help/mpc /ug/lane-keeping-assist-system-usingmodel-predictive-control.html, 2020. 3
12. Comma2k19 LD. https://www.kaggle.com/tkm22 61/comma2k19-ld, 2022. 6
13. Brandon Amos, Ivan Jimenez, Jacob Sacks, Byron Boots, and J Zico Kolter. Differentiable MPC for end-to-end planning and control. In NeurIPS, 2018. 13
14. Mayank Bansal, Alex Krizhevsky, and Abhijit S. Ogale. ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst. In RSS, 2019. 3
15. Karsten Behrendt and Ryan Soussan. Unsupervised Labeled Lane Marker Dataset Generation Using Maps. In IEEE International Conference on Computer Vision, 2019. 1, 3
16. Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al. End to End Learning for Self-Driving Cars. arXiv preprint arXiv:1604.07316, 2016. 3
17. Amardeep Boora, Indrajit Ghosh, and Satish Chandra. Identification of free flowing vehicles on two lane intercity highways under heterogeneous traffic condition. Transportation Research Procedia, 21:130–140, 2017. 12
18. Tom Brown, Dandelion Mane, Aurko Roy, Martin Abadi, and Justin Gilmer. Adversarial Patch. arXiv preprint arXiv:1712.09665, 2017. 4
19. Sergio Casas, Abbas Sadat, and Raquel Urtasun. MP3: A Unified Model to Map, Perceive, Predict and Plan. In CVPR, 2021. 3
20. Ming-Fang Chang, John Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, et al. Argoverse: 3D Tracking and Forecasting with Rich Maps. In CVPR, 2019. 8
21. Sihong Chen, Kai Ma, and Yefeng Zheng. Med3d: Transfer Learning for 3D Medical Image Analysis. arXiv preprint arXiv:1904.00625, 2019. 8
22. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL, 2019. 8
23. Richard C Dorf and Robert H Bishop. Modern Control Systems. Pearson, 2011. 4
24. Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An Open Urban Driving Simulator. In CoRL, 2017. 2
25. Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An Open Urban Driving Simulator. In CoRL, pages 1–16, 2017. 4
26. Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Florian Tramer, Atul Prakash, Tadayoshi Kohno, and Dawn Song. Physical Adversarial Examples for Object Detectors. In WOOT, 2018. 4
27. Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and Harnessing Adversarial Examples. arXiv preprint arXiv:1412.6572, 2014. 4
28. Richard Hartley and Andrew Zisserman. Multiple View Geometry in Computer Vision. Cambridge University Press, 2 edition, 2003. 4, 12, 13
29. Aharon Bar Hillel, Ronen Lerner, Dan Levi, and Guy Raz. Recent Progress in Road and Lane Detection: A Survey. Machine vision and applications, 25(3):727–745, 2014. 1
30. Yuenan Hou, Zheng Ma, Chunxiao Liu, Tak-Wai Hui, and Chen Change Loy. Inter-Region Affinity Distillation for Road Marking Segmentation. In CVPR, 2020. 3
31. Yuenan Hou, Zheng Ma, Chunxiao Liu, and Chen Change Loy. Learning Lightweight Lane Detection CNNs by Self Attention Distillation. In CVPR, 2019. 2, 3
32. Yen-Chang Hsu, Zheng Xu, Zsolt Kira, and Jiawei Huang. Learning to Cluster for Proposal-Free Instance Segmentation. In IJCNN, 2018. 2
33. Ashesh Jain, Luca Del Pero, Hugo Grimmett, and Peter Ondruska. Autonomy 2.0: Why Is Self-Driving Always 5 Years Away? arXiv preprint arXiv:2107.08142, 2021. 2
34. Pengfei Jing, Qiyi Tang, Yuefeng Du, Lei Xue, Xiapu Luo, Ting Wang, Sen Nie, and Shi Wu. Too Good to Be Safe: Tricking Lane Detection in Autonomous Driving with Crafted Perturbations. In USENIX Security, 2021. 2
35. Jason Kong, Mark Pfeiffer, Georg Schildbach, and Francesco Borrelli. Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design. In IV, 2015. 6, 10, 13
36. Jin-Woo Lee and Bakhtiar Litkouhi. A Unified Framework of the Automated Lane Centering/Changing Control for Motion Smoothness Adaptation. In ITSC, 2012. 3
37. Xiang Li, Jun Li, Xiaolin Hu, and Jian Yang. Line-CNN: End-to-End Traffic Line Detection with Line Proposal Unit. ITSC, 2019. 3 9
38. Lizhe Liu, Xiaohao Chen, Siyu Zhu, and Ping Tan. CondLaneNet: A Top-To-Down Lane Detection Framework Based on Conditional Convolution. In ICCV, 2021. 2, 3
39. Davy Neven, Bert De Brabandere, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool. Towards End-to-End Lane Detection: An Instance Segmentation Approach. In IV, 2018. 2
40. Xingang Pan, Jianping Shi, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Spatial as Deep: Spatial CNN for Traffic Scene Understanding. In AAAI, 2018. 1, 2, 3, 5, 6, 7, 8, 12, 13
41. Jonah Philion. FastDraw: Addressing the Long Tail of Lane Detection by Adapting a Sequential Prediction Network. In CVPR, 2019. 3
42. Qin, Zequn and Wang, Huanyu and Li, Xi. Ultra Fast Structure-Aware Deep Lane Detection. In ECCV, 2020. 3, 5, 6, 7, 8, 13
43. Zhan Qu, Huan Jin, Yang Zhou, Zhen Yang, and Wei Zhang. Focus on Local: Detecting Lane Marker From Bottom Up via Key Point. In CVPR, 2021. 3
44. Rajesh Rajamani. Vehicle Dynamics and Control. Springer Science & Business Media, 2011. 4, 12, 13
45. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In NeurIPS, 2015. 3
46. Richalet, J. and Rault, A. and Testud, J. L. and Papon, J. Paper: Model Predictive Heuristic Control. Automatica, 14(5):413–428, Sept. 1978. 4, 6, 12
47. Takami Sato, Junjie Shen, Ningfei Wang, Yunhan Jia, Xue Lin, and Qi Alfred Chen. Dirty Road Can Attack: Security of Deep Learning based Automated Lane Centering under Physical-World Attack. USENIX Security Symposium, 2021. 2, 4, 5, 6, 10, 12, 13
48. Harald Schafer, Eder Santana, Andrew Haden, and Riccardo Biasini. A Commute in Data: The comma2k19 Dataset. arXiv preprint arXiv:1812.05752, 2018. 6
49. Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing Properties of Neural Networks. In ICLR, 2014. 4
50. Lucas Tabelini, Rodrigo Berriel, Thiago M. Paix ao, Claudine Badue, Alberto Ferreira De Souza, and Thiago OliveiraSantos. Keep your Eyes on the Lane: Real-time Attentionguided Lane Detection. In CVPR, 2021. 2, 3, 5, 6, 7, 8, 13
51. Lucas Tabelini, Rodrigo Berriel, Thiago M Paixao, Claudine Badue, Alberto F De Souza, and Thiago Oliveira-Santos. Polylanenet: Lane Estimation via Deep Polynomial Regression. In ICPR, 2021. 2, 3, 5, 6, 7, 8, 13
52. Shiho Tanaka, Kenichi Yamada, Toshio Ito, and Takenao Ohkawa. Vehicle Detection Based on Perspective Transformation Using Rear-View Camera. Hindawi Publishing Corporation International Journal of Vehicular Technology, 9, 03 2011. 4, 12, 13
53. Jigang Tang, Songbin Li, and Peng Liu. A Review of Lane Detection Methods Based on Deep Learning. Pattern Recognition, 111:107623, 2021. 1, 3
54. Yuval Tassa, Nicolas Mansard, and Emo Todorov. Controllimited differential dynamic programming. In ICRA, 2014. 13
55. Daniel Watzenig and Martin Horn. Automated Driving: Safer and More Efficient Future Driving. Springer, 2016. 6
56. Seungwoo Yoo, Hee Seok Lee, Heesoo Myeong, Sungrack Yun, Hyoungwoo Park, Janghoon Cho, and Duck Hoon Kim. End-to-End Lane Marker Detection via Row-Wise Classification. In CVPR Workshops, 2020. 3
57. Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A Diverse Driving Dataset for Heterogeneous Multitask Learning. In CVPR, 2020. 2
58. Tu Zheng, Hao Fang, Yi Zhang, Wenjian Tang, Zheng Yang, Haifeng Liu, and Deng Cai. RESA: Recurrent Feature-Shift Aggregator for Lane Detection, 2020. 2
59. Tu Zheng, Hao Fang, Yi Zhang, Wenjian Tang, Zheng Yang, Haifeng Liu, and Deng Cai. RESA: Recurrent Feature-Shift Aggregator for Lane Detection. AAAI, 2021. 2 


### A. Detailed Settings of Lane Detection Metrics
Table 5 shows the parameters and input required to calculate each metric for lane detection. As shown, only E2ELD requires multiple frames for computation. For E2E-LD and PSLD, we adopt the kinematic bicycle model [35] to simulate the vehicle motion. The only parameter in the kinematic bicycle model is the wheelbase. We use WheelBase=2.65 meters, which is the wheelbase of Toyota RAV4 2017.

Table 5. Parameter and input of metrics for Lane Detection

Parameter Input Per-frame

Accuracy α X0 ✓

F1 score α, β X0 ✓

E2E-LD TE, WheelBase X0,...,XTE , C

PSLD Tp, WheelBase X0, C ✓

### B. Detailed Attack Implementation
We use the official implementation of the DRP attack [47]. We also use parameters that are reported to have the best balance between effectiveness and secrecy: the learning rate is 10−2 , the regularization parameter λ is 10−3 , and the perturbable area ratio (PAR) is 50%. We run 200 iterations to generate the patch in all experiments.

### C. Details of Comma2k19-LD dataset
Fig. 6 shows the first frame of all 100 scenarios and its lane line annotations. For annotation, human annotators mark the lane line points and check if the linear interpolation results of the markings align with the lane line information. To convert the annotations to the TuSimple dataset format, we sample points every 10 pixels in the y-axis from the interpolated results. 10

Figure 6. The first frame of all 100 scenarios and its lane line annotations (green line)

### D. Adaptation to TuSimple Challenge Camera
Frames Geometory

In the evaluation of the comma2k19-LD dataset, we use the same pretrained models trained on the TuSimple Challenge training dataset. To deal with the differences in the datasets, we convert the camera frames in the comma2k19-

LD dataset to have geometry similar to the camera frames in the TuSimple challenge dataset. Fig. 7 illustrates the overview of the conversion. We remove the surrounding area and use only the central part of the Comma2k19-LD camera frame to have the same sky-ground area ratio and the same lane occupation ratio in the image width with the ones in the TuSimple dataset.

### E. Evaluation of the Domain Shift Effect
In this study, we use lane detection models pretrained on the TuSimple dataset and evaluate them on the

Comma2k19-LD dataset. Although both datasets are similar driver’s view images, there can be some domain shifts between them. To understand the impact, we trained the 4 models with another 100 scenarios extracted from the

Comma2k19 dataset. We run 10 epochs with the data on top of the models pretrained on the TuSimple dataset. For the lane line labels, we use OpenPilot’s lane detection results in the dataset. We conduct the same evaluation in §4.2

Sky area: 140px

Lane occupies 78% of the width.

Ground area: 580px

Remove hood area

Remove sky are to make the sky-ground area ratio 140:580.

Remove the side areas so that the lane occupies 78% of the width. (a) TuSimple camera frame (b) Comma2k19 camera frame

Figure 7. Overview of adapting the camera frames in

Comma2k19-LD dataset to the camera frame in the TuSimple

Challenge dataset. We remove the surrounding area and use only the central part of the Comma2k19 camera frame to ensure that the comma2k19-LD camera frames have a geometry similar to that of the TuSimple challenge camera frames. and §4.3 of the main paper. As shown in Table 6 and Table 7, the observations are consistent: SCNN outperforms in the conventional metrics; PolyLaneNet is the most robust in attack scenarios. The Pearson correlation coefficients show almost the same results as the ones in §4.2 and §4.3 that the conventional metrics have strong negative correlations in the benign scenarios and the correlations in the attack scenarios are not statistically significant.

However, the E2E-LD in the attack scenarios are gen11 erally higher than the results in §4.2 and §4.3 of the main paper while the E2E-LD in the benign scenarios is generally lower. This indicates that this additional fine-tuning improves the performance in the benign scenarios, but it harms the robustness against adversarial attacks.

### F. Alternative metric design
To improve the conventional metrics, one of the most intuitive approaches is the L1 or L2 distance in the bird’s eye view because they do not suffer from the problem of the ill parameters discussed in §2.2, and lane detection results from a bird’s eye view may be a more adequate to measure of drivability than detection results from a front camera. We actually have considered such metrics before, but we did not finally choose them because, without some form of control simulation, we find it fundamentally nontrivial to accurately predict the combined effects of detection errors at different lane line positions and with different error amounts on the downstream AD driving. This can be concretely shown in Table 8. As shown, both such 3D-L1 and 3D-L2 distance metrics have considerably lower correlation coefficient r with E2E-LD compared to our PSLD.

They are indeed better than conventional accuracy and F1 score metrics. However, they are still leading to erroneous judgment on downstream AD performance similar to the accuracy and F1 score: e.g., PolyLaneNet is 2nd-worst based on 3D-L1/L2 distance metrics in the attack scenarios, but in E2E-LD it is the best. With our PSLD, such judgment is strictly consistent with E2E-LD (Table 7). One reason we observe is that the 3D-L1/L2 metrics can be greatly biased by farther points; those points by design have much less impact on the downstream AD control, but suffer from more detection errors (due to the far distance). One thought is to assign smaller weights to farther points, but how to systematically decide such weights without any form of control simulation is fundamentally nontrivial. Additionally, such a weight-based design can still be fundamentally limited in achieving sufficient AD control relating capabilities.

### G. Details of OpenPilot ALC and its integration with lane detection models
In this section, we explain the details of OpenPilot

ALC [5] and the details of its integration with the 4 lane detection models we evaluate in this study. As described in [47], the OpenPilot ALC system consists of 3 steps: lane detection, lateral control, and vehicle actuation.

#### G.1. Lane detection
The image frame from the front camera is input to the lane detection model in every frame (20Hz). Since the original OpenPilot lane detection model is a recurrent neural network model, the recurrent input from the previous frame is fed to the model with the image. All 4 models used in this study do not have a recurrent structure, i.e., they detect lanes only in the current frame. This is because the

TuSimple Challenge has a runtime limit of less than 200 ms for each frame. Another famous dataset, CULane [40], does not provide even continuous frames. In autonomous driving, the recurrent structure is a reasonable choice since past frame information is always available. Hence, the runtime calculation latency imposed in the TuSimple challenge is one of the gaps between the practicality for autonomous driving and the conventional evaluation.

#### G.2. Lateral control
Based on the detected lane line, the lateral control decides steering angle decisions to follow the lane center (i.e., the desired driving path or waypoints) as much as possible.

The original OpenPilot model outputs 3 line information: left lane line, right lane line, and driving path. The desired driving path is calculated as the average of the driving path and the center line of the left and right lane lines. The steering decision is decided by the model predictive control (MPC) [46]. The detected lane lines are represented in the bird’s-eye-view (BEV) because the steering decision needs to be decided in a world coordinate system.

On the contrary, all 4 models used in this study detect the lane lines in the front-camera view. We thus project the detected lane lines into the BEV space with perspective transformation [28, 52]. The transformation matrix for this projection is created manually based on road objects such as lane markings, and then calibrated to be able to drive in a straight lane. We create the transformation matrix for each scenario as the position of the camera and the tilt of the ground are different for each scenario. The desired driving path is calculated by the average of the left and right lane lines and fed to the MPC to decide the steering angle decisions.

In addition to the desired driving path, the MPC receives the current speed and steering angle to decide the steering angle decisions. For the steering angle, we use the human driver’s steering angle in the Comma2k19 dataset in the first frame. In the following frames, the steering angle is updated by the kinematic bicycle model [44], which is the most widely-used motion model for vehicles. For the vehicle speed, we use the speed of human driving in the in the comma2k19 dataset as we assume that the vehicle speed is not changed largely in the free-flow scenario, in which a vehicle has at least 5–9 seconds clear headway [17].

#### G.3. Vehicle actuation
The step sends steering change messages to the vehicle based on the steering angle decisions. In OpenPlot, this step operates at 100 Hz control frequency. As the lane detection and lateral control outputs the steering angle decisions in 20 12

Table 6. Evaluation results of the E2E-LD and the conventional metrics, accuracy and F1 in the benign and attack scenarios. For each metric, the corresponding Pearson correlation coefficient with E2E-LD in the bottom rows. The bold and underlined letters indicate the highest and lowest performance or correlation, respectively.

Benign Attack

Original Parameters (α = 20, β = 0.85)

Original Parameters (α = 20, β = 0.85)

E2E-LD [m] Accuracy F1 E2E-LD [m] Accuracy F1

Metric

SCNN [40] 0.20 0.93 0.84 0.52 0.67 0.30

UltraFast [42] 0.18 0.92 0.84 0.62 0.49 0.16

PolyLaneNet [51] 0.13 0.93 0.86 0.54 0.62 0.33

LaneATT [50] 0.14 0.93 0.85 0.71 0.51 0.12

Corr.

SCNN [40] - -0.65∗∗∗ -0.51∗∗∗ - -0.09ns -0.04ns

UltraFast [42] - -0.63∗∗∗ -0.60∗∗∗ - 0.14ns 0.07ns

PolyLaneNet [51] - -0.32∗∗∗ -0.62∗∗∗ - 0.14ns 0.04ns

LaneATT [50] - -0.57∗∗∗ -0.26∗∗∗ - -0.02ns -0.06ns ns Not Significant (p > 0.05), ∗ p ≤ 0.05, ∗∗ p ≤ 0.01, ∗∗∗ p ≤ 0.001

Table 7. Evaluation results of the E2E-LD and PSLD in the benign and attack scenarios. The format is the same as Table 6.

Benign Attack

E2E-LD [m] PSLD [m] E2E-LD [m] PSLD [m]

Metric

SCNN [40] 0.20 0.04 0.52 0.61

UltraFast [42] 0.18 0.02 0.62 0.66

PolyLaneNet [51] 0.13 0.02 0.54 0.55

LaneATT [50] 0.14 0.03 0.71 0.82

Corr.

SCNN [40] - 0.93∗∗∗ - 0.93∗∗∗

UltraFast [42] - 0.60∗∗∗ - 0.99∗∗∗

PolyLaneNet [51] - 0.65∗∗∗ - 0.99∗∗∗

LaneATT [50] - 0.55∗∗∗ - 0.78∗∗∗ ns Not Significant (p > 0.05), ∗ p ≤ 0.05, ∗∗ p ≤ 0.01, ∗∗∗ p ≤ 0.001

Table 8. Pearson correlation coefficient r with E2E-LD. 3D-L1/L2 denote the L1/L2 distances in 3D space following Reviewer 1’s suggestion. Bold and underline denote highest and lowest scores.

Benign Attack

PSLD (ours) 3D-L1 3D-L2 PSLD 3D-L1 3D-L2

SCNN 0.93 0.71 0.65 0.96 0.38 0.34

UltraFast 0.54 0.24 0.19 0.93 0.24 0.21

PolyLaneNet 0.49 0.47 0.44 0.97 0.33 0.38

LaneATT 0.38 0.23 0.17 0.95 0.23 0.23

Average 0.59 0.41 0.36 0.95 0.29 0.29

Hz, the vehicle actuation sends 5 messages every steering angle decision. The steering changes are limited to a maximum value due to the physical constraints of vehicle and for stable and for stability and safety. In this study, we limit the steering angle change to 0.25◦ following prior work, which is the steering limit for production ALC systems [47].

We update the vehicle states with the kinematic bicycle model based on the steering change. Note that like all motion models, the kinematic bicycle model does have approximation errors to the real vehicle dynamics [35]. However, more accurate motion models require more complex parameters such as vehicle mass, tire stiffness, and air resistance [4]. In this study, since our focus is on understanding the impact of lane detection model robustness on end-to-end driving, the most widely-used kinematic bicycle model is a sufficient choice for simulating closed-loop control behaviors.

### H. Additional Discussions and Results
#### H.1. Additional Discussions
Ground-Truth Road Center. We obtain the ground truth waypoints based on the human driving traces. Ideally, the waypoints should be obtained by measuring roads.

However, since this study focuses on the general trends of the 4 lane detection approaches, we consider that the impact of this factor should not have a major effect. If you want to use PSLD to capture more subtle differences between models, the ground truth should be more accurate.

Differentiable PSLD Regularization. We show that

PSLD works as a good surrogate for E2E-LD. Next, we may want to minimize this metric directly in the model training.

Since the only non-differentiable computation in PSLD is the lateral controller, we can replace this part with a differentiable controller [13, 54] and incorporate it as a regularization term in the loss function for training. Detailed study of this problem is left to future work.

#### H.2. TuSimple Challenge Dataset
Fig. 8 shows the examples of lane detection results and the accuracy metric in benign scenarios on the TuSimple

Challenge dataset [9]. The limitations of the conventional metrics can be found in benign cases as well. As shown,

SCNN has always higher accuracy than PolyLaneNet (at most 18% edge). Such a large leading edge is across the dataset as in Table 2 (89% vs 72% in Accuracy, 75% vs 50% in F1 score). However, for downstream AD their performances are almost the same, with PolyLaneNet actually slightly better (Table 6 of the main paper).

#### H.3. Comma2k19 LD Dataset
We synthesize font-camera frames with a vehicle motion model [44] and perspective transformation [28, 52]. 13

SCNN UltraFast PolyLaneNet

Accuracy: 91% Accuracy: 100% Accuracy: 73%

Accuracy: 85% Accuracy: 90% Accuracy: 83%

Accuracy: 88% Accuracy: 100% Accuracy: 76%

LaneATT

Accuracy: 100%

Accuracy: 90%

Accuracy: 100%

Figure 8. Examples of lane detection results and the accuracy metric in benign scenarios on TuSimple Challenge dataset [9]. As shown, the conventional accuracy metric does not necessarily indicate drivability if used in autonomous driving

Fig. 9, 10, 11, 12, show the first 20 frames under attack and their detection results of the 4 lane detection methods, respectively. As shown, the generated images are generally complete and the distortion is very slight.

Figure 9. The first 20 frames (from left-top to right-bottom) of an attack scenario on SCNN. The vehicle is deviating to right due to the attack.

Figure 10. The first 20 frames (from left-top to right-bottom) of an attack scenario on UltraFast. The vehicle is deviating to right due to the attack.

Figure 11. The first 20 frames (from left-top to right-bottom) of an attack scenario on PolyLaneNet. The vehicle is deviating to right due to the attack.

Figure 12. The first 20 frames (from left-top to right-bottom) of an attack scenario on LaneATT. The vehicle is deviating to right due to the attack. 14
