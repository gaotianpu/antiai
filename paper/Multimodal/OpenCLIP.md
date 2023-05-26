# Reproducible scaling laws for contrastive language-image learning
用于对比语言图像学习的可复现缩放定律 2022.12.14 https://arxiv.org/abs/2212.07143

## Abstract
Scaling up neural networks has led to remarkable performance across a wide range of tasks. Moreover, performance often follows reliable scaling laws as a function of training set size, model size, and compute, which offers valuable guidance as large-scale experiments are becoming increasingly expensive. However, previous work on scaling laws has primarily used private data & models or focused on uni-modal language or vision learning. To address these limitations, we investigate scaling laws for contrastive language-image pre-training (CLIP) with the public LAION dataset and the open-source OpenCLIP repository. Our large-scale experiments involve models trained on up to two billion image-text pairs and identify power law scaling for multiple downstream tasks including zero-shot classification, retrieval, linear probing, and end-to-end fine-tuning. We find that the training distribution plays a key role in scaling laws as the OpenAI and OpenCLIP models exhibit different scaling behavior despite identical model architectures and similar training recipes. We open-source our evaluation workflow and all models, including the largest public CLIP models, to ensure reproducibility and make scaling laws research more accessible. Source code and instructions to reproduce this study will be available at https://github.com/LAION-AI/scaling-laws-openclip

扩大神经网络的规模已经在广泛的任务中取得了显著的性能。此外，性能通常遵循可靠的缩放定律，作为训练集大小、模型大小和计算的函数，随着大规模实验变得越来越昂贵，这提供了有价值的指导。然而，以前关于缩放定律的工作主要使用私人数据和模型，或者专注于单模态语言或视觉学习。为了解决这些限制，我们研究了对比语言图像预训练(CLIP)与公共LAION数据集和开源OpenCLIP存储库的缩放定律。我们的大规模实验包括在多达20亿个图像-文本对上训练模型，并确定多个下游任务的幂律缩放，包括零样本分类、检索、线性探测和端到端微调。我们发现，训练分布在缩放定律中起着关键作用，因为尽管模型架构相同，训练配置相似，但OpenAI和OpenCLIP模型表现出不同的缩放行为。我们开放了我们的评估工作流程和所有模型，包括最大的公共CLIP模型，以确保可复现，并使缩放定律研究更容易获得。复x现本研究的源代码和说明在 https://github.com/LAION-AI/scaling-laws-openclip

## 1 Introduction
Large pre-trained models now achieve state-of-the-art performance on a wide range of tasks. In particular, large models have led to substantial advances in speech [56], language [17, 57, 8, 28], vision [38, 84], and multi-modal language-vision settings [55, 33, 54, 59, 62]. A key ingredient in these breakthroughs has been self- or weakly-supervised learning, which enabled the use of Internetharvested training sets and reduced the need for explicit human annotation of training data. In addition, recent pre-trained models relied on increasing the compute, model, and data scale by orders of magnitude.

经过预训练的大型模型现在可以在各种任务中实现最先进的性能。特别是，大型模型在语音[56]、语言[17，57，8，28]、视觉[38，84]和多模态语言视觉设置[55，33，54，59，62]方面取得了实质性进展。这些突破的一个关键因素是自我或弱监督学习，这使得能够使用互联网支持的训练集，并减少了对训练数据的明确人工标注的需要。此外，最近的预训练模型依赖于将计算、模型和数据规模增加几个数量级。

When varying model size, compute amount, and data quantity, several papers have empirically observed that both pre-training loss and downstream task performance reliably improve with scale. Specifically, researchers have postulated scaling laws in the form of power law relationships between model performance and model compute, or data scale [35, 73, 61, 84]. Such scaling laws allow practitioners to predict model performance for a given model and compute scale, extrapolate to larger scales, and can be used to determine pre-training regimes that obtain optimal model performance for a fixed amount of compute [35, 28].

当改变模型大小、计算量和数据量时，几篇论文已经实证观察到，预训练损失和下游任务性能都会随着规模的扩大而可靠地提高。具体而言，研究人员以模型性能和模型计算或数据尺度之间的幂律关系的形式假设了尺度定律[35，73，61，84]。这种缩放定律允许从业者预测给定模型和计算规模的模型性能，外推到更大的规模，并可用于确定预训练机制，以获得固定计算量的最佳模型性能[35，28]。

||Data| Arch.| ImageNet| VTAB+| COCO
---|---|---|---|---|---
CLIP [55]| WIT-400M |L/14| 75.5 |55.8| 61.1
Ours |LAION-2B |L/14 |75.2 |54.6 |71.1
Ours | LAION-2B |H/14 |78.0 |56.4 |73.4

Table 1: We study the scaling behavior of large CLIP models using fully open-source training code and data. All models in our investigation will be made available and include the largest public CLIP models. This table shows zero-shot performance at 224 pixel resolution, displaying accuracy on ImageNet [15], average accuracy on 35 VTAB+ datasets [65, 85], and image retrieval recall at 5 on MS-COCO image retrieval [46].
表1：我们使用完全开源的训练代码和数据研究了大型CLIP模型的伸缩行为。我们调查中的所有模型都将提供，其中包括最大的公共CLIP模型。此表显示了224像素分辨率下的零样本性能，ImageNet上的显示精度[15]，35个VTAB+数据集上的平均精度[65，85]，MS-COCO图像检索上的图像检索召回率为5[46]。

So far, the literature on empirical scaling laws has focused on language-only [35, 73, 28] or visiononly models [83, 25, 61]. In the multimodal domain of language and vision, contrastive language-image models such as CLIP [55] have recently achieved large performance gains in zero-image classification, for instance improving zero-shot ImageNet accuracy from the prior state-of-the-art of 12% to 76%. Moreover, these models demonstrate unprecedented robustness to distribution shifts compared to prior supervised models [71, 55, 78]. However, there is currently no systematic investigation for scaling trends in contrastive language-image learning. One substantial challenge in this direction is that until recently, there were no datasets of sufficiently large scale openly available for the research community to undertake such experiments.

到目前为止，关于经验标度定律的文献主要集中在纯语言[35，73，28]或纯视觉模型[83，25，61]。在语言和视觉的多模态领域，CLIP等对比语言图像模型最近在零图像分类方面取得了巨大的性能提升，例如，将零样本ImageNet的准确率从先前的12%提高到76%。此外，与先前的监督模型相比，这些模型对分布变化表现出前所未有的稳健性[71，55，78]。然而，目前还没有系统地研究对比语言图像学习中的缩放趋势。在这个方向上的一个重大挑战是，直到最近，还没有足够大规模的数据集可供研究界进行此类实验。

In this work, we conduct a scaling laws study for contrastive language-vision learning by utilizing the recently released LAION-5B [65] dataset of 5 billion image-text pairs. To ensure that our experiments are fully reproducible, we use the open source OpenCLIP [32] code to train CLIP models while varying model, data, and samples seen. We evaluate our CLIP models on several downstream tasks, including zero-shot classification, image retrieval, and fine-tuning via linear probing and end-to-end optimization. We observe a consistent increase in performance when scaling model, data, and compute, and derive scaling laws of power law form across different downstream tasks (Figure 1a, 1b). Interestingly, when comparing our OpenCLIP and OpenAI’s original CLIP models, we find larger scaling coefficients for OpenCLIP models on zero-shot retrieval, while OpenAI CLIP models show stronger scaling for zero-shot classification. Table 1 shows two of our models and their results on image classification and retrieval benchmarks.

在这项工作中，我们利用最近发布的50亿个图像-文本对的LAION-5B[65]数据集，对对比语言视觉学习进行了比例律研究。为了确保我们的实验是完全可复制的，我们使用开源的OpenCLIP[32]代码来训练CLIP模型，同时改变所看到的模型、数据和样本。我们在几个下游任务上评估了CLIP模型，包括零样本分类、图像检索和通过线性探测和端到端优化进行微调。我们观察到，当缩放模型、数据和计算时，性能会持续提高，并推导出不同下游任务的幂律形式的缩放定律(图1a、1b)。有趣的是，当比较我们的OpenCLIP和OpenAI的原始CLIP模型时，我们发现OpenCLIP模型在零样本检索上具有更大的缩放系数，而OpenAI CLIP模型对于零样本分类显示出更强的缩放系数。表1显示了我们的两个模型及其在图像分类和检索基准方面的结果。

We hypothesize that the training dataset is responsible for the task-dependent differences in scaling behavior between the OpenCLIP and OpenAI models. Our experiments have used the same ViT architectures as the OpenAI models, and the training recipes are largely matched. The main difference in training recipes is the batch size due to different compute environments, and our experiments with varying batch sizes suggest that the batch size changes do not explain the change in scaling trends.

我们假设训练数据集是OpenCLIP和OpenAI模型之间缩放行为的任务依赖性差异的原因。我们的实验使用了与OpenAI模型相同的ViT架构，并且训练配置基本匹配。训练配置的主要差异是由于不同的计算环境导致的批量大小，我们对不同批量大小的实验表明，批量大小的变化并不能解释缩放趋势的变化。

Overall our findings highlight the design of pre-training datasets as an important direction to further improve image-text models. Dataset designers should measure scaling behavior so that the generalization capabilities of image-text models can continue to improve as we increase model size and the amount of compute. Moreover, pre-training datasets should be evaluated on a broad range of downstream tasks because model scaling can differ substantially by task with different pre-training sources leading to different scaling behavior by task. We hope that our open-source and reproducible

总的来说，我们的研究结果强调了预训练数据集的设计是进一步改进图像-文本模型的重要方向。数据集设计者应该测量缩放行为，以便随着模型大小和计算量的增加，图像-文本模型的泛化能力能够不断提高。此外，应在广泛的下游任务上评估预训练数据集，因为不同的预训练源导致不同任务的缩放行为，模型缩放可能因任务而异。我们希望我们的开源和可复制

CLIP-WIT (a) Relationship between total training compute and zero-shot classification performance on downstream tasks. Left: ImageNet performance. Right: average performance on five ImageNet robustness datasets (ImageNet-V2 [60], ImageNet-R [22], ImageNet-Sketch [75], ObjectNet [5], and ImageNet-A [24]). Scaling model size, data size, and samples seen leads to better performance on zero-shot classification. Models trained on OpenAI’s WebImageText (WIT) show a stronger scaling than models trained on LAION. 10 11 10 12

CLIP-WIT(a)总训练计算与下游任务零样本分类性能之间的关系。左图：ImageNet性能。右图：五个ImageNet稳健性数据集(ImageNet-V2[60]、ImageNet-R[22]、ImageNet Sketch[75]、ObjectNet[5]和ImageNet-A[24])的平均性能。缩放模型大小、数据大小和所看到的样本可以提高零样本分类的性能。在OpenAI的WebImageText(WIT)上训练的模型显示出比在LAION上训练的模型更强的伸缩性。10 11 10 12

CLIP-WIT (b) Relationship between total training compute and zero-shot image retrieval performance on MS-COCO (Left) and Flickr30K (Right). Scaling model size, data size, and samples seen leads to better performance on zero-shot image retrieval. Interestingly, in contrast to zero-shot classification (Figure 1a), models trained on LAION show a stronger scaling trend than OpenAI CLIP models trained on OpenAI’s WebImageText (WIT) dataset.

CLIP-WIT(b)MS-COCO(左)和Flickr30K(右)上的总训练计算和零样本图像检索性能之间的关系。缩放模型大小、数据大小和看到的样本可以提高零样本图像检索的性能。有趣的是，与零样本分类(图1a)相比，在LAION上训练的模型显示出比在OpenAI的WebImageText(WIT)数据集上训练的OpenAI CLIP模型更强的缩放趋势。

Figure 1: Relationship between total training compute and performance in zero-shot classification (1a) and retrieval (1b). We fit a power-law on the Pareto frontier of the available models. Since total compute budgets (measured in GMAC) of different trained models are not exactly aligned, we divide the total compute scale into bins and select the best model performance from each bin. scaling trends offer concrete starting points for improving current image-text datasets and models.

图1：总训练计算与零样本分类(1a)和检索(1b)性能之间的关系。我们在可用模型的Pareto边界上拟合了一个幂律。由于不同训练模型的总计算预算(以GMAC衡量)并不完全一致，我们将总计算规模划分为多个区间，并从每个区间中选择最佳的模型性能。缩放趋势为改进当前的图像文本数据集和模型提供了具体的起点。

## 2 Background and related work
Scaling laws for generalization and transfer. Strong empirical evidence that increasing model or data scale is beneficial was initially studied in the context of deep learning and computer vision [70, 26]. For instance, in [26], the power law relation between scale and model performance was highlighted. Empirical work stimulated theoretical studies that provided justification for the observed generalization boost with scale, investigating generalization error in overparameterized networks in the interpolation regime [6, 9].

泛化和迁移的标度定律。最初在深度学习和计算机视觉的背景下研究了增加模型或数据规模是有益的有力经验证据[70，26]。例如，在[26]中，强调了规模和模型性能之间的幂律关系。经验工作刺激了理论研究，为观察到的随尺度的泛化增广提供了理由，研究了插值机制下过参数化网络的泛化误差[6，9]。

Early empirical studies focused on the effect of training scale on upstream performance, measuring the test loss from the same distribution used for training. Subsequent studies of large language models such as GPT-3 [8] demonstrated broad generalization capabilities in models with substantially larger scale. Moreover, neural scaling laws of the power law form were derived for language models, connecting model, data, and training compute scale to performance [35, 73, 28]. This also allowed accurate prediction of model performance at larger scales, and researchers were able to determine the scale parameters for achieving optimal performance given a fixed amount of compute [28, 39]. Scaling law studies were then also studied in the vision domain [61, 84], also observing a power law dependency of performance on scale.

早期的实证研究侧重于训练规模对上游绩效的影响，测量了用于训练的相同分布中的测试损失。随后对GPT-3[8]等大型语言模型的研究表明，在规模更大的模型中具有广泛的泛化能力。此外，还为语言模型推导了幂律形式的神经缩放定律，将模型、数据和训练计算规模与性能联系起来[35，73，28]。这也允许在更大的尺度上准确预测模型性能，并且研究人员能够在给定固定计算量的情况下确定实现最佳性能的尺度参数[28，39]。随后，还对视觉领域的标度定律研究进行了研究[61，84]，还观察到了性能对标度的幂律依赖性。

Scaling law studies were also conducted for transfer and out-of-distribution performance [35, 73, 84]. In these studies, researchers observed that performance on downstream tasks benefits from increasing model, data, and training compute scale [38, 8, 35, 84]. Interestingly, upstream performance does not always correlate with downstream performance [73, 72]. Since downstream performance most accurately reflects a practical use cases, examining scaling behavior on downstream tasks is increasingly important. Recent work has also studied the effect of scale on other model characteristics, such as performance after pruning and compression [64, 11] and on susceptibility to catastrophic forgetting [58].

还对迁移和分布外性能进行了标度律研究[35，73，84]。在这些研究中，研究人员观察到，下游任务的性能得益于模型、数据和训练计算规模的增加[38，8，35，84]。有趣的是，上游性能并不总是与下游性能相关[73，72]。由于下游性能最准确地反映了实际用例，因此检查下游任务的伸缩行为变得越来越重要。最近的工作还研究了规模对其他模型特征的影响，如修剪和压缩后的性能[64，11]以及对灾难性遗忘易感性的影响[58]。

Scaling up language-vision learning. Learning from very large amounts of weakly aligned imagetext pairs has led to the development of models with broad generalization capabilities. Notably, work on contrastive language-image pre-training (CLIP [55]) showed dramatic improvement compared to the previous state-of-the-art in zero-shot transfer and unprecendented robustness to distribution shift [71, 48, 51, 18]. The success of the initial CLIP study, which used a private WIT-400M image-text pairs dataset and ViT-L/14 as the largest scale vision encoder, motivated further developments and numerous extensions that increased model and data scale. ALIGN [33] used a private dataset of 1.8B text-image pairs and a large EfficientNet-L2 as an image encoder. BASIC [54] employed a large CoAttNet-7 model with 2.4B parameters for the image encoder, also further increasing dataset size up to 6.6B image-text pairs, using supervised visual encoder pre-training and private datasets (ALIGN and JFT-5B). LiT [86] used a private dataset of 4B image-text samples for contrastive learning on a total of 18B samples, scaling the visual encoder up to ViT-g/14, which was pre-trained in a supervised manner using another private dataset (JFT-3B). CoCa [81] used ViT-g/14 as a visual encoder and both the ALIGN and JFT private datasets, and an additional text captioning loss based on autoregressive language modeling during pre-training. LiMoE [49] trained a sparse mixture-of-experts (MoE) single tower architecture that share a common backbone for both vision and text using both private 3.6B image-text data from LiT and JFT-4B [84], obtaining a ViT H/14 model at the largest scale. Flamingo [3] uses a large private interleaved image-text dataset, using NFNet-F6 as a visual encoder while scaling up the text encoder from 1.4B to 70B parameters. PaLI [12] trained a multi-language multi-task text-image model using ViT-e (4B parameters) as a visual encoder and mT5-XXL (13B parameters) as a text encoder, trained on a private dataset (WebLI) with 29B image-text pairs. While these studies already show clear merits of scaling up, they do not conduct a thorough scaling investigation by systematically scaling model, data and, training compute. Moreover, most studies involve a customized multi-stage training procedure, where encoders may be pre-trained separately with uni-modal losses, and then tuned further with a contrastive image-text 4 loss, while also potentially freezing one of the encoders [54, 86]. This makes it difficult to derive conclusions about the effect of scale as pre-training procedures are heterogeneous. In addition, the private nature of the employed datasets impairs reproduction and validation of the results, especially in cases where pre-trained models are also not publicly available.

扩大语言视觉学习。从大量弱对齐的图像-文本对中学习，导致了具有广泛泛化能力的模型的开发。值得注意的是，对比语言-图像预训练(CLIP[55])的工作与之前的零样本迁移技术相比有了显著改进，并且对分布偏移的稳健性前所未有[71，48，51，18]。最初的CLIP研究使用了私人的WIT-400M图像-文本对数据集和ViT-L/14作为最大规模的视觉编码器，该研究的成功推动了进一步的开发和大量扩展，增加了模型和数据规模。ALIGN[33]使用1.8B文本图像对的私有数据集和大型EfficientNet-L2作为图像编码器。BASIC[54]使用了一个具有2.4B参数的大型CoAttNet-7模型作为图像编码器，还使用监督视觉编码器预训练和私有数据集(ALIGN和JFT-5B)将数据集大小进一步增加到6.6B的图像-文本对。LiT[86]使用4B图像文本样本的私有数据集对总共18B个样本进行对比学习，将视觉编码器缩放到ViT-g/14，该ViT-g/114是使用另一个私有数据集(JFT-3B)以监督方式预训练的。CoCa[81]使用ViT-g/14作为视觉编码器，同时使用ALIGN和JFT私有数据集，以及预训练期间基于自回归语言建模的额外文本字幕损失。LiMoE[49]使用来自LiT和JFT-4B[84]的私人3.6B图像文本数据，训练了一种稀疏的专家混合(MoE)单塔架构，该架构共享视觉和文本的公共主干，从而获得了最大规模的ViT H/14模型。Flamingo[3]使用大型私人交错图像-文本数据集，使用NFNet-F6作为视觉编码器，同时将文本编码器的参数从1.4B放大到70B。PaLI[12]使用ViT-e(4B参数)作为视觉编码器，mT5 XXL(13B参数)作为文本编码器，在具有29B图像-文本对的私有数据集(WebLI)上训练多语言多任务文本图像模型。虽然这些研究已经表明了扩展的明显优点，但它们并没有通过系统地扩展模型、数据和训练计算来进行彻底的扩展调查。此外，大多数研究都涉及定制的多阶段训练程序，其中编码器可以单独进行单模态损失的预训练，然后通过对比图像文本4损失进行进一步调整，同时也可能冻结其中一个编码器[54，86]。这使得很难得出关于规模效应的结论，因为预训练程序是异质的。此外，所用数据集的私有性损害了结果的再现和验证，尤其是在预先训练的模型也无法公开的情况下。

Open large-scale language-vision datasets. Conducting scaling law studies requires sufficiently large pre-training datasets. Earlier efforts to provide open image-text datasets like MS-COCO [46], Visual Genome [42], YFCC-100M [74], Conceptual Captions CC3M and CC12M [67, 30] do not match the current scale of private data used to train large-scale language vision models. More recently, larger image-text datasets have been collected from Common Crawl [1]. The resulting datasets, LAION-400M [66] and LAION-5B [65] are publicly available, enabling training language-vision models at larger scale [63, 49, 27]. Using the LAION toolset [65], it also became possible to construct additional open datasets, such as COYO-700M [10].

开放大规模的语言视觉数据集。进行缩放定律研究需要足够大的预训练数据集。早期提供开放图像文本数据集的努力，如MS-COCO[46]、视觉基因组[42]、YFCC-100M[74]、概念标题CC3M和CC12M[67，30]，与用于训练大规模语言视觉模型的私人数据的当前规模不匹配。最近，从Common Crawl[1]中收集了更大的图像文本数据集。由此产生的数据集LAION-400M[66]和LAION-5B[65]是公开的，能够在更大范围内训练语言视觉模型[63，49，27]。使用LAION工具集[65]，还可以构建额外的开放数据集，如COYO-700M[10]。

## 3 Datasets and Methods 数据集和方法
### 3.1 Open large-scale datasets LAION-400M/2B 开放式大型数据集LAION-400M/2B
We use the LAION-400M [66] and LAION-5B [65] datasets which are open, public image-text datasets validated by the pre-training of state-of-the art multi-modal models such as CLIP [55] and Stable Diffusion [63]. LAION-5B contains an English image-text subset of 2.32 billion samples, which we refer to as LAION-2B in this work. Due to its scale, transparency and open-source nature, LAION has already been adopted by various works on language-vision modelling, validating its suitability for systematic scaling law studies.

我们使用LAION-400M[66]和LAION-5B[65]数据集，它们是开放的公共图像文本数据集，通过对现有多模态模型(如CLIP[55]和稳定扩散[63])的预训练进行验证。LAION-5B包含一个由23.2亿个样本组成的英文图像文本子集，我们在这项工作中称之为LAION-2B。由于其规模、透明度和开源性质，LAION已经被各种语言视觉建模工作所采用，验证了其适用于系统的缩放定律研究。

### 3.2 Pre-training OpenCLIP across various scales 不同规模的OpenCLIP预训练
To systematically vary model scale, data scale and the number of samples seen during pre-training, we selected a scale range for each dimension. For model scale, we choose CLIP architectures with ViT-B/32, ViT-B/16, ViT-L/14, ViT-H/14 and ViT-g/14 as visual encoders, scaling the text encoder in accord (see Appendix Table 24). For data scale, we use LAION-80M (an 80M subset of LAION- 400M), LAION-400M, and LAION-2B. For training duration, we choose 3B, 13B and 34B samples seen scales. Due to compute constraints, for the larger H/14 and g/14 model scales, we conduct only restricted measurements (done for LAION-2B, with 34B samples seen for H/14, and with 13B samples seen for g/14). This selection provides coverage at the scale where we cannot afford to sample with the same density as at the intermediate and lower model scales. To verify that LAION-80M and LAION-400M are valid subsets of LAION-2B, we conduct a control experiment by extracting a random 400M subset of LAION-2B and comparing our reference OpenCLIP ViT-B/32 models pre-trained on both datasets. When doing so, we found no significant difference (see Appendix Sec. B.2.3).

为了系统地改变模型规模、数据规模和预训练期间看到的样本数量，我们为每个维度选择了一个规模范围。对于模型规模，我们选择了具有ViT-B/32、ViT-B/16、ViT-L/14、ViT-H/14和ViT-g/14作为视觉编码器的CLIP架构，从而按照比例缩放文本编码器(见附录表24)。对于数据规模，我们使用LAION-80M(LAION-400M的80M子集)、LAION-400M和LAION-2B。对于训练持续时间，我们选择3B、13B和34B样本观察量表。由于计算限制，对于较大的H/14和g/14模型尺度，我们只进行有限的测量(对LAION-2B进行测量，对H/14观察到34B个样本，对g/14观察到13B个样本)。这种选择提供了覆盖范围，在这种范围内，我们无法承受与中等和较低模型范围相同密度的样本。为了验证LAION-80M和LAION-400M是LAION-2B的有效子集，我们通过提取LAION-2B的随机400M子集并比较我们在两个数据集上预训练的参考OpenCLIP ViT-B/32模型来进行对照实验。这样做时，我们没有发现显著差异(见附录第B.2.3节)。

Compared to the original CLIP training procedure [55], we work with larger batch sizes and adapt the learning rate accordingly. We opt for larger batch sizes to allow for more efficient distributed training; maximizing the local batch size per GPU and using close to one thousand GPUs lead us to global batch sizes in the range of 86-88K samples. In order to assess the validity of re-using measurements obtained with different batch sizes, we perform a number of control experiments varying batch size from 32K to 86-88K, and observe a difference of 0.2−0.5% across different settings (see Appendix Sec. B.2.3), which is small enough not to confound observations on the effect of scale.

与原始的CLIP训练程序[55]相比，我们使用更大的批量，并相应地调整学习率。我们选择更大的批量，以实现更高效的分布式训练;最大化每个GPU的本地批处理大小并使用近1000个GPU使我们获得86-88K样本范围内的全局批处理大小。为了评估重新使用不同批量获得的测量结果的有效性，我们进行了一系列控制实验，批量从32K到86-88K不等，并在不同的设置中观察到0.2−0.5%的差异(见附录第B.2.3节)，这一差异足够小，不会混淆对量表影响的观察结果。

For each number of samples seen scale, we execute a separate training experiment with a cosine annealing learning schedule adapted to the number of samples. This allows us to assess performance of models pre-trained with different training durations and avoid suboptimal training when using the same schedule for runs of different length [28]. We tune a small number of hyper-parameters (see Appendix Table 17), each scale point to optimize validation loss and prevent training instabilities, and otherwise closely follow the original CLIP training procedure [55], using the InfoNCE loss, Adam with decoupled weight regularization [47] (i.e., AdamW) as an optimizer, with β1 = 0.9, β2 = 0.98 and weight decay of 0.2. We train the models using mixed precision. For larger model scales (ViT-L/14, H/14, g/14), we observed loss spikes during training which had an adverse effect on performance. We fixed the issue by switching from mixed precision with float16 to mixed precision with bfloat16.1 We hypothesize that bfloat16 fixed the issue due to larger models typically showing larger activation values as observed by [16], making bfloat16 more suitable with its wider dynamic range (8 exponent bits).

对于看到的每一个样本数量，我们执行一个单独的训练实验，其中余弦退火学习计划适用于样本数量。这使我们能够评估用不同训练持续时间预先训练的模型的性能，并在使用相同的时间表进行不同长度的跑步时避免次优训练[28]。我们调整了少量超参数(见附录表17)，每个标度点都是为了优化验证损失和防止训练不稳定性，并在其他方面密切遵循原始CLIP训练程序[55]，使用InfoNCE损失，Adam和解耦权重正则化[47](即AdamW)作为优化器，β1=0.9，β2=0.98，权重衰减为0.2。我们使用混合精度训练模型。对于较大的模型规模(ViT-L/14、H/14、g/14)，我们在训练过程中观察到损失峰值，这对表现产生了不利影响。我们通过从float16的混合精度切换到bfloat161的混合精度来解决这个问题。我们假设bfloat16解决了这个问题，因为正如[16]所观察到的，较大的模型通常显示较大的激活值，使bfloat6更适合其更宽的动态范围(8个指数位)。

CLIP pre-training experiments on larger scales require distributed training, as otherwise experiment execution times are intractable. We use OpenCLIP [32], an open source software that was adapted for distributed training on supercomputers. Using data parallel training via PyTorch DDP [45, 53], we conduct experiments with up to 1520 NVIDIA A100 GPUs. Distributed training was executed on JUWELS Booster [34], the supercomputer at Juelich Supercomputing Center (JSC, Germany), and partly also at Stability AI AWS supercomputer [2] For more details on distributed training procedure and on experiment compute budgets and runtimes, see Appendix Sec.A and Sec.B.2.4.

大规模的CLIP预训练实验需要分布式训练，否则实验执行时间很难解决。我们使用OpenCLIP[32]，这是一种开源软件，适用于超级计算机上的分布式训练。通过PyTorch DDP[45，53]使用数据并行训练，我们使用多达1520个NVIDIA A100 GPU进行了实验。分布式训练在Juelich超级计算中心(德国JSC)的超级计算机JUWELS Booster[34]上执行，部分也在Stability AI AWS超级计算机[2]上执行。有关分布式训练程序、实验计算预算和运行时间的更多详情，请参阅附录第A节和第B.2.4节。

## 4 Scaling laws for different downstream tasks 不同下游任务的缩放定律
### 4.1 Zero-shot transfer and robustness 零样本迁移和稳健性
One of the biggest advantages of open-vocabulary models like CLIP is that they can be used on downstream classification tasks by carefully designing text prompts corresponding to class descriptions, without requiring any labeled training example. Moreover, pre-trained CLIP models are observed to excel on out-of-distribution robustness benchmarks [55, 48]. In this section, we study the effect of scale on zero-shot classification, including an investigation on robustness benchmarks. We evaluate the models on ImageNet [15], ImageNet distribution shift datasets [22, 23, 24, 75, 5], and the visual task adaptation benchmark (VTAB) [85]. We conduct a simple duplication check for downstream datasets based on the perceptual image hash library pHash [82], revealing no or very little overlap with pre-training datasets (see Appendix Sec. B.1).

像CLIP这样的开放词汇模型的最大优势之一是，它们可以通过仔细设计与类描述相对应的文本提示来用于下游分类任务，而不需要任何令牌的训练样本。此外，观察到预先训练的CLIP模型在分布外稳健性基准方面表现出色[55，48]。在本节中，我们研究了规模对零样本分类的影响，包括对稳健性基准的调查。我们在ImageNet[15]、ImageNet分布偏移数据集[22，23，24，75，5]和视觉任务自适应基准(VTAB)[85]上评估了模型。我们基于感知图像哈希库pHash[82]对下游数据集进行了简单的重复检查，发现与预训练数据集没有或几乎没有重叠(见附录第B.1节)。

Evaluation setup. We follow the setup of Radford et al. [55]. For each downstream dataset, we use a set of pre-defined prompts for each class, which we collected from prior works [55, 86]. We compute the embedding of each class by averaging over the embeddings of the prompts obtained using the text tower, then we L2-normalize them. Given a dataset {(xi , yi)} n i=1, we classify each image as the class that has the largest cosine similarity with the (L2-normalized) image embedding, yˆi = argmaxj (φ(xi) T cj ). We evaluate the models using top-1 accuracy. For comparison to OpenAI CLIP, we take ViT-B/32, B/16, and L/14 models pre-trained on the private WIT-400M dataset. 1We also tried to reduce the learning rate, change the learning rate schedule, and use gradient clipping but none of these changes helped to avoid the training instabilities.

评估设置。我们遵循Radford等人[55]的设置。对于每个下游数据集，我们为每个类使用一组预定义的提示，这些提示是我们从以前的工作中收集的[55，86]。我们通过对使用文本塔获得的提示的嵌入进行平均来计算每个类的嵌入，然后对它们进行L2归一化。给定一个数据集{(xi，yi)}n i=1，我们将每个图像分类为与(L2归一化)图像嵌入具有最大余弦相似性的类，yˆi=argmaxj(φ(xi)T cj)。我们使用排名前1的精度来评估模型。为了与OpenAI CLIP进行比较，我们采用了在私有WIT-400M数据集上预先训练的ViT-B/32、B/16和L/14模型。1我们还试图降低学习率，改变学习率时间表，并使用梯度剪裁，但这些变化都无助于避免训练的不稳定性。

Effect of scale. Accuracy consistently improves when increasing model, data and samples seen scale hand-in-hand. Accuracy follows power laws, such that larger models benefit from larger data and samples seen scale (Figure 1a). The strongest ImageNet accuracy (78%) is obtained with the largest total pre-training compute, using ViT-H/14 pre-trained on LAION-2B data scale and 34B samples seen. For additional results, see Appendix Sec. B.2.4.

规模效应。当同时增加模型、数据和样本时，准确性会不断提高。精度遵循幂律，因此较大的模型受益于较大的数据和样本(图1a)。使用在LAION-2B数据量表上预训练的ViT-H/14和34B样本，使用最大的总预训练计算获得了最强的ImageNet准确率(78%)。有关其他结果，请参见附录第B.2.4节。

Fitting power-law (E = βCα) on the Pareto frontier of the available models, we measure scaling coefficients αopenCLIP = −0.11 and αCLIP = −0.16 for zero-shot top-1 ImageNet and αopenCLIP = −0.13 and αCLIP = −0.24 for ImageNet robustness datasets performance [22, 23, 24, 75]. For those tasks, we observe a scaling advantage for CLIP pre-trained on WIT-400M over OpenCLIP pretrained on LAION-400M/2B. αopenCLIP is similar for ImageNet and robustness datasets, suggesting that improving accuracy with scale leads to corresponding improvement on robustness benchmarks for OpenCLIP pre-trained on LAION.

在可用模型的帕累托前沿拟合幂律(E=βCα)，我们测量了零样本top-1 ImageNet的缩放系数αopenCLIP=−0.11和αCLIP=–0.16，以及ImageNetwork稳健性数据集性能的缩放系数βopenCLIP=−0.13和αCLIP=−0.24[22，23，24，75]。对于这些任务，我们观察到在WIT-400M上预训练的CLIP比在LAION-400M/2B上预训练过的OpenCLIP具有扩展优势。αopenCLIP对于ImageNet和稳健性数据集是相似的，这表明随着规模的提高精度会导致在LAION上预训练的openCLIP的稳健性基准的相应提高。

We also find bottleneck effects when scaling. For instance, OpenCLIP ViT-B/32 and ViT-B/16 models show no change or deterioration of performance when increasing data scale from 400M to 2B when using a smaller samples seen scale (3B or 13B). Moving to the largest samples seen scale (34B) then shows clear improvement for the larger 2B data scale, indicating that the number samples seen is a bottleneck (see also Appendix Table 18).

我们还发现了扩展时的瓶颈效应。例如，OpenCLIP ViT-B/32和ViT-B/16模型在使用较小的样本观测规模(3B或13B)时，当数据规模从400M增加到2B时，性能没有变化或恶化。移动到最大样本量表(34B)，显示出较大2B数据量表的明显改善，表明所看到的样本数量是一个瓶颈(另见附录表18)。

Using the obtained power law, we can make a prediction for the performance of a well-tuned ViT-g/14 model when using the largest data scale of 2B and samples seen scale of 34B, giving us error estimate of 20.9% (79.1% top-1 accuracy) on ImageNet. We predict even stronger performance at larger scales. For instance, assuming 68B samples seen we estimate top-1 accuracies of 79.7%, 80.7%, and 81.9% for ViT-H/14, ViT-g/14 and ViT-G/14, respectively (see also Appendix Sec. B.2.1).

使用所获得的幂律，当使用2B的最大数据尺度和34B的样本观测尺度时，我们可以对微调良好的ViT-g/14模型的性能进行预测，在ImageNet上给出20.9%(79.1%的前1精度)的误差估计。我们预测在更大范围内会有更强的表现。例如，假设看到68B个样本，我们估计ViT-H/14、ViT-g/14和ViT-g/14的前1级准确率分别为79.7%、80.7%和81.9%(另见附录B.2.1节)。

### 4.2 Retrieval 检索
Retrieval is another common way to evaluate zero-shot capabilities of the models. In this section, we study the effect of scale on both text and image zero-shot retrieval.

检索是评估模型零样本能力的另一种常见方法。在本节中，我们研究了缩放对文本和图像零样本检索的影响。

Evaluation setup. We compute text-image scores using the cosine similarity between image and text embeddings and rank the top-K images (resp. text captions) for each text caption (resp. images) when evaluating on image (resp. text) retrieval. We evaluate on MS-COCO [46] and Flickr30K [80], following the evaluation setup and test splits from [36]. We use Recall@K as an evaluation metric where K = 5.

评估设置。我们使用图像和文本嵌入之间的余弦相似性来计算文本图像得分，并在评估图像(对应文本)检索时对每个文本标题(对应图像)的前K个图像(对应文字标题)进行排序。我们在MS-COCO[46]和Flickr30K[80]上进行评估，遵循[36]中的评估设置和测试拆分。我们使用Recall@K作为评估度量，其中K＝5。

Effect of scale. Again we observe performance consistently improves when increasing scale following power law trends (Figure 1b). We measure scaling coefficients αopenCLIP = −0.08 and αCLIP = −0.05 for zero-shot retrieval on MS-COCO and αopenCLIP = −0.19 and αCLIP = −0.10 for Flickr30K. In contrast to zero-shot accuracy, retrieval performance shows a scaling advantage for OpenCLIP pretrained on LAION-400M/2B over CLIP pre-trained on WIT-400M. We also observe scale bottleneck effects. For instance, OpenCLIP ViT-L/14 model shows almost no improvement on LAION-400M when increasing the number of samples seen scale from 13B to 34B, indicatating a data scale bottleneck. When increasing data scale to 2B, we then observe clear improvements when going from 13B to 34B samples (see also Appendix Table 20 and 21).

规模效应。我们再次观察到，随着幂律趋势的扩大，性能不断提高(图1b)。我们测量了MS-COCO上零样本检索的缩放系数αopenCLIP=−0.08和αCLIP=-0.05，以及Flickr30K上的αopenCLIP=−0.19和αCLIP=-0.10。与零样本准确性相比，在LAION-400M/2B上预处理的OpenCLIP的检索性能显示出比在WIT-400M上预处理CLIP的缩放优势。我们还观察到规模瓶颈效应。例如，OpenCLIP-ViT-L/14模型显示，当看到的样本数量从13B增加到34B时，LAION-400M几乎没有改善，这表明存在数据规模瓶颈。当将数据规模增加到2B时，当从13B样本增加到34B样本时，我们观察到明显的改善(另见附录表20和21)。

Figure 2: Scaling model and data size leads to lower error linear classifers on ImageNet [15] and CIFAR-100 [43] in both the few-shot and full dataset regime. We train linear probes for models with at least 13B samples seen (also see corresponding Table 4). As discussed in Figure 1, we fit a power-law on the Pareto frontier of the available models.
图2：缩放模型和数据大小可以在ImageNet[15]和CIFAR-100[43]上的少样本和全数据集模型下降低误差线性分类器。我们为至少有13B样本的模型训练线性探针(也见相应的表4)。如图1所示，我们在可用模型的Pareto边界上拟合幂律。

Figure 3: Scaling model and data size leads to lower error linear classifers on the visual task adaptation benchmark (VTAB) [85]. We train linear probes for models with at least 13B samples seen (also see corresponding Table 4). As discussed in Figure 1, we fit a power-law on the Pareto frontier of the available models.
图3：缩放模型和数据大小导致视觉任务自适应基准(VTAB)上的误差较低的线性分类器[85]。我们为至少有13B样本的模型训练线性探针(也见相应的表4)。如图1所示，我们在可用模型的Pareto边界上拟合幂律。

### 4.3 Full and few-shot linear probing  全部和少样本线性探测
Another common way to measure the quality of learned representations is by training a linear classifier. While this technique underperforms end-to-end fine-tuning, it is often preferred because it requires far less compute [40, 55]. In this section we train linear classifiers, also referred to as linear probes, on the frozen representations of various CLIP models and examine the effect of data and model scale.

衡量学习表示质量的另一种常见方法是训练线性分类器。虽然这种技术的性能不如端到端微调，但它通常是首选，因为它需要的计算量要少得多[40，55]。在本节中，我们在各种CLIP模型的冻结表示上训练线性分类器，也称为线性探针，并检查数据和模型规模的影响。

Evaluation setup. Given a CLIP model with an image tower φ, our goal is to learn W such that W> φ(x) classifies x as its label y. Given a dataset {(xi , yi)} n i=1, we begin by saving the image features and labels for the dataset. That is, for all image label pairs (x, y) in the dataset we cache (φ(x), y). We then train a linear classifier W to minimize the cross entropy loss between softmax  W> φ(x)  and y. In preliminary experiments we found that this softmax regression achieved higher accuracy than linear regression. We use mini-batch stochastic optimization with the Adam optimizer [37]. We use batch size 256 and select the best result in a hyper-parameter sweep over learning rate {0.1, 0.01, 0.001} and epochs {10, 20, 40} individually for each model and dataset. For the ImageNet [15] and CIFAR100 datasets [43] we consider 10-shot, 25-shot, and full-dataset linear classifers (Figure 2). Additionally, we train linear classifiers on the visual task adaptation benchmark (VTAB) [85] (Figure 3).

评估设置。给定一个带有图像塔φ的CLIP模型，我们的目标是学习W，以便W>φ(x)将x分类为其标签y。给定一个数据集{(xi，yi)}n i=1，我们首先保存数据集的图像特征和标签。即，对于数据集中的所有图像标签对(x，y)，我们缓存(φ(x)，y)。然后，我们训练线性分类器W，以最小化softmax W>φ(x)和y之间的交叉熵损失。在初步实验中，我们发现这种softmax回归比线性回归获得了更高的精度。我们将小批量随机优化与Adam优化器一起使用[37]。我们使用批量大小256，并为每个模型和数据集分别选择超参数扫描学习率{0.1，0.01，0.001}和周期{10，20，40}的最佳结果。对于ImageNet[15]和CIFAR100数据集[43]，我们考虑10样本、25样本和全数据集线性分类器(图2)。此外，我们在视觉任务自适应基准(VTAB)[85]上训练线性分类器(图3)。

Effect of scale. For ImageNet, CIFAR100, and VTAB, scaling up consistently improves the accuracy of a linear classifier (Figure 2, 3). For ImageNet and CIFAR100, this is true in both the few-shot and full regimes. Moreover, among models trained on the same data distribution, scaling up follows a linear trend on a log-log plot. These results are perhaps not too surprising given similar observations for power laws on zero-shot downstream tasks in Section 4.1 as well as the correlation between zero-shot and linear probe performance observed by Radford et al. [55]. Nonetheless, this result re-affirms that scaling up model and data size leads to contunied accuracy improvements.

规模效应。对于ImageNet、CIFAR100和VTAB，按比例放大可以持续提高线性分类器的准确性(图2、3)。对于ImageNet和CIFAR100来说，无论是少量拍摄还是完整拍摄都是如此。此外，在相同数据分布上训练的模型中，在对数-对数图上，按比例放大遵循线性趋势。考虑到第4.1节中对零样本下游任务的幂律的类似观察，以及Radford等人[55]观察到的零样本和线性探头性能之间的相关性，这些结果可能并不太令人惊讶。尽管如此，这一结果再次证实，扩大模型和数据大小会带来持续的准确性提高。

### 4.4 Fine-tuning 微调
Next, we evaluate the effect of scale on fine-tuning performance. Since fine-tuning is much more compute-intensive than zero-shot and linear probing, we only evaluate a subset of the pre-trained models.

接下来，我们评估规模对微调性能的影响。由于微调比零样本和线性探测需要更多的计算，所以我们只评估预训练模型的子集。

Evaluation setup. We fine-tune and evaluate on ImageNet with the timm [77] library, using the image encoder from CLIP models trained on 2B data, 34B samples seen scale. To get the best results, we consider two different schemes, (A) fine-tune directly on ImageNet (B) first fine-tune on a subset of the full ImageNet-22k we call ImageNet-12k2 then continue fine-tuning on ImageNet, similar to [4]. We compare the results with OpenAI CLIP models fine-tuned with the same settings, evaluating the models using top-1 accuracy on ImageNet and the ImageNet distribution shift datasets [22, 23, 24, 75, 5]. The OpenCLIP models range from 82.6 to 88.5% top-1 on ImageNet, comparable to the best released ImageNet models pretrained on public datasets[4]. For additional details, including strong supervised baselines, see Appendix sec. B.2.2.

评估设置。我们在ImageNet上使用timm[77]库进行微调和评估，使用在2B数据上训练的CLIP模型的图像编码器，34B样本按比例观察。为了获得最佳结果，我们考虑了两种不同的方案，(A)直接在ImageNet上进行微调(B)首先在我们称之为ImageNet-12k2的完整ImageNet-22k的子集上进行微调，然后在ImageNet中继续进行微调，类似于[4]。我们将结果与使用相同设置微调的OpenAI CLIP模型进行比较，使用ImageNet和ImageNet分布偏移数据集上的前1精度评估模型[22，23，24，75，5]。OpenCLIP模型在ImageNet上的排名从82.6%到88.5%不等，与在公共数据集上预训练的最佳发布ImageNet模型相当[4]。有关其他详情，包括强有力的监督基线，请参见附录第B.2.2节。

In addition, we fine-tune and evaluate on eight diverse datasets where zero-shot models perform poorly [55, 31]: Cars [41], DTD [14], EuroSAT [21], GTSRB [69], MNIST [44], RESISC45 [13], SUN397 [79], and SVHN [50]. We fine-tune a single model jointly on the eight downstream tasks following Ilharco et al. [31], fine-tuning only the parameters of the vision encoder. The classification heads for each task are obtained using the zero-shot text encoder, and are kept frozen during fine-tuning. We fine-tune for 2000 iterations with a batch size of 128, learning rate 1e-5 and a cosine annealing learning rate schedule with 200 warm-up steps and the AdamW optimizer [47], with weight decay 0.1. We further explore the effect of fine-tuning on zero-shot ImageNet accuracy in the Appendix Sec. B.2.2.

此外，我们对零样本模型表现不佳的八个不同数据集进行了微调和评估[55，31]：Cars[41]、DTD[14]、EuroSAT[21]、GTSRB[69]、MNIST[44]、RESISC45[13]、SUN397[79]和SVHN[50]。在Ilharco等人[31]之后，我们在八个下游任务上联合微调单个模型，仅微调视觉编码器的参数。每个任务的分类头都是使用零样本文本编码器获得的，并在微调期间保持冻结。我们对2000次迭代进行微调，批量大小为128，学习率为1e-5，余弦退火学习率计划为200个预热步骤，AdamW优化器[47]，权重衰减为0.1。我们在附录第B.2.2节中进一步探讨了微调对零样本ImageNet精度的影响。

2We filter classes with few examples from the full ImageNet-22k with 14M examples to get a better balanced subset and we end up with 12K classes, 12M training examples, 470K validation examples.

2我们从完整的ImageNet-22k中筛选出几个样本的类，其中有14M个样本，以获得更好的平衡子集，最终我们得到了12K个类、12M个训练样本和470K个验证样本。

Figure 4: ImageNet and ImageNet robustness datasets classification performance for fine-tuned models.
图4：微调模型的ImageNet和ImageNet稳健性数据集分类性能。

Figure 5: Scaling model and data size leads to lower error after jointly fine-tuning on eight downstream image classification tasks. In this experiment, we fine-tune a single model jointly on all eight tasks, alternating batches from each task. We fine-tune only the parameters of the vision encoder, using a fixed classification head for each task initialized with the weights from the zero-shot model.
图5：在对八个下游图像分类任务进行联合微调后，缩放模型和数据大小可以降低误差。在这个实验中，我们在所有八个任务上联合微调单个模型，从每个任务中交替批次。我们只微调视觉编码器的参数，对每个任务使用一个固定的分类头，用零样本模型中的权重初始化。

Effect of scale. For ImageNet fine-tuning, only the models with the largest data and samples seen were fine-tuned. Despite the narrower scale range, a similar relationship in the slope of the OpenAI CLIP vs OpenCLIP fit lines is observed across the model scales (Figure 4). Moreover, scale consistently improves accuracy when fine-tuning on other downstream tasks (Figure 5). While trends vary with the task, we find that the slope of the linear trend relating accuracy and total compute used for pre-training depends on the pre-training dataset, typically favors CLIP WIT-400M, as we observe in zero-shot experiments.

规模效应。对于ImageNet的微调，只有具有最大数据和样本的模型被微调。尽管尺度范围较窄，但在模型尺度上观察到OpenAI CLIP与OpenCLIP拟合线的斜率存在类似关系(图4)。此外，在对其他下游任务进行微调时，规模不断提高准确性(图5)。虽然趋势随任务而变化，但我们发现用于预训练的线性趋势相关精度和总计算的斜率取决于预训练数据集，如我们在零样本实验中观察到的那样，通常有利于CLIP WIT-400M。

## 5 Discussion 讨论
Larger scale improves performance across different downstream tasks. In line with previous studies [35, 73, 61, 84], our work observes scaling laws of power law form across various downstream tasks. We empirically find that scaling model, data and training samples seen results in consistent improvements on downstream zero-shot classification, retrieval, linear probing, and fine-tuning performance.

更大的规模可以提高不同下游任务的性能。与之前的研究[35，73，61，84]一致，我们的工作观察到了各种下游任务中幂律形式的比例定律。我们通过实证发现，缩放模型、数据和训练样本在下游零样本分类、检索、线性探测和微调性能方面取得了一致的改进。

We also observe bottleneck behaviors [35, 84] that occur when fixing one scaling dimension while increasing others. For instance, OpenCLIP ViT-B/32 and ViT-B/16 are bottlenecked by the number of samples seen at the 13B scale. Increasing the number of samples seen to 34B reveals that LAION-2B brings clear improvement over LAION-400M, which would remain hidden when fixing the number of samples seen scale to a lower value. Similar observations may occur along other scaling dimensions. OpenCLIP ViT L/14 shows an example of data scale bottleneck on LAION-400M scale, as increasing the number of samples seen from 13B to 34B does not lead to improvements. The benefit of using a larger number of samples seen is then revealed when going to the larger LAION-2B dataset.

我们还观察到在固定一个缩放维度同时增加其他缩放维度时发生的瓶颈行为[35，84]。例如，OpenCLIP ViT-B/32和ViT-B/16被13B规模的样本数量所限制。将观察到的样本数量增加到34B表明，LAION-2B比LAION-400M带来了明显的改进，当将观察到规模的样本数量固定到较低的值时，LAION-400M将保持隐藏状态。类似的观察结果可能出现在其他缩放维度上。OpenCLIP ViT L/14显示了LAION-400M规模的数据规模瓶颈的一个样本，因为将看到的样本数量从13B增加到34B并不能带来改进。当使用更大的LAION-2B数据集时，可以看到使用更多样本的好处。

Having derived scaling laws from our experimental observations, we are able to make predictions for both smaller and larger scales. Extrapolation has its limits, as saturation effects at both lower and higher scale ranges have been previously observed. We can however extrapolate to scales close to the ones we have already measured. A prediction for larger ViT-g/14 trained on LAION-2B with 34B samples delivers an estimate of 79.1% ImageNet top-1 accuracy. This may appear at first sight modest compared to results reported by BASIC (85.7% [54]), LiT (85.2% [86]) or CoCA (86.1% [81]).

从我们的实验观测中导出了标度定律，我们能够对更小和更大的标度进行预测。外推有其局限性，因为先前已经观察到较低和较高尺度范围的饱和效应。然而，我们可以推断出接近我们已经测量过的尺度。在具有34B样本的LAION-2B上训练的较大ViT-g/14的预测提供了79.1%的ImageNet top 1准确度估计。与BASIC(85.7%[54])、LiT(85.2%[86])或CoCA(86.1%[81])报告的结果相比，乍一看，这可能是适度的。

However, these works leverage an internal JFT dataset with labels which can be used for supervised pre-training. Moreover, for 973/1000 ImageNet classes, researchers were able to manually identify a correspondance from a JFT class [78]. These works also use larger encoders, larger private data, and pre-train the encoders in multiple stages. Nonetheless, we estimate based on our empirical findings that further increasing model and data scale could result in competitive models even without using labeled data, additional supervised pre-training stages or additional losses. Finally, we observe that the improvement of zero-shot ImageNet accuracy due to scaling up is accompanied by closely aligned improvements on robustness benchmarks.

然而，这些工作利用了带有标签的内部JFT数据集，该数据集可用于监督预训练。此外，对于973/1000个ImageNet类，研究人员能够手动识别JFT类的对应项[78]。这些工作还使用了更大的编码器、更大的私有数据，并在多个阶段对编码器进行预训练。尽管如此，我们根据经验发现估计，即使不使用令牌数据、额外的监督预训练阶段或额外的损失，进一步增加模型和数据规模也可能导致竞争模型。最后，我们观察到，由于放大，零样本ImageNet精度的提高伴随着健壮性基准的紧密一致的改进。

Scaling behavior depends on task type and pre-training dataset. When measuring scaling coefficients for the observed power laws, we see that OpenAI CLIP and OpenCLIP have distinct scaling advantages over each other depending on the downstream task. OpenCLIP pre-trained on LAION-400M/2B data has stronger scaling trends for zero-shot retrieval, while OpenAI CLIP pre-trained on private WIT-400M data shows stronger scaling for zero-shot ImageNet classification. We hypothesize that the observed differences are due to differences in the pre-training data, as we closely follow the architectures and pre-training recipes used for the OpenAI CLIP models. WIT-400M may have a stronger affinity to ImageNet as a result of the curation procedure, while LAION-400M/2B was filtered by a pre-trained OpenAI ViT-B/32 model relying on its similarity measurements for image-text pairs, which may have rendered the dataset more suitable for retrieval based tasks. This hypothesis can be tested by systematically varying dataset composition procedure (for example by using a stronger L/14 model for filtering crawled data) and observing the effect on scaling behavior across various task types.

缩放行为取决于任务类型和预训练数据集。当测量观测到的幂律的缩放系数时，我们发现OpenAI CLIP和OpenCLIP根据下游任务的不同，彼此具有明显的缩放优势。在LAION-400M/2B数据上预处理的OpenCLIP对于零样本检索具有更强的缩放趋势，而在私有WIT-400M数据上预训练的OpenAI CLIP对于零样本ImageNet分类显示出更强的缩放。我们假设观察到的差异是由于预训练数据的差异，因为我们密切关注用于OpenAI CLIP模型的架构和预训练配置。由于策展过程，WIT-400M可能对ImageNet具有更强的亲和力，而LAION-400M/2B是由预先训练的OpenAI ViT-B/32模型根据其对图像-文本对的相似性测量进行过滤的，这可能使数据集更适合于基于检索的任务。这一假设可以通过系统地改变数据集组成过程来测试(例如，通过使用更强的L/14模型来过滤抓取的数据)，并观察对各种任务类型的缩放行为的影响。

Limitations of the current study. Observed scaling laws are based on points we were able to obtain with available compute resources. Therefore, the density of sampling the scales space is low. It is also not possible to conduct full hyper-parameter tuning, especially on larger scales, due to high compute costs. We rely thus on control experiments that look at few hyper-parameters at early pre-training stages and on tuning already performed in previous work to suggest that pre-training for each scale is not far from optimal. It was also not possible to obtain more points for OpenAI CLIP due to the private nature of the WIT-400M dataset. Moreover, we conduct only a simple duplication check for downstream data, which may leave few duplicates undetected. Previous studies [55, 86] also reported that duplication in test sets do not significantly alter most results, potentially due to the very large scale and diversity of pre-training data.

当前研究的局限性。观察到的缩放定律是基于我们能够利用可用的计算资源获得的点。因此，尺度空间的采样密度较低。由于计算成本高，也不可能进行全超参数调整，尤其是在较大规模上。因此，我们依赖于在早期预训练阶段观察少数超参数的控制实验，以及在先前工作中已经进行的调整，以表明每个量表的预训练离最佳并不远。由于WIT-400M数据集的私有性，也不可能为OpenAI CLIP获得更多积分。此外，我们只对下游数据进行简单的重复检查，这可能会留下很少的重复未被发现。先前的研究[55，86]也报告称，测试集中的重复不会显著改变大多数结果，这可能是由于预训练数据的规模和多样性非常大。

## 6 Conclusion
We present a systematic study of scaling laws for contrastive language-image learning, investigating how scale affects performance on several downstream tasks and across adaptation methods. We find—in accord with previous works on uni-modal learning [35, 84]—a power law relation between scale (model, data and the number of samples seen) and downstream performance in a broad range of settings, including zero-shot classification, retrieval, few- and full-shot linear probing and fine-tuning. Interestingly, the scaling behavior for OpenCLIP-LAION pre-trained models and for OpenAI-WIT-400M pre-trained models differ, showing distinct benefits of one over another on different downstream tasks. We hypothesize that such task-specific scaling differences originate from the different pre-training datasets. Predictions for model performance on larger scales made on the basis of the scaling laws estimate 81.9% zero-shot top-1 accuracy on ImageNet for a ViT-G/14 CLIP model trained on 68B image-text samples from scratch.

我们对对比语言图像学习的尺度规律进行了系统研究，研究了尺度如何影响几个下游任务和跨适应方法的表现。我们发现，与以往关于单模学习的工作[35，84]一致，单模学习是一种规模(模型、数据和看到的样本数)与下游性能之间的幂律关系，在广泛的设置中，包括零样本分类、检索、少量和全面线性探测和微调。有趣的是，OpenCLIP LAION预训练模型和OpenAI-WIT-400M预训练模型的缩放行为不同，在不同的下游任务上显示出不同的优势。我们假设这种特定任务的缩放差异源于不同的预训练数据集。基于缩放定律对大尺度模型性能的预测估计，对于在68B图像文本样本上从头开始训练的ViT-G/14 CLIP模型，ImageNet上的零样本top-1精确度为81.9%。

Our study opens many directions for further investigations. Obtaining more data points for smaller and intermediate scales can provide enough sampling density to better understand the optimal configuration of model size, dataset size and number of samples seen given a fixed compute, similar to works such as [28, 39]. Scaling laws for robustness benchmarks [71] can be derived when controlling for larger accuracies observed at larger scales. Further, treating vision and text encoder scales separately may lead to modality specific scaling laws. A promising direction is to study the effect of the pre-training dataset on scaling behavior. Our observations so far hint that the data source may strongly influence task-specific scaling. This paves the road for studies on foundation datasets [68]. Having open datasets [66, 65] and open source tools [32] at hand, such experiments can be conducted and reproduced in a common effort by the broader research community.

我们的研究为进一步的研究开辟了许多方向。在较小和中等规模下获得更多的数据点可以提供足够的采样密度，以更好地理解在给定固定计算的情况下模型大小、数据集大小和样本数量的最佳配置，类似于[28，39]等工作。当控制在更大尺度上观察到的更大精度时，可以导出稳健性基准[71]的缩放定律。此外，单独处理视觉和文本编码器标度可能导致模态特定的标度定律。一个很有前途的方向是研究预训练数据集对缩放行为的影响。到目前为止，我们的观察结果表明，数据源可能会强烈影响特定任务的缩放。这为基础数据集的研究铺平了道路[68]。有了开放的数据集[66，65]和开源工具[32]，这样的实验可以由更广泛的研究界共同进行和复制。

## Acknowledgments.
We would like to express gratitude to all the people who are working on making code, models and data publicly available, advancing community based research and making research more reproducible. Specifically, we would like to thank all the members of the LAION discord server3 community that was pivotal for the effort to compose LAION-400m and LAION-5B datasets without which this study would be impossible, and openAI for making their pre-trained CLIP models publicly available. We want to thank Hugging Face for providing hosting space for open datasets and models and Stability AI for providing supercomputing resources and storage space.

我们要向所有致力于公开代码、模型和数据、推进基于社区的研究并使研究更具可复制性的人表示感谢。具体而言，我们要感谢LAION discord-server3社区的所有成员，他们在撰写LAION-400和LAION-5B数据集方面发挥了关键作用，如果没有这些数据集，本研究将不可能进行，并感谢openAI公开他们预先训练的CLIP模型。我们要感谢Hugging Face为开放数据集和模型提供托管空间，感谢Stability AI提供超级计算资源和存储空间。

3 https://discord.gg/BZqhreFazY

The authors gratefully acknowledge the Gauss Centre for Supercomputing e.V. 4 for funding this work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster [34] at Jülich Supercomputing Centre (JSC). We also acknowledge storage resources on JUST [20] granted and operated by JSC, as well as computing resources from the Helmholtz Data Federation (HDF). Further thank goes for support provided by JSC supercomputing facility administration team, especially to Damian Alvarez for his endurance and patience during the long "de-micing" sessions on JUWELS Booster.

作者感谢高斯超级计算中心e.V.4通过约翰·冯·诺伊曼计算研究所(NIC)在Jülich超级计算中心(JSC)的GCS超级计算机JUWELS Booster[34]上提供计算时间，为这项工作提供资金。我们还承认JSC授予和运营的JUST[20]上的存储资源，以及亥姆霍兹数据联合会(HDF)的计算资源。进一步感谢JSC超级计算设施管理团队提供的支持，特别是达米安·阿尔瓦雷斯在JUWELS Booster的长时间“除米”会议中表现出的耐力和耐心。

Special thanks goes also to Richard Vencu (LAION, Stability AI) for his on-going dedication towards enabling a HPC system and infrastructure around it that can be used by broad community of researchers and citizen scientists.

还要特别感谢Richard Vencu(LAION，Stability AI)，他一直致力于实现HPC系统及其周围的基础设施，供广大研究人员和公民科学家使用。

## References
1. Common Crawl. https://commoncrawl.org. 5, 20
2. Stability AI HPC facility, https://hpc.stability.ai. 6
3. Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022. 4
4. Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021. 9, 26
5. Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In Advances in Neural Information Processing Systems (NeurIPS), 2019. 3, 6, 9
6. Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine-learning practice and the classical bias-variance trade-off. Proceedings of the National Academy of Sciences of the United States of America, 116:15849–15854, Aug. 2019. 4
7. Lucas Beyer, Olivier J Hénaff, Alexander Kolesnikov, Xiaohua Zhai, and Aäron van den Oord. Are we done with imagenet? arXiv preprint arXiv:2006.07159, 2020. 28
8. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. 1, 4
9. Sebastien Bubeck and Mark Sellke. A universal law of robustness via isoperimetry. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 28811–28822. Curran Associates, Inc., 2021. 4
10. Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022. 5
11. Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Michael Carbin, and Zhangyang Wang. The lottery tickets hypothesis for supervised and self-supervised pre-training in computer vision 4 https://gauss-centre.eu 13 models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16306–16316, 2021. 4
12. Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022. 4
13. Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the Institute of Electrical and Electronics Engineers (IEEE), 2017. https://ieeexplore.ieee.org/abstract/document/7891544. 9
14. Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Conference on Computer Vision and Pattern Recognition (CVPR), 2014. https://openaccess.thecvf.com/content_cvpr_2014/html/Cimpoi_Describing_Textures_ in_2014_CVPR_paper.html. 9
15. J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proc. IEEE Conf. Computer Vision and Pattern Recognition, pages 248–255, June 2009. 2, 6, 8, 9, 27
16. Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm. int8 (): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022. 6
17. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. 1
18. Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, and Ludwig Schmidt. Data determines distributional robustness in contrastive language image pre-training (clip). arXiv preprint arXiv:2205.01397, 2022. 4
19. Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. arXiv preprint arXiv:2211.07636, 2022. 21
20. Stephan Graf and Olaf Mextorf. Just: Large-scale multi-tier storage infrastructure at the jülich supercomputing centre. Journal of large-scale research facilities JLSRF, 7:180, 2021. 13
21. Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019. https://arxiv.org/abs/1709.00029. 9
22. Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. International Conference on Computer Vision (ICCV), 2021. https://arxiv.org/abs/2006.16241. 3, 6, 7, 9, 28
23. Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. International Conference on Learning Representations (ICLR), 2019. https: //arxiv.org/abs/1903.12261. 6, 7, 9
24. Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. Conference on Computer Vision and Pattern Recognition (CVPR), 2021. https://arxiv. org/abs/1907.07174. 3, 6, 7, 9, 28
25. Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, Chris Hallacy, Benjamin Mann, Alec Radford, Aditya Ramesh, Nick Ryder, Daniel M. Ziegler, John Schulman, Dario Amodei, and Sam McCandlish. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020. 2
26. Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Patwary, Mostofa Ali, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017. 3 14
27. Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 5, 21
28. Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 1, 2, 4, 6, 12
29. Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 328–339, Melbourne, Australia, July 2018. Association for Computational Linguistics. 26
30. Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17980–17989, June 2022. 5
31. Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, and Ludwig Schmidt. Patching open-vocabulary models by interpolating weights. arXiv preprint arXiv:2208.05592, 2022. 9, 10, 23, 26, 28
32. Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. 2, 6, 12, 19
33. Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904–4916. PMLR, 2021. 1, 4
34. Juelich Supercomputing Center. JUWELS Booster Supercomputer, 2020. https://apps.fz-juelich.de/jsc/hps/juwels/configuration.html# hardware-configuration-of-the-system-name-booster-module. 6, 13, 19
35. Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 1, 2, 4, 11, 12
36. Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3128–3137, 2015. 7
37. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 9
38. Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (bit): General visual representation learning. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision – ECCV 2020, pages 491–507, Cham, 2020. Springer International Publishing. 1, 4
39. Skanda Koppula, Yazhe Li, Evan Shelhamer, Andrew Jaegle, Nikhil Parthasarathy, Relja Arandjelovic, João Carreira, and Olivier Hénaff. Where should i spend my flops? efficiency evaluations of visual pre-training methods. arXiv preprint arXiv:2209.15589, 2022. 4, 12
40. Simon Kornblith, Jonathon Shlens, and Quoc V Le. Do better imagenet models transfer better? In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. https://arxiv.org/abs/ 1805.08974. 9
41. Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for finegrained categorization. In International Conference on Computer Vision Workshops (ICML), 2013. https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W19/html/ Krause_3D_Object_Representations_2013_ICCV_paper.html. 9
42. Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):32–73, 2017. 5 15
43. Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009. 8, 9, 27
44. Yann LeCun. The mnist database of handwritten digits, 1998. http://yann.lecun.com/exdb/mnist/. 9
45. Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, et al. Pytorch distributed: Experiences on accelerating data parallel training. arXiv preprint arXiv:2006.15704, 2020. 6
46. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 2, 5, 7
47. Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 6, 10
48. John P Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy Liang, Yair Carmon, and Ludwig Schmidt. Accuracy on the line: on the strong correlation between out-of-distribution and in-distribution generalization. In International Conference on Machine Learning, pages 7721–7735. PMLR, 2021. 4, 6
49. Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal contrastive learning with limoe: the language-image mixture of experts. arXiv preprint arXiv:2206.02770, 2022. 4, 5, 21
50. Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. In Advances in Neural Information Processing Systems (NeurIPS) Workshops, 2011. https://storage.googleapis.com/ pub-tools-public-publication-data/pdf/37648.pdf. 9
51. Thao Nguyen, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, and Ludwig Schmidt. Quality not quantity: On the interaction between dataset design and robustness of clip. arXiv preprint arXiv:2208.05516, 2022. 4
52. Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 19
53. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems, 32, 2019. 6
54. Hieu Pham, Zihang Dai, Golnaz Ghiasi, Hanxiao Liu, Adams Wei Yu, Minh-Thang Luong, Mingxing Tan, and Quoc V Le. Combined scaling for zero-shot transfer learning. arXiv preprint arXiv:2111.10050, 2021. 1, 4, 5, 11
55. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 1, 2, 4, 5, 6, 9, 12, 19, 21
56. Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. Technical report, Tech. Rep., Technical report, OpenAI, 2022. 1
57. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. 1
58. Vinay Venkatesh Ramasesh, Aitor Lewkowycz, and Ethan Dyer. Effect of scale on catastrophic forgetting in neural networks. In International Conference on Learning Representations, 2021. 4
59. Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 1
60. Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classifiers generalize to ImageNet? In International Conference on Machine Learning (ICML), 2019. https: //arxiv.org/abs/1902.10811. 3, 28 16
61. Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34, 2021. 1, 2, 4, 11
62. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 1
63. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 5, 21
64. Jonathan S Rosenfeld, Jonathan Frankle, Michael Carbin, and Nir Shavit. On the predictability of pruning across scales. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 9075–9083. PMLR, 18–24 Jul 2021. 4
65. Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2022. 2, 5, 12, 20, 21, 31, 39
66. Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION-400M: Open dataset of CLIP-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 5, 12, 20
67. Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, Melbourne, Australia, July 2018. Association for Computational Linguistics. 5
68. Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S Morcos. Beyond neural scaling laws: beating power law scaling via data pruning. arXiv preprint arXiv:2206.14486, 2022. 12
69. Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. The german traffic sign recognition benchmark: a multi-class classification competition. In International Joint Conference on Neural Networks (IJCNN), 2011. https://ieeexplore.ieee.org/document/6033395. 9
70. Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In Proceedings of the IEEE international conference on computer vision, pages 843–852, 2017. 3
71. Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, and Ludwig Schmidt. Measuring robustness to natural distribution shifts in image classification. Advances in Neural Information Processing Systems, 33:18583–18599, 2020. 2, 4, 12
72. Yi Tay, Mostafa Dehghani, Samira Abnar, Hyung Won Chung, William Fedus, Jinfeng Rao, Sharan Narang, Vinh Q Tran, Dani Yogatama, and Donald Metzler. Scaling laws vs model architectures: How does inductive bias influence scaling? arXiv preprint arXiv:2207.10551, 2022. 4
73. Yi Tay, Mostafa Dehghani, Jinfeng Rao, William Fedus, Samira Abnar, Hyung Won Chung, Sharan Narang, Dani Yogatama, Ashish Vaswani, and Donald Metzler. Scale efficiently: Insights from pretraining and finetuning transformers. In International Conference on Learning Representations, 2021. 1, 2, 4, 11
74. Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016. 5
75. Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. In Advances in Neural Information Processing Systems (NeurIPS), 2019. https://arxiv.org/abs/1905.13549. 3, 6, 7, 9, 28 17
76. Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, et al. Internimage: Exploring large-scale vision foundation models with deformable convolutions. arXiv preprint arXiv:2211.05778, 2022. 21
77. Ross Wightman. Pytorch image models. https://github.com/rwightman/pytorch-image-models, 2019. 9
78. Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust fine-tuning of zero-shot models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7959–7971, 2022. 2, 11
79. Jianxiong Xiao, Krista A Ehinger, James Hays, Antonio Torralba, and Aude Oliva. Sun database: Exploring a large collection of scene categories. International Journal of Computer Vision (IJCV), 2016. https://link.springer.com/article/10.1007/s11263-014-0748-y. 9
80. Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. 7
81. Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. CoCa: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research, 2022. 4, 11
82. Christoph Zauner. Implementation and benchmarking of perceptual image hash functions. 2010. 6, 21, 22
83. Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. arXiv preprint arXiv:2106.04560, 2021. 2
84. Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12104–12113, 2022. 1, 4, 11, 12
85. Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867, 2019. 2, 6, 8, 9, 27
86. Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. LiT: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18123–18133, 2022. 4, 5, 6, 11, 12, 21 18 


## A Further details on distributed training A关于分布式训练的进一步细节
### A.1 Supercomputer specifications A.1超级计算机规范
The JUWELS Booster [34] supercomputer used for training consists of 936 compute nodes that host four NVIDIA A100 GPUs each, providing 3744 GPUs in total. The installed A100 Tensor Core GPUs (40 GB) provide 19.5 TFLOP/s of FP64TC computing performance each. The GPUs are hosted by AMD EPYC 7402 CPUs with 2 × 24 cores (SMT-2) per node, clocked with 2.8 GHz. Each node is diskless and is equipped with 512 GB of RAM. The network is based on Mellanox HDR200 InfiniBand, with four Mellanox ConnectX 6 devices per node, each providing 200 Gbit/s bandwidth per direction.

用于训练的JUWELS Booster[34]超级计算机由936个计算节点组成，每个节点托管四个NVIDIA A100 GPU，总共提供3744个GPU。已安装的A100 Tensor Core GPU(40 GB)每个可提供19.5 TFLOP/s的FP64TC计算性能。GPU由AMD EPYC 7402 CPU托管，每个节点有2×24核(SMT-2)，时钟频率为2.8 GHz。每个节点都是无盘的，并配备了512 GB的RAM。该网络基于Mellanox HDR200 InfiniBand，每个节点有四个Mellanox ConnectX 6设备，每个设备每个方向提供200 Gbit/s的带宽。

The NVIDIA A100 GPUs reach peak efficiency of 48.75 GFLOP/(sW) when utilizing the FP64 Tensor Cores. This made the employed machine rank highest in the Green500 list as of November 2020 as the most energy efficient supercomputer among the first 100 machines of the Top500 list with 25 GFLOP/(sW).

当使用FP64张量内核时，NVIDIA A100 GPU达到48.75 GFLOP/(sW)的峰值效率。这使得截至2020年11月，该机器在绿色500强榜单中排名最高，成为前500强榜单前100台机器中能效最高的超级计算机，拥有25 GFLOP/(sW)。

### A.2 Scaling and training time A.2缩放和训练时间
Here, we report scaling behavior during large-scale pre-training using ViT-L/14 as a vision backbone with OpenCLIP [32]. We performed scaling experiments to assess the scalability of data parallel training distributed across many GPUs on multiple nodes using PyTorch DDP. The efficiency in Figure 6b is computed using the following formula: E(N) = 100 × T(N) N×T(1) . T(N) is the total measured throughput in Im/s for N GPUs. The best achievable efficiency, when scaling is perfect, is 100%.We observe that scaling is sufficiently close to ideal linear, staying above ≈ 84% for 1024 GPUs (256 nodes). We also provide the raw throughput (Im/s) numbers in Figure 6a.

在这里，我们报告了大规模预训练期间的缩放行为，使用ViT-L/14作为OpenCLIP[32]的视觉骨干。我们使用PyTorch DDP进行了扩展实验，以评估分布在多个节点上的多个GPU上的数据并行训练的可扩展性。图6b中的效率使用以下公式计算：E(N)=100×T(N)N×T(1)。T(N)是N个GPU的总测量吞吐量，单位为Im/s。当缩放完美时，可实现的最佳效率是100%。我们观察到，缩放足够接近理想线性，对于1024个GPU(256个节点)，保持在≈84%以上。我们还在图6a中提供了原始吞吐量(Im/s)数字。

### A.3 Sharding contrastive loss A.3分割对比损失
The InfoNCE loss [52] used by CLIP can be thought of as a method to maximize the mutual information between text and image representations. Formally, Oord et al. express that I(X; Y ) ≥ log(N) − LN , N denoting batch size and LN representing the InfoNCE loss. As a result of this lower bound, maximizing the batch size will maximize our mutual information.

CLIP使用的InfoNCE损失[52]可以被认为是一种最大化文本和图像表示之间相互信息的方法。从形式上讲，Oord等人表示I(X;Y)≥log(N)−LN，N表示批量大小，LN表示InfoNCE损失。由于这个下限，最大化批量大小将最大化我们的相互信息。

Radford et al. [55] take advantage of this bound and use N = 32, 768 to train CLIP. Such a batch size necessitates the sharding of computation. Although the original CLIP paper points towards this notion, the implementation details are nontrivial.

Radford等人[55]利用这一界限，并使用N=32768来训练CLIP。这样的批量大小需要对计算进行分片。尽管最初的CLIP论文指向了这个概念，但实现细节并不重要。

Before sharding, the similarity scores will take up O(N2 ) memory on each worker, totalling to 4 GB of VRAM in FP32. After sharding memory reduces to instantiating two n × N matrices, n being the batch size allocated to each worker. Using a local batch size of 256, the similarity matrices now occupy 64 MB of memory in FP32.

在分片之前，相似性分数将占用每个工作线程的O(N2)内存，FP32中的VRAM总计为4GB。分片内存后，减少到实例化两个n×n矩阵，n是分配给每个工作者的批大小。使用256的局部批量大小，相似性矩阵现在在FP32中占据64MB的内存。

To achieve this memory reduction, we can eliminate redundant computations and compute the similarities of local features versus all features. When aggregated across all machines, this achieves identical gradients. However, it should be noted that the all-gather method is imperative for correct gradient calculation. PyTorch’s standard torch.distributed.all_gather can not be differentiated through, while torch.distributed.nn.functional.all_gather can be. Thus, we require the use of the latter to correctly calculate the gradients in a distributed manner.

为了实现这种内存减少，我们可以消除冗余计算，并计算局部特征与所有特征的相似性。当在所有机器上进行聚合时，可以实现相同的梯度。然而，应该注意的是，所有聚集方法对于正确的梯度计算是必不可少的。PyTorch的标准torch.distributed.all_gather不能通过区分，而torch.ddistributed.nn.functional.all_gath可以区分。因此，我们要求使用后者以分布式方式正确计算梯度。

Figure 6: Distributed training for OpenCLIP ViT-L/14, scaling behavior on the supercomputer using A100 GPUs while varying the number of GPUs. In Figure 6a, we show the raw throughputs and in Figure 6b we show speedup and efficiency we obtain in the same setup, relative to training with a single node (each node contains 4 GPUs).
图6:OpenCLIP ViT-L/14的分布式训练，使用A100 GPU在超级计算机上缩放行为，同时改变GPU的数量。在图6a中，我们显示了原始吞吐量，在图6b中，我们展示了相对于使用单个节点(每个节点包含4个GPU)进行训练，我们在相同设置中获得的加速和效率。

### A.4 Training instabilities A.4训练不稳定性
As parameterization increased within our training runs, so did model model instability. Half-way through the runs of ViT L/14 H/14 and g/14, NaN values and loss spikes began occurring.

随着训练过程中参数化的增加，模型的不稳定性也随之增加。ViT L/14 H/14和g/14运行到一半时，NaN值和损耗峰值开始出现。

To address these issues, we attempted to use extra normalization layers, add scaled cosine attention, resume many steps before crashes, and implement other architecture tweaks with no success. What ended up solving the stability issues was increasing precision.

为了解决这些问题，我们尝试使用额外的归一化层，添加缩放余弦注意力，在崩溃前恢复许多步骤，并实现其他架构调整，但没有成功。最终解决稳定性问题的是不断提高的精确度。

Using Automatic Mixed Precision (AMP) with bfloat16 over float16, or float32 with tensor-float32 resolved the issues mentioned above. We also have observed that even the smaller ViT-B models with AMP can become unstable when learning rate and batch size become sufficiently large, suggesting a generic scheme behind the phenomenon where frequency of instabilities occurring during the training is a function of model scale and global batch size.

在float16上使用bfloat16的自动混合精度(AMP)，或在float32上使用tensor-float32解决了上述问题。我们还观察到，当学习率和批量大小变得足够大时，即使是具有AMP的较小的ViT-B模型也可能变得不稳定，这表明在训练过程中发生的不稳定频率是模型规模和全局批量大小的函数的现象背后有一个通用方案。

### B Experimental details B实验细节
#### B.1 Datasets employed in experiments. B.1实验中使用的数据集。
LAION-400M and LAION-5B. Both LAION-400M [66] and LAION-5B [65] are open, public image-text datasets that were composed by obtaining links from Common Crawl [1]. While LAION- 400M contains 414M english image-text pairs, LAION-5B is currently the largest public image-text dataset containing over 5.8 billion multi-lingual image-text examples. In both cases, samples are obtained by filtering a subset of Common Crawl with a pre-trained OpenAI ViT B/32 model. LAION- 5B contains an English image-text subset of 2.32 billion samples, to which we refer as LAION-2B in this work. Besides the open nature of the datasets, a further advantage is full transparency about the dataset composition and assembly, with software stack and tools around LAION-400M and LAION-5B released as open-source, increasing reproducibility of experiments. This already resulted in numerous works using the datasets for training state-of-the-art language-vision models [63, 49, 27, 76, 19], validating the usage of those datasets for studying scaling laws in this work.

LAION-400M和LAION-5B。LAION-400M[66]和LAION-5B[65]都是开放的公共图像文本数据集，通过从Common Crawl[1]中获取链接而组成。LAION-400M包含4.14亿个英语图像文本对，而LAION-5B是目前最大的公共图像文本数据集，包含超过58亿个多语言图像文本样本。在这两种情况下，样本都是通过使用预先训练的OpenAI ViT B/32模型过滤Common Crawl的子集来获得的。LAION-5B包含一个由23.2亿个样本组成的英文图像文本子集，我们在这项工作中称之为LAION-2B。除了数据集的开放性之外，另一个优势是数据集的组成和组装完全透明，LAION-400M和LAION-5B的软件堆栈和工具作为开源发布，增加了实验的可复现。这已经导致了许多使用数据集来训练最先进的语言视觉模型的工作[63，49，27，76，19]，验证了这些数据集在这项工作中用于研究缩放定律的使用。

Table 2: Open LAION datasets used for pre-training in this study. Adapted from [65]. LAION-2B is a subset of multi-lingual LAION-5B and is more than 20 times larger than other public English image-text datasets. The scale of LAION-2B is comparable to the largest private dataset used for language-vision model training.
表2：本研究中用于预训练的开放LAION数据集。改编自[65]。LAION-2B是多语言LAION-5B的子集，比其他公共英语图像文本数据集大20多倍。LAION-2B的规模与用于语言视觉模型训练的最大私有数据集相当。

Downstream transfer and fine-tuning datasets. For downstream classification tasks, in addition to standard ImageNet, we follow [65] and use VTAB+, a collection of datasets in VTAB together with ImageNet derived robustness datasets and additional datasets, forming a comprehensive set of 35 tasks. For evaluating retrieval, we make use of MS-COCO and Flickr30K. For fine-tuning, we make use of a dedicated ImageNet-12k dataset (12M training examples, 470K validation examples) which is a subset of the full ImageNet-22k (14M examples) that we employ for the multi-stage fine tuning procedure described in Sec. 4.4. For more details on downstream datasets, refer to Table 25.

下游迁移和微调数据集。对于下游分类任务，除了标准的ImageNet之外，我们遵循[65]并使用VTAB+，VTAB+是VTAB中的数据集集合，与ImageNet衍生的稳健性数据集和其他数据集一起，形成了一个由35个任务组成的综合集合。为了评估检索，我们使用MS-COCO和Flickr30K。对于微调，我们使用专用的ImageNet-12k数据集(12M个训练样本，470K个验证样本)，它是完整ImageNet-22k(14M个样本)的子集，我们用于第4.4节中描述的多阶段微调过程。有关下游数据集的更多详情，请参阅表25。

Duplication check for pre-training and downstream datasets.. To ensure that images from downstream datasets are not contained in LAION, we conduct a simple duplication check based on the perceptual image hash library pHash [82]. We apply pHash’s discrete cosine transform (DCT) method on LAION-400M images and images from downstream datasets. Afterwards, for each downstream dataset, we count the number of duplicates by finding the hashes that are also present in LAION-400M. We provide the overlap percentage found on a subset of downstream datasets in Table 3. In Figure 7, we also provide a sample of images from downstream datasets detected as duplicates in LAION-400M. Overall, the ratio of detected duplicates is around 1%, except on ImageNet-R (3.80%) and ImageNet-Sketch (5.15%). We investigate further and re-evaluate zero-shot performance of our pre-trained Vit-H/14 on ImageNet-R and ImageNet-Sketch by removing duplicates from their test sets. For ImageNet-R, zero-shot top-1 accuracy goes from 89.32% to 89.21% after removing duplicates. For ImageNet-Sketch, zero-shot top-1 accuracy goes from 66.57% to 66.59% after removing duplicates. We conclude, based on those results, that it is unlikely that downstream results would be affected by the duplicates. This would be in line with previous works[55, 86] which explicitly measured and compared performance on deduplicated downstream datasets, reporting that duplication in test sets do not significantly alter most results. This is likely due to the very large scale and diversity of pre-training data. We leave more elaborated duplication detection procedures for future work.

预训练和下游数据集的重复检查。。为了确保来自下游数据集的图像不包含在LAION中，我们基于感知图像哈希库pHash[82]进行了简单的重复检查。我们将pHash的离散余弦变换(DCT)方法应用于LAION-400M图像和来自下游数据集的图像。然后，对于每个下游数据集，我们通过查找LAION-400M中也存在的哈希来计算重复的数量。我们在表3中提供了在下游数据集的子集上发现的重叠百分比。在图7中，我们还提供了来自下游数据集的图像样本，这些图像在LAION-400M中被检测为重复。总的来说，除了ImageNet-R(3.80%)和ImageNet-Sketch(5.15%)之外，检测到的重复项的比率约为1%。我们进一步调查并通过从测试集中删除重复项来重新评估我们的预训练Vit-H/14在ImageNet-R和ImageNet-Sketchs上的零样本性能。对于ImageNet-R，删除重复项后，零样本top-1精确度从89.32%提高到89.21%。对于ImageNet-Sketch，删除重复项后，零样本top-1精度从66.57%提高到66.59%。根据这些结果，我们得出的结论是，下游结果不太可能受到重复的影响。这与之前的工作[55，86]一致，之前的工作明确测量并比较了消除重复的下游数据集的性能，报告称测试集中的重复不会显著改变大多数结果。这可能是由于训练前数据的规模和多样性非常大。我们将更详细的重复检测程序留给未来的工作。

Table 3: Ratio of images (%) on downstream datasets that were detected on LAION-400M, using pHash [82].
表3：使用pHash[82]在LAION-400M上检测到的下游数据集上的图像比率(%)。

Figure 7: Duplicate images detected using pHash[82] between downstream datasets and LAION- 400M. Top row shows images from downstream datasets, while bottom row show corresponding detected duplicates in LAION-400M. We observe near-duplicate detection for a variety of image transformations: blurring, text blitting, color transformations, cropping, and scaling. Last two columns show false positive examples detected on ImageNet-Sketch dataset. In general, we observed that most of false positive cases had a uniform background, which pHash seems to be sensitive to.
图7：使用pHash[82]在下游数据集和LAION-400M之间检测到重复图像。顶行显示来自下游数据集的图像，而底行显示LAION-400M中相应的检测到的重复。我们观察到各种图像变换的近重复检测：模糊、文本闪电、颜色变换、裁剪和缩放。最后两列显示在ImageNet Sketch数据集上检测到的假阳性样本。总的来说，我们观察到大多数假阳性病例都有一个统一的背景，pHash似乎对此很敏感。

### B.2 Further experimental results B.2进一步的实验结果
#### B.2.1 Predictions derived from scaling laws B.2.1根据比例定律得出的预测
We can use scaling laws derived from our measurements to predict model performance for larger scales on different downstream tasks. To perform predictions, we fit a power-law on the Pareto frontier5 . Fig.8a and Fig.8b show extrapolation of performance for ImageNet and MS-COCO, respectively. According to the predictions, H/14 (68B samples seen) would achieve 79.73% (+1.76%) zero-shot top-1 accuracy on ImageNet and 75.10% (+1.60%) image retrieval Recall@5 on MS-COCO, compared to our trained H/14 (34B samples seen). For g/14 (68B samples seen), we predict 80.66% (+4%) zero-shot top-1 accuracy on ImageNet and 75.85% (+3.45%) image retrieval Recall@5 on MS-COCO, compared to our trained g/14 (13B samples seen). On the largest compute budget we consider, G/14 (68B samples seen), we predict 81.92% zero-shot top-1 accuracy on ImageNet and 76.99% image retrieval Recall@5 on MS-COCO.

我们可以使用从我们的测量中得出的缩放定律来预测不同下游任务上更大规模的模型性能。为了进行预测，我们在帕累托前沿拟合幂律5。图8a和图8b分别显示了ImageNet和MS-COCO的性能外推。根据预测，H/14(看到68B个样本)将在ImageNet上实现79.73%(+1.76%)的零样本top-1准确率和75.10%(+1.60%)的图像检索Recall@5在MS-COCO上，与我们训练的H/14(看到34B样本)相比。对于g/14(看到68B个样本)，我们预测ImageNet上80.66%(+4%)的零样本top-1准确率和75.85%(+3.45%)的图像检索Recall@5在MS-COCO上，与我们训练的g/14(见13B样本)相比。在我们考虑的最大计算预算G/14(已看到68B个样本)下，我们预测ImageNet上81.92%的零样本top-1准确率和76.99%的图像检索Recall@5在MS-COCO上。

5Since total compute budget (measured in GMAC) of different trained models are not exactly aligned, we adopt a binning approach. We bin the GMAC compute budget axis and compute the optimal performance within each bin, then fit a line in log-log space on the resulting bins.

5由于不同训练模型的总计算预算(以GMAC衡量)并不完全一致，我们采用了装箱方法。我们对GMAC计算预算轴进行分类，并计算每个分类中的最佳性能，然后在所得分类上的日志空间中拟合一行。

Figure 8: Zero-shot performance extrapolation of g/14, H/14 and G/14 on larger scales. We fit a power-law on the Pareto frontier of available models. In Fig.8a we show the predictions for ImageNet classification, while in Fig.8b we show the predictions for MS-COCO image retrieval.
图8：g/14、H/14和g/14在较大尺度上的零样本性能推断。我们在可用模型的帕累托边界上拟合了一个幂律。在图8a中，我们显示了对ImageNet分类的预测，而在图8b中，我们展示了对MS-COCO图像检索的预测。

#### B.2.2 Fine-tuning B.2.2微调
In Table 8, we show detailed results of fine-tuning on ImageNet with and without extra data (Imagenet-12k), and show results of the fine-tuned models on five ImageNet robustness test sets. Also, complementing the results shown in Figure 5 in Section 4.4, we show a per-task breakdown of the the zero-shot and fine-tuned performance on the eight classification tasks in Figures 9 and 10. Exact numbers are shown in Tables 5, 6, and 7.

在表8中，我们显示了在有和没有额外数据的情况下对ImageNet进行微调的详细结果(ImageNet-12k)，并显示了在五个ImageNet稳健性测试集上对模型进行微调的结果。此外，为了补充第4.4节图5所示的结果，我们在图9和图10中显示了八个分类任务的零样本和微调性能的每任务分解。具体数字如表5、表6和表7所示。

Moreover, since fine-tuning on some downstream tasks can decrease accuracy on others, we experiment with model patching by interpolating between the weights of fine-tuned and zero-shot models, as in Ilharco et al. [31].6 We choose the mixing coefficient α ∈ 0, 0.1, ..., 1.0 that maximizes average accuracy on the eight downstream tasks, while accuracy on ImageNet—used as a control— decreases by one percentage point or less. In Figure 11, we show how scale affects performance on the eight tasks we fine-tune one, along with that on ImageNet.

此外，由于一些下游任务的微调可能会降低其他任务的准确性，我们通过在微调和零样本模型的权重之间进行插值来进行模型修补，如Ilharco等人[31].6。我们选择混合系数α∈0，0.1，…，1.0，使八个下游任务的平均准确性最大化，而用作对照的ImageNet上的准确度则下降了一个百分点或更低。在图11中，我们展示了规模如何影响八项任务的性能，我们对其中一项任务进行了微调，并在ImageNet上进行了微调。

Finally, Tables 9 and 10 include hparam templates for reproducing ImageNet fine-tune results. Once published, the individual model weights will include their specific training hyper-parameters as there is some variation in specific instances (i.e. at different upscale sizes, from 12k to 1k). Motivated by BEiT [4], all ImageNet fine-tune runs make use of layer-wise learning-rate decay (also known as discriminative fine-tuning [29]); this is an important parameter that needs tuning per model size along with the learning-rate itself.

最后，表9和表10包括用于再现ImageNet微调结果的hparam模板。一旦发布，单个模型权重将包括其特定的训练超参数，因为在特定情况下(即在不同的高档尺寸下，从12k到1k)会有一些变化。受BEiT[4]的启发，所有ImageNet微调运行都使用逐层学习速率衰减(也称为判别微调[29]);这是一个重要的参数，需要根据模型大小以及学习率本身进行调整。

6The weights θpatched of the patched model are obtained via the equation θpatched = (1 − α)θzero-shot + αθfine-tuned, where α ∈ [0, 1] is the mixing coefficient.
6通过方程θpatched=(1−α)θ零样本+αθ精细微调，得到了修补模型的权重θpatchid，其中α∈[0，1]是混合系数。

Figure 9: Scaling trends of zero-shot models on the eight other downstream tasks used for the fine-tuning experiments in Section 4.4 and on ImageNet.
图9：第4.4节和ImageNet中用于微调实验的其他八个下游任务的零样本模型的缩放趋势。

Figure 10: Scaling trends of fine-tuned models on the eight other downstream tasks used for the fine-tuning experiments in Section 4.4.
图10：用于第4.4节微调实验的其他八个下游任务的微调模型的缩放趋势。

Figure 11: Scaling trends of patched models [31], on ImageNet and eight other downstream tasks used for the fine-tuning expe
图11：ImageNet上的修补模型[31]和用于微调expe的其他八个下游任务的缩放趋势

#### B.2.3 Control experiments B.2.3对照实验
Batch size during pre-training. To be able to train efficiently on a large number of GPUs (up to 1520 in this work), it is desired to maximize the local batch size for each GPU worker for performing data parallel distributed training. For this large amount of GPUs, it leads to training with global batch sizes of 86K-88K. As we would like to also re-use experiments that were already performed with smaller batch sizes of 32K-45K, we execute control experiments to reassure that varying batch size in those ranges does not alter observed model performance on downstream tasks strongly. The experiments summarized in Table 11 provide evidence that performance variation due to changes in batch size is small, in the range of 0.2 − 0.5% across different settings, which is small enough not to

预训练期间的批量大小。为了能够在大量GPU上高效地进行训练(本工作中最多1520个)，需要最大化每个GPU工作者的本地批大小，以执行数据并行分布式训练。对于如此大量的GPU，它导致了全局批量大小为86K-88K的训练。由于我们也想重复使用已经用32K-45K的较小批量进行的实验，我们执行对照实验，以确保在这些范围内改变批量不会强烈改变下游任务中观察到的模型性能。表11中总结的实验提供了证据，证明由于批次大小的变化而导致的性能变化很小，在不同设置下在0.2−0.5%的范围内，这足够小，不会

Table 4: Scaling model and data size leads to lower error linear classifers on ImageNet [15], CIFAR100 [43], and the visual task adaptation benchmark (VTAB) [85]. We train linear probes for models with at least 13B samples seen. We train probes by first caching the image features, thus no data augmentation is used. k shot denotes that k images per-class are used to train the linear probe.
表4：缩放模型和数据大小导致ImageNet[15]、CIFAR100[43]和视觉任务自适应基准(VTAB)[85]上的误差较低的线性分类器。我们为至少有13B样本的模型训练线性探针。我们通过首先缓存图像特征来训练探针，因此不使用数据增广。k shot表示每类的k个图像用于训练线性探针。

Table 5: Zero-shot accuracy for various models on downstream tasks from Section B.2.2.
表5：第B.2.2节中下游任务的各种模型的零样本精度。

Table 6: Accuracy after fine-tuning for various models on downstream tasks from Section B.2.2. We fine-tune jointly on the eight downstream image classification tasks, alternating batches from each task. We fine-tune only the parameters of the vision encoder, using a fixed classification head for each task initialized with the weights from the zero-shot model.
表6：第B.2.2节中下游任务的各种模型微调后的准确性。我们对八个下游图像分类任务进行联合微调，每个任务的批次交替进行。我们只微调视觉编码器的参数，对每个任务使用一个固定的分类头，用零样本模型中的权重初始化。

Table 7: Accuracy after joint patching [31] for various models on downstream tasks from Section B.2.2. Patching by jointly fine-tuning on the eight tasks with the exception of ImageNet (used only as control), then interpolating the weights of the fine-tuned model with the weights of the zero-shot model. The mixing coefficient for the interpolation is chosen so it maximizes average accuracy on the eight downstream tasks while maintaining ImageNet accuracy within 1 percentage point of the corresponding zero-shot model.
表7：第B.2.2节下游任务中各种模型的接缝修补后的精度[31]。通过对八个任务(ImageNet除外)进行联合微调进行修补(仅用作控制)，然后用零样本模型的权重插值微调模型的权重。选择插值的混合系数，以便最大化八个下游任务的平均精度，同时将ImageNet精度保持在相应零样本模型的1%以内。

Table 8: Fine-tune results for ImageNet-1k and associated robustness test sets (ImageNet-ReaL [7], ImageNet-V2 [60], ImageNet-A [24], Imagenet-R [22], and ImageNet-Sketch [75]). Rows with the ’Extra FT’ set to IN-12k were fine-tuned on a 12k class subset of ImageNet-22k before fine-tuning on ImageNet.
表8：ImageNet-1k和相关稳健性测试集(ImageNet-ReaL[7]、ImageNet-V2[60]、ImageNet-A[24]、ImageNet-R[22]和ImageNet-Sketch[75])的微调结果。在ImageNet上进行微调之前，在ImageNet-22k的12k类子集上对“Extra FT”设置为IN-12k的行进行了微调。

Table 9: ImageNet fine-tune hyper-parameters.
表9:ImageNet微调超参数。

Table 10: ImageNet-12k intermediate fine-tune hyper-parameters.  distort the trends observed in the effect of scale, where the changes are substantially larger.
表10:ImageNet-12k中间微调超参数。扭曲了在规模效应中观察到的趋势，其中变化要大得多。

Table 11: Batch size control experiments, zero-shot ImageNet top-1 accuracy. Executed on LAION- 400M, 13B samples seen (32 full epochs).
表11：批量大小控制实验，零样本ImageNet顶级精度。在LAION上执行-400M，看到13B样本(32个完整周期)。

LAION-400M and 400M subset of LAION-2B size. For 400M data scale, we are using LAION-400M dataset, as it was already validated by numerous previous works. This is not a subset of LAION-2B, as both were obtained by the same, but separately executed composition procedure using Common Crawl. To test that LAION-400M and LAION-2B can be considered as two different scale of the same data distribution, we extracted a random 400M subset from LAION-2B and conducted a pre-training experiment using our reference OpenCLIP ViT-B/32 model, 13B samples seen scale. We evaluated the pre-trained model on ImageNet zero-shot classification task, comparing it to same model pre-trained on LAION-400M. The outcome shows no significant difference between the performance of both models. This provides evidence that LAION-400M is comparable to a 400M subset extracted from LAION-2B, and can be thus considered to be a smaller scale of same data distribution.

LAION-400M和LAION-2B大小的400M子集。对于400M的数据规模，我们使用的是LAION-400M数据集，因为它已经被之前的许多工作所验证。这不是LAION-2B的子集，因为两者都是通过相同的、但使用Common Crawl单独执行的组合过程获得的。为了测试LAION-400M和LAION-2B可以被视为相同数据分布的两个不同规模，我们从LAION-2B中提取了一个随机的400M子集，并使用我们的参考OpenCLIP ViT-B/32模型进行了预训练实验，13B样本见规模。我们在ImageNet零样本分类任务中评估了预训练模型，并将其与在LAION-400M上预训练的相同模型进行了比较。结果显示，两种模型的性能之间没有显著差异。这提供了证据，表明LAION-400M与从LAION-2B中提取的400M子集相当，因此可以被认为是相同数据分布的较小规模。

Table 12: 400M data scale subset control experiments, zero-shot ImageNet top-1 accuracy. Executed either on 400M subset of LAION-2B or on LAION-400M, 13B samples seen (32 full epochs).
表12:400M数据规模子集控制实验，零样本ImageNet精度排名第一。在LAION-2B的400M子集上或在LAION-400M上执行，看到13B个样本(32个完整周期)。

Pre-training trial-to-trial variance. To have a sanity check of trial-to-trial variance for model pre-training, we trained our reference ViT-B/32 model, 13B samples seen scale, for two trials using exactly the same hyper-parameters (lr=0.001, batch size 86K, warm up 2K). We evaluated the two trials on ImageNet zero-shot classification task. The result suggests a small variance of around 0.1%, which is much smaller than variations observed when changing the scales. This allows us to conclude that scaling trends we observe are not distorted by variance caused by trial-to-trial pre-training.

训练前试验到试验的差异。为了对模型预训练的试验间方差进行健全性检查，我们使用完全相同的超参数(lr=0.001，批量大小86K，预热2K)训练了我们的参考ViT-B/32模型，即13B样本。我们对ImageNet零样本分类任务的两个试验进行了评估。结果表明，变化幅度较小，约为0.1%，远小于改变尺度时观察到的变化。这使我们能够得出结论，我们观察到的缩放趋势不会因试验到试验的预训练引起的差异而失真。

Table 13: Trial-to-trial variance control experiment. Executed on LAION-400M, 13B samples seen (32 full epochs) using ViT B/32 model.
表13：试验间方差控制实验。在LAION-400M上执行，使用ViT B/32模型看到13B样本(32个完整周期)。

Table 14: Resampling vs. full shuffling control experiments, zero-shot ImageNet top-1 accuracy. Executed on LAION-400M, 13B samples seen (32 full epochs).
表14：重采样与全混洗控制实验，零样本ImageNet的精确度排名第一。在LAION-400M上执行，看到13B样本（32个完整时期）。

Resampling vs full shuffled training. During our larger scale pre-training experiments featuring LAION-2B, it became important to allow for frequent checkpoint saving. Saving within a running epoch would require to memorize which samples were already seen, to be able to resume training in such a way that only previously not seen samples would be taken. To simplify the  procedure, we have tested a version that does not perform epoch-wise training, taking a pre-defined number of samples instead for a virtual "step" through data. Such a resampling procedure can have repeated samples in the subset of data that contains in total the number of samples equal to number of samples in one full epoch through the dataset. As such training procedure differs from standard epoch-wise training, we conducted test experiments to check whether this results in differences in performance of pre-trained models when comparing to standard epoch-wise shuffling training. We trained our reference ViT-B/32 model and ViT-B/16 model on LAION-400M either using standard epoch-wise training with shuffling or the training that involves described resampling procedure. We observed only negligible differences of 0.1%-0.3%, concluding that using simple resampling cannot distort scaling trends observed in the study.

重新采样与完全混洗训练。在我们以LAION-2B为特征的大规模预训练实验中，允许频繁保存检查点变得很重要。在一个运行的历元内保存需要记住哪些样本已经被看到，以便能够以这样一种方式恢复训练，即只采集以前没有看到的样本。为了简化程序，我们测试了一个不执行历元训练的版本，而是采用预定义数量的样本来进行数据的虚拟“步骤”。这样的重新采样过程可以在数据子集中具有重复的样本，该数据子集总共包含等于通过数据集的一个完整周期中的样本数量的样本数量。由于这种训练过程不同于标准的历元训练，我们进行了测试实验，以检查与标准的历世混洗训练相比，这是否会导致预训练模型的性能差异。我们在LAION-400M上训练了我们的参考ViT-B/32模型和ViT-B/16模型，要么使用带有混洗的标准历元训练，要么使用涉及所描述的重采样过程的训练。我们只观察到0.1%-0.3%的可忽略不计的差异，得出的结论是，使用简单的重新采样不会扭曲研究中观察到的缩放趋势。

Table 15: Detailed results on VTAB+ [65] zero-shot classification, where we average over 35 tasks.
表15：VTAB+[65]零样本分类的详细结果，其中我们平均超过35项任务。

Details of zero-shot classification results. Complementing results from the Section 4, we provide summary tables for the performance measure on different downstream tasks: ImageNet (Tab. 18), ImageNet robustness(Tab. 19), MS-COCO image retrieval (Tab. 20) and text retrieval (Tab. 21), Flickr30K image retrieval (Tab. 22) and text retrieval (Tab. 23), and VTAB+ (Tab. 15 and 16).

零样本分类结果的详情。作为对第4节结果的补充，我们提供了不同下游任务的性能度量汇总表：ImageNet(表18)、ImageNet稳健性(表19)、MS-COCO图像检索(表20)和文本检索(表21)、Flickr30K图像检索(图22)和文本检取(表23)以及VTAB+(表15和16)。

Details of linear probing results. To supplement Figures 2 and 3, we provide the corresponding Table 4 with detailed results.

线性探测结果的详情。为了补充图2和图3，我们在相应的表4中提供了详细的结果。

Architecture and training hyperparameters. We provide overview for architecture (Tab. 24) and pre-training hyper-parameters (Tab. 17) that we have used in the experiments.

架构和训练超参数。我们提供了我们在实验中使用的架构(表24)和预训练超参数(表17)的概述。

### C Code and Data availability C代码和数据可用性
We will provide source code used for running experiments and producing figures in this study at https://github.com/LAION-AI/scaling-laws-openclip. Links to pre-trained models obtained in this study and links to instructions for obtaining LAION-400m and LAION-5B used for pre-training experiments will be also made available there. All datasets used in the study are openly available and are listed together with references to the original work in Table 25.

我们将在https://github.com/LAION-AI/scaling-laws-openclip.本研究中获得的预训练模型的链接以及用于预训练实验的LAION-4000和LAION-5B的获取说明的链接也将在那里提供。研究中使用的所有数据集都是公开的，并在表25中与原始工作的参考文献一起列出。

### Broader and Social Impact 更广泛的社会影响
Safety aspect. Our work deals with studying function and properties of pre-trained models on large scales. Releasing these models to public can have both positive and negative implications, like with any research artefact that possesses generic functionality. We would like to stress that we consider the released pre-trained language-vision models as research artefacts that are there to advance the studies of scaling laws and allow analysis of the properties and behavior of such models for the broader research community. These models are not meant to be incorporated into end products or even used for applications in sensitive areas like interpretation of medical imaging in hospitals or security surveillance. There is potential for abuse of technology based on large-scale pre-trained generalist models, and it is the task of democratic institutions to work out rules for sensitive applications that might involve those. Open release of models gives the broad research community also opportunity to study safety related aspects of such models, such to preventively design measures that make such abuse by malicious parties less probable, in a common transparent effort. Same applies to the common effort of studying yet not systematically understood biases that such models may contain due to pre-training on either largely uncurated, imbalanced data or on data filtered by models that already contain unknown biases (like OpenAI’s CLIP that was trained on the private WIT-400M dataset), and due to the simplistic nature of the contrastive InfoNCE loss that drives learning.

安全方面。我们的工作是研究大规模预训练模型的函数和属性。向公众发布这些模型既有积极的影响，也有消极的影响，就像任何具有通用功能的研究成果一样。我们想强调的是，我们将发布的预训练语言视觉模型视为研究人工制品，用于推进标度律的研究，并允许为更广泛的研究群体分析此类模型的属性和行为。这些模型不打算被纳入最终产品中，甚至不打算用于敏感领域的应用，如医院医学成像的解释或安全监控。基于大规模预先训练的多面手模型的技术有可能被滥用，民主机构的任务是为可能涉及这些应用的敏感应用制定规则。模型的公开发布也为广泛的研究界提供了研究此类模型的安全相关方面的机会，例如通过共同透明的努力，预防性地设计措施，降低恶意方滥用的可能性。同样的情况也适用于研究尚未系统理解的偏见的共同努力，这些偏见可能是由于对基本上未评级、不平衡的数据或对已经包含未知偏见的模型过滤的数据进行预训练(如在私人WIT-400M数据集上训练的OpenAI的CLIP)而包含的，以及由于驱动学习的对比InfoNCE损失的简单化性质。

Energy cost. There is high computational cost bound to pre-training experiments on large scale. Supercomputers used in our studies are highly ranked in the Green Top-500 list, ensuring that energy costs are dampened. In addition, strongly transferable pre-trained models save energy on numerous downstream tasks where they can perform in data-efficient and thus in an energy saving manner. Releasing such pre-trained models to public incurs additional energy savings, as research community can re-use already validated models without necessity to train those from scratch again.

能源成本。大规模预训练实验的计算成本很高。我们研究中使用的超级计算机在绿色500强名单中排名很高，确保了能源成本的降低。此外，强可迁移的预训练模型可以在许多下游任务中节省能量，在这些任务中，它们可以高效地执行数据，从而以节能的方式执行。向公众发布这种预先训练的模型会带来额外的能源节约，因为研究界可以重复使用已经验证的模型，而无需重新从头开始训练。

### Author Contributions  作者贡献
* Mehdi Cherti: Planned and executed experiments on JUWELS Booster and Stability AI HPC, coordinated and performed results collection, distillation and analysis, performed deduplication analysis on LAION and downstream targest datasets, manuscript writing and revision. 
* Romain Beaumont: Planned and executed experiments on JUWELS Booster and Stability AI HPC, pioneered training of large scale openCLIP ViT H-14 and g-14, manuscript revision. 
* Ross Wightman: Planned and executed experiments on JUWELS Booster and Stability AI HPC, performed full fine-tuning experiments, manuscript writing and revision.  
* Mitchell Wortsman: experiment design and planning, performed experiments evaluating linear probing fine-tuning performance and robustness on ImageNet and other downstream datasets, manuscript writing and revision. 
* Gabriel Ilharco: experiment design and planning, performed experiments evaluating full finetuning performance and robustness on ImageNet and other downstream datasets, manuscript writing and revision. 
* Cade Gordon: implemented local contrastive loss for efficient distributed training, manuscript writing and revision. 
* Christoph Schuhmann: supervised larger scale experiments, resource and community organization, compute and storage resource acquisition, manuscript revision. 
* Ludwig Schmidt: provided advice on experiment design and study directions, manuscript writing and revision, general supervision 
* Jenia Jitsev: led the project; conducted experiments on various scales using JUWELS Booster and Stability AI HPC, scientific organization & manuscript writing, ethical and social content, experiments planning and design, compute and storage resource acquisition, general supervision. 

Table 16: Detailed zero-shot top-1 classification results of LAION-2B models on VTAB+ 35 tasks.
表16：关于VTAB+35任务的LAION-2B模型的详细零样本top-1分类结果。

Table 17: Training hyper-parameters and resources used to for pre-training our models on LAION 80M, 400M, and 2B subsets. Note that BS refer to batch size per GPU worker (with global the corresponding global batch size), LR to base learning rate, Warm to the total number of warmup steps, Time to total training time in hours, GPU-h to GPU hours, MWh to the total energy consumed in Megawatt hours.
表17：用于在LAION 80M、400M和2B子集上预训练我们的模型的训练超参数和资源。请注意，BS指的是每个GPU工作人员的批量大小(全局为相应的全局批量大小)，LR指的是基本学习率，Warm指的是预热步骤的总数，Time指的是总训练时间(以小时为单位)，GPU-h指的是GPU小时，MWh指消耗的总能量(以兆瓦时为单位)。

Table 18: Detailed results on ImageNet zero-shot accuracy.
表18：ImageNet零样本精度的详细结果。

Table 19: Detailed results on ImageNet five robustness datasets zero-shot accuracy (average over the five datasets is reported).
表19：ImageNet五个稳健性数据集零样本准确度的详细结果(报告了五个数据集的平均值)。

Table 20: Detailed results on MS-COCO image retrieval Recall@5.
表20：MS-COCO图像检索的详细结果Recall@5.

Table 21: Detailed results on MS-COCO text retrieval Recall@5.
表21：MS-COCO文本检索的详细结果Recall@5.

Table 22: Detailed results on Flickr30K image retrieval Recall@5.
表22：Flickr30K图像检索的详细结果Recall@5.

Table 23: Detailed results on Flickr30K text retrieval Recall@5.
表23：Flickr30K文本检索的详细结果Recall@5.

Table 24: Hyper-parameters of different architectures we consider. Emb refers to embedding size,
表24：我们考虑的不同架构的超参数。Emb是指嵌入尺寸，

Table 25: Datasets used for evaluating downstream performance. Adapted from [65].
表25：用于评估下游性能的数据集。改编自[65]。

