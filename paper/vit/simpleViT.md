# Better plain ViT baselines for ImageNet-1k
ImageNet-1k的更简单的ViT基线 2022.5.3 https://arxiv.org/abs/2205.01580

## Abstract
It is commonly accepted that the Vision Transformer model requires sophisticated regularization techniques to excel at ImageNet-1k scale data. Surprisingly, we find this is not the case and standard data augmentation is sufficient. This note presents a few minor modifications to the original Vision Transformer (ViT) vanilla training setting that dramatically improve the performance of plain ViT models. Notably, 90 epochs of training surpass 76% top-1 accuracy in under seven hours on a TPUv3-8, similar to the classic ResNet50 baseline, and 300 epochs of training reach 80% in less than one day.

人们普遍认为，视觉转换器模型需要复杂的正则化技术才能在ImageNet-1k尺度的数据上表现出色。令人惊讶的是，我们发现事实并非如此，标准的数据增广就足够了。本说明对原始视觉转换器(ViT)普通训练设置进行了一些小修改，显著提高了普通ViT模型的性能。值得注意的是，在TPVv3-8上，90个周期的训练在不到7小时的时间内超过了76%的前1名准确率，类似于经典的ResNet50基线，300个周期的训练在不到一天的时间内达到了80%。https://github.com/google-research/big_vision

## 1. Introduction
The ViT paper [4] focused solely on the aspect of largescale pre-training, where ViT models outshine well tuned ResNet [6] (BiT [8]) models. The addition of results when pre-training only on ImageNet-1k was an afterthought, mostly to ablate the effect of data scale. Nevertheless, ImageNet-1k remains a key testbed in the computer vision research and it is highly beneficial to have as simple and effective a baseline as possible.

ViT论文[4]仅关注大规模预训练方面，其中ViT模型胜过调整良好的ResNet[6](BiT[8])模型。仅在ImageNet-1k上进行预训练时添加结果是事后才想到的，主要是为了消除数据规模的影响。尽管如此，ImageNet-1k仍然是计算机视觉研究中的一个关键试验台，并且拥有尽可能简单有效的基线是非常有益的。

Thus, coupled with the release of the big vision codebase used to develop ViT [4], MLP-Mixer [14], ViT-G [19], LiT [20], and a variety of other research projects, we now provide a new baseline that stays true to the original ViT’s simplicity while reaching results competitive with similar approaches [15, 17] and concurrent [16], which also strives for simplification.

因此，随着用于开发ViT[4]、MLP-Mixer[14]、ViT-G[19]、LiT[20]和各种其他研究项目的大愿景代码库的发布，我们现在提供了一个新的基线，该基线保持了原始ViT的简单性，同时达到了与类似方法[15，17]和并发[16]相竞争的结果，该方法也致力于简化。

## 2. Experimental setup
We focus entirely on the ImageNet-1k dataset (ILSVRC- 2012) for both (pre)training and evaluation. We stick to the original ViT model architecture due to its widespread acceptance [1, 2, 5, 9, 15], simplicity and scalability, and revisit only few very minor details, none of which are novel. We choose to focus on the smaller ViT-S/16 variant introduced by [15] as we believe it provides a good tradeoff between iteration velocity with commonly available hardware and final accuracy. However, when more compute and data is available, we highly recommend iterating with ViT-B/32 or ViT-B/16 instead [12,19], and note that increasing patch-size is almost equivalent to reducing image resolution.

我们完全专注于ImageNet-1k数据集(ILSVRC-2012)，用于(预)训练和评估。由于其广泛接受[1，2，5，9，15]、简单性和可扩展性，我们坚持使用原始的ViT模型架构，并且只回顾了一些非常小的细节，没有一个是新颖的。我们选择关注[15]引入的较小的ViT-S/16变体，因为我们认为它在使用常用硬件的迭代速度和最终精度之间提供了良好的折衷。然而，当有更多的计算和数据可用时，我们强烈建议使用ViT-B/32或ViT-B/16进行迭代[12，19]，并注意，增加分块大小几乎等同于降低图像分辨率。

Figure 1. Comparison of ViT model for this note to state-of-the-art ViT and ResNet models. Left plot demonstrates how performance depends on the total number of epochs, while the right plot uses TPUv3-8 wallclock time to measure compute. We observe that our simple setting is highly competitive, even to the canonical ResNet-50 setups.
图1。本说明中的ViT模型与最先进的ViT和ResNet模型的比较。左图演示了性能如何取决于周期的总数，而右图使用TPUv3-8壁时钟时间来测量计算。我们观察到，我们的简单设置具有很强的竞争力，即使对于标准的ResNet-50设置也是如此。

All experiments use “inception crop” [13] at 224px² resolution, random horizontal flips, RandAugment [3], and Mixup augmentations. We train on the first 99% of the training data, and keep 1% for minival to encourage the community to stop selecting design choices on the validation (de-facto test) set. The full setup is shown in Appendix A.

所有实验都使用224px²分辨率的“初始裁剪”[13]、随机水平翻转、RandAugment[3]和Mixup增广。我们在前99%的训练数据上进行训练，并保留1%用于minival，以鼓励社区停止在验证(事实测试)集上选择设计选项。完整设置如附录A所示。

## 3. Results
The results for our improved setup are shown in Figure 1, along with a few related important baselines. It is clear that a simple, standard ViT trained this way can match both the seminal ResNet50 at 90 epochs baseline, as well as more modern ResNet [17] and ViT [16] training setups. Furthermore, on a small TPUv3-8 node, the 90 epoch run takes only 6h30, and one can reach 80% accuracy in less than a day when training for 300 epochs.

我们改进设置的结果如图1所示，还有一些相关的重要基线。很明显，以这种方式训练的简单、标准的ViT既可以匹配90个周期基线的开创性ResNet50，也可以匹配更现代的ResNet[17]和ViT[16]训练设置。此外，在一个小的TPUv3-8节点上，90个历元的运行只需要6h30，当训练300个历元时，可以在不到一天的时间内达到80%的准确率。

The main differences from [4, 12] are a batch-size of 1024 instead of 4096, the use of global average-pooling (GAP) instead of a class token [2, 11], fixed 2D sin-cos position embeddings [2], and the introduction of a small amount of RandAugment [3] and Mixup [21] (level 10 and probability 0.2 respectively, which is less than [12]). These small changes lead to significantly better performance than that originally reported in [4].

与[4，12]的主要区别是批大小为1024而不是4096，使用全局平均池(GAP)而不是类令牌[2，11]，固定的2D sin-cos位置嵌入[2]，以及引入少量RandAugment[3]和Mixup[21](分别为级别10和概率0.2，小于[12])。这些微小的变化导致了比[4]中最初报告的性能明显更好的性能。

Notably absent from this baseline are further architectural changes, regularizers such as dropout or stochastic depth [7], advanced optimization schemes such as SAM [10], extra augmentations such as CutMix [18], repeated augmentations [15], or blurring, “tricks” such as high-resolution finetuning or checkpoint averaging, as well as supervision from a strong teacher via knowledge distillation.

值得注意的是，该基线中没有进一步的架构变化、正则化因子(如丢弃或随机深度[7])、高级优化方案(如SAM[10])、额外增广(如CutMix[18])、重复增广[15]或模糊、“技巧”(如高分辨率微调或检查点平均)，以及通过知识蒸馏由强大的教师进行监督。

||90ep|150ep|300ep
---|---|---|---
Our improvements|76.5|78.5|80.0
no RandAug+MixUp|73.6|73.7|73.7
Posemb: sincos2d → learned|75.0|78.0|79.6
Batch-size: 1024 → 4096|74.7|77.3|78.6
Global Avgpool → [cls] token|75.0|76.9|78.2
Head: MLP → linear 76.7|78.6|79.8
Original + RandAug + MixUp|71.6|74.8|76.1
Original|66.8|67.2|67.1

Table 1. Ablation of our trivial modifications.
表1。消融我们琐碎的修改。

Table 1 shows an ablation of the various minor changes we propose. It exemplifies how a collection of almost trivial changes can accumulate to an important overall improvement. The only change which makes no significant difference in classification accuracy is whether the classification head is a single linear layer, or an MLP with one hidden tanh layer as in the original Transformer formulation.

表1显示了我们提出的各种微小变化的消融。它举例说明了一组几乎微不足道的更改是如何累积到一个重要的整体改进的。在分类精度上没有显著差异的唯一变化是分类头是单个线性层，还是像原始Transformer公式中那样具有一个隐藏tanh层的MLP。

## 4. Conclusion
It is always worth striving for simplicity.

追求简单总是值得的。

## Acknowledgements.
We thank Daniel Suo and Naman Agarwal for nudging for 90 epochs and feedback on the report, as well as the Google Brain team for a supportive research environment.

我们感谢Daniel Suo和Naman Agarwal对报告的90个周期的迭代和反馈，以及谷歌大脑团队提供的支持性研究环境。

||Top-1|ReaL|v2
---|---|---|---
Original (90ep)|66.8|72.8|52.2
Our improvements (90ep)|76.5|83.1|64.2
Our improvements (150ep)|78.5|84.5|66.4
Our improvements (300ep)|80.0|85.4|68.3

Table 2. A few more standard metrics.
表2. 还有一些标准指标。

## References
1. Wuyang Chen, Xianzhi Du, Fan Yang, Lucas Beyer, Xiaohua Zhai, Tsung-Yi Lin, Huizhong Chen, Jing Li, Xiaodan Song, Zhangyang Wang, and Denny Zhou. A simple singlescale vision transformer for object localization and instance segmentation. CoRR, abs/2112.09747, 2021. 1
2. Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In International Conference on Computer Vision (ICCV), 2021. 1, 2
3. Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. RandAugment: Practical data augmentation with no separate search. CoRR, abs/1909.13719, 2019. 1, 2
4. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021. 1, 2
5. Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross B. Girshick. Masked autoencoders are scalable vision learners. CoRR, abs/2111.06377, 2021. 1
6. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 1
7. Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q. Weinberger. Deep networks with stochastic depth. CoRR, abs/1603.09382, 2016. 2
8. Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big Transfer (BiT): General visual representation learning. In European Conference on Computer Vision (ECCV), 2020. 1
9. Yanghao Li, Hanzi Mao, Ross B. Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. CoRR, abs/2203.16527, 2022. 1
10. Yong Liu, Siqi Mai, Xiangning Chen, Cho-Jui Hsieh, and Yang You. Towards efficient and scalable sharpness-aware minimization. CoRR, abs/2203.02714, 2022. 2
11. Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? CoRR, abs/2108.08810, 2021. 2
12. Andreas Steiner, Alexander Kolesnikov, Xiaohua Zhai, Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train your ViT? data, augmentation, and regularization in vision transformers. CoRR, abs/2106.10270, 2021. 1, 2
13. Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 1 2
14. Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, et al. Mlpmixer: An all-mlp architecture for vision. Advances in Neural Information Processing Systems, 34, 2021. 1
15. Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herv´e J´egou. Training data-efficient image transformers & distillation through attention. In International Conference on Machine Learing (ICML), 2021. 1, 2
16. Hugo Touvron, Matthieu Cord, and Herv´e J´egou. DeiT III: revenge of the ViT. CoRR, abs/2204.07118, 2022. 1
17. Ross Wightman, Hugo Touvron, and Herv´e J´egou. ResNet strikes back: An improved training procedure in timm. CoRR, abs/2110.00476, 2021. 1
18. Sangdoo Yun, Dongyoon Han, Sanghyuk Chun, Seong Joon Oh, Youngjoon Yoo, and Junsuk Choe. CutMix: Regularization strategy to train strong classifiers with localizable features. In International Conference on Computer Vision (ICCV), 2019. 2
19. Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1
20. Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. LiT: Zero-shot transfer with locked-image text tuning. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1
21. Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations (ICLR), 2018. 2 

## A. big vision experiment configuration
```python
def get_config():
    config = mlc.ConfigDict()
    config.dataset = 'imagenet2012'
    config.train_split = 'train[:99%]'
    config.cache_raw = True
    config.shuffle_buffer_size = 250_000
    config.num_classes = 1000
    config.loss = 'softmax_xent'
    config.batch_size = 1024
    config.num_epochs = 90

    pp_common = (
        '|value_range(-1, 1)'
        '|onehot(1000, key="{lbl}", key_result="
        labels")'
        '|keep("image", "labels")'
    )
    config.pp_train = (
        'decode_jpeg_and_inception_crop(224)' +
        '|flip_lr|randaug(2,10)' +
        pp_common.format(lbl='label')
    )
    pp_eval = 'decode|resize_small(256)|
    central_crop(224)' + pp_common

    config.log_training_steps = 50
    config.log_eval_steps = 1000
    config.checkpoint_steps = 1000

    # Model section
    config.model_name = 'vit'
    config.model = dict(
        variant='S/16',
        rep_size=True,
        pool_type='gap',
        posemb='sincos2d',
    )

    # Optimizer section
    config.grad_clip_norm = 1.0
    config.optax_name = 'scale_by_adam'
    config.optax = dict(mu_dtype='bfloat16')
    config.lr = 0.001
    config.wd = 0.0001
    config.schedule = dict(warmup_steps=10_000,
    decay_type='cosine')
    config.mixup = dict(p=0.2, fold_in=None)

    # Eval section
    config.evals = [
        ('minival', 'classification'),
        ('val', 'classification'),
        ('real', 'classification'),
        ('v2', 'classification'),
    ]
    eval_common = dict(
        pp_fn=pp_eval.format(lbl='label'),
        loss_name=config.loss,
        log_steps=1000,
    )

    config.minival = dict(**eval_common)
    config.minival.dataset = 'imagenet2012'
    config.minival.split = 'train[99%:]'
    config.minival.prefix = 'minival_'

    config.val = dict(**eval_common)
    config.val.dataset = 'imagenet2012'
    config.val.split = 'validation'
    config.val.prefix = 'val_'

    config.real = dict(**eval_common)
    config.real.dataset = 'imagenet2012_real'
    config.real.split = 'validation'
    config.real.pp_fn = pp_eval.format(lbl='
    real_label')
    config.real.prefix = 'real_'

    config.v2 = dict(**eval_common)
    config.v2.dataset = 'imagenet_v2'
    config.v2.split = 'test'
    config.v2.prefix = 'v2_'

    return config
```
Listing 1. Full recommended config