# TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models TrOCR：
基于变换器的预训练模型的光学字符识别

## Abstract 摘要
Text recognition is a long-standing research problem for document digitalization. Existing approaches are usually built based on CNN for image understanding and RNN for charlevel text generation. In addition, another language model is usually needed to improve the overall accuracy as a postprocessing step. In this paper, we propose an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR, which leverages the Transformer architecture for both image understanding and wordpiece-level text generation. The TrOCR model is simple but effective, and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. Experiments show that the TrOCR model outperforms the current state-of-the-art models on the printed, handwritten and scene text recognition tasks. The TrOCR models and code are publicly available at https://aka.ms/trocr.

文本识别是文档数字化中一个长期存在的研究问题。现有的方法通常基于用于图像理解的CNN和用于charlevel文本生成的RNN。此外，作为后处理步骤，通常需要另一种语言模型来提高整体准确性。在本文中，我们提出了一种端到端的文本识别方法，该方法具有预先训练的图像转换器和文本转换器模型，即TrOCR，它利用转换器架构来进行图像理解和分词级别的文本生成。TrOCR模型简单但有效，可以使用大规模合成数据进行预训练，并使用人类标记的数据集进行微调。实验表明，TrOCR模型在打印、手写和场景文本识别任务上优于当前最先进的模型。TrOCR模型和代码可在https://aka.ms/trocr.

## Introduction 引言
Optical Character Recognition (OCR) is the electronic or mechanical conversion of images of typed, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene photo or from subtitle text superimposed on an image. Typically, an OCR system includes two main modules: a text detection module and a text recognition module. Text detection aims to localize all text blocks within the text image, either at word-level or textline-level. The text detection task is usually considered as an object detection problem where conventional object detection models such as YoLOv5 and DBNet (Liao et al. 2019) can be applied. Meanwhile, text recognition aims to understand the text image content and transcribe the visual signals into natural language tokens. The text recognition task is usually framed as an encoderdecoder problem where existing methods leveraged CNNbased encoder for image understanding and RNN-based decoder for text generation. In this paper, we focus on the text recognition task for document images and leave text detection as the future work.

光学字符识别（OCR）是将打字、手写或打印文本的图像电子或机械转换为机器编码文本，无论是从扫描文档、文档照片、场景照片还是从叠加在图像上的字幕文本。通常，OCR系统包括两个主要模块：文本检测模块和文本识别模块。文本检测旨在定位文本图像中的所有文本块，无论是在单词级别还是在文本行级别。文本检测任务通常被认为是一个对象检测问题，其中可以应用传统的对象检测模型，如YoLOv5和DBNet（Liao et al.2019）。同时，文本识别旨在理解文本图像内容，并将视觉信号转录为自然语言表征。文本识别任务通常被定义为编码器-编码器问题，其中现有的方法利用基于CNN的编码器来理解图像，利用基于RNN的解码器来生成文本。在本文中，我们专注于文档图像的文本识别任务，并将文本检测作为未来的工作。

Recent progress in text recognition (Diaz et al. 2021) has witnessed the significant improvements by taking advantage of the Transformer (Vaswani et al. 2017) architec- *Work done during internship at Microsoft Research Asia. tures. However, existing methods are still based on CNNs as the backbone, where the self-attention is built on top of CNN backbones as encoders to understand the text image.

文本识别的最新进展（Diaz等人，2021）见证了利用Transformer（Vaswani等人，2017）architec-*在微软亚洲研究院实习期间所做的工作的显著改进。图尔斯。然而，现有的方法仍然是基于CNN作为主干，其中自我关注建立在作为编码器的CNN主干之上，以理解文本图像。

For decoders, Connectionist Temporal Classification (CTC) (Graves et al. 2006) is usually used compounded with an external language model on the character-level to improve the overall accuracy. Despite the great success achieved by the hybrid encoder/decoder method, there is still a lot of room to improve with pre-trained CV and NLP models: 1) the network parameters in existing methods are trained from scratch with synthetic/human-labeled datasets, leaving large-scale pre-trained models unexplored. 2) as image Transformers become more and more popular (Dosovitskiy et al. 2021; Touvron et al. 2021), especially the recent selfsupervised image pre-training (Bao, Dong, and Wei 2021), it is straightforward to investigate whether pre-trained image Transformers can replace CNN backbones, meanwhile exploiting the pre-trained image Transformers to work together with the pre-trained text Transformers in a single framework on the text recognition task.

对于解码器，连接主义时间分类（CTC）（Graves等人，2006）通常与字符级别的外部语言模型结合使用，以提高整体准确性。尽管混合编码器/解码器方法取得了巨大成功，但预训练的CV和NLP模型仍有很大的改进空间：1）现有方法中的网络参数是用合成/人类标记的数据集从头开始训练的，而大规模的预训练模型尚未探索。2） 随着图像转换器越来越流行（Dosovitskiy et al.2021；Touvron et al.2011），特别是最近的自我监督图像预训练（Bao，Dong，and Wei 2021），很容易研究预训练的图像转换器是否可以取代CNN骨干，同时利用预先训练的图像转换器与预先训练的文本转换器在一个单一的框架中协同工作完成文本识别任务。

To this end, we propose TrOCR, an end-to-end Transformer-based OCR model for text recognition with pre-trained CV and NLP models, which is shown in Figure 1. Distinct from the existing text recognition models, TrOCR is a simple but effective model which does not use the CNN as the backbone. Instead, following (Dosovitskiy et al. 2021), it first resizes the input text image into 384×384 and then the image is split into a sequence of 16×16 patches which are used as the input to image Transformers. Standard Transformer architecture with the self-attention mechanism is leveraged on both encoder and decoder parts, where wordpiece units are generated as the recognized text from the input image. To effectively train the TrOCR model, the encoder can be initialized with pre-trained ViT-style models (Dosovitskiy et al. 2021; Touvron et al. 2021; Bao, Dong, and Wei 2021) while the decoder can be initialized with pre-trained BERT-style models (Devlin et al. 2019; Liu et al. 2019; Dong et al. 2019; Wang et al. 2020b), respectively. Therefore, the advantage of TrOCR is three-fold.

为此，我们提出了TrOCR，这是一种基于Transformer的端到端OCR模型，用于使用预先训练的CV和NLP模型进行文本识别，如图1所示。与现有的文本识别模型不同，TrOCR是一种简单而有效的模型，它不使用CNN作为主干。相反，接下来（Dosovitskiy等人，2021），它首先将输入的文本图像调整为384×384的大小，然后将图像分割为16×16个补丁的序列，这些补丁用作图像转换器的输入。具有自注意机制的标准Transformer架构在编码器和解码器部分都得到了利用，其中字片单元被生成为来自输入图像的已识别文本。为了有效地训练TrOCR模型，编码器可以用预先训练的ViT风格模型初始化（Dosovitskiy等人2021；Touvron等人2021；Bao、Dong和Wei 2021），而解码器可以用预先培训的BERT风格模型来初始化（Devlin等人2019；刘等人2019；Dong等人2019；王等人2020b）。因此，TrOCR的优点有三个方面。

First, TrOCR uses the pre-trained image Transformer and text Transformer models, which take advantages of largescale unlabeled data for image understanding and language modeling, with no need for an external language model. Second, TrOCR does not require any convolutional network for the backbone and does not introduce any image-specific inductive biases, which makes the model very easy to implement and maintain. Finally, experiment results on OCR benchmark datasets show that the TrOCR can achieve stateof-the-art results on printed, handwritten and scene text image datasets without any complex pre/post-processing steps. Furthermore, we can easily extend the TrOCR for multilingual text recognition with minimum efforts, where just leveraging multilingual pre-trained models in the decoderside and expand the dictionary.

首先，TrOCR使用预先训练的图像转换器和文本转换器模型，它们利用大规模未标记数据进行图像理解和语言建模，而不需要外部语言模型。其次，TrOCR不需要任何卷积网络作为主干，也不引入任何图像特定的归纳偏差，这使得模型非常容易实现和维护。最后，在OCR基准数据集上的实验结果表明，TrOCR可以在打印、手写和场景文本图像数据集上实现最先进的结果，而无需任何复杂的前/后处理步骤。此外，我们可以毫不费力地将TrOCR扩展到多语言文本识别，只需在解码器中利用多语言预先训练的模型并扩展字典。

Figure 1: The architecture of TrOCR, where an encoder-decoder model is designed with a pre-trained image Transformer as the encoder and a pre-trained text Transformer as the decoder. 

图1:TrOCR的体系结构，其中编码器-解码器模型设计为以预训练的图像转换器作为编码器，以预先训练的文本转换器作为解码器。

The contributions of this paper are summarized as follows:

本文的贡献总结如下：

1. We propose TrOCR, an end-to-end Transformer-based OCR model for text recognition with pre-trained CV and NLP models. To the best of our knowledge, this is the first work that jointly leverages pre-trained image and text Transformers for the text recognition task in OCR.

1.我们提出了TrOCR，这是一种基于Transformer的端到端OCR模型，用于具有预先训练的CV和NLP模型的文本识别。据我们所知，这是第一项联合利用预先训练的图像和文本转换器进行OCR中的文本识别任务的工作。

2. TrOCR achieves state-of-the-art results with a standard Transformer-based encoder-decoder model, which is convolution free and does not rely on any complex pre/post-processing steps.

2.TrOCR使用标准的基于Transformer的编码器-解码器模型实现了最先进的结果，该模型无卷积，不依赖于任何复杂的前/后处理步骤。

3. The TrOCR models and code are publicly available at https://aka.ms/trocr.

3.TrOCR模型和代码可在https://aka.ms/trocr.

## TrOCR 转换OCR
### Model Architecture 模型体系结构
TrOCR is built up with the Transformer architecture, including an image Transformer for extracting the visual features and a text Transformer for language modeling. We adopt the vanilla Transformer encoder-decoder structure in TrOCR. The encoder is designed to obtain the representation of the image patches and the decoder is to generate the wordpiece sequence with the guidance of the visual features and previous predictions.

TrOCR是使用Transformer架构构建的，包括用于提取视觉特征的图像Transformer和用于语言建模的文本Transformer。我们在TrOCR中采用了香草变压器编码器-解码器结构。编码器被设计为获得图像块的表示，解码器在视觉特征和先前预测的指导下生成单词序列。

#### Encoder  编码器
The encoder receives an input image ximg ∈ < 3×H0×W0 , and resizes it to a fixed size (H, W). Since the Transformer encoder cannot process the raw images unless they are a sequence of input tokens, the encoder decomposes the input image into a batch of N = HW/P2 foursquare patches with a fixed size of (P, P), while the width W and the height H of the resized image are guaranteed to be divisible by the patch size P. Subsequently, the patches are flattened into vectors and linearly projected to D-dimensional vectors, aka the patch embeddings. D is the hidden size of the Transformer through all of its layers.

编码器接收输入图像ximg∈<3×H0×W0，并将其调整为固定大小（H，W）。由于Transformer编码器不能处理原始图像，除非它们是输入令牌序列，因此编码器将输入图像分解为一批具有固定大小（P，P）的N＝HW/P2四方形块，同时保证调整大小的图像的宽度W和高度H可被块大小P整除。随后，补丁被展平为向量并线性投影为D维向量，也就是补丁嵌入。D是Transformer在其所有层中隐藏的大小。

Similar to ViT (Dosovitskiy et al. 2021) and DeiT (Touvron et al. 2021), we keep the special token “[CLS]” that is usually used for image classification tasks. The “[CLS]” token brings together all the information from all the patch embeddings and represents the whole image. Meanwhile, we also keep the distillation token in the input sequence when using the DeiT pre-trained models for encoder initialization, which allows the model to learn from the teacher model. The patch embeddings and two special tokens are given learnable 1D position embeddings according to their absolute positions.

类似于ViT（Dosovitskiy et al.2021）和DeiT（Touvron et al.2011），我们保留了通常用于图像分类任务的特殊标记“[CLS]”。“[CLS]”标记将来自所有补丁嵌入的所有信息汇集在一起，并表示整个图像。同时，当使用DeiT预训练模型进行编码器初始化时，我们还将提取令牌保留在输入序列中，这允许模型从教师模型中学习。根据补丁嵌入和两个特殊令牌的绝对位置，它们被赋予可学习的1D位置嵌入。

Unlike the features extracted by the CNN-like network, the Transformer models have no image-specific inductive biases and process the image as a sequence of patches, which makes the model easier to pay different attention to either the whole image or the independent patches.

与类CNN网络提取的特征不同，Transformer模型没有图像特定的归纳偏差，并将图像处理为一系列补丁，这使得模型更容易对整个图像或独立补丁给予不同的关注。

#### Decoder  解码器
We use the original Transformer decoder for TrOCR. The standard Transformer decoder also has a stack of identical layers, which have similar structures to the layers in the encoder, except that the decoder inserts the “encoder-decoder attention” between the multi-head selfattention and feed-forward network to distribute different attention on the output of the encoder. In the encoder-decoder attention module, the keys and values come from the en- coder output, while the queries come from the decoder input. In addition, the decoder leverages the attention masking in the self-attention to prevent itself from getting more information during training than prediction. Based on the fact that the output of the decoder will right shift one place from the input of the decoder, the attention mask needs to ensure the output for the position i can only pay attention to the previous output, which is the input on the positions less than i: 

我们使用原始的Transformer解码器进行TrOCR。标准Transformer解码器也有一堆相同的层，这些层与编码器中的层具有相似的结构，只是解码器在多头自注意和前馈网络之间插入“编码器-解码器注意”，以在编码器的输出上分配不同的注意。在编码器-解码器注意力模块中，键和值来自编码器的输出，而查询来自解码器的输入。此外，解码器利用自注意中的注意掩蔽来防止自己在训练期间获得比预测更多的信息。基于解码器的输出将从解码器的输入右移一位的事实，注意力掩码需要确保位置i的输出只能关注前一个输出，即小于i的位置上的输入：

hi = P roj(Emb(T okeni)) σ(hij ) = e hij P V k=1 e hik for j = 1, 2, . . . , V

hi=P roj（Emb（T okeni））σ（hij）=e hij P V k=1 e hik对于j=1，2，五、

The hidden states from the decoder are projected by a linear layer from the model dimension to the dimension of the vocabulary size V , while the probabilities over the vocabulary are calculated on that by the softmax function. We use beam search to get the final output.

解码器的隐藏状态由线性层从模型维度投影到词汇表大小V的维度，而词汇表上的概率由softmax函数计算。我们使用波束搜索来获得最终输出。

### Model Initialization 模型初始化
Both the encoder and the decoder are initialized by the public models pre-trained on large-scale labeled and unlabeled datasets.

编码器和解码器都是由在大规模标记和未标记数据集上预先训练的公共模型初始化的。

#### Encoder Initialization  编码器初始化
The DeiT (Touvron et al. 2021) and BEiT (Bao, Dong, and Wei 2021) models are used for the encoder initialization in the TrOCR models. DeiT trains the image Transformer with ImageNet (Deng et al. 2009) as the sole training set. The authors try different hyperparameters and data augmentation to make the model dataefficient. Moreover, they distill the knowledge of a strong image classifier to a distilled token in the initial embedding, which leads to a competitive result compared to the CNNbased models.

DeiT（Touvron等人，2021）和BEiT（Bao，Dong和Wei 2021）模型用于TrOCR模型中的编码器初始化。DeiT使用ImageNet（Deng等人，2009）作为唯一的训练集来训练图像转换器。作者尝试使用不同的超参数和数据扩充来提高模型的数据效率。此外，他们在初始嵌入中将强图像分类器的知识提取为提取的令牌，这导致了与基于CNN的模型相比具有竞争力的结果。

Referring to the Masked Language Model pre-training task, BEiT proposes the Masked Image Modeling task to pre-train the image Transformer. Each image will be converted to two views: image patches and visual tokens. They tokenize the original image into visual tokens by the latent codes of discrete VAE (Ramesh et al. 2021), randomly mask some image patches, and make the model recover the original visual tokens. The structure of BEiT is the same as the image Transformer and lacks the distilled token when compared with DeiT.

参考掩蔽语言模型预训练任务，BEiT提出了掩蔽图像建模任务来预训练图像转换器。每个图像将转换为两个视图：图像补丁和视觉标记。他们通过离散VAE的潜在代码将原始图像标记为视觉标记（Ramesh等人，2021），随机屏蔽一些图像补丁，并使模型恢复原始视觉标记。BEiT的结构与图像转换器相同，并且与DeiT相比缺少提取的令牌。

#### Decoder Initialization  解码器初始化
We use the RoBERTa (Liu et al. 2019) models and the MiniLM (Wang et al. 2020b) models to initialize the decoder. Generally, RoBERTa is a replication study of (Devlin et al. 2019) that carefully measures the impact of many key hyperparameters and training data size. Based on BERT, they remove the next sentence prediction objective and dynamically change the masking pattern of the Masked Language Model.

我们使用RoBERTa（Liu等人，2019）模型和MiniLM（Wang等人，2020b）模型来初始化解码器。一般来说，RoBERTa是（Devlin等人，2019）的一项复制研究，它仔细测量了许多关键超参数和训练数据大小的影响。基于BERT，他们去除了下一句预测目标，并动态地改变了掩蔽语言模型的掩蔽模式。

The MiniLM are compressed models of the large pretrained Transformer models while retaining 99% performance. Instead of using the soft target probabilities of masked language modeling predictions or intermediate representations of the teacher models to guide the training of the student models in the previous work. The MiniLM models are trained by distilling the self-attention module of the last Transformer layer of the teacher models and introducing a teacher assistant to assist with the distillation.

MiniLM是大型预训练变压器模型的压缩模型，同时保持99%的性能。而不是在先前的工作中使用掩蔽语言建模预测的软目标概率或教师模型的中间表示来指导学生模型的训练。MiniLM模型是通过提取教师模型的最后一个Transformer层的自注意模块并引入教师助理来帮助提取来训练的。

When loading the above models to the decoders, the structures do not precisely match since both of them are only the encoder of the Transformer architecture. For example, the encoder-decoder attention layers are absent in these models. To address this, we initialize the decoders with the RoBERTa and MiniLM models by manually setting the corresponding parameter mapping, and the absent parameters are randomly initialized.

当将上述模型加载到解码器时，结构并不精确匹配，因为它们都只是Transformer架构的编码器。例如，在这些模型中不存在编码器-解码器注意力层。为了解决这一问题，我们通过手动设置相应的参数映射，用RoBERTa和MiniLM模型初始化解码器，并随机初始化缺失的参数。

### Task Pipeline 任务管道
In this work, the pipeline of the text recognition task is that given the textline images, the model extracts the visual features and predicts the wordpiece tokens relying on the image and the context generated before. The sequence of ground truth tokens is followed by an “[EOS]” token, which indicates the end of a sentence. During training, we shift the sequence backward by one place and add the “[BOS]” token to the beginning indicating the start of generation. The shifted ground truth sequence is fed into the decoder, and the output of that is supervised by the original ground truth sequence with the cross-entropy loss. For inference, the decoder starts from the “[BOS]” token to predict the output iteratively while continuously taking the newly generated output as the next input.

在这项工作中，文本识别任务的管道是，给定文本行图像，模型提取视觉特征，并根据之前生成的图像和上下文预测单词标记。地面实况标记的序列后面跟着一个“[EOS]”标记，表示句子的结尾。在训练过程中，我们将序列向后移动一位，并在开头添加“[BOS]”标记，指示生成的开始。移位的地面实况序列被馈送到解码器，并且其输出由具有交叉熵损失的原始地面实况序列监督。为了进行推理，解码器从“[BOS]”令牌开始迭代预测输出，同时连续地将新生成的输出作为下一个输入。

### Pre-training 预培训
We use the text recognition task for the pre-training phase, since this task can make the models learn the knowledge of both the visual feature extraction and the language model. The pre-training process is divided into two stages that differ by the used dataset. In the first stage, we synthesize a largescale dataset consisting of hundreds of millions of printed textline images and pre-train the TrOCR models on that. In the second stage, we build two relatively small datasets corresponding to printed and handwritten downstream tasks, containing millions of textline images each. We use the existed and widely adopted synthetic scene text datasets for the scene text recognition task. Subsequently, we pre-train separate models on these task-specific datasets in the second stage, all initialized by the first-stage model.

我们在预训练阶段使用文本识别任务，因为该任务可以使模型学习视觉特征提取和语言模型的知识。预训练过程分为两个阶段，这两个阶段因使用的数据集而异。在第一阶段，我们合成了一个由数亿打印文本行图像组成的大规模数据集，并在此基础上预训练TrOCR模型。在第二阶段，我们构建了两个相对较小的数据集，分别对应于打印和手写的下游任务，每个数据集包含数百万个文本行图像。我们使用现有的和广泛采用的合成场景文本数据集进行场景文本识别任务。随后，我们在第二阶段中在这些特定于任务的数据集上预训练单独的模型，所有模型都由第一阶段模型初始化。

### Fine-tuning 微调
Except for the experiments regarding scene text recognition, the pre-trained TrOCR models are fine-tuned on the downstream text recognition tasks. The outputs of the TrOCR models are based on Byte Pair Encoding (BPE) (Sennrich, Haddow, and Birch 2015) and SentencePiece (Kudo and Richardson 2018) and do not rely on any task-related vocabularies.

除了关于场景文本识别的实验外，预训练的TrOCR模型在下游文本识别任务上进行了微调。TrOCR模型的输出基于字节对编码（BPE）（Sennrich、Haddow和Birch 2015）和句子片段（Kudo和Richardson 2018），不依赖于任何与任务相关的词汇。

### Data Augmentation 数据扩充
We leverage data augmentation to enhance the variety of the pre-training and fine-tuning data. Six kinds of image transformations plus keeping the original are taken for printed and handwritten datasets, which are random rotation (-10 to 10 degrees), Gaussian blurring, image dilation, image erosion, downscaling, and underlining. We randomly decide which image transformation to take with equal possibilities for each sample. For scene text datasets, RandAugment (Cubuk et al. 2020) is applied following (Atienza 2021), and the augmentation types include inversion, curving, blur, noise, distortion, rotation, etc.

我们利用数据扩充来增强预训练和微调数据的多样性。打印和手写数据集采用了六种图像转换加上保持原始，即随机旋转（-10到10度）、高斯模糊、图像膨胀、图像侵蚀、缩小和下划线。我们随机决定对每个样本进行相同可能性的图像变换。对于场景文本数据集，RandAugment（Cubuk et al.2020）应用如下（Atienza 2021），增强类型包括反转、弯曲、模糊、噪声、失真、旋转等。

Table 1: Ablation study on the SROIE dataset, where all the models are trained using the SROIE dataset only.

表1:SROIE数据集上的消融研究，其中所有模型仅使用SROIE数据集进行训练。

Table 2: Ablation study of pretrained model initialization, data augmentation and two stages of pre-training on the SROIE dataset. 

表2：SROIE数据集上预训练模型初始化、数据扩充和两个阶段预训练的消融研究。

## Experiments 实验
### Data 数据
#### Pre-training Dataset  预训练数据集
To build a large-scale high-quality dataset, we sample two million document pages from the publicly available PDF files on the Internet. Since the PDF files are digital-born, we can get pretty printed textline images by converting them into page images and extracting the textlines with their cropped images. In total, the first-stage pre-training dataset contains 684M textlines.

为了构建一个大规模的高质量数据集，我们从互联网上公开的PDF文件中抽取了200万个文档页面。由于PDF文件是数字生成的，我们可以通过将它们转换为页面图像并用裁剪后的图像提取文本线来获得漂亮的打印文本线图像。总的来说，第一阶段的预训练数据集包含684M个文本行。

We use 5,427 handwritten fonts1 to synthesize handwritten textline images by the TRDG2 , an open-source text recognition data generator. The text used for generation is crawled from random pages of Wikipedia. The handwritten dataset for the second-stage pre-training consists of 17.9M textlines, including IIIT-HWS dataset (Krishnan and Jawahar 2016). In addition, we collect around 53K receipt images in the real world and recognize the text on them by commercial OCR engines. According to the results, we crop the textlines by their coordinates and rectify them into normalized images. We also use TRDG to synthesize 1M printed textline images with two receipt fonts and the builtin printed fonts. In total, the printed dataset consists of 3.3M textlines. The second-stage pre-training data for the scene text recognition are MJSynth (MJ) (Jaderberg et al. 2014) and SynthText (ST) (Gupta, Vedaldi, and Zisserman 2016), totaling about 16M text images.

我们使用5427个手写字体1，通过开源文本识别数据生成器TRDG2合成手写文本行图像。用于生成的文本是从维基百科的随机页面中抓取的。第二阶段预训练的手写数据集由179M个文本行组成，包括IIIT-HWS数据集（Krishnan和Jawahar，2016）。此外，我们在现实世界中收集了大约53K张收据图像，并通过商业OCR引擎识别上面的文本。根据结果，我们根据文本线的坐标对其进行裁剪，并将其校正为归一化图像。我们还使用TRDG来合成具有两种收据字体和内置打印字体的1M打印文本行图像。总的来说，打印的数据集由330万条文本行组成。用于场景文本识别的第二阶段预训练数据是MJSynth（MJ）（Jaderberg等人，2014）和SynthText（ST）（Gupta、Vedaldi和Zisserman，2016），总计约1600万个文本图像。

1 The fonts are obtained from https://fonts.google.com/?category=Handwriting and https:// www.1001fonts.com/handwritten-fonts.html. 

1字体来自https://fonts.google.com/?category=Handwriting以及https://www.1001fonts.com/handwritten-fonts.html。

2 https://github.com/Belval/TextRecognitionDataGenerator 

2.https://github.com/Belval/TextRecognitionDataGenerator

#### Benchmarks  基准
The SROIE (Scanned Receipts OCR and Information Extraction) dataset (Task 2) focuses on text recognition in receipt images. There are 626 receipt images and 361 receipt images in the training and test sets of SROIE. Since the text detection task is not included in this work, we use cropped images of the textlines for evaluation, which are obtained by cropping the whole receipt images according to the ground truth bounding boxes.

SROIE（扫描收据OCR和信息提取）数据集（任务2）专注于收据图像中的文本识别。SROIE的训练集和测试集中分别有626张和361张收据图像。由于文本检测任务不包括在这项工作中，我们使用文本行的裁剪图像进行评估，这些图像是通过根据基本事实边界框裁剪整个收据图像而获得的。

The IAM Handwriting Database is composed of handwritten English text, which is the most popular dataset for handwritten text recognition. We use the Aachen’s partition of the dataset3 : 6,161 lines from 747 forms in the train set, 966 lines from 115 forms in the validation set and 2,915 lines from 336 forms in the test set.

IAM手写数据库由手写英文文本组成，是最流行的手写文本识别数据集。我们使用了数据集3的亚琛划分：列车集中747个表格中的6161行，验证集中115个表格的966行，测试集中336个表格的2915行。

Recognizing scene text images is more challenging than printed text images, as many images in the wild suffer from blur, occlusion, or low-resolution problems. Here we leverage some widely-used benchmarks, including IIIT5K-3000 (Mishra, Alahari, and Jawahar 2012), SVT-647 (Wang, Babenko, and Belongie 2011), IC13-857, IC13-1015 (Karatzas et al. 2013), IC15-1811, IC15-2077 (Karatzas et al. 2015), SVTP-645 (Phan et al. 2013), and CT80-288 (Risnumawan et al. 2014) to evaluate the capacity of the proposed TrOCR.

识别场景文本图像比打印文本图像更具挑战性，因为野外的许多图像都存在模糊、遮挡或低分辨率问题。在这里，我们利用了一些广泛使用的基准，包括IIIT5K-3000（Mishra、Alahari和Jawahar，2012年）、SVT-647（Wang、Babenko和Belongie，2011年）、IC13-857、IC13-1015（Karatzas et al.2013）、IC15-1811、IC15-2077（Karatza等人，2015）、SVTP-645（Phan等人，2013）和CT80-288（Risnumawan等人，2014）来评估拟议的TrOCR的能力。

Table 3: Evaluation results (word-level Precision, Recall, F1) on the SROIE dataset, where the baselines come from the SROIE leaderboard (https://rrc.cvc.uab.es/?ch= 13&com=evaluation&task=2).

表3：SROIE数据集的评估结果（单词级精度、回忆、F1），其中基线来自SROIE排行榜(https://rrc.cvc.uab.es/?ch=13=com=评估&任务=2）。

### Settings 设置
The TrOCR models are built upon the Fairseq (Ott et al. 2019) which is a popular sequence modeling toolkit. For the model initialization, the DeiT models are implemented and initialized by the code and the pre-trained models from the timm library (Wightman 2019) while the BEiT models and the MiniLM models are from the UniLM’s official repository4 . The RoBERTa models come from the corresponding page in the Fairseq GitHub repository. We use 32 V100 GPUs with the memory of 32GBs for pre-training and 8 V100 GPUs for fine-tuning. For all the models, the batch size is set to 2,048 and the learning rate is 5e-5. We use the BPE and sentencepiece tokenizer from Fairseq to tokenize the textlines to wordpieces.

TrOCR模型建立在Fairseq（Ott等人，2019）的基础上，Fairseq是一个流行的序列建模工具包。对于模型初始化，DeiT模型由来自timm库（Wightman 2019）的代码和预训练模型实现和初始化，而BEiT模型和MiniLM模型来自UniLM的官方存储库4。RoBERTa模型来自FairseqGitHub存储库中的相应页面。我们使用32个具有32GB内存的V100GPU进行预训练，使用8个V100GPU用于微调。对于所有模型，批量大小设置为2048，学习率为5e-5。我们使用Fairseq中的BPE和句子片段标记器将文本行标记为单词片段。

3 https://github.com/jpuigcerver/Laia/tree/master/egs/iam 

3.https://github.com/jpuigcerver/Laia/tree/master/egs/iam

4 https://github.com/microsoft/unilm

4.https://github.com/microsoft/unilm

We employ the 384×384 resolution and 16×16 patch size for DeiT and BEiT encoders. The DeiTSMALL has 12 layers with 384 hidden sizes and 6 heads. Both the DeiTBASE and the BEiTBASE have 12 layers with 768 hidden sizes and 12 heads while the BEiTLARGE has 24 layers with 1024 hidden sizes and 16 heads. We use 6 layers, 256 hidden sizes and 8 attention heads for the small decoders, 512 hidden sizes for the base decoders and 12 layers, 1,024 hidden sizes and 16 heads for the large decoders. For this task, we only use the last half of all layers from the corresponding RoBERTa model, which are the last 6 layers for the RoBERTaBASE and the last 12 layers for the RoBERTaLARGE. The beam size is set to 10 for TrOCR models.

我们为DeiT和BEiT编码器采用384×384分辨率和16×16补丁大小。DeiTSMALL有12层，384个隐藏尺寸和6个头。DeiTBASE和BEiTBASE都有12层768隐藏尺寸和12个头，而BEiTLARGE有24层1024隐藏尺寸和16个头。我们对小型解码器使用6层、256个隐藏大小和8个注意头，对基本解码器使用512个隐藏大小，对大型解码器使用12层、1024个隐藏大小、16个注意头。对于这项任务，我们只使用相应RoBERTa模型中所有层的最后一半，即RoBERTaBASE的最后6层和RoBERTaLARGE的最后12层。对于TrOCR模型，光束大小设置为10。

We take the CRNN model (Shi, Bai, and Yao 2016) as the baseline model. The CRNN model is composed of convolutional layers for image feature extraction, recurrent layers for sequence modeling and the final frame label prediction, and a transcription layer to translate the frame predictions to the final label sequence. To address the character alignment issue, they use the CTC loss to train the CRNN model. For a long time, the CRNN model is the dominant paradigm for text recognition. We use the PyTorch implementation5 and initialized the parameters by the provided pre-trained model.

我们以CRNN模型（Shi，Bai，and Yao 2016）作为基线模型。CRNN模型由用于图像特征提取的卷积层、用于序列建模和最终帧标签预测的递归层以及用于将帧预测转换为最终标签序列的转录层组成。为了解决字符对齐问题，他们使用CTC损失来训练CRNN模型。长期以来，CRNN模型是文本识别的主导范式。我们使用PyTorch实现5，并通过提供的预训练模型初始化参数。

### Evaluation Metrics 评估指标
The SROIE dataset is evaluated using the word-level precision, recall and f1 score. If repeated words appear in the ground truth, they are also supposed to appear in the prediction. The precision, recall and f1 score are described as:

SROIE数据集使用单词级精度、召回率和f1分数进行评估。如果重复的单词出现在基本事实中，它们也应该出现在预测中。精确度、召回率和f1分数描述如下：

P recision = Correct matches The number of the detected words

P recision=正确匹配检测到的单词数

Recall = Correct matches The number of the ground truth words

Recall=正确匹配基本事实单词的数量

F1 = 2 × Precision × Recall Precision + Recall .

F1=2×精度×召回精度+召回。

The IAM dataset is evaluated by the case-sensitive Character Error Rate (CER). The scene text datasets are evaluated by the Word Accuracy. For fair comparison, we filter the final output string to suit the popular 36-character charset (lowercase alphanumeric) in this task.

IAM数据集通过区分大小写的字符错误率（CER）进行评估。场景文本数据集由单词准确性评估。为了进行公平的比较，我们过滤最终的输出字符串，以适应此任务中流行的36个字符的字符集（小写字母数字）。

### Results 结果
#### Architecture Comparison  架构比较
We compare different combinations of the encoder and decoder to find the best settings.

我们比较编码器和解码器的不同组合以找到最佳设置。

For encoders, we compare DeiT, BEiT and the ResNet-50 network. Both the DeiT and BEiT are the base models in their original papers. For decoders, we compare the base decoders initialized by RoBERTaBASE and the large decoders initialized by RoBERTaLARGE. For further comparison, we also evaluate the CRNN baseline model and the Tesseract OCR in this section, while the latter is an open-source OCR Engine using the LSTM network. 

对于编码器，我们比较了DeiT、BEiT和ResNet-50网络。DeiT和BEiT都是他们原始论文中的基本模型。对于解码器，我们比较了由RoBERTaBASE初始化的基本解码器和由RoBERTaLARGE初始化的大型解码器。为了进一步比较，我们还在本节中评估了CRNN基线模型和Tesseract OCR，而后者是使用LSTM网络的开源OCR引擎。

5 https://github.com/meijieru/crnn.pytorch

5.https://github.com/meijieru/crnn.pytorch

Table 1 shows the results of combined models. From the results, we observe that the BEiT encoders show the best performance among the three types of encoders while the best decoders are the RoBERTaLARGE decoders. Apparently, the pre-trained models on the vision task improve the performance of text recognition models, and the pure Transformer models are better than the CRNN models and the Tesseract on this task. According to the results, we mainly use three settings on the subsequent experiments: TrOCRSMALL (total parameters=62M) consists of the encoder of DeiTSMALL and the decoder of MiniLM, TrOCRBASE (total parameters=334M) consists of the encoder of BEiTBASE and the decoder of RoBERTaLARGE, TrOCRLARGE (total parameters=558M) consists of the encoder of BEiTLARGE and the decoder of RoBERTaLARGE. In Table 2, we have also done some ablation experiments to verify the effect of pre-trained model initialization, data augmentation, and two stages of pre-training. All of them have great improvements to the TrOCR models.

表1显示了组合模型的结果。从结果中，我们观察到，在三种类型的编码器中，BEiT编码器表现出最好的性能，而最好的解码器是RoBERTaLARGE解码器。显然，在视觉任务上预先训练的模型提高了文本识别模型的性能，并且纯Transformer模型在该任务上优于CRNN模型和Tesseract模型。根据结果，我们在随后的实验中主要使用了三种设置：TrOCRSMALL（总参数=62M）由DeiTSMALL的编码器和MiniLM的解码器组成，TrOCRBASE（总参数=334M）由BEiTBASE的编码器和RoBERTaLARGE的解码器组成。在表2中，我们还进行了一些消融实验，以验证预训练模型初始化、数据扩充和两个阶段的预训练的效果。所有这些都对TrOCR模型有很大的改进。

#### SROIE Task 2 SROIE任务2
Table 3 shows the results of the TrOCR models and the current SOTA methods on the leaderboard of the SROIE dataset. To capture the visual information, all of these baselines leverage CNN-based networks as the feature extractors while the TrOCR models use the image Transformer to embed the information from the image patches. For language modeling, MSO Lab (Sang and Cuong 2019) and CLOVA OCR (Sang and Cuong 2019) use LSTM layers and H&H Lab (Shi, Bai, and Yao 2016) use GRU layers while the TrOCR models use the Transformer decoder with a pure attention mechanism. According to the results, the TrOCR models outperform the existing SOTA models with pure Transformer structures. It is also confirmed that Transformer-based text recognition models get competitive performance compared to CNN-based networks in visual feature extraction and RNN-based networks in language modeling on this task without any complex pre/post-process steps.

表3显示了SROIE数据集排行榜上的TrOCR模型和当前SOTA方法的结果。为了捕获视觉信息，所有这些基线都利用基于CNN的网络作为特征提取器，而TrOCR模型使用图像转换器来嵌入图像补丁中的信息。对于语言建模，MSO实验室（Sang and Cuong 2019）和CLOVA OCR（Sang和Cuong 2019年）使用LSTM层，H&H实验室（Shi，Bai和Yao 2016）使用GRU层，而TrOCR模型使用具有纯注意力机制的Transformer解码器。根据结果，TrOCR模型优于具有纯Transformer结构的现有SOTA模型。在没有任何复杂的前/后处理步骤的情况下，基于Transformer的文本识别模型在视觉特征提取方面与基于CNN的网络和基于RNN的网络在该任务的语言建模方面相比具有竞争力。

#### IAM Handwriting Database  IAM手写数据库
Table 4 shows the results of the TrOCR models and the existing methods on the IAM Handwriting Database. According to the results, the methods with CTC decoders show good performance on this task and the external LM will result in a significant reduction in CER. By comparing the methods (Bluche and Messina 2017) with the TrOCR models, the TrOCRLARGE achieves a better result, which indicates that the Transformer decoder is more competitive than the CTC decoder in text recognition and has enough ability for language modeling instead of relying on an external LM. Most of the methods use sequence models in their encoders after the CNN-based backbone except the FCN encoders in (Wang et al. 2020a), which leads to a significant improvement on CER. Instead of relying on the features from the CNN-based backbone, the TrOCR models using the information from the image patches get similar and even better results, illustrating that the Transformer structures are competent to extract visual features well after pre-training. From the experiment results, the TrOCR models exceed all the methods which only use synthetic/IAM as the sole training set with pure Transformer structures and 

表4显示了IAM手写数据库上的TrOCR模型和现有方法的结果。根据结果，使用CTC解码器的方法在该任务上表现出良好的性能，并且外部LM将显著降低CER。通过将这些方法（Bluche和Messina 2017）与TrOCR模型进行比较，TrOCRLARGE获得了更好的结果，这表明Transformer解码器在文本识别方面比CTC解码器更有竞争力，并且具有足够的语言建模能力，而不是依赖于外部LM。除了（Wang等人，2020a）中的FCN编码器之外，大多数方法在基于CNN的主干之后的编码器中使用序列模型，这导致了CER的显著改进。使用来自图像补丁的信息的TrOCR模型获得了类似甚至更好的结果，而不是依赖于来自基于CNN的主干的特征，这表明Transformer结构能够在预训练后很好地提取视觉特征。从实验结果来看，TrOCR模型超过了仅使用合成/IAM作为纯Transformer结构的唯一训练集的所有方法，并且

Table 4: Evaluation results (CER) on the IAM Handwriting dataset.

表4：IAM手写数据集的评估结果（CER）。

Table 5: Inference time on the IAM Handwriting dataset. achieve a new state-of-the-art CER of 2.89. Without leveraging any extra human-labeled data, TrOCR even gets comparable results with the methods in (Diaz et al. 2021) using the additional internal human-labeled dataset.

表5:IAM手写数据集上的推断时间。实现了2.89的最先进的CER。在没有利用任何额外的人类标记数据的情况下，TrOCR甚至可以使用额外的内部人类标记数据集获得与（Diaz等人，2021）中的方法相当的结果。

#### Scene Text Datasets  场景文本数据集
In Table 6, we compare the TrOCRBASE and TrOCRLARGE models of fine-tuning with synthetic data only and fine-tuning with synthetic data and benchmark datasets (the training sets of IC13, IC15, IIIT5K, SVT) to the popular and recent SOTA methods. Compared to all, the TrOCR models establish five new SOTA results of eight experiments while getting comparable results on the rest. Our model underperforms on the IIIT5K dataset, and we find some scene text sample images contain symbols, but the ground truth does not. It is inconsistent with the behavior in our pre-training data (retaining symbols in ground truth), causing the model to tend still to process symbols. There are two kinds of mistakes: outputting symbols but truncating the output in advance to ensure that the number of wordpieces is consistent with the ground truth, or identifying symbols as similar characters.

在表6中，我们将仅使用合成数据进行微调以及使用合成数据和基准数据集（IC13、IC15、IIIT5K、SVT的训练集）进行微调的TrOCRBASE和TrOCRLARGE模型与流行的和最近的SOTA方法进行了比较。与所有实验相比，TrOCR模型建立了八个实验的五个新的SOTA结果，而在其他实验中获得了可比较的结果。我们的模型在IIIT5K数据集上表现不佳，我们发现一些场景文本样本图像包含符号，但事实并非如此。它与我们的预训练数据中的行为不一致（在基本事实中保留符号），导致模型仍然倾向于处理符号。有两种错误：输出符号但提前截断输出以确保单词的数量与基本事实一致，或者将符号识别为相似字符。

#### Inference Speed  推理速度
Table 5 shows the inference speed of different settings TrOCR models on the IAM Handwriting Database. We can conclude that there is no significant margin in inference speed between the base models and the large models. In contrast, the small model shows comparable results for printed and handwriting text recognition even though the number of parameters is an order of magnitude smaller and the inference speed is as twice as fast. The low number of parameters and high inference speed means fewer computational resources and user waiting time, making it more suitable for deployment in industrial applications.

表5显示了IAM手写数据库上不同设置的TrOCR模型的推理速度。我们可以得出结论，在基本模型和大型模型之间，推理速度没有显著的差距。相比之下，尽管参数数量少了一个数量级，推理速度是原来的两倍，但小模型在打印和手写文本识别方面显示出了可比较的结果。低参数数量和高推理速度意味着更少的计算资源和用户等待时间，使其更适合在工业应用中部署。

## Related Work 相关工作
### Scene Text Recognition 场景文本识别
For text recognition, the most popular approaches are usually based on the CTC-based models. (Shi, Bai, and Yao 2016) proposed the standard CRNN, an end-to-end architecture combined by CNN and RNN. The convolutional layers are used to extract the visual features and convert them to sequence by concatenating the columns, while the recurrent layers predict the per-frame labels. They use a CTC decoding strategy to remove the repeated symbols and all the blanks from the labels to achieve the final prediction. (Su and Lu 2014) used the Histogram of Oriented Gradient (HOG) features extracted from the image patches in the same column of the input image, instead of the features from the CNN network. A BiLSTM is then trained for labeling the sequential data with the CTC technique to find the best match. (Gao et al. 2019) extracted the feature by the densely connected network incorporating the residual attention block and capture the contextual information and sequential dependency by the CNN network. They compute the probability distribution on the output of the CNN network instead of using an RNN network to model them. After that, CTC translates the probability distributions into the final label sequence.

对于文本识别，最流行的方法通常是基于CTC的模型。（施，白，姚2016）提出了标准的CRNN，一种由CNN和RNN结合的端到端架构。卷积层用于提取视觉特征，并通过连接列将其转换为序列，而递归层则预测每帧标签。他们使用CTC解码策略来去除标签中的重复符号和所有空白，以实现最终预测。（Su和Lu 2014）使用了从输入图像的同一列中的图像块中提取的定向梯度直方图（HOG）特征，而不是来自CNN网络的特征。然后训练BiLSTM，用CTC技术标记序列数据，以找到最佳匹配。（Gao et al.2019）通过结合剩余注意力块的密集连接网络提取特征，并通过CNN网络捕获上下文信息和顺序依赖性。他们计算CNN网络输出的概率分布，而不是使用RNN网络对其进行建模。然后，CTC将概率分布转换为最终的标签序列。

The Sequence-to-Sequence models (Zhang et al. 2020b; Wang et al. 2019; Sheng, Chen, and Xu 2019; Bleeker and de Rijke 2019; Lee et al. 2020; Atienza 2021) are gradually attracting more attention, especially after the advent of the Transformer architecture (Vaswani et al. 2017). SaHAN (Zhang et al. 2020b), standing for the scale-aware hierarchical attention network, are proposed to address the character scale-variation issue. The authors use the FPN network and the CRNN models as the encoder as well as a hierarchical attention decoder to retain the multi-scale features. (Wang et al. 2019) extracted a sequence of visual features from the input images by the CNN with attention module and BiLSTM. The decoder is composed of the proposed Gated Cascade Attention Module (GCAM) and generates the target characters from the feature sequence extracted by the encoder. For the Transformer models, (Sheng, Chen, and Xu 2019) first applied the Transformer to Scene Text Recognition. Since the input of the Transformer architecture is required to be a sequence, a CNN-based modality-transform block is employed to transform 2D input images to 1D sequences. (Bleeker and de Rijke 2019) added a direction embedding to the input of the decoder for the bidirectional text decoding with a single decoder, while (Lee et al. 2020) utilized the two-dimensional dynamic positional embedding to keep the spatial structures of the intermediate feature maps for recognizing texts with arbitrary arrangements and large inter-character spacing. (Yu et al. 2020) proposed semantic reasoning networks to replace RNN-like structures for more accurate text recognition. (Atienza 2021) only used the image Transformer without text Transformer for the text recognition in a non-autoregressive way.

序列到序列模型（Zhang et al.2020b；Wang et al.2019；Sheng，Chen和Xu 2019；Bleeker和de Rijke 2019；Lee et al.2020；Atienza 2021）正逐渐吸引更多的关注，尤其是在Transformer架构出现之后（Vaswani et al.2017）。SaHAN（Zhang et al.2020b），代表尺度感知的层次注意力网络，被提出来解决字符尺度变化问题。作者使用FPN网络和CRNN模型作为编码器和层次注意力解码器来保留多尺度特征。（Wang et al.2019）通过具有注意力模块和BiLSTM的CNN从输入图像中提取了一系列视觉特征。解码器由所提出的门控级联注意模块（GCAM）组成，并根据编码器提取的特征序列生成目标字符。对于Transformer模型，（Sheng，Chen，and Xu 2019）首次将Transformer应用于场景文本识别。由于Transformer架构的输入需要是序列，因此采用基于CNN的模态变换块来将2D输入图像变换为1D序列。（Bleeker和de Rijke 2019）在解码器的输入端添加了方向嵌入，用于使用单个解码器进行双向文本解码，而（Lee等人2020）利用二维动态位置嵌入来保持中间特征图的空间结构，用于识别具有任意排列和大字符间间距的文本。（Yu et al.2020）提出了语义推理网络来取代类RNN结构，以实现更准确的文本识别。（Atienza 2021）仅使用图像转换器而不使用文本转换器以非自回归的方式进行文本识别。

Table 6: Word accuracy on the six benchmark datasets (36-char), where “Syn” indicates the model using synthetic data only and “Syn+Benchmark” indicates the model using synthetic data and benchmark datasets.

表6：六个基准数据集（36个字符）的单词准确性，其中“Syn”表示仅使用合成数据的模型，“Syn+benchmark”表示使用合成数据和基准数据集的模型。

The texts in natural images may appear in irregular shapes caused by perspective distortion. (Shi et al. 2016; Baek et al. 2019; Litman et al. 2020; Shi et al. 2018; Zhan and Lu 2019) addressed this problem by processing the input images with an initial rectification step. For example, thin-plate spline transformation (Shi et al. 2016; Baek et al. 2019; Litman et al. 2020; Shi et al. 2018) is applied to find a smooth spline interpolation between a set of fiducial points and normalize the text region to a predefined rectangle, while (Zhan and Lu 2019) proposed an iterative rectification network to model the middle line of scene texts as well as the orientation and boundary of textlines. (Baek et al. 2019; Diaz et al. 2021) proposed universal architectures for comparing different recognition models.

自然图像中的文本可能由于透视失真而呈现出不规则的形状。（Shi等人2016；Baek等人2019；Litman等人2020；Shi等人2018；Zhan和Lu 2019）通过用初始校正步骤处理输入图像来解决这个问题。例如，薄板样条变换（Shi等人2016；Baek等人2019；Litman等人2020；Shi等人2018）被应用于寻找一组基准点之间的平滑样条插值，并将文本区域归一化为预定义的矩形，而（詹和鲁2019）提出了一种迭代校正网络来对场景文本的中线以及文本线的方向和边界进行建模。（Baek等人2019；Diaz等人2021）提出了用于比较不同识别模型的通用架构。

### Handwritten Text Recognition  手写文本识别
(Memon et al. 2020) gave a systematic literature review about the modern methods for handwriting recognition. Various attention mechanisms and positional encodings are compared in the (Michael et al. 2019) to address the alignment between the input and output sequence. The combination of RNN encoders (mostly LSTM) and CTC decoders (Bluche and Messina 2017; Graves and Schmidhuber 2008; Pham et al. 2014) took a large part in the related works for a long time. Besides, (Graves and Schmidhuber 2008; Voigtlaender, Doetsch, and Ney 2016; Puigcerver 2017) have also tried multidimensional LSTM encoders. Similar to the scene text recognition, the seq2seq methods and the scheme for attention decoding have been verified in (Michael et al. 2019; Kang et al. 2020; Chowdhury and Vig 2018; Bluche 2016). (Ingle et al. 2019) addressed the problems in building a large-scale system.

（Memon等人，2020）对现代手写识别方法进行了系统的文献综述。在（Michael等人，2019）中比较了各种注意力机制和位置编码，以解决输入和输出序列之间的对齐问题。RNN编码器（主要是LSTM）和CTC解码器的组合（Bluche和Messina 2017；Graves和Schmidhuber 2008；Pham等人2014）在很长一段时间内参与了相关工作。此外，（Graves和Schmidhuber 2008；Voigtlander、Doetsch和Ney 2016；Puigcerver 2017）也尝试了多维LSTM编码器。与场景文本识别类似，seq2seq方法和注意力解码方案已在中得到验证（Michael等人，2019；Kang等人2020；Chowdhury和Vig 2018；Bluche 2016）。（Ingle等人，2019）解决了建立大规模系统的问题。

## Conclusion 结论
In this paper, we present TrOCR, an end-to-end Transformer-based OCR model for text recognition with pre-trained models. Distinct from existing approaches,

在本文中，我们提出了TrOCR，这是一种基于Transformer的端到端OCR模型，用于预训练模型的文本识别。与现有方法不同，

TrOCR does not rely on the conventional CNN models for image understanding. Instead, it leverages an image Transformer model as the visual encoder and a text Transformer model as the textual decoder. Moreover, we use the wordpiece as the basic unit for the recognized output instead of the character-based methods, which saves the computational cost introduced by the additional language modeling. Experiment results show that TrOCR achieves state-of-the-art results on printed, handwritten and scene text recognition with just a simple encoder-decoder model, without any post-processing steps.

TrOCR不依赖于传统的CNN模型来进行图像理解。相反，它利用图像转换器模型作为视觉编码器，利用文本转换器模型作为文本解码器。此外，我们使用分词作为识别输出的基本单元，而不是基于字符的方法，这节省了额外的语言建模带来的计算成本。实验结果表明，TrOCR只需一个简单的编码器-解码器模型，就可以在打印、手写和场景文本识别方面获得最先进的结果，而无需任何后处理步骤。

