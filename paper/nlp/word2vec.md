# Efficient Estimation of Word Representations in Vector Space
向量空间中单词表示的有效估计 2013-1-16 https://arxiv.org/abs/1301.3781

## Abstract
We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities. 

我们提出了两种新的模型架构，用于从非常大的数据集中计算单词的连续向量表示。在单词相似性任务中测量这些表示的质量，并将结果与基于不同类型的神经网络的SOTA技术进行比较。我们观察到在低得多的计算成本下，准确度有了很大的提高，即从16亿个单词数据集学习高质量的单词向量只需不到一天的时间。此外，我们还表明，这些向量在我们的测试集上提供了最先进的性能，用于测量句法和语义词的相似性。

## 1 Introduction
Many current NLP systems and techniques treat words as atomic units - there is no notion of similarity between words, as these are represented as indices in a vocabulary. This choice has several good reasons - simplicity, robustness and the observation that simple models trained on huge amounts of data outperform complex systems trained on less data. An example is the popular N-gram model used for statistical language modeling - today, it is possible to train N-grams on virtually all available data (trillions of words [3]).

许多当前的NLP系统和技术将单词视为原子单位 - 单词之间没有相似性的概念，因为它们在词汇表中被表示为索引。这种选择有几个很好的理由 —— 简单、稳健，以及观察到在大量数据上训练的简单模型优于在较少数据上训练过的复杂系统。一个例子是用于统计语言建模的流行的N-gram模型 —— 今天，可以在几乎所有可用数据(万亿单词[3])上训练N-gram。

However, the simple techniques are at their limits in many tasks. For example, the amount of relevant in-domain data for automatic speech recognition is limited - the performance is usually dominated by the size of high quality transcribed speech data (often just millions of words). In machine translation, the existing corpora for many languages contain only a few billions of words or less. Thus, there are situations where simple scaling up of the basic techniques will not result in any significant progress, and we have to focus on more advanced techniques.

然而，这些简单的技术在许多任务中都处于极限。例如，用于自动语音识别的相关域内数据的数量是有限的 —— 性能通常取决于高质量转录语音数据的大小(通常只有数百万字)。在机器翻译中，许多语言的现有语料库只包含几十亿个单词或更少的单词。因此，在某些情况下，简单地扩大基本技术并不会带来任何重大进展，我们必须专注于更先进的技术。

With progress of machine learning techniques in recent years, it has become possible to train more complex models on much larger data set, and they typically outperform the simple models. Probably the most successful concept is to use distributed representations of words [10]. For example, neural network based language models significantly outperform N-gram models [1, 27, 17].

随着近年来机器学习技术的进步，已经有可能在更大的数据集上训练更复杂的模型，并且它们通常优于简单模型。可能最成功的概念是使用单词的分布式表示[10]。例如，基于神经网络的语言模型显著优于N-gram模型[1，27，17]。

### 1.1 Goals of the Paper
The main goal of this paper is to introduce techniques that can be used for learning high-quality word vectors from huge data sets with billions of words, and with millions of words in the vocabulary. As far as we know, none of the previously proposed architectures has been successfully trained on more than a few hundred of millions of words, with a modest dimensionality of the word vectors between 50 - 100.

本文的主要目标是介绍可用于从具有数十亿单词和数百万单词的庞大数据集学习高质量单词向量的技术。据我们所知，之前提出的架构中没有一个成功地训练了超过数亿个单词，单词向量的维数在50-100之间。

We use recently proposed techniques for measuring the quality of the resulting vector representations, with the expectation that not only will similar words tend to be close to each other, but that words can have multiple degrees of similarity [20]. This has been observed earlier in the context of inflectional languages - for example, nouns can have multiple word endings, and if we search for similar words in a subspace of the original vector space, it is possible to find words that have similar endings [13, 14].

我们使用最近提出的技术来测量生成的向量表示的质量，期望不仅相似的单词会彼此接近，而且单词可以具有多个相似程度[20]。这在屈折(inflectional)语言的上下文中已经观察到了 —— 例如，名词可以有多个词尾，如果我们在原始向量空间的子空间中搜索相似的单词，就有可能找到具有相似词尾的单词[13，14]。

Somewhat surprisingly, it was found that similarity of word representations goes beyond simple syntactic regularities. Using a word offset technique where simple algebraic operations are performed on the word vectors, it was shown for example that vector(”King”) - vector(”Man”) + vector(”Woman”) results in a vector that is closest to the vector representation of the word Queen [20].

令人惊讶的是，人们发现单词表示的相似性超出了简单的句法规律。使用单词偏移技术，在单词向量上执行简单的代数运算，例如，向量(“King”)- 向量(“Man”)+ 向量(“Woman”)生成最接近单词Queen的向量表示的向量[20]。

In this paper, we try to maximize accuracy of these vector operations by developing new model architectures that preserve the linear regularities among words. We design a new comprehensive test set for measuring both syntactic and semantic regularities1 , and show that many such regularities can be learned with high accuracy. Moreover, we discuss how training time and accuracy depends on the dimensionality of the word vectors and on the amount of the training data.

在本文中，我们试图通过开发新的模型架构来保持单词之间的线性规律，从而最大限度地提高这些向量运算的准确性。我们设计了一个新的综合测试集，用于测量句法和语义规则1，并表明可以高精度地学习许多这样的规则。此外，我们讨论了训练时间和准确性如何取决于单词向量的维数和训练数据的数量。

### 1.2 Previous Work
Representation of words as continuous vectors has a long history [10, 26, 8]. A very popular model architecture for estimating neural network language model (NNLM) was proposed in [1], where a feedforward neural network with a linear projection layer and a non-linear hidden layer was used to learn jointly the word vector representation and a statistical language model. This work has been followed by many others.

将单词表示为连续向量有着悠久的历史[10，26，8]。[1]中提出了一种用于估计神经网络语言模型(NNLM)的非常流行的模型架构，其中使用具有线性投影层和非线性隐藏层的前馈神经网络来联合学习单词向量表示和统计语言模型。这项工作已被许多其他人效仿。

Another interesting architecture of NNLM was presented in [13, 14], where the word vectors are first learned using neural network with a single hidden layer. The word vectors are then used to train the NNLM. Thus, the word vectors are learned even without constructing the full NNLM. In this work, we directly extend this architecture, and focus just on the first step where the word vectors are learned using a simple model.

[13，14]中提出了NNLM的另一个有趣的架构，其中首先使用具有单个隐藏层的神经网络学习单词向量。然后使用字向量来训练NNLM。因此，即使不构造完整的NNLM，也可以学习单词向量。在这项工作中，我们直接扩展了这个架构，并将重点放在使用简单模型学习单词向量的第一步。

It was later shown that the word vectors can be used to significantly improve and simplify many NLP applications [4, 5, 29]. Estimation of the word vectors itself was performed using different model architectures and trained on various corpora [4, 29, 23, 19, 9], and some of the resulting word vectors were made available for future research and comparison2 . However, as far as we know, these architectures were significantly more computationally expensive for training than the one proposed in [13], with the exception of certain version of log-bilinear model where diagonal weight matrices are used [23]. 

后来的研究表明，单词向量可以用于显著改进和简化许多NLP应用[4，5，29]。单词向量本身的估计是使用不同的模型架构进行的，并在不同的语料库上进行训练[4，29，23，19，9]，并且得到的一些单词向量可用于未来的研究和比较2。然而，据我们所知，这些架构在训练方面的计算成本明显高于[13]中提出的架构，除了使用对角权重矩阵的对数双线性模型的某些版本[23]。

2http://ronan.collobert.com/senna/
http://metaoptimize.com/projects/wordreprs/ 
http://www.fit.vutbr.cz/˜imikolov/rnnlm/ 
http://ai.stanford.edu/˜ehhuang/


## 2 Model Architectures
Many different types of models were proposed for estimating continuous representations of words, including the well-known Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA). In this paper, we focus on distributed representations of words learned by neural networks, as it was previously shown that they perform significantly better than LSA for preserving linear regularities among words [20, 31]; LDA moreover becomes computationally very expensive on large data sets.

许多不同类型的模型被提出用于估计单词的连续表示，包括众所周知的潜在语义分析(LSA)和潜在狄利克雷分配(LDA)。 在本文中，我们关注由神经网络学习的单词的分布式表示，因为之前已经表明，在保持单词之间的线性规律方面，它们比LSA表现得更好[20，31]; 此外，LDA在大型数据集上的计算变得非常昂贵。

Similar to [18], to compare different model architectures we define first the computational complexity of a model as the number of parameters that need to be accessed to fully train the model. Next, we will try to maximize the accuracy, while minimizing the computational complexity. 

类似于[18]，为了比较不同的模型架构，我们首先将模型的计算复杂性定义为需要访问的参数的数量，以完全训练模型。接下来，我们将尝试最大化精度，同时最小化计算复杂性。

1The test set is available at www.fit.vutbr.cz/˜imikolov/rnnlm/word-test.v1.txt 

For all the following models, the training complexity is proportional to
对于以下所有模型，训练复杂性与

O = E × T × Q, (1) 

where E is number of the training epochs, T is the number of the words in the training set and Q is defined further for each model architecture. Common choice is E = 3 − 50 and T up to one billion. All models are trained using stochastic gradient descent and backpropagation [26].

其中E是训练时期的数量，T是训练集合中的单词数量，Q是为每个模型架构进一步定义的。通常的选择是E=3−50，T高达10亿。所有模型都使用随机梯度下降和反向传播进行训练[26]。

### 2.1 Feedforward Neural Net Language Model (NNLM)  前馈神经网络语言模型
The probabilistic feedforward neural network language model has been proposed in [1]. It consists of input, projection, hidden and output layers. At the input layer, N previous words are encoded using 1-of-V coding, where V is size of the vocabulary. The input layer is then projected to a projection layer P that has dimensionality N × D, using a shared projection matrix. As only N inputs are active at any given time, composition of the projection layer is a relatively cheap operation.

概率前馈神经网络语言模型已在[1]中提出。它由输入层、投影层、隐藏层和输出层组成。在输入层，使用1-of-V编码对N个先前单词进行编码，其中V是词汇的大小。然后使用共享投影矩阵将输入层投影到维度为N×D的投影层P。由于在任何给定时间只有N个输入是有效的，所以投影层的组成是一种相对廉价的操作。

The NNLM architecture becomes complex for computation between the projection and the hidden layer, as values in the projection layer are dense. For a common choice of N = 10, the size of the projection layer (P) might be 500 to 2000, while the hidden layer size H is typically 500 to 1000 units. Moreover, the hidden layer is used to compute probability distribution over all the words in the vocabulary, resulting in an output layer with dimensionality V . Thus, the computational complexity per each training example is

由于投影层中的值密集，NNLM架构在投影和隐藏层之间的计算变得复杂。对于N＝10的常见选择，投影层(P)的大小可以是500到2000，而隐藏层大小H通常是500到1000个单位。此外，隐藏层用于计算词汇表中所有单词的概率分布，从而生成维度为V的输出层。因此，每个训练样本的计算复杂性为

Q = N × D + N × D × H + H × V, (2) 

where the dominating term is H × V . However, several practical solutions were proposed for avoiding it; either using hierarchical versions of the softmax [25, 23, 18], or avoiding normalized models completely by using models that are not normalized during training [4, 9]. With binary tree representations of the vocabulary, the number of output units that need to be evaluated can go down to around $log_2(V)$. Thus, most of the complexity is caused by the term N × D × H.

其中主导项是H×V。然而，为避免这种情况，提出了一些切实可行的解决办法; 或者使用softmax的分层版本[25，23，18]，或者通过使用在训练期间未标准化的模型来完全避免标准化模型[4，9]。使用词汇的二叉树表示，需要评估的输出单元的数量可以减少到$log_2(V)$左右。因此，大多数复杂性是由N×D×H项引起的。

In our models, we use hierarchical softmax where the vocabulary is represented as a Huffman binary tree. This follows previous observations that the frequency of words works well for obtaining classes in neural net language models [16]. Huffman trees assign short binary codes to frequent words, and this further reduces the number of output units that need to be evaluated: while balanced binary tree would require $log_2(V)$ outputs to be evaluated, the Huffman tree based hierarchical softmax requires only about $log_2(Unigram perplexity(V))$. For example when the vocabulary size is one million words, this results in about two times speedup in evaluation. While this is not crucial speedup for neural network LMs as the computational bottleneck is in the N ×D×H term, we will later propose architectures that do not have hidden layers and thus depend heavily on the efficiency of the softmax normalization.

在我们的模型中，我们使用分层softmax，其中词汇表表示为霍夫曼二叉树。这遵循了先前的观察结果，即单词的频率对于在神经网络语言模型中获得类非常有效[16]。霍夫曼树将短二进制代码分配给频繁的单词，这进一步减少了需要评估的输出单元的数量：虽然平衡二进制树需要评估$log_2(V)$个输出，但基于霍夫曼树的分层softmax只需要大约$log_2(Unigram perplexity(V))$。例如，当词汇量为一百万个单词时，这将导致评估速度提高约两倍。虽然这对于神经网络LMs来说不是关键的加速，因为计算瓶颈在N×D×H项，但我们稍后将提出不具有隐藏层的架构，因此严重依赖于softmax归一化的效率。

### 2.2 Recurrent Neural Net Language Model (RNNLM) 循环神经网络语言模型
Recurrent neural network based language model has been proposed to overcome certain limitations of the feedforward NNLM, such as the need to specify the context length (the order of the model N), and because theoretically RNNs can efficiently represent more complex patterns than the shallow neural networks [15, 2]. The RNN model does not have a projection layer; only input, hidden and output layer. What is special for this type of model is the recurrent matrix that connects hidden layer to itself, using time-delayed connections. This allows the recurrent model to form some kind of short term memory, as information from the past can be represented by the hidden layer state that gets updated based on the current input and the state of the hidden layer in the previous time step.

基于循环神经网络的语言模型已被提出以克服前馈NNLM的某些限制，例如需要指定上下文长度(模型N的阶数)，并且因为理论上RNN可以有效地表示比浅层神经网络更复杂的模式[15，2]。RNN模型没有投影层; 仅输入、隐藏和输出层。这类模型的特殊之处在于使用延时连接将隐藏层连接到自身的循环矩阵。这允许循环模型形成某种短期记忆，因为来自过去的信息可以由隐藏层状态表示，该隐藏层状态基于当前输入和前一时间步骤中隐藏层的状态进行更新。

The complexity per training example of the RNN model is

RNN模型的每个训练样本的复杂性为

Q = H × H + H × V, (3) 

where the word representations D have the same dimensionality as the hidden layer H. Again, the term H × V can be efficiently reduced to $H × log_2(V)$ by using hierarchical softmax. Most of the complexity then comes from H × H. 

其中单词表示D具有与隐藏层H相同的维度。再次，通过使用分层softmax，术语H×V可以有效地降为$H × log_2(V)$。然后，大部分复杂性来自H×H.

### 2.3 Parallel Training of Neural Networks 并行训练
To train models on huge data sets, we have implemented several models on top of a large-scale distributed framework called DistBelief [6], including the feedforward NNLM and the new models proposed in this paper. The framework allows us to run multiple replicas of the same model in parallel, and each replica synchronizes its gradient updates through a centralized server that keeps all the parameters. For this parallel training, we use mini-batch asynchronous gradient descent with an adaptive learning rate procedure called Adagrad [7]. Under this framework, it is common to use one hundred or more model replicas, each using many CPU cores at different machines in a data center. 

为了在巨大的数据集上训练模型，我们在称为DistBelief[6]的大型分布式框架上实现了几个模型，包括前馈NNLM和本文中提出的新模型。该框架允许我们并行运行同一模型的多个副本，每个副本通过保存所有参数的集中服务器同步其梯度更新。对于这种并行训练，我们使用小批量异步梯度下降和称为Adgrad[7]的自适应学习速率程序。在此框架下，通常使用100个或更多模型副本，每个副本在数据中心的不同机器上使用多个CPU内核。

## 3 New Log-linear Models  对数线性模型
In this section, we propose two new model architectures for learning distributed representations of words that try to minimize computational complexity. The main observation from the previous section was that most of the complexity is caused by the non-linear hidden layer in the model. While this is what makes neural networks so attractive, we decided to explore simpler models that might not be able to represent the data as precisely as neural networks, but can possibly be trained on much more data efficiently.

在本节中，我们提出了两种新的模型架构，用于学习单词的分布式表示，以尽量减少计算复杂性。上一节的主要观察结果是，大部分复杂性是由模型中的非线性隐藏层造成的。虽然这正是神经网络如此吸引人的原因，但我们决定探索更简单的模型，这些模型可能无法像神经网络那样精确地表示数据，但可能可以在更有效的数据上进行训练。

The new architectures directly follow those proposed in our earlier work [13, 14], where it was found that neural network language model can be successfully trained in two steps: first, continuous word vectors are learned using simple model, and then the N-gram NNLM is trained on top of these distributed representations of words. While there has been later substantial amount of work that focuses on learning word vectors, we consider the approach proposed in [13] to be the simplest one. Note that related models have been proposed also much earlier [26, 8].

新的架构直接遵循我们早期工作[13，14]中提出的架构，其中发现神经网络语言模型可以通过两个步骤成功训练：首先，使用简单的模型学习连续的单词向量，然后在这些单词的分布式表示上训练N-gram NNLM。虽然后来有大量的工作集中在学习单词向量上，但我们认为[13]中提出的方法是最简单的方法。 请注意，相关模型也在更早的时候提出[26，8]。

### 3.1 Continuous Bag-of-Words Model 连续单词袋模型
The first proposed architecture is similar to the feedforward NNLM, where the non-linear hidden layer is removed and the projection layer is shared for all words (not just the projection matrix); thus, all words get projected into the same position (their vectors are averaged). We call this architecture a bag-of-words model as the order of words in the history does not influence the projection.Furthermore, we also use words from the future; we have obtained the best performance on the task introduced in the next section by building a log-linear classifier with four future and four history words at the input, where the training criterion is to correctly classify the current (middle) word. Training complexity is then

第一个提出的架构类似于前馈NNLM，其中去除了非线性隐藏层，并为所有单词(而不仅仅是投影矩阵)共享投影层; 因此，所有单词被投影到相同的位置(它们的向量被平均)。我们称这种架构为一个词袋模型，因为历史上单词的顺序不会影响投影。 此外，我们还使用来自未来的词语; 在下一节介绍的任务中，我们通过在输入端构建具有四个未来和四个历史单词的对数线性分类器，获得了最佳性能，其中训练标准是正确地对当前(中间)单词进行分类。 那么训练的复杂性就是:

$Q = N × D + D × log_2(V )$. (4)

We denote this model further as CBOW, as unlike standard bag-of-words model, it uses continuous distributed representation of the context. The model architecture is shown at Figure 1. Note that the weight matrix between the input and the projection layer is shared for all word positions in the same way as in the NNLM.

我们将该模型进一步表示为CBOW，因为与标准的词袋模型不同，它使用上下文的连续分布式表示。模型架构如图1所示。注意，输入和投影层之间的权重矩阵以与NNLM中相同的方式为所有单词位置共享。

### 3.2 Continuous Skip-gram Model 
The second architecture is similar to CBOW, but instead of predicting the current word based on the context, it tries to maximize classification of a word based on another word in the same sentence. More precisely, we use each current word as an input to a log-linear classifier with continuous projection layer, and predict words within a certain range before and after the current word. We found that increasing the range improves quality of the resulting word vectors, but it also increases the computational complexity. Since the more distant words are usually less related to the current word than those close to it, we give less weight to the distant words by sampling less from those words in our training examples.

第二种架构类似于CBOW，但它不是基于上下文来预测当前单词，而是试图基于同一句子中的另一个单词来最大化对单词的分类。更准确地说，我们使用每个当前单词作为具有连续投影层的对数线性分类器的输入，并预测当前单词前后一定范围内的单词。我们发现，增加范围可以提高生成的单词向量的质量，但也会增加计算复杂性。由于较远的单词与当前单词的相关性通常小于与其相近的单词，因此我们在训练样本中通过减少对这些单词的采样来减少对较远单词的权重。

The training complexity of this architecture is proportional to 
该架构的训练复杂性与

$Q = C × (D + D × log_2(V))$, (5) 

where C is the maximum distance of the words. Thus, if we choose C = 5, for each training word we will select randomly a number R in range < 1; C >, and then use R words from history and R words from the future of the current word as correct labels. This will require us to do R × 2 word classifications, with the current word as input, and each of the R + R words as output. In the following experiments, we use C = 10. 

其中C是单词的最大距离。因此，如果我们选择C=5，对于每个训练字，我们将R随机选择范围<1 ; C>，然后使用历史中的R个单词和当前单词未来的R个词汇作为正确的标签。这将要求我们进行R×2单词分类，将当前单词作为输入，将每个R+R单词作为输出。在以下实验中，我们使用C=10。

Figure 1: New model architectures. The CBOW architecture predicts the current word based on the context, and the Skip-gram predicts surrounding words given the current word.
图1：新模型架构。CBOW架构基于上下文预测当前单词，Skip gram预测给定当前单词的周围单词。

## 4 Results
To compare the quality of different versions of word vectors, previous papers typically use a table showing example words and their most similar words, and understand them intuitively. Although it is easy to show that word France is similar to Italy and perhaps some other countries, it is much more challenging when subjecting those vectors in a more complex similarity task, as follows. We follow previous observation that there can be many different types of similarities between words, for example, word big is similar to bigger in the same sense that small is similar to smaller. Example of another type of relationship can be word pairs big - biggest and small - smallest [20]. We further denote two pairs of words with the same relationship as a question, as we can ask: ”What is the word that is similar to small in the same sense as biggest is similar to big?”

为了比较不同版本的单词向量的质量，以前的论文通常使用一个表来显示样本单词及其最相似的单词，并直观地理解它们。尽管很容易证明单词France与意大利以及其他一些国家相似，但在进行更复杂的相似性任务时，这就更具挑战性了，如下所示。我们遵循先前的观察，单词之间可能有许多不同类型的相似性，例如，单词big与bigger相似，意思是small与smallest相似。另一类关系的例子可以是单词对大小[20]。我们进一步将两对具有相同关系的单词表示为一个问题，正如我们可以问的那样：“在biggest与big相同的意义上，与small相似的单词是什么？”

Somewhat surprisingly, these questions can be answered by performing simple algebraic operations with the vector representation of words. To find a word that is similar to small in the same sense as biggest is similar to big, we can simply compute vector X = vector(”biggest”)−vector(”big”) + vector(”small”). Then, we search in the vector space for the word closest to X measured by cosine distance, and use it as the answer to the question (we discard the input question words during this search). When the word vectors are well trained, it is possible to find the correct answer (word smallest) using this method.

有些令人惊讶的是，这些问题可以通过对单词的向量表示进行简单的代数运算来解决。要找到一个与“小”相似的词，就像“biggest”与“big”相似一样，我们可以简单地计算向量X=向量(“biggest”)−向量(“big”)+向量(”small”)。然后，我们在向量空间中搜索通过余弦距离测量的最接近X的单词，并将其用作问题的答案(我们在搜索过程中丢弃输入的问题单词)。当单词向量经过良好训练时，使用该方法可以找到正确的答案(单词最小)。

Finally, we found that when we train high dimensional word vectors on a large amount of data, the resulting vectors can be used to answer very subtle semantic relationships between words, such as a city and the country it belongs to, e.g. France is to Paris as Germany is to Berlin. Word vectors with such semantic relationships could be used to improve many existing NLP applications, such as machine translation, information retrieval and question answering systems, and may enable other future applications yet to be invented. 

最后，我们发现，当我们在大量数据上训练高维单词向量时，生成的向量可以用来回答单词之间非常微妙的语义关系，例如城市和它所属的国家，例如法国到巴黎就像德国到柏林。具有这种语义关系的词向量可以用于改进许多现有的NLP应用，例如机器翻译、信息检索和问题解答系统，并且可以实现其他尚未发明的未来应用。

Table 1: Examples of five types of semantic and nine types of syntactic questions in the SemanticSyntactic Word Relationship test set.
表1：语义句法单词关系测试集中五种类型的语义和九种类型的句法问题的样本。

### 4.1 Task Description
To measure quality of the word vectors, we define a comprehensive test set that contains five types of semantic questions, and nine types of syntactic questions. Two examples from each category are shown in Table 1. Overall, there are 8869 semantic and 10675 syntactic questions. The questions in each category were created in two steps: first, a list of similar word pairs was created manually. Then, a large list of questions is formed by connecting two word pairs. For example, we made a list of 68 large American cities and the states they belong to, and formed about 2.5K questions by picking two word pairs at random. We have included in our test set only single token words, thus multi-word entities are not present (such as New York).

为了衡量单词向量的质量，我们定义了一个综合测试集，该测试集包含五种类型的语义问题和九种类型的句法问题。表1中显示了每个类别的两个样本。总体而言，共有8869个语义问题和10675个句法问题。每个类别中的问题分两个步骤创建：首先，手动创建一个类似单词对列表。然后，通过连接两个单词对形成一个大的问题列表。例如，我们列出了68个美国大城市及其所属的州，并通过随机抽取两个单词对形成了大约2.5K个问题。我们在测试集中只包含了单个标记词，因此不存在多词实体(例如纽约)。

We evaluate the overall accuracy for all question types, and for each question type separately (semantic, syntactic). Question is assumed to be correctly answered only if the closest word to the vector computed using the above method is exactly the same as the correct word in the question; synonyms are thus counted as mistakes. This also means that reaching 100% accuracy is likely to be impossible, as the current models do not have any input information about word morphology. However, we believe that usefulness of the word vectors for certain applications should be positively correlated with this accuracy metric. Further progress can be achieved by incorporating information about structure of words, especially for the syntactic questions.

我们评估所有问题类型的总体准确性，并分别评估每个问题类型(语义、句法)的总体准确性。只有当使用上述方法计算的向量的最近单词与问题中的正确单词完全相同时，问题才被假定为正确答案; 同义词因此被算作错误。这也意味着要达到100%的精度是不可能的，因为当前的模型没有任何关于单词形态的输入信息。然而，我们认为单词向量在某些应用中的有用性应该和这个准确度度量正相关。通过结合有关单词结构的信息，特别是句法问题，可以取得进一步的进展。

### 4.2 Maximization of Accuracy
We have used a Google News corpus for training the word vectors. This corpus contains about 6B tokens. We have restricted the vocabulary size to 1 million most frequent words. Clearly, we are facing time constrained optimization problem, as it can be expected that both using more data and higher dimensional word vectors will improve the accuracy. To estimate the best choice of model architecture for obtaining as good as possible results quickly, we have first evaluated models trained on subsets of the training data, with vocabulary restricted to the most frequent 30k words. The results using the CBOW architecture with different choice of word vector dimensionality and increasing amount of the training data are shown in Table 2.

我们使用了谷歌新闻语料库来训练单词向量。这个语料库包含大约6B个tokens。我们将词汇量限制在100万个最频繁的单词。显然，我们面临着时间约束的优化问题，因为可以预期，使用更多的数据和更高维度的词向量将提高准确性。为了估计快速获得尽可能好的结果的模型架构的最佳选择，我们首先评估了在训练数据子集上训练的模型，词汇限制在最频繁的30k个单词。表2显示了使用CBOW架构的结果，该架构具有不同的词向量维度选择和不断增加的训练数据量。

It can be seen that after some point, adding more dimensions or adding more training data provides diminishing improvements. So, we have to increase both vector dimensionality and the amount of the training data together. While this observation might seem trivial, it must be noted that it is currently popular to train word vectors on relatively large amounts of data, but with insufficient size (such as 50 - 100). Given Equation 4, increasing amount of training data twice results in about the same increase of computational complexity as increasing vector size twice.

可以看出，在某一点之后，添加更多维度或添加更多训练数据提供的改进逐渐减少。所以，我们必须同时增加向量维数和训练数据量。虽然这一观察可能看起来微不足道，但必须注意，目前流行的是在相对大量的数据上训练单词向量，但其大小不足(例如50-100)。给定等式4，两次增加训练数据量导致的计算复杂性增加与两次增加向量大小相同。

Table 2: Accuracy on subset of the Semantic-Syntactic Word Relationship test set, using word vectors from the CBOW architecture with limited vocabulary. Only questions containing words from the most frequent 30k words are used.
表2：使用有限词汇的CBOW架构中的单词向量，语义句法单词关系测试集子集的准确性。只使用包含最常用的30k个单词的问题。

Table 3: Comparison of architectures using models trained on the same data, with 640-dimensional word vectors. The accuracies are reported on our Semantic-Syntactic Word Relationship test set, and on the syntactic relationship test set of [20]
表3：使用在相同数据上训练的模型与640维单词向量进行的架构比较。我们的语义句法词关系测试集和[20]的句法关系测试集报告了准确性

For the experiments reported in Tables 2 and 4, we used three training epochs with stochastic gradient descent and backpropagation. We chose starting learning rate 0.025 and decreased it linearly, so that it approaches zero at the end of the last training epoch.

对于表2和表4中报告的实验，我们使用了具有随机梯度下降和反向传播的三个训练阶段。我们选择了0.025的起始学习率，并将其线性降低，以使其在最后一个训练时期结束时接近零。

### 4.3 Comparison of Model Architectures
First we compare different model architectures for deriving the word vectors using the same training data and using the same dimensionality of 640 of the word vectors. In the further experiments, we use full set of questions in the new Semantic-Syntactic Word Relationship test set, i.e. unrestricted to the 30k vocabulary. We also include results on a test set introduced in [20] that focuses on syntactic similarity between words(3We thank Geoff Zweig for providing us the test set. ).

首先，我们比较用于使用相同的训练数据和使用640个单词向量的相同维度来导出单词向量的不同模型架构。在进一步的实验中，我们在新的语义句法单词关系测试集中使用了全套问题，即不受30k词汇的限制。我们还包括[20]中引入的测试集的结果，该测试集关注单词之间的句法相似性。(3我们感谢Geoff Zweig为我们提供测试集。)

The training data consists of several LDC corpora and is described in detail in [18] (320M words, 82K vocabulary). We used these data to provide a comparison to a previously trained recurrent neural network language model that took about 8 weeks to train on a single CPU. We trained a feedforward NNLM with the same number of 640 hidden units using the DistBelief parallel training [6], using a history of 8 previous words (thus, the NNLM has more parameters than the RNNLM, as the projection layer has size 640 × 8).

训练数据由几个LDC语料库组成，在[18]中有详细描述(320M个单词，82K个词汇)。我们使用这些数据与之前训练的循环神经网络语言模型进行了比较，该模型在单个CPU上训练了大约8周。我们使用DistBelief并行训练[6]，使用8个先前单词的历史，训练了具有相同数量640个隐藏单元的前馈NNLM(因此，NNLM比RNNLM具有更多的参数，因为投影层的大小为640×8)。

In Table 3, it can be seen that the word vectors from the RNN (as used in [20]) perform well mostly on the syntactic questions. The NNLM vectors perform significantly better than the RNN - this is not surprising, as the word vectors in the RNNLM are directly connected to a non-linear hidden layer. The CBOW architecture works better than the NNLM on the syntactic tasks, and about the same on the semantic one. Finally, the Skip-gram architecture works slightly worse on the syntactic task than the CBOW model (but still better than the NNLM), and much better on the semantic part of the test than all the other models.

在表3中，可以看到RNN(如[20]中所用)的词向量在句法问题上表现良好。NNLM向量的性能明显优于RNN——这并不奇怪，因为RNNLM中的单词向量直接连接到非线性隐藏层。CBOW架构在句法任务上比NNLM更好，在语义任务上也差不多。最后，Skip gram架构在语法任务上比CBOW模型稍差(但仍比NNLM更好)，在测试的语义部分比所有其他模型好得多。

Next, we evaluated our models trained using one CPU only and compared the results against publicly available word vectors. The comparison is given in Table 4. The CBOW model was trained on subset of the Google News data in about a day, while training time for the Skip-gram model was about three days.

接下来，我们评估了仅使用一个CPU训练的模型，并将结果与公开可用的单词向量进行了比较。表4给出了比较结果。CBOW模型在大约一天内对谷歌新闻数据子集进行了训练，而Skip gram模型的训练时间大约为三天。

Table 4: Comparison of publicly available word vectors on the Semantic-Syntactic Word Relationship test set, and word vectors from our models. Full vocabularies are used.
表4：语义句法词关系测试集上公开可用的词向量与我们模型中的词向量的比较。使用完整的词汇表。

For experiments reported further, we used just one training epoch (again, we decrease the learning rate linearly so that it approaches zero at the end of training). Training a model on twice as much data using one epoch gives comparable or better results than iterating over the same data for three epochs, as is shown in Table 5, and provides additional small speedup.

对于进一步报道的实验，我们只使用了一个训练时期(同样，我们线性地降低学习率，使其在训练结束时接近零)。如表5所示，使用一个时期在两倍多的数据上训练模型，与在三个时期内对相同的数据进行迭代相比，可以获得类似或更好的结果，并提供额外的小加速。

Table 5: Comparison of models trained for three epochs on the same data and models trained for one epoch. Accuracy is reported on the full Semantic-Syntactic data set.
表5：在相同数据上训练了三个时期的模型与训练了一个时期的模式的比较。准确度报告在完整的语义句法数据集上。

### 4.4 Large Scale Parallel Training of Models 模型的大规模并行训练
As mentioned earlier, we have implemented various models in a distributed framework called DistBelief. Below we report the results of several models trained on the Google News 6B data set, with mini-batch asynchronous gradient descent and the adaptive learning rate procedure called Adagrad [7]. We used 50 to 100 model replicas during the training. The number of CPU cores is an estimate since the data center machines are shared with other production tasks, and the usage can fluctuate quite a bit. Note that due to the overhead of the distributed framework, the CPU usage of the CBOW model and the Skip-gram model are much closer to each other than their single-machine implementations. The result are reported in Table 6.

如前所述，我们在一个名为DistBelief的分布式框架中实现了各种模型。下面，我们报告了在谷歌新闻6B数据集上训练的几个模型的结果，包括小批量异步梯度下降和称为Adgrad[7]的自适应学习率程序。我们在训练期间使用了50到100个模型副本。由于数据中心机器与其他生产任务共享，因此CPU核心的数量是一个估计值，使用情况可能会有很大的波动。注意，由于分布式框架的开销，CBOW模型和Skip gram模型的CPU使用情况比它们的单机实现更接近。结果见表6。

Table 6: Comparison of models trained using the DistBelief distributed framework. Note that training of NNLM with 1000-dimensional vectors would take too long to complete.
表6：使用DistBelief分布式框架训练的模型的比较。注意，用1000维向量训练NNLM需要太长时间才能完成。

### 4.5 Microsoft Research Sentence Completion Challenge Microsoft研究句子完成挑战
The Microsoft Sentence Completion Challenge has been recently introduced as a task for advancing language modeling and other NLP techniques [32]. This task consists of 1040 sentences, where one word is missing in each sentence and the goal is to select word that is the most coherent with the rest of the sentence, given a list of five reasonable choices. Performance of several techniques has been already reported on this set, including N-gram models, LSA-based model [32], log-bilinear model [24] and a combination of recurrent neural networks that currently holds the state of the art performance of 55.4% accuracy on this benchmark [19].

微软句子完成挑战最近被引入，作为推进语言建模和其他NLP技术的一项任务[32]。这项任务由1040个句子组成，每个句子中缺少一个单词，目标是在给出五个合理选择的列表后，选择与句子其余部分最连贯的单词。已经报道了几种技术在这一集合上的性能，包括N-gram模型、基于LSA的模型[32]、对数双线性模型[24]和循环神经网络的组合，该组合目前在该基准上保持55.4%的最新性能[19]。

We have explored the performance of Skip-gram architecture on this task. First, we train the 640- dimensional model on 50M words provided in [32]. Then, we compute score of each sentence in the test set by using the unknown word at the input, and predict all surrounding words in a sentence. The final sentence score is then the sum of these individual predictions. Using the sentence scores, we choose the most likely sentence.

我们探讨了Skip gram架构在这项任务中的性能。首先，我们在[32]中提供的50M个单词上训练640维模型。然后，我们通过使用输入处的未知单词来计算测试集中每个句子的得分，并预测句子中所有周围的单词。最后的句子得分是这些单独预测的总和。使用句子得分，我们选择最可能的句子。

A short summary of some previous results together with the new results is presented in Table 7. While the Skip-gram model itself does not perform on this task better than LSA similarity, the scores from this model are complementary to scores obtained with RNNLMs, and a weighted combination leads to a new state of the art result 58.9% accuracy (59.2% on the development part of the set and 58.7% on the test part of the set).

表7中列出了一些先前结果以及新结果的简要总结。虽然Skip gram模型本身在这项任务上的表现不如LSA相似性，但该模型中的分数与RNNLM获得的分数是互补的，加权组合导致了58.9%的最新技术结果准确性(开发部分为59.2%，测试部分为58.7%)。

Table 7: Comparison and combination of models on the Microsoft Sentence Completion Challenge.
表7:Microsoft句子完成挑战模型的比较和组合。

## 5 Examples of the Learned Relationships 学习关系的例子
Table 8 shows words that follow various relationships. We follow the approach described above: the relationship is defined by subtracting two word vectors, and the result is added to another word. Thus for example, Paris - France + Italy = Rome. As it can be seen, accuracy is quite good, although there is clearly a lot of room for further improvements (note that using our accuracy metric that assumes exact match, the results in Table 8 would score only about 60%). We believe that word vectors trained on even larger data sets with larger dimensionality will perform significantly better, and will enable the development of new innovative applications. Another way to improve accuracy is to provide more than one example of the relationship. By using ten examples instead of one to form the relationship vector (we average the individual vectors together), we have observed improvement of accuracy of our best models by about 10% absolutely on the semantic-syntactic test. 

表8显示了遵循各种关系的单词。我们遵循上述方法：通过减去两个单词向量来定义关系，并将结果添加到另一个单词。例如，巴黎-法国+意大利=罗马。可以看出，准确度相当好，但显然还有很大的空间可以进一步改进(注意，使用我们假设精确匹配的准确度度量，表8中的结果将仅得分约60%)。我们相信，在更大维度的更大数据集上训练的单词向量将表现得更好，并将有助于开发新的创新应用程序。提高准确性的另一种方法是提供一个以上的关系样本。通过使用十个样本而不是一个来形成关系向量(我们将各个向量平均在一起)，我们观察到在语义句法测试中，我们的最佳模型的准确度绝对提高了约10%。

Table 8: Examples of the word pair relationships, using the best word vectors from Table 4 (Skipgram model trained on 783M words with 300 dimensionality).
表8：单词对关系的样本，使用表4中的最佳单词向量(在783M个300维单词上训练的Skipgram模型)。

It is also possible to apply the vector operations to solve different tasks. For example, we have observed good accuracy for selecting out-of-the-list words, by computing average vector for a list of words, and finding the most distant word vector. This is a popular type of problems in certain human intelligence tests. Clearly, there is still a lot of discoveries to be made using these techniques. 

还可以应用向量运算来解决不同的任务。例如，通过计算单词列表的平均向量并找到最远的单词向量，我们观察到了从列表中选择单词的良好准确性。这是某些人类智力测试中常见的一类问题。显然，使用这些技术仍有很多发现。

## 6 Conclusion
In this paper we studied the quality of vector representations of words derived by various models on a collection of syntactic and semantic language tasks. We observed that it is possible to train high quality word vectors using very simple model architectures, compared to the popular neural network models (both feedforward and recurrent). Because of the much lower computational complexity, it is possible to compute very accurate high dimensional word vectors from a much larger data set. Using the DistBelief distributed framework, it should be possible to train the CBOW and Skip-gram models even on corpora with one trillion words, for basically unlimited size of the vocabulary. That is several orders of magnitude larger than the best previously published results for similar models.

在本文中，我们研究了在一组句法和语义语言任务上由各种模型导出的词的向量表示的质量。我们观察到，与流行的神经网络模型(前馈和循环)相比，使用非常简单的模型架构训练高质量的单词向量是可能的。由于计算复杂度低得多，因此可以从更大的数据集计算非常精确的高维单词向量。使用DistBelief分布式框架，即使在拥有一万亿单词的语料库上，也应该可以训练CBOW和Skip语法模型，因为基本上不受词汇大小的限制。这比之前发布的同类模型的最佳结果大了几个数量级。

An interesting task where the word vectors have recently been shown to significantly outperform the previous state of the art is the SemEval-2012 Task 2 [11]. The publicly available RNN vectors were used together with other techniques to achieve over 50% increase in Spearman’s rank correlation over the previous best result [31]. The neural network based word vectors were previously applied to many other NLP tasks, for example sentiment analysis [12] and paraphrase detection [28]. It can be expected that these applications can benefit from the model architectures described in this paper.

一项有趣的任务是SemEval-2012任务2[11]，其中单词向量最近被证明显著优于先前的技术状态。公开可用的RNN向量与其他技术一起使用，使Spearman的等级相关性比之前的最佳结果提高了50%以上[31]。基于神经网络的词向量先前被应用于许多其他NLP任务，例如情感分析[12]和释义检测[28]。可以预期，这些应用程序可以从本文描述的模型架构中受益。

Our ongoing work shows that the word vectors can be successfully applied to automatic extension of facts in Knowledge Bases, and also for verification of correctness of existing facts. Results from machine translation experiments also look very promising. In the future, it would be also interesting to compare our techniques to Latent Relational Analysis [30] and others. We believe that our comprehensive test set will help the research community to improve the existing techniques for estimating the word vectors. We also expect that high quality word vectors will become an important building block for future NLP applications. 

我们正在进行的工作表明，单词向量可以成功地应用于知识库中事实的自动扩展，也可以用于验证现有事实的正确性。机器翻译实验的结果也很有希望。将来，将我们的技术与潜在关系分析[30]和其他技术进行比较也是很有意思的。我们相信，我们的综合测试集将有助于研究社区改进现有的单词向量估计技术。我们还期望高质量的词向量将成为未来NLP应用的重要构建块。

## 7 Follow-Up Work 后续工作
After the initial version of this paper was written, we published single-machine multi-threaded C++ code for computing the word vectors, using both the continuous bag-of-words and skip-gram architectures(The code is available at https://code.google.com/p/word2vec/) . The training speed is significantly higher than reported earlier in this paper, i.e. it is in the order of billions of words per hour for typical hyperparameter choices. We also published more than 1.4 million vectors that represent named entities, trained on more than 100 billion words. Some of our follow-up work will be published in an upcoming NIPS 2013 paper [21].

在本文的初始版本编写完成后，我们发布了用于计算单词向量的单机多线程C++代码，同时使用了连续的词袋和skip-gram结构。训练速度明显高于本文前面报道的速度，即对于典型的超参数选择，训练速度大约为每小时数十亿字。我们还发布了140多万个表示命名实体的向量，这些向量训练了1000多亿个单词。我们的一些后续工作将发表在即将发布的NIPS 2013论文[21]中。

## References
1. Y. Bengio, R. Ducharme, P. Vincent. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137-1155, 2003.
2. Y. Bengio, Y. LeCun. Scaling learning algorithms towards AI. In: Large-Scale Kernel Machines, MIT Press, 2007.
3. T. Brants, A. C. Popat, P. Xu, F. J. Och, and J. Dean. Large language models in machine translation. In Proceedings of the Joint Conference on Empirical Methods in Natural Language Processing and Computational Language Learning, 2007.
4. R. Collobert and J. Weston. A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning. In International Conference on Machine Learning, ICML, 2008.
5. R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu and P. Kuksa. Natural Language Processing (Almost) from Scratch. Journal of Machine Learning Research, 12:2493- 2537, 2011.
6. J. Dean, G.S. Corrado, R. Monga, K. Chen, M. Devin, Q.V. Le, M.Z. Mao, M.A. Ranzato, A. Senior, P. Tucker, K. Yang, A. Y. Ng., Large Scale Distributed Deep Networks, NIPS, 2012.
7. J.C. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 2011.
8. J. Elman. Finding Structure in Time. Cognitive Science, 14, 179-211, 1990.
9. Eric H. Huang, R. Socher, C. D. Manning and Andrew Y. Ng. Improving Word Representations via Global Context and Multiple Word Prototypes. In: Proc. Association for Computational Linguistics, 2012.
10. G.E. Hinton, J.L. McClelland, D.E. Rumelhart. Distributed representations. In: Parallel distributed processing: Explorations in the microstructure of cognition. Volume 1: Foundations, MIT Press, 1986.
11. D.A. Jurgens, S.M. Mohammad, P.D. Turney, K.J. Holyoak. Semeval-2012 task 2: Measuring degrees of relational similarity. In: Proceedings of the 6th International Workshop on Semantic Evaluation (SemEval 2012), 2012.
12. A.L. Maas, R.E. Daly, P.T. Pham, D. Huang, A.Y. Ng, and C. Potts. Learning word vectors for sentiment analysis. In Proceedings of ACL, 2011.
13. T. Mikolov. Language Modeling for Speech Recognition in Czech, Masters thesis, Brno University of Technology, 2007.
14. T. Mikolov, J. Kopeck´y, L. Burget, O. Glembek and J. ˇCernock´y. Neural network based language models for higly inflective languages, In: Proc. ICASSP 2009.
15. T. Mikolov, M. Karafi´at, L. Burget, J. ˇCernock´y, S. Khudanpur. Recurrent neural network based language model, In: Proceedings of Interspeech, 2010.
16. T. Mikolov, S. Kombrink, L. Burget, J. ˇCernock´y, S. Khudanpur. Extensions of recurrent neural network language model, In: Proceedings of ICASSP 2011.
17. T. Mikolov, A. Deoras, S. Kombrink, L. Burget, J. ˇCernock´y. Empirical Evaluation and Combination of Advanced Language Modeling Techniques, In: Proceedings of Interspeech, 2011. 4The code is available at https://code.google.com/p/word2vec/ 11
18. T. Mikolov, A. Deoras, D. Povey, L. Burget, J. ˇCernock´y. Strategies for Training Large Scale Neural Network Language Models, In: Proc. Automatic Speech Recognition and Understanding, 2011.
19. T. Mikolov. Statistical Language Models based on Neural Networks. PhD thesis, Brno University of Technology, 2012.
20. T. Mikolov, W.T. Yih, G. Zweig. Linguistic Regularities in Continuous Space Word Representations. NAACL HLT 2013.
21. T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean. Distributed Representations of Words and Phrases and their Compositionality. Accepted to NIPS 2013.
22. A. Mnih, G. Hinton. Three new graphical models for statistical language modelling. ICML, 2007.
23. A. Mnih, G. Hinton. A Scalable Hierarchical Distributed Language Model. Advances in Neural Information Processing Systems 21, MIT Press, 2009.
24. A. Mnih, Y.W. Teh. A fast and simple algorithm for training neural probabilistic language models. ICML, 2012.
25. F. Morin, Y. Bengio. Hierarchical Probabilistic Neural Network Language Model. AISTATS, 2005.
26. D. E. Rumelhart, G. E. Hinton, R. J. Williams. Learning internal representations by backpropagating errors. Nature, 323:533.536, 1986.
27. H. Schwenk. Continuous space language models. Computer Speech and Language, vol. 21, 2007.
28. R. Socher, E.H. Huang, J. Pennington, A.Y. Ng, and C.D. Manning. Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection. In NIPS, 2011.
29. J. Turian, L. Ratinov, Y. Bengio. Word Representations: A Simple and General Method for Semi-Supervised Learning. In: Proc. Association for Computational Linguistics, 2010.
30. P. D. Turney. Measuring Semantic Similarity by Latent Relational Analysis. In: Proc. International Joint Conference on Artificial Intelligence, 2005.
31. A. Zhila, W.T. Yih, C. Meek, G. Zweig, T. Mikolov. Combining Heterogeneous Models for Measuring Relational Similarity. NAACL HLT 2013.
32. G. Zweig, C.J.C. Burges. The Microsoft Research Sentence Completion Challenge, Microsoft Research Technical Report MSR-TR-2011-129, 2011. 12
