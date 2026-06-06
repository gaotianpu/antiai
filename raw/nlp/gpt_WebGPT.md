# WebGPT: Browser-assisted question-answering with human feedback
WebGPT：带有人工反馈的浏览器辅助问答 https://arxiv.org/abs/2112.09332

## 阅读笔记
* 对齐技术
* 模仿学习, imitation learning 
* behavior cloning, 行为克隆， rejection sampling 

## Abstract
We fine-tune GPT-3 to answer long-form questions using a text-based webbrowsing environment, which allows the model to search and navigate the web. By setting up the task so that it can be performed by humans, we are able to train models on the task using imitation learning, and then optimize answer quality with human feedback. To make human evaluation of factual accuracy easier, models must collect references while browsing in support of their answers. We train and evaluate our models on ELI5, a dataset of questions asked by Reddit users. Our best model is obtained by fine-tuning GPT-3 using behavior cloning, and then performing rejection sampling against a reward model trained to predict human preferences. This model’s answers are preferred by humans 56% of the time to those of our human demonstrators, and 69% of the time to the highest-voted answer from Reddit.

我们微调 GPT-3 以使用基于文本的网络浏览环境回答长篇问题，该环境允许模型搜索和导航网络。 通过设置任务使其可以由人类执行，我们能够使用模仿学习训练任务模型，然后通过人类反馈优化答案质量。 为了让人类更容易评估事实的准确性，模型必须在浏览时收集参考资料以支持他们的答案。 我们在 ELI5 上训练和评估我们的模型，ELI5 是 Reddit 用户提出的问题数据集。 我们最好的模型是通过使用行为克隆对 GPT-3 进行微调，然后针对经过训练以预测人类偏好的奖励模型执行拒绝抽样而获得的。 这个模型的答案在 56% 的情况下比我们的人类演示者的答案更受人类的青睐，而在 69% 的情况下人类更喜欢 Reddit 上投票最高的答案。

## 1 Introduction
A rising challenge in NLP is long-form question-answering (LFQA), in which a paragraph-length answer is generated in response to an open-ended question. LFQA systems have the potential to become one of the main ways people learn about the world, but currently lag behind human performance [Krishna et al., 2021]. Existing work tends to focus on two core components of the task, information retrieval and synthesis.

NLP 中一个日益增长的挑战是长篇问答 (LFQA)，其中针对开放式问题生成段落长度的答案。 LFQA 系统有可能成为人们了解世界的主要方式之一，但目前落后于人类表现 [Krishna et al., 2021]。 现有工作倾向于关注任务的两个核心组成部分，即信息检索和综合。

In this work we leverage existing solutions to these components: we outsource document retrieval to the Microsoft Bing Web Search API,(2 https://www.microsoft.com/en-us/bing/apis/bing-web-search-api ) and utilize unsupervised pre-training to achieve high-quality synthesis by fine-tuning GPT-3 [Brown et al., 2020]. Instead of trying to improve these ingredients, we focus on combining them using more faithful training objectives. Following Stiennon et al. [2020], we use human feedback to directly optimize answer quality, allowing us to achieve performance competitive with humans.

在这项工作中，我们利用这些组件的现有解决方案：我们将文档检索外包给 Microsoft Bing Web 搜索 API，(2 https://www.microsoft.com/en-us/bing/apis/bing-web-search-api ) 并利用无监督预训练通过微调 GPT-3 实现高质量合成 [Brown et al., 2020]。 我们没有尝试改进这些成分，而是专注于使用更忠实的训练目标将它们结合起来。 在 Stiennon et al. [2020]之后，我们使用人类反馈直接优化答案质量，使我们能够实现与人类竞争的性能。

We make two key contributions: 
* We create a text-based web-browsing environment that a fine-tuned language model can interact with. This allows us to improve both retrieval and synthesis in an end-to-end fashion using general methods such as imitation learning and reinforcement learning. 
* We generate answers with references: passages extracted by the model from web pages while browsing. This is crucial for allowing labelers to judge the factual accuracy of answers, without engaging in a difficult and subjective process of independent research.

我们做出了两个关键贡献：
* 创建了一个基于文本的网络浏览环境，微调的语言模型可以与之交互。 这使我们能够使用模仿学习和强化学习等通用方法以端到端的方式改进检索和合成。
* 生成带有参考的答案：模型在浏览时从网页中提取的段落。 这对于允许标注人员判断答案的事实准确性至关重要，而无需参与独立研究的困难和主观过程。

Our models are trained primarily to answer questions from ELI5 [Fan et al., 2019], a dataset of questions taken from the “Explain Like I’m Five” subreddit. We collect two additional kinds of data: demonstrations of humans using our web-browsing environment to answer questions, and comparisons between two model-generated answers to the same question (each with their own set of references). Answers are judged for their factual accuracy, coherence, and overall usefulness.

我们的模型主要经过训练以回答 ELI5 [Fan et al., 2019] 中的问题，ELI5 是从“像我五岁一样解释”subreddit 中提取的问题数据集。 我们收集了另外两种数据：人类使用我们的网络浏览环境回答问题的演示，以及对同一问题的两个模型生成答案之间的比较（每个都有自己的参考集）。 答案是根据事实的准确性、连贯性和整体实用性来判断的。

We use this data in four main ways: behavior cloning (i.e., supervised fine-tuning) using the demonstrations, reward modeling using the comparisons, reinforcement learning against the reward model, and rejection sampling against the reward model. Our best model uses a combination of behavior cloning and rejection sampling. We also find reinforcement learning to provide some benefit when inference-time compute is more limited.

我们以四种主要方式使用这些数据：使用演示的行为克隆（即监督微调）、使用比较的奖励建模、针对奖励模型的强化学习以及针对奖励模型的拒绝抽样。 我们最好的模型结合了行为克隆和拒绝抽样。 我们还发现强化学习在推理时计算更受限制时提供了一些好处。

We evaluate our best model in three different ways. First, we compare our model’s answers to answers written by our human demonstrators on a held-out set of questions. Our model’s answers are preferred 56% of the time, demonstrating human-level usage of the text-based browser. Second, we compare our model’s answers (with references stripped, for fairness) to the highest-voted answer provided by the ELI5 dataset. Our model’s answers are preferred 69% of the time. Third, we evaluate our model on TruthfulQA [Lin et al., 2021], an adversarial dataset of short-form questions. Our model’s answers are true 75% of the time, and are both true and informative 54% of the time, outperforming our base model (GPT-3), but falling short of human performance.

我们以三种不同的方式评估我们的最佳模型。 首先，我们将模型的答案与人类演示者针对一组保留的问题所写的答案进行比较。 我们模型的答案在 56% 的情况下是首选的，展示了基于文本的浏览器的人类使用水平。 其次，我们将模型的答案（为了公平起见，参考文献被剥离）与 ELI5 数据集提供的投票最高的答案进行比较。 我们模型的答案在 69% 的情况下是首选。 第三，我们在 TruthfulQA [Lin et al., 2021] 上评估我们的模型，TruthfulQA 是一种简短问题的对抗性数据集。 我们模型的答案在 75% 的情况下是正确的，在 54% 的情况下是真实且信息丰富的，优于我们的基本模型 (GPT-3)，但不及人类表现。

The remainder of the paper is structured as follows: 
* In Section 2, we describe our text-based web-browsing environment and how our models interact with it. 
* In Section 3, we explain our data collection and training methods in more detail. 
* In Section 4, we evaluate our best-performing models (for different inference-time compute budgets) on ELI5 and TruthfulQA. 
* In Section 5, we provide experimental results comparing our different methods and how they scale with dataset size, parameter count, and inference-time compute. 
* In Section 6, we discuss the implications of our findings for training models to answer questions truthfully, and broader impacts. 

在本文的其余结构如下：
* 在第 2 节中，描述了我们基于文本的网络浏览环境以及我们的模型如何与之交互。
* 在第 3 节中，更详细地解释了我们的数据收集和训练方法。
* 在第 4 节中，评估了我们在 ELI5 和 TruthfulQA 上表现最佳的模型（针对不同的推理时间计算预算）。
* 在第 5 节中，提供了实验结果，比较了我们的不同方法以及它们如何随数据集大小、参数计数和推理时间计算进行缩放。
* 在第 6 节中，讨论了我们的发现对训练模型如实回答问题的影响，以及更广泛的影响。

## 2 Environment design
Previous work on question-answering such as REALM [Guu et al., 2020] and RAG [Lewis et al., 2020a] has focused on improving document retrieval for a given query. Instead, we use a familiar existing method for this: a modern search engine (Bing). This has two main advantages. First, modern search engines are already very powerful, and index a large number of up-to-date documents. Second, it allows us to focus on the higher-level task of using a search engine to answer questions, something that humans can do well, and that a language model can mimic.

REALM [Guu et al., 2020] 和 RAG [Lewis et al., 2020a] 以前的问答工作侧重于改进给定查询的文档检索。 相反，我们为此使用熟悉的现有方法：现代搜索引擎 (Bing)。 这有两个主要优点。 首先，现代搜索引擎已经非常强大，可以索引大量最新的文档。 其次，它使我们能够专注于使用搜索引擎回答问题的更高层次的任务，这是人类可以做得很好并且语言模型可以模仿的事情。

For this approach, we designed a text-based web-browsing environment. The language model is prompted with a written summary of the current state of the environment, including the question, the text of the current page at the current cursor location, and some other information (see Figure 1(b)). In response to this, the model must issue one of the commands given in Table 1, which performs an action such as running a Bing search, clicking on a link, or scrolling around. This process is then repeated with a fresh context (hence, the only memory of previous steps is what is recorded in the summary).

对于这种方法，我们设计了一个基于文本的网络浏览环境。 语言模型会收到当前环境状态的书面摘要提示，包括问题、当前光标位置处当前页面的文本以及一些其他信息（参见图1(b)）。 为此，模型必须发出 表1 中给出的命令之一，该命令执行诸如运行 Bing 搜索、单击链接或四处滚动等操作。 然后使用新的上下文重复此过程（因此，之前步骤的唯一记忆是摘要中记录的内容）。

![Figure 1](../images/WebGPT/fig_1.png)<br/>
Figure 1: An observation from our text-based web-browsing environment, as shown to human demonstrators (left) and models (right). The web page text has been abridged for illustrative purposes. 
图1：从我们基于文本的 Web 浏览环境中观察到的结果，展示给演示者（左）和模型（右）。 出于说明目的，网页文本已被删节。

![Table 1](../images/WebGPT/tab_1.png)<br/>
Table 1: Actions the model can take. If a model generates any other text, it is considered to be an invalid action. Invalid actions still count towards the maximum, but are otherwise ignored.
表1：模型可以采取的行动。 如果模型生成任何其他文本，则视为无效操作。 无效操作仍计入最大值，但会被忽略。

While the model is browsing, one of the actions it can take is to quote an extract from the current page. When this is performed, the page title, domain name and extract are recorded to be used later as a reference. Browsing then continues until either the model issues a command to end browsing, the maximum number of actions has been reached, or the maximum total length of references has been reached. At this point, as long as there is at least one reference, the model is prompted with the question and the references, and must compose its final answer.

在模型浏览时，它可以执行的操作之一是引用当前页面的摘录。 执行此操作时，页面标题、域名和摘录将被记录下来以备后用。 然后继续浏览，直到模型发出结束浏览的命令、已达到最大操作数或已达到最大引用总长度。 此时，只要有至少一个参考，模型就会被提示问题和参考，并且必须组成它的最终答案。

Further technical details about our environment can be found in Appendix A. 

有关我们环境的更多技术细节可以在附录 A 中找到。

## 3 Methods
### 3.1 Data collection
Guidance from humans is central to our approach. A language model pre-trained on natural language would not be able to use our text-based browser, since it does not know the format of valid commands. We therefore collected examples of humans using the browser to answer questions, which we call demonstrations. However, training on demonstrations alone does not directly optimize answer quality, and is unlikely to lead far beyond human performance [Stiennon et al., 2020]. We therefore collected pairs of model-generated answers to the same question, and asked humans which one they preferred, which we call comparisons.

来自人类的指导是我们方法的核心。 在自然语言上预训练的语言模型将无法使用我们基于文本的浏览器，因为它不知道有效命令的格式。 因此，我们收集了人类使用浏览器回答问题的样本，我们称之为演示。 然而，仅靠演示训练并不能直接优化答案质量，而且不太可能远远超过人类的表现 [Stiennon et al., 2020 年]。 因此，我们收集了针对同一问题的成对模型生成答案，并询问人类他们更喜欢哪一个，我们称之为比较。

For both demonstrations and comparisons, the vast majority of questions were taken from ELI5 [Fan et al., 2019], a dataset of long-form questions. For diversity and experimentation, we also mixed in a small number of questions from other sources, such as TriviaQA [Joshi et al., 2017]. In total, we collected around 6,000 demonstrations, 92% of which were for questions from ELI5, and around 21,500 comparisons, 98% of which were for questions from ELI5. A more detailed breakdown of the questions we used along with post-processing details can be found in Appendix B. 

对于演示和比较，绝大多数问题都来自 ELI5 [Fan et al., 2019]，这是一个长格式问题的数据集。 为了多样性和实验性，我们还混合了来自其他来源的少量问题，例如 TriviaQA [Joshi et al., 2017]。 我们总共收集了大约 6,000 个演示，其中 92% 来自 ELI5 的问题，以及大约 21,500 个比较，其中 98% 来自 ELI5 的问题。 我们使用的问题的更详细分类以及后处理细节可以在附录 B 中找到。

![Table 2](../images/WebGPT/tab_2.png)<br/>
Table 2: An answer produced by our 175B best-of-64 model to a randomly-chosen question from the ELI5 test set (not cherry-picked). The full text of the references can be found in Appendix J, along with answers from our human demonstrators and the ELI5 dataset. Further samples are available at https://openaipublic.blob.core.windows.net/webgpt-answer-viewer/index.html.
表2：我们的 175B best-of-64 模型对 ELI5 测试集中随机选择的问题（非精选）产生的答案。 可以在附录 J 中找到参考文献的全文，以及我们的人类演示者和 ELI5 数据集的答案。 如需更多样本，请访问 https://openaipublic.blob.core.windows.net/webgpt-answer-viewer/index.html。

To make it easier for humans to provide demonstrations, we designed a graphical user interface for the environment (see Figure 1(a)). This displays essentially the same information as the text-based interface and allows any valid action to be performed, but is more human-friendly. For comparisons, we designed a similar interface, allowing auxiliary annotations as well as comparison ratings to be provided, although only the final comparison ratings (better, worse or equally good overall) were used in training.

为了让人类更容易提供演示，我们为环境设计了一个图形用户界面（见图1(a)）。 这显示与基于文本的界面基本相同的信息，并允许执行任何有效的操作，但更人性化。 为了进行比较，我们设计了一个类似的界面，允许提供辅助注释和比较评级，尽管在训练中只使用了最终的比较评级（总体上更好、更差或同样好）。

For both demonstrations and comparisons, we emphasized that answers should be relevant, coherent, and supported by trustworthy references. Further details about these criteria and other aspects of our data collection pipeline can be found in Appendix C.

对于演示和比较，我们强调答案应该是相关的、连贯的，并有可信赖的参考资料支持。 有关这些标准和我们数据收集管道其他方面的更多详情，请参见附录 C。

We are releasing a dataset of comparisons, the details of which can be found in Appendix K.

我们正在发布一个比较数据集，其详情可以在附录 K 中找到。

### 3.2 Training
The use of pre-trained models is crucial to our approach. Many of the underlying capabilities required to successfully use our environment to answer questions, such as reading comprehension and answer synthesis, emerge as zero-shot capabilities of language models [Brown et al., 2020]. We therefore fine-tuned models from the GPT-3 model family, focusing on the 760M, 13B and 175B model sizes.

使用预训练模型对我们的方法至关重要。 成功使用我们的环境来回答问题所需的许多基础能力，例如阅读理解和答案合成，都作为语言模型的零样本能力出现 [Brown et al., 2020]。 因此，我们对 GPT-3 模型系列中的模型进行了微调，重点关注 760M、13B 和 175B 模型尺寸。

Starting from these models, we used four main training methods:
1. Behavior cloning (BC). We fine-tuned on the demonstrations using supervised learning, with the commands issued by the human demonstrators as labels.
2. Reward modeling (RM). Starting from the BC model with the final unembedding layer removed, we trained a model to take in a question and an answer with references, and output a scalar reward. Following Stiennon et al. [2020], the reward represents an Elo score, scaled such that the difference between two scores represents the logit of the probability that one will be preferred to the other by the human labelers. The reward model is trained using a cross-entropy loss, with the comparisons as labels. Ties are treated as soft 50% labels.
3. Reinforcement learning (RL). Once again following Stiennon et al. [2020], we fine-tuned the BC model on our environment using PPO [Schulman et al., 2017]. For the environment reward, we took the reward model score at the end of each episode, and added this to a KL penalty from the BC model at each token to mitigate overoptimization of the reward model.
4. Rejection sampling (best-of-n). We sampled a fixed number of answers (4, 16 or 64) from either the BC model or the RL model (if left unspecified, we used the BC model), and selected the one that was ranked highest by the reward model. We used this as an alternative method of optimizing against the reward model, which requires no additional training, but instead uses more inference-time compute. 

从这些模型开始，我们使用了四种主要的训练方法：
1. 行为克隆（BC）。 我们使用监督学习对演示进行了微调，并将人类演示者发出的命令作为标签。
2. 奖励模型（RM）。 从移除了最终反嵌入层的 BC 模型开始，我们训练了一个模型来接收问题和带有参考的答案，并输出标量奖励。 在 Stiennon et al. [2020]之后，奖励代表一个 Elo 分数，经过缩放使得两个分数之间的差异代表人类标注人员优先选择另一个的概率的对数。 奖励模型使用交叉熵损失进行训练，并将比较作为标签。 领带被视为软 50% 标签。
3. 强化学习（RL）。 再次跟随 Stiennon et al. [2020]，我们使用 PPO [Schulman et al., 2017] 在我们的环境中微调了 BC 模型。 对于环境奖励，我们在每一集结束时获取奖励模型分数，并将其添加到每个令牌的 BC 模型的 KL 惩罚中，以减轻奖励模型的过度优化。
4. 拒绝抽样（best-of-n）。 我们从 BC 模型或 RL 模型（如果未指定，我们使用 BC 模型）中抽取固定数量的答案（4、16 或 64），并选择奖励模型排名最高的答案。 我们将其用作针对奖励模型进行优化的替代方法，它不需要额外的训练，而是使用更多的推理时间计算。

We used mutually disjoint sets of questions for each of BC, RM and RL. 
For BC, we held out around 4% of the demonstrations to use as a validation set.

我们对 BC、RM 和 RL 中的每一个都使用了相互不相交的问题集。 对于 BC，我们保留了大约 4% 的演示用作验证集。

For RM, we sampled answers for the comparison datasets in an ad-hoc manner, using models of various sizes (but primarily the 175B model size), trained using various combinations of methods and hyperparameters, and combined them into a single dataset. This was for data efficiency: we collected many comparisons for evaluation purposes, such as for tuning hyperparameters, and did not want to waste this data. Our final reward models were trained on around 16,000 comparisons, the remaining 5,500 being used for evaluation only.

对于 RM，我们使用各种大小的模型（但主要是 175B 模型大小）以临时方式对比较数据集的答案进行采样，使用方法和超参数的各种组合进行训练，并将它们组合成一个数据集。 这是为了提高数据效率：我们收集了许多用于评估目的的比较，例如用于调整超参数，并且不想浪费这些数据。 我们最终的奖励模型接受了大约 16,000 次比较的训练，其余 5,500 次仅用于评估。

For RL, we trained on a mixture of 90% questions from ELI5 and 10% questions from TriviaQA. To improve sample efficiency, at the end of each episode we inserted 15 additional answering-only episodes using the same references as the previous episode. We were motivated to try this because answering explained slightly more of the variance in reward model score than browsing despite taking many fewer steps, and we found it to improve sample efficiency by approximately a factor of 2. We also randomized the maximum number of browsing actions, sampling uniformly from the range 20–100 inclusive.

对于 RL，我们训练了 90% 来自 ELI5 的问题和 10% 来自 TriviaQA 的问题。 为了提高样本效率，在每一集的结尾，我们使用与上一集相同的参考插入了 15 个额外的仅回答集。 我们有动力尝试这个，因为尽管采取的步骤少得多，但回答比浏览更能解释奖励模型分数的差异，我们发现它可以将样本效率提高大约 2 倍。我们还随机化了浏览操作的最大数量 ，从 20-100 范围内统一抽样。

Hyperparameters for all of our training methods can be found in Appendix E. 

我们所有训练方法的超参数都可以在附录 E 中找到。

## 4 Evaluation
In evaluating our approach, we focused on three “WebGPT” models, each of which was trained with behavior cloning followed by rejection sampling against a reward model of the same size: a 760M best-of-4 model, a 13B best-of-16 model and a 175B best-of-64 model. As discussed in Section 5.2, these are compute-efficient models corresponding to different inference-time compute budgets. We excluded RL for simplicity, since it did not provide significant benefit when combined with rejection sampling (see Figure 4).

在评估我们的方法时，我们专注于三个“WebGPT”模型，每个模型都经过行为克隆训练，然后针对相同大小的奖励模型进行拒绝采样：一个 760M best-of-4 模型，一个 13B best-of- 16 模型和 175B 最佳 64 模型。 如第 5.2 节所述，这些是对应于不同推理时间计算预算的计算高效模型。 为简单起见，我们将 RL 排除在外，因为它在与拒绝抽样结合时没有提供显著的好处（见图4）。

We evaluated all WebGPT models using a sampling temperature of 0.8, which was tuned using human evaluations, and with a maximum number of browsing actions of 100.

我们使用 0.8 的采样温度评估了所有 WebGPT 模型，该温度使用人工评估进行了调整，浏览操作的最大数量为 100。

### 4.1 ELI5
We evaluated WebGPT on the ELI5 test set in two different ways:
1. We compared model-generated answers to answers written by demonstrators using our web-browsing environment. For these comparisons, we used the same procedure as comparisons used for reward model training. We consider this to be a fair comparison, since the instructions for demonstrations and comparisons emphasize a very similar set of criteria.
2. We compared model-generated answers to the reference answers from the ELI5 dataset, which are the highest-voted answers from Reddit. In this case, we were concerned about ecological validity, since our detailed comparison criteria may not match those of real-life users. We were also concerned about blinding, since Reddit answers do not typically include citations. To mitigate these concerns, we stripped all citations and references from the model-generated answers, hired new contractors who were not familiar with our detailed instructions, and gave them a much more minimal set of instructions, which are given in Appendix F.

我们以两种不同的方式在 ELI5 测试集上评估 WebGPT：
1. 我们将模型生成的答案与演示者使用我们的网络浏览环境编写的答案进行了比较。 对于这些比较，我们使用了与用于奖励模型训练的比较相同的程序。 我们认为这是一个公平的比较，因为演示和比较的说明强调了一组非常相似的标准。
2. 我们将模型生成的答案与来自 ELI5 数据集的参考答案进行了比较，后者是 Reddit 上得票最高的答案。 在这种情况下，我们担心生态有效性，因为我们详细的比较标准可能与现实生活中的用户不符。 我们还担心致盲，因为 Reddit 答案通常不包含引用。 为了减轻这些担忧，我们从模型生成的答案中删除了所有引文和参考资料，聘请了不熟悉我们详细说明的新承包商，并给了他们一组更简单的说明，这些说明在附录 F 中给出。

In both cases, we treat ties as 50% preference ratings (rather than excluding them).

在这两种情况下，我们将领带视为 50% 的偏好评级（而不是排除它们）。

Our results are shown in Figure 2. Our best model, the 175B best-of-64 model, produces answers that are preferred to those written by our human demonstrators 56% of the time. This suggests that the use of human feedback is essential, since one would not expect to exceed 50% preference by imitating demonstrations alone (although it may still be possible, by producing a less noisy policy). The same model produces answers that are preferred to the reference answers from the ELI5 dataset 69% of the time. This is a substantial improvement over Krishna et al. [2021], whose best model’s answers are preferred 23% of the time to the reference answers, although they use substantially less compute than even our smallest model. 

我们的结果如图2 所示。我们最好的模型，即 175B 64 中最佳模型，在 56% 的情况下产生的答案优于我们的人类演示者所写的答案。 这表明使用人类反馈是必不可少的，因为人们不会期望仅通过模仿示范就超过 50% 的偏好（尽管通过制定噪音较小的策略，这仍然是可能的）。 在 69% 的情况下，同一模型生成的答案优于 ELI5 数据集的参考答案。 这是对 Krishna 等人的重大改进。 [2021]，其最佳模型的答案在 23% 的情况下比参考答案更受欢迎，尽管它们使用的计算量甚至比我们最小的模型少得多。

![Figure 2](../images/WebGPT/fig_2.png)<br/>
Figure 2: Human evaluations on ELI5 comparing against (a) demonstrations collected using our web browser, (b) the highest-voted answer for each question. The amount of rejection sampling (the n in best-of-n) was chosen to be compute-efficient (see Figure 8). Error bars represent ±1 standard error.
图2：人类对 ELI5 的评估与 (a) 使用我们的网络浏览器收集的演示，(b) 每个问题的最高投票答案进行比较。 选择拒绝采样的数量（best-of-n 中的 n）以提高计算效率（参见图8）。 误差线表示±1 个标准误差。

Although the evaluations against the ELI5 reference answers are useful for comparing to prior work, we believe that the evaluations against human demonstrations are more meaningful, for several reasons: 
* Fact-checking. It is difficult to assess the factual accuracy of answers without references: even with the help of a search engine, expertise is often required. However, WebGPT and human demonstrators provide answers with references. 
* Objectivity. The use of minimal instructions makes it harder to know what criteria are being used to choose one answer over another. Our more detailed instructions enable more interpretable and consistent comparisons. 
* Blinding. Even with citations and references stripped, WebGPT composes answers that are different in style to Reddit answers, making the comparisons less blinded. In contrast, WebGPT and human demonstrators compose answers in similar styles. Additionally, some ELI5 answers contained links, which we instructed labelers not to follow, and this could have biased labelers against those answers. 
* Answer intent. People ask questions on ELI5 to obtain original, simplified explanations rather than answers that can already be found on the web, but these were not criteria we wanted answers to be judged on. Moreover, many ELI5 questions only ever get a small number of low-effort answers. With human demonstrations, it is easier to ensure that the desired intent and level of effort are used consistently.

尽管针对 ELI5 参考答案的评估对于与之前的工作进行比较很有用，但我们认为针对人类演示的评估更有意义，原因如下：
* 事实核查。 没有参考文献很难评估答案的事实准确性：即使借助搜索引擎，通常也需要专业知识。 然而，WebGPT 和人类演示者提供了带有参考的答案。
* 客观性。 使用最少的说明使得更难知道使用什么标准来选择一个答案而不是另一个答案。 我们更详细的说明可实现更具可解释性和一致性的比较。
*致盲。 即使删除了引文和参考文献，WebGPT 也会编写与 Reddit 答案风格不同的答案，从而减少比较的盲目性。 相比之下，WebGPT 和人类演示者以相似的风格撰写答案。 此外，一些 ELI5 答案包含链接，我们指示标记者不要遵循这些链接，这可能会使标记者对这些答案产生偏见。
* 回答意图。 人们在 ELI5 上提问是为了获得原始的、简化的解释，而不是已经可以在网络上找到的答案，但这不是我们想要判断答案的标准。 此外，许多 ELI5 问题只能得到少量简单的答案。 通过人工演示，更容易确保始终如一地使用所需的意图和努力程度。

### 4.2 TruthfulQA
To further probe the abilities of WebGPT, we evaluated WebGPT on TruthfulQA [Lin et al., 2021], an adversarially-constructed dataset of short-form questions. TruthfulQA questions are crafted such that they would be answered falsely by some humans due to a false belief or misconception. Answers are scored on both truthfulness and informativeness, which trade off against one another (for example, “I have no comment” is considered truthful but not informative).

为了进一步探索 WebGPT 的能力，我们在 TruthfulQA [Lin et al., 2021] 上评估了 WebGPT，TruthfulQA 是一种对抗性构建的简短问题数据集。 TruthfulQA 问题的设计使得某些人会由于错误的信念或误解而错误地回答这些问题。 答案的真实性和信息量均有评分，两者相互权衡（例如，“我无可奉告”被认为是真实的，但不提供信息）。

We evaluated both the base GPT-3 models used by WebGPT and the WebGPT models themselves on TruthfulQA. For GPT-3, we used both the “QA prompt” and the “helpful prompt” from Lin et al. [2021], and used the automated metric, since this closely tracks human evaluation on answers produced by the GPT-3 model family. For WebGPT, we used human evaluation, since WebGPT’s answers are out-of-distribution for the automated metric. TruthfulQA is a short-form dataset, so we also truncated WebGPT’s answers to 50 tokens in length, and then removed any trailing partial sentences. (3 This inadvertently resulted in a small number of empty answers, which were considered truthful but not informative. This affected 74 answers in total, around 3% of answers)

我们在 TruthfulQA 上评估了 WebGPT 使用的基本 GPT-3 模型和 WebGPT 模型本身。 对于 GPT-3，我们同时使用了 Lin 等人的“QA 提示”和“有用提示”。 [2021]，并使用了自动化指标，因为这密切跟踪了人类对 GPT-3 模型系列产生的答案的评估。 对于 WebGPT，我们使用人工评估，因为 WebGPT 的答案对于自动指标而言是分布外的。 TruthfulQA 是一个短格式数据集，因此我们还将 WebGPT 的答案截断为 50 个标记的长度，然后删除了任何尾随的部分句子。(3 这无意中导致了少量空洞的答案，这些答案被认为是真实的，但没有提供信息。 这总共影响了 74 个答案，约占答案的 3%)

![Figure 3](../images/WebGPT/fig_3.png)<br/>
Figure 3: TruthfulQA results. The amount of rejection sampling (the n in best-of-n) was chosen to be compute-efficient (see Figure 8). Error bars represent ±1 standard error. 
图3：TruthfulQA 结果。 选择拒绝采样的数量（best-of-n 中的 n）以提高计算效率（参见图8）。 误差线表示±1 个标准误差。

Our results are shown in Figure 3. All WebGPT models outperform all GPT-3 models (with both prompts) on both the percentage of truthful answers and the percentage of truthful and informative answers. Moreover, the percentage of truthful and informative answers increases with model size for WebGPT, unlike GPT-3 with either prompt. Further qualitative analysis of WebGPT’s performance on TruthfulQA is given in Section 6.1.

我们的结果如图3 所示。所有 WebGPT 模型在真实答案的百分比以及真实和信息丰富的答案的百分比方面都优于所有 GPT-3 模型（具有两种提示）。 此外，与带有任何提示的 GPT-3 不同，真实和信息丰富的答案的百分比随着 WebGPT 的模型大小而增加。 6.1 节给出了 WebGPT 在 TruthfulQA 上的性能的进一步定性分析。

### 4.3 TriviaQA
We also evaluated the WebGPT 175B BC model on TriviaQA [Joshi et al., 2017]. These results are given in Appendix G. 

我们还在 TriviaQA [Joshi et al., 2017] 上评估了 WebGPT 175B BC 模型。 这些结果在附录 G 中给出。

## 5 Experiments
### 5.1 Comparison of training methods
We ran a number of additional experiments comparing reinforcement learning (RL) and rejection sampling (best-of-n) with each other and with the behavior cloning (BC) baseline. Our results are shown in Figures 4 and 5. Rejection sampling provides a substantial benefit, with the 175B best-of-64 BC model being preferred 68% of the time to the 175B BC model. Meanwhile, RL provides a smaller benefit, with the 175B RL model being preferred 58% of the time to the 175B BC model.

我们进行了一些额外的实验，将强化学习 (RL) 和拒绝抽样 (best-of-n) 相互比较，并与行为克隆 (BC) 基线进行比较。 我们的结果显示在图4 和图5 中。拒绝抽样提供了实质性的好处，175B 最佳 64 BC 模型在 68% 的情况下比 175B BC 模型更受欢迎。 同时，RL 提供的好处较小，175B RL 模型在 58% 的情况下比 175B BC 模型更受青睐。

Even though both rejection sampling and RL optimize against the same reward model, there are several possible reasons why rejection sampling outperforms RL: 
* It may help to have many answering attempts, simply to make use of more inference-time compute. 
* The environment is unpredictable: with rejection sampling, the model can try visiting many more websites, and then evaluate the information it finds with the benefit of hindsight. 
* The reward model was trained primarily on data collected from BC and rejection sampling policies, which may have made it more robust to overoptimization by rejection sampling than by RL. 
* RL requires hyperparameter tuning, whereas rejection sampling does not.

尽管拒绝抽样和 RL 都针对相同的奖励模型进行了优化，但拒绝抽样优于 RL 的原因可能有以下几个：
* 进行多次回答尝试可能会有所帮助，只是为了利用更多的推理时间计算。
* 环境是不可预测的：通过拒绝抽样，模型可以尝试访问更多的网站，然后事后评估它找到的信息。
* 奖励模型主要根据从 BC 收集的数据和拒绝抽样策略进行训练，这可能使其通过拒绝抽样比 RL 更能抵抗过度优化。
* RL 需要调整超参数，而拒绝采样则不需要。

![Figure 4](../images/WebGPT/fig_4.png)<br/>
Figure 4: Preference of RL models over BC models, with (right) and without (left) using rejection sampling. RL slightly improves preference, but only when not using rejection sampling. Error bars represent ±1 standard error. 
图4：RL 模型优于 BC 模型，使用（右）和不使用（左）拒绝抽样。 RL 略微提高了偏好，但仅在不使用拒绝抽样时有效。 误差线表示±1 个标准误差。

![Figure 5](../images/WebGPT/fig_5.png)<br/>
Figure 5: Preference of the 175B best-of-n BC model over the BC model. The validation RM prediction is obtained using the estimator described in Appendix I, and predicts human preference well in this setting. The shaded region represents ±1 standard error. 
图5：175B best-of-n BC 模型优于 BC 模型。 验证 RM 预测是使用附录 I 中描述的估计器获得的，并在此设置中很好地预测了人类偏好。 阴影区域代表±1 个标准误差。

The combination of RL and rejection sampling also fails to offer much benefit over rejection sampling alone. One possible reason for this is that RL and rejection sampling are optimizing against the same reward model, which can easily be overoptimized (especially by RL, as noted above). In addition to this, RL reduces the entropy of the policy, which hurts exploration. Adapting the RL objective to optimize rejection sampling performance is an interesting direction for future research.

RL 和拒绝抽样的组合也无法提供比单独的拒绝抽样更多的优势。 一个可能的原因是 RL 和拒绝抽样正在针对相同的奖励模型进行优化，这很容易被过度优化（尤其是 RL，如上所述）。 除此之外，RL 降低了策略的熵，这会损害探索。 调整 RL 目标以优化拒绝采样性能是未来研究的一个有趣方向。

It is also worth highlighting the importance of carefully tuning the BC baseline for these comparisons. As discussed in Appendix E, we tuned the number of BC epochs and the sampling temperature using a combination of human evaluations and reward model score. This alone closed much of the gap we originally saw between BC and RL.

还值得强调的是，为这些比较仔细调整 BC 基线的重要性。 正如附录 E 中所讨论的，我们结合人类评估和奖励模型得分来调整 BC 时期的数量和采样温度。 仅此一项就缩小了我们最初在 BC 和 RL 之间看到的大部分差距。

### 5.2 Scaling experiments
We also conducted experiments to investigate how model performance varied with the size of the dataset, the number of model parameters, and the number of samples used for rejection sampling. Since human evaluations can be noisy and expensive, we used the score of a 175B “validation” reward model (trained on a separate dataset split) for these experiments. We found this to be a good predictor of human preference when not optimizing against a reward model using RL (see Figure 5). Recall that the reward represents an Elo score, with a difference of 1 point representing a preference of sigmoid(1) ≈ 73%.

我们还进行了实验来研究模型性能如何随数据集的大小、模型参数的数量以及用于拒绝抽样的样本数量而变化。 由于人工评估可能嘈杂且昂贵，因此我们使用 175B“验证”奖励模型（在单独的数据集拆分上训练）的分数进行这些实验。 我们发现这可以很好地预测人类在不针对使用 RL 的奖励模型进行优化时的偏好（见图5）。 回想一下，奖励代表一个 Elo 分数，1 分的差异代表sigmoid(1) ≈ 73% 的偏好。

Scaling trends with dataset size and parameter count are shown in Figures 6 and 7. For dataset size, doubling the number of demonstrations increased the policy’s reward model score by about 0.13, and doubling the number of comparisons increased the reward model’s accuracy by about 1.8%. For parameter count, the trends were noisier, but doubling the number of parameters in the policy increased its reward model score by roughly 0.09, and doubling the number of parameters in the reward model increased its accuracy by roughly 0.4%.

图6 和图7 显示了数据集大小和参数计数的缩放趋势。对于数据集大小，将演示次数加倍可使策略的奖励模型得分增加约 0.13，将比较次数增加一倍可使奖励模型的准确性增加约 1.8% . 对于参数计数，趋势更加嘈杂，但将策略中的参数数量增加一倍可将其奖励模型得分提高约 0.09，将奖励模型中的参数数量增加一倍可将其准确性提高约 0.4%。

For rejection sampling, we analyzed how to trade off the number of samples against the number of model parameters for a given inference-time compute budget (see Figure 8). We found that it is generally compute-efficient to use some amount of rejection sampling, but not too much. The models for our main evaluations come from the Pareto frontier of this trade-off: the 760M best-of-4 model, the 13B best-of-16 model, and the 175B best-of-64 model. 

对于拒绝采样，我们分析了如何针对给定的推理时间计算预算权衡样本数量与模型参数数量（参见图8）。 我们发现使用一定数量的拒绝抽样通常可以提高计算效率，但不要太多。 我们主要评估的模型来自这种权衡的帕累托边界：760M 4 种最佳模型、13B 16 种最佳模型和 175B 64 种最佳模型。

![Figure 6](../images/WebGPT/fig_6.png)<br/>
Figure 6: BC scaling, varying the proportion of the demonstration dataset and parameter count of the policy. 
图6：BC 缩放，改变演示数据集的比例和策略的参数计数。

![Figure 7](../images/WebGPT/fig_7.png)<br/>
Figure 7: RM scaling, varying the proportion of the comparison dataset and parameter count of the reward model. 
图7：RM 缩放，改变比较数据集的比例和奖励模型的参数计数。

![Figure 8](../images/WebGPT/fig_8.png)<br/>
Figure 8: Best-of-n scaling, varying the parameter count of the policy and reward model together, as well as the number of answers sampled. 
图8：Best-of-n 缩放，一起改变策略和奖励模型的参数计数，以及采样的答案数量。


## 6 Discussion
### 6.1 Truthfulness of WebGPT
As NLP systems improve and become more widely deployed, it is becoming increasingly important to develop techniques for reducing the number of false statements they make [Evans et al., 2021]. To assess the contribution of WebGPT to this aim, it is helpful to distinguish two categories of false statement made by a model:
1. Imitative falsehoods. These are false statements that are incentivized by the training objective (even in the limit of infinite data and compute), such as reproducing common misconceptions [Lin et al., 2021].
2. Non-imitative falsehoods. These are false statements that are the result of the model failing to achieve its training objective, including most hallucinations, which are statements that are false, but look plausible at a glance [Maynez et al., 2020].

随着 NLP 系统的改进和更广泛的部署，开发减少它们做出的错误陈述数量的技术变得越来越重要 [Evans et al., 2021 年]。 为了评估 WebGPT 对此目标的贡献，区分模型做出的两类虚假陈述是有帮助的：
1.模仿谎言。 这些是由训练目标（即使在无限数据和计算的限制下）激励的错误陈述，例如重现常见的误解 [Lin et al., 2021]。
2. 非模仿性谎言。 这些是错误陈述，是模型未能实现其训练目标的结果，包括大多数幻觉，这些陈述是错误的，但乍一看似乎有道理 [Maynez et al., 2020]。

Our TruthfulQA results suggest that WebGPT produces fewer imitative falsehoods than GPT-3. We believe this is because WebGPT is incentivized to prefer reliable sources (both because of filtering performed by the Bing API, and because we specify this in our instructions). Nevertheless, as shown in Table 3, WebGPT still sometimes quotes from highly unreliable sources in response to TruthfulQA questions. We hypothesize that this is because of the distribution shift from ELI5 to TruthfulQA, and that training on adversarially-selected questions is a promising way to improve this. It would be important in such an endeavor to pay close attention to labeler judgments of source trustworthiness (see Appendix C).

我们的 TruthfulQA 结果表明，与 GPT-3 相比，WebGPT 产生的模仿性虚假信息更少。 我们认为这是因为 WebGPT 被激励选择可靠的来源（既因为 Bing API 执行的过滤，也因为我们在说明中指定了这一点）。 尽管如此，如表3 所示，WebGPT 有时仍会引用高度不可靠的来源来回答 TruthfulQA 问题。 我们假设这是因为分布从 ELI5 转移到 TruthfulQA，并且对对抗性选择的问题进行训练是一种有前途的改进方法。 在这种努力中，密切关注标签商对来源可信度的判断是很重要的（见附录 C）。

Our results on ELI5 suggest that WebGPT also produces fewer non-imitative falsehoods than GPT-3. We did not test this hypothesis directly, since we found that it was challenging for labelers to spot subtle hallucinations. However, prior work shows that the use of retrieval reduces the rate of hallucinations [Shuster et al., 2021], and moreover WebGPT performs about as well as human demonstrations for factual accuracy on ELI5 (see Figure 2(a)). Nevertheless, WebGPT still sometimes produces non-imitative falsehoods, which are typically mistakes when attempting to paraphrase or synthesize information rather than wild hallucinations. 

我们在 ELI5 上的结果表明，与 GPT-3 相比，WebGPT 产生的非模仿性虚假信息也更少。 我们没有直接检验这个假设，因为我们发现标记者很难发现细微的幻觉。 然而，先前的工作表明，检索的使用降低了幻觉的发生率 [Shuster et al., 2021]，而且 WebGPT 在 ELI5 上的事实准确性与人类演示的表现大致相同（见图2(a)）。 尽管如此，WebGPT 有时仍会产生非模仿性的错误信息，这通常是在试图解释或合成信息时出现的错误，而不是疯狂的幻觉。

![Table 3](../images/WebGPT/tab_3.png)<br/>
Table 3: Two questions from TruthfulQA, cherry-picked to highlight a success and a failure of WebGPT. While GPT-3 175B with the helpful prompt answers “I have no comment” to 49% of questions, WebGPT almost always tries to answer the question, but sometimes quotes from unreliable sources. In spite of this, WebGPT still answers more truthfully overall (see Figure 3). Key: ✗ = false, ✓ = true but uninformative, ✓ = true and informative
表3：来自 TruthfulQA 的两个问题，经过精心挑选以突出 WebGPT 的成功和失败。 虽然 GPT-3 175B 的有用提示对 49% 的问题回答“我无可奉告”，但 WebGPT 几乎总是试图回答这个问题，但有时会引用不可靠的来源。 尽管如此，WebGPT 总体上还是回答得比较真实（见图3）。 关键：✗ = 错误，✓ = 正确但没有提供信息，✓ = 正确且提供信息

### 6.2 Perceived truthfulness of WebGPT  感知真实性
In order to assess the benefits and risks of WebGPT, it is necessary to consider not only how often it makes false statements, but also how likely users are to rely on those statements. Although WebGPT makes false statements less frequently than GPT-3, its answers also appear more authoritative, partly because of the use of citations. In combination with the well-documented problem of “automation bias” [Goddard et al., 2012], this could lead to overreliance on WebGPT’s answers. This is particularly problematic because, as discussed in Section 6.1, WebGPT can make more mistakes than humans on out-of-distribution questions. Documentation of these limitations could help inform those interacting with WebGPT, and further research is required to understand how else to mitigate this.

为了评估 WebGPT 的好处和风险，不仅要考虑它做出虚假陈述的频率，还要考虑用户依赖这些陈述的可能性有多大。 尽管 WebGPT 做出错误陈述的频率低于 GPT-3，但它的答案也显得更权威，部分原因是引用的使用。 结合有据可查的“自动化偏见”问题 [Goddard et al., 2012 年]，这可能会导致过度依赖 WebGPT 的答案。 这尤其成问题，因为如第 6.1 节所述，WebGPT 在分布外问题上犯的错误比人类多。 这些限制的文档可以帮助告知那些与 WebGPT 交互的人，并且需要进一步的研究来了解如何缓解这种情况。

### 6.3 Reinforcement of bias 偏差强化
There are a number of ways in which WebGPT tends to perpetuate and reinforce existing assumptions and biases. Firstly, WebGPT inherits the biases of the base model from which it is fine tuned, GPT-3 [Brown et al., 2020], and this influences the way in which it chooses to search for and synthesize information. Search and synthesis both depend on the ability to include and exclude material depending on some measure of its value, and by incorporating GPT-3’s biases when making these decisions, WebGPT can be expected to perpetuate them further. Secondly, the fact that WebGPT synthesizes information from existing sources gives it the potential to reinforce and entrench existing beliefs and norms. Finally, WebGPT usually accepts the implicit assumptions made by questions, and more generally seems to be influenced by the stance taken by questions. This is something that could exacerbate confirmation bias in users.

WebGPT 倾向于通过多种方式延续和强化现有的假设和偏见。 首先，WebGPT 继承了对其进行微调的基本模型 GPT-3 [Brown et al., 2020] 的偏差，这影响了它选择搜索和合成信息的方式。 搜索和综合都取决于根据其价值的某种衡量标准来包含和排除材料的能力，并且通过在做出这些决定时结合 GPT-3 的偏见，WebGPT 有望进一步延续它们。 其次，WebGPT 从现有来源综合信息这一事实使其有可能加强和巩固现有的信念和规范。 最后，WebGPT 通常接受问题做出的隐含假设，并且更普遍地似乎受到问题采取的立场的影响。 这可能会加剧用户的确认偏差。

These problems could be mitigated with improvements both to WebGPT’s base model and to WebGPT’s training objective, and we discuss some alternative objectives in the next section. It may also be important to control how WebGPT is used, both by limiting access and by tailoring the design and documentation of applications.

这些问题可以通过改进 WebGPT 的基本模型和 WebGPT 的训练目标来缓解，我们将在下一节讨论一些替代目标。 通过限制访问和定制应用程序的设计和文档来控制 WebGPT 的使用方式可能也很重要。

Additional analysis of the effect of question stance and of reference point bias is given in Appendix H. 

附录 H 给出了对问题立场和参考点偏差影响的额外分析。

### 6.4 Using references to evaluate factual accuracy 使用参考文献评估事实准确性
Central to our approach is the use of references collected by the model to aid human evaluation of factual accuracy. This was previously suggested by Metzler et al. [2021], and has several benefits: 
* More accurate feedback. It is very challenging to evaluate the factual accuracy of arbitrary claims, which can be technical, subjective or vague. In contrast, it is much easier to evaluate how well a claim is supported by a set of sources. 
* Less noisy feedback. It is also easier to specify an unambiguous procedure for evaluating how well a claim is supported by a set of sources, compared to evaluating the factual accuracy of an arbitrary claim. This improves agreement rates between labelers, which helps data efficiency. 
* Transparency. It is much easier to understand how WebGPT composes answers than it is for GPT-3, since the entire browsing process can be inspected. It is also straightforward for end-users to follow up on sources to better judge factual accuracy for themselves.

我们方法的核心是使用模型收集的参考来帮助人类评估事实的准确性。 这是 Metzler 等人先前提出的。 [2021]，并有几个好处：
* 更准确的反馈。 评估任意声明的事实准确性非常具有挑战性，这些声明可能是技术性的、主观的或模糊的。 相比之下，评估一组来源对声明的支持程度要容易得多。
* 减少嘈杂的反馈。 与评估任意声明的事实准确性相比，指定一个明确的程序来评估一组来源对声明的支持程度也更容易。 这提高了标记者之间的协议率，从而有助于提高数据效率。
* 透明度。 与 GPT-3 相比，了解 WebGPT 如何撰写答案要容易得多，因为可以检查整个浏览过程。 最终用户也可以直接跟进来源以更好地自己判断事实的准确性。

Despite these benefits, references are far from a panacea. Our current procedure incentivizes models to cherry-pick references that they expect labelers to find convincing, even if those references do not reflect a fair assessment of the evidence. As discussed in Section 6.3, there are early signs of this happening, with WebGPT accepting the implicit assumptions of questions, and the problem is likely to be exacerbated by more capable models and more challenging or subjective questions. We could mitigate this using methods like debate [Irving et al., 2018], in which models are trained to find evidence both for and against different claims. Such setups can also be viewed as simple cases of recursive reward modeling [Leike et al., 2018] and Iterated Amplification [Christiano et al., 2018], in which the model assists its own evaluation.

尽管有这些好处，参考文献远非灵丹妙药。 我们目前的程序会激励模型挑选他们希望标签商认为有说服力的参考资料，即使这些参考资料没有反映对证据的公平评估。 正如第 6.3 节中所讨论的，有这种情况发生的早期迹象，WebGPT 接受了问题的隐含假设，并且更强大的模型和更具挑战性或主观性的问题可能会加剧该问题。 我们可以使用辩论 [Irving et al., 2018] 等方法来缓解这种情况，在这些方法中，训练模型以寻找支持和反对不同主张的证据。 此类设置也可以视为递归奖励建模 [Leike et al., 2018 年] 和迭代放大 [Christiano et al., 2018 年] 的简单案例，其中模型协助其自身评估。

Our approach also raises a challenging problem with societal implications: how should factual accuracy be evaluated when training AI systems? Evans et al. [2021, Section 2] propose a number of desiderata, but a substantial gap remains between these and the highly specific criteria needed to train current AI systems with reasonable data efficiency. We made a number of difficult judgment calls, such as how to rate the trustworthiness of sources (see Appendix C), which we do not expect universal agreement with. While WebGPT did not seem to take on much of this nuance, we expect these decisions to become increasingly important as AI systems improve, and think that cross-disciplinary research is needed to develop criteria that are both practical and epistemically sound.

我们的方法还提出了一个具有社会影响的挑战性问题：在训练 AI 系统时应如何评估事实准确性？ 埃文斯等人。 [2021，第 2 节] 提出了一些迫切需要，但这些与以合理的数据效率训练当前 AI 系统所需的高度具体的标准之间仍然存在很大差距。 我们做出了一些困难的判断，例如如何评价来源的可信度（见附录 C），我们不希望得到普遍同意。 虽然 WebGPT 似乎并没有体现出太多这种细微差别，但我们预计随着 AI 系统的改进，这些决定将变得越来越重要，并且认为需要跨学科研究来制定既实用又合理的标准。

### 6.5 Risks of live web access
At both train and inference time, WebGPT has live access to the web via our text-based browsing environment. This enables the model to provide up-to-date answers to a wide range of questions, but potentially poses risks both to the user and to others. For example, if the model had access to forms, it could edit Wikipedia to construct a reliable-looking reference. Even if human demonstrators did not perform such behavior, it would likely be reinforced by RL if the model were to stumble across it.

在训练和推理时间，WebGPT 都可以通过我们基于文本的浏览环境实时访问网络。 这使模型能够为范围广泛的问题提供最新的答案，但可能会给用户和其他人带来风险。 例如，如果模型可以访问表单，它可以编辑维基百科以构建看起来可靠的参考。 即使人类示威者没有执行此类行为，如果模型偶然发现这种行为，强化学习也可能会加强这种行为。

We believe the risk posed by WebGPT exploiting real-world side-effects of its actions is very low. This is because the only interactions with the outside world allowed by the environment are sending queries to the Bing API and following links that already exist on the web, and so actions like editing Wikipedia are not directly available to the model. While a capable enough system could escalate these privileges [Harms, 2016], WebGPT’s capabilities seem far below what would be required to achieve this.

我们认为 WebGPT 利用其行为在现实世界中产生的副作用所带来的风险非常低。 这是因为环境允许的与外部世界的唯一交互是向 Bing API 发送查询并跟踪 Web 上已经存在的链接，因此编辑维基百科等操作不能直接用于模型。 虽然一个足够强大的系统可以提升这些权限 [Harms，2016]，但 WebGPT 的功能似乎远低于实现这一目标所需的能力。

Nevertheless, much more capable models could potentially pose much more serious risks [Bostrom, 2014]. For this reason, we think as the capabilities of models increase, so should the burden of proof of safety for giving them access to the web, even at train time. As part of this, measures such as tripwire tests could be used to help catch exploitative model behavior early. 

然而，功能更强大的模型可能会带来更严重的风险 [Bostrom，2014 年]。 出于这个原因，我们认为随着模型功能的增加，即使在训练时也应该让模型访问网络的安全证明负担也随之增加。 作为其中的一部分，可以使用诸如绊网测试之类的措施来帮助及早发现利用模型的行为。

## 7 Related work
Combining machine learning with an external knowledge base, for the task of question-answering, preceded the rise of pre-trained language models in the late 2010s. One notable system of this kind was DeepQA (also known as IBM Watson), which was used to beat the best humans at Jeopardy [Ferrucci et al., 2010]. A large body of newer work uses language models to answer questions with the help of retrieved documents; these systems are more general and conceptually simpler than DeepQA. One approach is to use inner product search to retrieve relevant documents and then generate an answer given these documents: 

将机器学习与外部知识库相结合，以完成问答任务，这早于 2010 年代末预训练语言模型的兴起。 这种类型的一个著名系统是 DeepQA（也称为 IBM Watson），它被用来在 Jeopardy [Ferrucci et al., 2010] 中击败最优秀的人类。 大量较新的工作使用语言模型在检索到的文档的帮助下回答问题； 这些系统比 DeepQA 更通用，概念上更简单。 一种方法是使用内积搜索来检索相关文档，然后根据这些文档生成答案：

p(passage∣query) ∝ exp(embed(passage) ⋅ embed(query)). (1)

Given a training dataset that specifies relevant passages for each question, dense passage retrieval (DPR) trains the retriever directly using a contrastive objective [Karpukhin et al., 2020]. Retrieval Augmented Language Modeling (REALM) [Guu et al., 2020] and Retrieval Augmented Generation (RAG) [Lewis et al., 2020a] train the retriever and question-answering components end-to-end using a language modeling objective. Unlike DPR, RAG, and REALM, which focus on benchmarks with short answers, Krishna et al. [2021] use a similar system to tackle long-form question-answering on the ELI5 dataset [Fan et al., 2019]. They find that automated metrics like ROUGE-L are not meaningful, which motivates our choice to use human comparisons as the main metric. Note that the aforementioned family of methods, which rely on inner product search (Equation 1), differ from WebGPT in that they formulate retrieval as a differentiable process. Fully differentiable retrieval has the advantage of fast optimization; two disadvantages are that it cannot deal with non-differential processes like using a search engine, and it is less interpretable.

给定一个为每个问题指定相关段落的训练数据集，密集段落检索 (DPR) 使用对比目标直接训练检索器 [Karpukhin et al., 2020]。 检索增强语言建模 (REALM) [Guu et al., 2020] 和检索增强生成 (RAG) [Lewis et al., 2020a] 使用语言建模目标端到端地训练检索器和问答组件。 与 DPR、RAG 和 REALM 不同，它们专注于具有简短答案的基准，Krishna 等人。 [2021] 使用类似的系统来解决 ELI5 数据集上的长格式问答问题 [Fan et al., 2019]。 他们发现像 ROUGE-L 这样的自动化指标没有意义，这促使我们选择使用人类比较作为主要指标。 请注意，上述依赖于内积搜索（等式 1）的方法系列与 WebGPT 的不同之处在于，它们将检索制定为可微分过程。 全可微检索具有快速优化的优势； 两个缺点是它不能像使用搜索引擎那样处理非差异过程，而且它的可解释性较差。

Like WebGPT, some other recent work defines document retrieval or web browsing as a reinforcement learning (RL) problem. Yuan et al. [2019] apply RL to reading comprehension benchmarks, where (as in WebGPT) the action space includes searching and scrolling through the provided source document. They suggest web-level QA (like WebGPT) as a direction for future work. Adolphs et al. [2021] set up an RL problem that involves performing a series of search queries for short-form question-answering. They train their system in two alternative ways: behavior cloning (BC) on synthetically-generated sequences and RL. Finally, there is another body of work that uses BC and RL to control web browsers, for automating other tasks besides question-answering [Shi et al., 2017, Gur et al., 2018]. 

与 WebGPT 一样，最近的一些其他工作将文档检索或 Web 浏览定义为强化学习 (RL) 问题。 元等。 [2019] 将 RL 应用于阅读理解基准测试，其中（如在 WebGPT 中）动作空间包括搜索和滚动提供的源文档。 他们建议将网络级 QA（如 WebGPT）作为未来工作的方向。 阿道夫等人。 [2021] 设置了一个 RL 问题，涉及执行一系列搜索查询以进行简短的问答。 他们以两种替代方式训练他们的系统：对合成生成的序列进行行为克隆 (BC) 和强化学习。 最后，还有另一项工作使用 BC 和 RL 来控制网络浏览器，用于自动化除问答之外的其他任务 [Shi et al., 2017, Gur et al., 2018]。

## 8 Conclusion
We have demonstrated a novel approach to long-form question-answering, in which a language model is fine-tuned to use a text-based web-browsing environment. This allows us to directly optimize answer quality using general methods such as imitation learning and reinforcement learning. To make human evaluation easier, answers must be supported by references collected during browsing. Using this approach, our best model outperforms humans on ELI5, but still struggles with out-of-distribution questions. 

我们展示了一种新颖的长篇问答方法，其中语言模型经过微调以使用基于文本的网络浏览环境。 这使我们能够使用模仿学习和强化学习等通用方法直接优化答案质量。 为了使人工评估更容易，答案必须得到浏览期间收集的参考资料的支持。 使用这种方法，我们最好的模型在 ELI5 上的表现优于人类，但仍然难以解决分布外的问题。

## 9 Author contributions
Reiichiro Nakano, Jacob Hilton, Suchir Balaji and John Schulman jointly led the project, developed the codebase, ran all data collection and experiments, and wrote the paper.

Jeff Wu, Long Ouyang, Xu Jiang and Karl Cobbe provided invaluable advice on a multitude of topics over the course of the project.

Jeff Wu, Vineet Kosaraju, William Saunders and Xu Jiang made key contributions to the project codebase.

Christina Kim, Christopher Hesse and Shantanu Jain built and supported infrastructure used for model training and inference.

Tyna Eloundou and Gretchen Krueger conducted the analysis of bias and contributed to the paper.

Kevin Button and Matthew Knight provided computer security support.

Benjamin Chess provided computer networking support. 

## 10 Acknowledgments
We would like to thank Leo Gao, Hyeonwoo Noh and Chelsea Voss for working on future directions;

Steve Dowling, Christian Gibson, Peter Hoeschele, Fraser Kelton, Bianca Martin, Bob McGrew, 12

Felipe Such and Hannah Wong for technical, logistical and communications support; Steven Adler,

Miles Brundage, David Farhi, William Guss, Oleg Klimov, Jan Leike, Ryan Lowe, Diogo Moitinho de

Almeida, Arvind Neelakantan, Alex Ray, Nick Ryder and Andreas Stuhlmüller for helpful discussions;

Owen Cotton-Barratt, Owain Evans, Jared Kaplan, Girish Sastry, Carl Shulman, Denis Yarats and

Daniel Ziegler for helpful discussions and feedback on drafts; Beth Barnes and Paul Christiano for helpful discussions and feedback on drafts, and in particular for suggesting the project; and Dario

Amodei for suggesting to work on factual inaccuracy in language models. We would also like to thank Surge AI for helping us with data collection, in particular Edwin Chen, Andrew Mauboussin,

Craig Pettit and Bradley Webb.

Finally, we would like to thank all of our contractors for providing demonstrations and comparisons, without which this project would not have been possible, including: Jamie Alexander, Andre Gooden,

Jacquelyn Johns, Rebecca Kientz, Ashley Michalski, Amy Dieu-Am Ngo, Alex Santiago, Alice

Sorel, Sam Thornton and Kelli W. from Upwork; and Elena Amaya, Michael Baggiano, Carlo Basile,

Katherine Beyer, Erica Dachinger, Joshua Drozd, Samuel Ernst, Rodney Khumalo, Andrew Kubai,

Carissa Lewis, Harry Mubvuma, William Osborne, Brandon P., Kimberly Quinn, Jonathan Roque,

Jensen Michael Ruud, Judie Anne Sigdel, Bora Son, JoAnn Stone, Rachel Tanks, Windy Thomas,

Laura Trivett, Katherine Vazquez, Brandy and Shannon from Surge AI.

## References
* L. Adolphs, B. Boerschinger, C. Buck, M. C. Huebscher, M. Ciaramita, L. Espeholt, T. Hofmann,and Y. Kilcher. Boosting search engines with interactive agents. arXiv preprint arXiv:2109.00527,2021.
* S. Bhakthavatsalam, D. Khashabi, T. Khot, B. D. Mishra, K. Richardson, A. Sabharwal, C. Schoenick,O. Tafjord, and P. Clark. Think you have solved direct-answer question answering? Try ARC-DA,the direct-answer AI2 reasoning challenge. arXiv preprint arXiv:2102.03315, 2021.
* N. Bostrom. Superintelligence: Paths, Dangers, Strategies. Oxford University Press, 2014.
* T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan,P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. arXiv preprintarXiv:2005.14165, 2020.
* H. Cheng, Y. Shen, X. Liu, P. He, W. Chen, and J. Gao. UnitedQA: A hybrid approach for opendomain question answering. arXiv preprint arXiv:2101.00178, 2021.
* D. Chong and J. N. Druckman. Framing theory. Annu. Rev. Polit. Sci., 10:103–126, 2007.
* P. Christiano, B. Shlegeris, and D. Amodei. Supervising strong learners by amplifying weak experts. arXiv preprint arXiv:1810.08575, 2018.
* O. Evans, O. Cotton-Barratt, L. Finnveden, A. Bales, A. Balwit, P. Wills, L. Righetti, and W. Saunders. Truthful AI: Developing and governing AI that does not lie. arXiv preprint arXiv:2110.06674,2021.
* A. Fan, Y. Jernite, E. Perez, D. Grangier, J. Weston, and M. Auli. ELI5: Long form questionanswering. arXiv preprint arXiv:1907.09190, 2019.
* D. Ferrucci, E. Brown, J. Chu-Carroll, J. Fan, D. Gondek, A. A. Kalyanpur, A. Lally, J. W. Murdock,E. Nyberg, J. Prager, et al. Building watson: An overview of the deepqa project. AI magazine, 31(3):59–79, 2010.
* K. Goddard, A. Roudsari, and J. C. Wyatt. Automation bias: a systematic review of frequency,effect mediators, and mitigators. Journal of the American Medical Informatics Association, 19(1):121–127, 2012.
* I. Gur, U. Rueckert, A. Faust, and D. Hakkani-Tur. Learning to navigate the web. arXiv preprintarXiv:1812.09195, 2018.
* K. Guu, K. Lee, Z. Tung, P. Pasupat, and M.-W. Chang. REALM: Retrieval-augmented languagemodel pre-training. arXiv preprint arXiv:2002.08909, 2020.
* M. Harms. Crystal Society. Crystal Trilogy. CreateSpace Independent Publishing Platform, 2016. ISBN 9781530773718.
* G. Irving, P. Christiano, and D. Amodei. AI safety via debate. arXiv preprint arXiv:1805.00899,2018.
* M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer. TriviaQA: A large scale distantly supervisedchallenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.
* V. Karpukhin, B. O˘guz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih. Dense passageretrieval for open-domain question answering. arXiv preprint arXiv:2004.04906, 2020.
* K. Krishna, A. Roy, and M. Iyyer. Hurdles to progress in long-form question answering. arXivpreprint arXiv:2103.06332, 2021.
* J. Leike, D. Krueger, T. Everitt, M. Martic, V. Maini, and S. Legg. Scalable agent alignment viareward modeling: a research direction. arXiv preprint arXiv:1811.07871, 2018.
* P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih,T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. arXivpreprint arXiv:2005.11401, 2020a.
* P. Lewis, P. Stenetorp, and S. Riedel. Question and answer test-train overlap in open-domain questionanswering datasets. arXiv preprint arXiv:2008.02637, 2020b.
* S. Lin, J. Hilton, and O. Evans. TruthfulQA: Measuring how models mimic human falsehoods. arXivpreprint arXiv:2109.07958, 2021.
* J. Maynez, S. Narayan, B. Bohnet, and R. McDonald. On faithfulness and factuality in abstractivesummarization. arXiv preprint arXiv:2005.00661, 2020.
* D. Metzler, Y. Tay, D. Bahri, and M. Najork. Rethinking search: Making experts out of dilettantes. arXiv preprint arXiv:2105.02274, 2021.
* B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAMjournal on control and optimization, 30(4):838–855, 1992.
* J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimizationalgorithms. arXiv preprint arXiv:1707.06347, 2017.
* T. Shi, A. Karpathy, L. Fan, J. Hernandez, and P. Liang. World of bits: An open-domain platform forweb-based agents. In International Conference on Machine Learning, pages 3135–3144. PMLR,2017.
* K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston. Retrieval augmentation reduces hallucinationin conversation. arXiv preprint arXiv:2104.07567, 2021.
* N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, andP. Christiano. Learning to summarize from human feedback. arXiv preprint arXiv:2009.01325,2020.
* X. Yuan, J. Fu, M.-A. Cote, Y. Tay, C. Pal, and A. Trischler. Interactive machine comprehension withinformation seeking agents. arXiv preprint arXiv:1908.10449, 2019.

## A Environment design details
Our text-based web-browsing environment is written mostly in Python with some JavaScript. For a high-level overview, see Section 2. Further details are as follows: 
* When a search is performed, we send the query to the Microsoft Bing Web Search API, and convert this to a simplified web page of results. 
* When a link to a new page is clicked, we call a Node.js script that fetches the HTML of the web page and simplifies it using Mozilla’s Readability.js. • We remove any search results or links to reddit.com or quora.com, to prevent the model copying answers from those sites. 
* We take the simplified HTML and convert links to the special format 【<link ID>†<link text>†<destination domain>】, or 【<link ID>†<link text>】 if the destination and source domains are the same. Here, the link ID is the index of the link on the page, which is also used for the link-clicking command. We use special characters such as 【 and 】 because they are rare and encoded in the same few ways by the tokenizer, and if they appear in the page text then we replace them by similar alternatives. 
* We convert superscripts and subscripts to text using ^ and _, and convert images to the special format [Image: <alt text>], or [Image] if there is no alt text. 
* We convert the remaining HTML to text using html2text. 
* For text-based content types other than HTML, we use the raw text. For PDFs, we convert them to text using pdfminer.six. For all other content types, and for errors and timeouts, we use an error message. 
* We censor any pages that contain a 10-gram overlap with the question (or reference answer, if provided) to prevent the model from cheating, and use an error message instead. 
* We convert the title of the page to text using the format <page title> (<page domain>). For search results pages, we use Search results for: <query>. • When a find in page or quote action is performed, we compare the text from the command against the page text with any links stripped (i.e., including only the text from each link). We also ignore case. For quoting, we also ignore whitespace, and allow the abbreviated format <start text>━<end text> to save tokens. 
* During browsing, the state of the browser is converted to text as shown in Figure 1(b). For the answering phase (the last step of the episode), we convert the question to text using the format <question>■, and follow this by each of the collected quotes in the format [<quote number>] <quote page title> (<quote page domain>) <double new line><quote extract>■. 

我们基于文本的网络浏览环境主要是用 Python 和一些 JavaScript 编写的。 有关高级概述，请参阅第 2 节。更多详情如下：
* 执行搜索时，我们将查询发送到 Microsoft Bing Web Search API，并将其转换为简化的网页结果。
* 单击指向新页面的链接时，我们会调用一个 Node.js 脚本来获取网页的 HTML 并使用 Mozilla 的 Readability.js 对其进行简化。 • 我们删除所有搜索结果或指向 reddit.com 或 quora.com 的链接，以防止模型从这些网站复制答案。
* 我们采用简化的 HTML 并将链接转换为特殊格式【<link ID>†<link text>†<destination domain>】，或【<link ID>†<link text>】，如果目标域和源域是 相同的。 这里的链接ID是链接在页面上的索引，也用于链接点击命令。 我们使用特殊字符，如【和】，因为它们很少见，并且分词器以相同的几种方式编码，如果它们出现在页面文本中，我们就会用类似的替代品替换它们。
* 我们使用 ^ 和 _ 将上标和下标转换为文本，并将图像转换为特殊格式 [Image: <alt text>]，如果没有 alt 文本，则转换为 [Image]。
* 我们使用 html2text 将剩余的 HTML 转换为文本。
* 对于 HTML 以外的基于文本的内容类型，我们使用原始文本。 对于 PDF，我们使用 pdfminer.six 将它们转换为文本。 对于所有其他内容类型，以及错误和超时，我们使用错误消息。
* 我们会审查任何包含与问题（或参考答案，如果提供）重叠 10 克的页面，以防止模型作弊，并改用错误消息。
* 我们使用 <page title> (<page domain>) 格式将页面标题转换为文本。 对于搜索结果页面，我们使用 Search results for: <query>。 • 当执行“在页面中查找”或“引用”操作时，我们将命令中的文本与删除任何链接的页面文本进行比较（即，仅包括来自每个链接的文本）。 我们也忽略大小写。 对于引用，我们也忽略空格，并允许缩写格式 <start text>━ <end text> 来保存标记。
* 在浏览过程中，浏览器的状态被转换为文本，如图1(b) 所示。 对于回答阶段（剧集的最后一步），我们使用格式 <question>■ 将问题转换为文本，然后以 [<quote number>] <quote page title> 格式跟随每个收集到的引述 (<quote page domain>) <double new line><quote extract>■. 


## B Question dataset details
For our demonstration and comparison datasets, the vast majority of questions were taken from ELI5 [Fan et al., 2019], to which we applied the follow post-processing:
1. We included URLs in full, rather than using special _URL_ tokens.
2. We filtered out questions with the title “[deleted by user]”, and ignored the selftext “[deleted]” and “[removed]”. (The “selftext” is the body of the post.)
3. We concatenated the title and any non-empty selftext, separated by a double new line.
4. We prepended “Explain: ” to questions that were not phrased as actual questions (e.g., we used “Explain: gravity” rather than simply “gravity”).

对于我们的演示和比较数据集，绝大多数问题来自 ELI5 [Fan et al., 2019]，我们对其应用了以下后处理：
1. 我们包含了完整的 URL，而不是使用特殊的 _URL_ 标记。
2. 我们过滤掉了标题为“[deleted by user]”的问题，忽略了selftext“[deleted]”和“[removed]”。 （“selftext”是帖子的正文。）
3. 我们将标题和任何非空的自文本连接起来，用双换行符分隔。
4. 我们在未表述为实际问题的问题前加上“解释：”（例如，我们使用“解释：引力”而不是简单的“引力”）。

The final step was performed because there is sometimes an implicit “Explain Like I’m Five” at the start of questions. We considered a question to be phrased as an actual question if it included either a question mark, or one of the following sequences of characters with a regex-word boundary at either end, case-insensitively: 

执行最后一步是因为有时在问题开始时会隐含“像我五岁一样解释”。 如果一个问题包含问号，或以下字符序列之一，两端带有正则表达式单词边界（不区分大小写），我们认为该问题被表述为实际问题：

explain, eli5, which, what, whats, whose, who, whos, whom, where, wheres, when, whens, how, hows, why, whys, am, is, isn, isnt, are, aren, arent, was, wasn, wasnt, were, weren, werent, do, don, dont, does, doesn, doesnt, did, didn, didnt, can, cant, could, couldn, couldnt, have, haven, havent, has, hasn, hasnt, may, might, must, mustn, mustnt, shall, shant, should, shouldn, shouldnt, will, wont, would, wouldn, wouldnt

For diversity and experimentation, we also mixed in a small number of questions from the following datasets: 
* TriviaQA. This is a dataset of short-form questions taken from trivia websites [Joshi et al., 2017]. 
* AI2 Reasoning Challenge (ARC). This is a dataset of grade-school level, multiple-choice science questions [Bhakthavatsalam et al., 2021], which we converted to free-form questions using the format <question><new line>A. <option A><new line>.... This dataset is sub-divided into two difficulties, “Challenge” and “Easy”. 
* Hand-written. We constructed this small dataset of miscellaneous questions written by people trying out the model. 
* ELI5 fact-check. We constructed this dataset using answers to questions from ELI5 given by an instruction-following model(https://beta.openai.com/docs/engines/instruct-series-beta ). Each question has the following format: Fact-check each of the claims in the following answer. <double new line>Question: <ELI5 question><double new line>Answer: <model answer>

为了多样性和实验性，我们还混合了以下数据集中的少量问题：
* 琐事问答。 这是从琐事网站 [Joshi et al., 2017] 获取的简短问题的数据集。
* AI2 推理挑战 (ARC)。 这是小学水平的多项选择科学问题的数据集 [Bhakthavatsalam et al., 2021]，我们使用 <question><new line>A 格式将其转换为自由形式的问题。 <option A><new line>.... 这个数据集被细分为两个难度，“Challenge”和“Easy”。
* 手写。 我们构建了这个由试用该模型的人编写的各种问题组成的小型数据集。
* ELI5 事实核查。 我们使用指令跟随模型 (https://beta.openai.com/docs/engines/instruct-series-beta) 给出的 ELI5 问题的答案构建了这个数据集。 每个问题都有以下格式：事实检查以下答案中的每个声明。 <double new line>问题：<ELI5 question><double new line>答案：<model answer>

The numbers of demonstrations and comparisons we collected for each of these datasets are given in Table 4.

表4 给出了我们为每个数据集收集的演示和比较的数量。

![Table 4](../images/WebGPT/tab_4.png)<br/>
Table 4: Breakdown of our demonstrations and comparisons by question dataset.
表4：按问题数据集对我们的演示和比较进行细分。

## C Data collection details
To collect demonstrations and comparisons, we began by hiring freelance contractors from Upwork (https://www.upwork.com), and then worked with Surge AI (https://www.surgehq.ai) to scale up our data collection. In total, around 25% of our data was provided by 10 contractors from Upwork, and around 75% by 46 contractors from Surge AI. The top 5 contractors provided around 50% of the data.

为了收集演示和比较，我们首先从 Upwork (https://www.upwork.com) 聘请自由承包商，然后与 Surge AI (https://www.surgehq.ai) 合作扩大我们的数据收集。 总共约 25% 的数据由 Upwork 的 10 个承包商提供，约 75% 的数据由 Surge AI 的 46 个承包商提供。 前 5 名承包商提供了大约 50% 的数据。

For both types of task, we provided contractors with a video and a detailed instruction document (linked below). Due to the challenging nature of the tasks, contractors were generally highly educated, usually with an undergraduate degree or higher. Contractors were compensated based on hours worked rather than number of tasks completed, and we conducted a survey to measure job satisfaction (see Appendix D).

对于这两种类型的任务，我们都为承包商提供了视频和详细的说明文件（链接如下）。 由于任务具有挑战性，承包商通常受过高等教育，通常拥有本科或更高学历。 承包商根据工作时间而不是完成的任务数量获得报酬，我们进行了一项调查来衡量工作满意度（见附录 D）。

For data quality, we put prospective contractors through a paid trial period lasting a few hours, and manually checked their work. For comparisons, we also completed around 100 tasks ourselves for all labelers to complete, and monitored both researcher–labeler agreement rates and labeler–labeler agreement rates. Treating the agreement rate between a neutral label and a non-neutral label as 50%, we measured a final researcher-labeler agreement rate of 74%, and a labeler-labeler agreement rate of 73%.

对于数据质量，我们让潜在承包商经历持续几个小时的付费试用期，并手动检查他们的工作。 为了进行比较，我们还自己完成了大约 100 项任务，供所有标注人员完成，并监测了研究人员与标注人员的一致率和标注人员与标注人员的一致率。 将中性标签和非中性标签之间的一致率视为 50%，我们测得最终研究人员-标注人员的一致率为 74%，标注人员-标注人员的一致率为 73%。

Demonstrations took an average of around 15 minutes each, and comparisons took an average of around 10 minutes each. Despite conventional wisdom that human labelling tasks should be quick and repeatable, we did not think it would be straightforward to decompose our tasks into significantly simpler ones, but we consider this to be a promising direction for further research.

每个演示平均花费大约 15 分钟，每个比较平均花费大约 10 分钟。 尽管传统观点认为人工标记任务应该快速且可重复，但我们认为将我们的任务分解为简单得多的任务并不是一件容易的事，但我们认为这是进一步研究的一个有希望的方向。

### C.1 Demonstrations
We designed the demonstration interface in such a way that, as a rule, the user is given the same information as the model, and has the same actions available. There were a couple of exceptions to this:
1. Unlike humans, the model has no memory of previous steps. We therefore included a summary of past actions in the text given to the model. However, we felt that it was unnecessary to display this to humans.
2. The Scrolled <up, down> <2, 3> actions are useful for reducing the number of actions taken, but humans are used to scrolling one step at a time. We therefore made these actions unavailable to humans, and instead simply merged any repeated Scrolled <up, down> 1 actions that they made.

我们以这样一种方式设计演示界面，即通常向用户提供与模型相同的信息，并提供相同的可用操作。 有几个例外：
1. 与人类不同，模型对之前的步骤没有记忆。 因此，我们在提供给模型的文本中包含了对过去行为的总结。 但是，我们觉得没有必要向人类展示它。
2. Scrolled <up, down> <2, 3> 动作对于减少采取的动作数量很有用，但人类习惯于一次滚动一步。 因此，我们让这些动作对人类不可用，而是简单地合并他们所做的任何重复的 Scrolled <up, down> 1 动作。

The full instruction document we provided to contractors for demonstrations can be viewed https://docs.google.com/document/d/1dqfhj1W8P0JhwMKD5lWbhppY9JDFfm7tCwZudolmpzg/edit.

我们提供给承包商进行演示的完整说明文件可以查看 ...

### C.2 Comparisons
To minimize label noise, it is important to make comparisons as unambiguous as possible. We therefore designed the following procedure for comparing two answers to a given question:
1. Read the question, and flag if it does not make sense or should not be answered (in which case the rest of the comparison is skipped).
2. Read the first answer and its references.
3. Rate the trustworthiness of any references relied upon by the answer.
4. Annotate each of the claims in the answer with the level of support it has and its relevance to the question. A screenshot of the annotation tool is shown in Figure 9.
5. Repeat steps 2–4 for the second answer and its references.
6. Give comparison ratings for the amount of unsupported and irrelevant information, the usefulness of information with different levels of support, and coherence.
7. Weighing everything up, give a final comparison rating for overall usefulness.

为了尽量减少标签噪声，尽可能明确地进行比较很重要。 因此，我们设计了以下程序来比较给定问题的两个答案：
1. 阅读问题，如果没有意义或不应回答则标记（在这种情况下跳过其余的比较）。
2. 阅读第一个答案及其参考资料。
3. 评价答案所依赖的任何参考资料的可信度。
4. 对答案中的每项主张进行注释，说明其支持程度及其与问题的相关性。 注释工具的屏幕截图如图9 所示。
5. 对第二个答案及其参考文献重复步骤 2–4。
6. 对不支持和不相关信息的数量、不同支持程度的信息的有用性以及连贯性给出比较评级。
7. 权衡一切，对整体实用性给出最终比较评级。

![Figure 9](../images/WebGPT/fig_9.png)<br/>
Figure 9: Screenshot from the comparison interface, showing the annotation tool.
图9：比较界面的屏幕截图，显示了注释工具。

For each of the comparison ratings, we used a 5-point Likert scale with the options “A much better”, “A better”, “Equally good”, “B better” and “B much better”.

对于每个比较评级，我们使用 5 点李克特量表，选项为“A 更好”、“A 更好”、“同样好”、“B 更好”和“B 更好”。

Importantly, we did not require contractors to perform independent research to judge the factual accuracy of answers, since this would have been difficult and subjective. Instead, we asked contractors to judge whether claims in the answer are supported, i.e., either backed up by a reliable reference, or common knowledge.

重要的是，我们不要求承包商进行独立研究来判断答案的事实准确性，因为这既困难又主观。 相反，我们要求承包商判断答案中的主张是否得到支持，即是否得到可靠参考或常识的支持。

For the final comparison rating, we encouraged contractors to use their best judgment, but to roughly consider the following criteria in descending order of priority: 
* Whether or not the answer contains unsupported information. 
* Whether or not the core question has been answered. 
* Whether or not there is additional helpful information, which does not necessarily need to answer the question directly. 
* How coherent the answer is, and whether or not there are any citation errors. 
* How much irrelevant information there is in the answer. (This can be higher priority in extreme cases.)

对于最终比较评级，我们鼓励承包商使用他们的最佳判断，但按优先级降序粗略考虑以下标准：
* 答案是否包含不受支持的信息。
* 是否回答了核心问题。
* 是否有额外的帮助信息，不一定需要直接回答问题。
* 答案的连贯性如何，是否有引用错误。
* 答案中有多少无关信息。 （在极端情况下，这可能具有更高的优先级。）

The full instruction document we provided to contractors for comparisons can be viewed here.

可以在此处查看我们提供给承包商进行比较的完整说明文件。

For most of the project, we made every part of this procedure required 10% of the time, and made every part except for the final comparison rating optional 90% of the time. Towards the end of the project, we removed the question flags from the first part since we felt that they were being overused, and made the comparison ratings for unsupported information and coherence required all of the time.

对于大部分项目，我们将此过程的每个部分都设为需要 10% 的时间，并将除最终比较评级之外的每个部分设为可选的 90% 时间。 在项目快结束时，我们从第一部分中删除了问题标记，因为我们认为它们被过度使用了，并且始终需要对不受支持的信息和连贯性进行比较评级。

Despite the complexity of this procedure, we only used the final comparison rating in training, even collapsing together the “much better” and “better” ratings. We experimented with predicting some of the other information as an auxiliary loss, but we were not able to significantly improve the validation accuracy of the reward model. Nevertheless, we consider this to be another promising direction for further research. 

尽管这个过程很复杂，但我们在训练中只使用了最终的比较评分，甚至将“更好”和“更好”的评分合并在一起。 我们尝试将其他一些信息预测为辅助损失，但我们无法显著提高奖励模型的验证准确性。 尽管如此，我们认为这是进一步研究的另一个有希望的方向。

## D Contractor survey 承包商调查
It was valuable to gather feedback from our contractors, both to understand and improve their process, and to monitor job satisfaction. To this end, we sent them a questionnaire with the following questions: 
* Please say how much you agree with each of the statements. (Required 5-point Likert rating and optional comments)
    1. It was clear from the instructions what I was supposed to do.
    2. I found the task enjoyable and engaging.
    3. I found the task repetitive.
    4. I was paid fairly for doing the task.
    5. Overall, I am glad that I did this task.
* What would you change about the task to make it more engaging or enjoyable? (Encouraged) 
* Are there any other tools you could be given that would make it easier to complete the task to a consistently high standard? (Encouraged) 
* Did you come up with any shortcuts that you used to do the task more quickly, and if so, what were they? (Encouraged) 
* Do you have any other comments? (Optional)

从我们的承包商那里收集反馈对于了解和改进他们的流程以及监控工作满意度非常有价值。 为此，我们向他们发送了一份调查问卷，其中包含以下问题：
* 请说出您对每个陈述的同意程度。 （要求 5 分李克特评分和可选评论）
     1. 从指示中可以清楚地看出我应该做什么。
     2. 我发现这项任务令人愉快且引人入胜。
     3. 我发现任务重复。
     4. 我完成这项任务得到了公平的报酬。
     5. 总的来说，我很高兴我完成了这项任务。
* 你会如何改变任务以使其更具吸引力或乐趣？ （鼓励）
* 是否有任何其他工具可以让您更轻松地以始终如一的高标准完成任务？ （鼓励）
* 你有没有想出任何用来更快完成任务的捷径，如果有，它们是什么？ （鼓励）
* 你还有什么意见吗？ （选修的）

The “encouraged” questions were required questions but with instructions to put “N/A” if they really could not think of anything (this was rare).

“鼓励”的问题是必答题，但如果他们真的想不出任何东西（这种情况很少见），则说明会填写“N/A”。

We surveyed all contractors who completed 32 or more tasks (thus we excluded people who dropped out after the trial period or shortly thereafter). We did this 3 times over the course of the project: once for demonstrations and twice for comparisons. The quantitative results from these surveys are given in Figure 10. The vast majority of respondents reported that they enjoyed the task, were paid fairly and were glad that they did the task overall. A significant minority of respondents also reported that they found the task repetitive. 

我们调查了所有完成 32 项或更多任务的承包商（因此我们排除了在试用期后或之后不久退出的人）。 我们在整个项目过程中做了 3 次：一次用于演示，两次用于比较。 图10 给出了这些调查的定量结果。绝大多数受访者表示他们喜欢这项任务，报酬公平，并且很高兴他们总体上完成了这项任务。 相当一部分受访者还表示，他们发现这项任务是重复的。

![Figure 10](../images/WebGPT/fig_10.png)<br/>
Figure 10: Likert ratings aggregated over all 3 of our contractor surveys. All ratings are weighted equally, even when the same contractor provided ratings in multiple surveys. In total, there are 41 ratings for each question. 
图10：我们所有 3 项承包商调查的李克特评级汇总。 即使同一承包商在多项调查中提供评级，所有评级的权重也相同。 每个问题总共有 41 个评分。

## E Hyperparameters
Hyperparameters for all of our training methods are given in Tables 6 and 7. We mostly used the same hyperparameters for the different model sizes, with the caveat that we expressed the Adam step sizes as multiples of the pre-training Adam step sizes, which are given in Table 5.

我们所有训练方法的超参数在表6 和表7 中给出。我们主要对不同的模型大小使用相同的超参数，但需要注意的是，我们将 Adam 步长表示为预训练 Adam 步长的倍数，即 在表5中给出。  

For each training method, we implemented some form of early stopping:
1. For BC, we stopped after a certain number of epochs based on reward model score (which usually improves past the point of minimum validation loss).
2. For RM, we stopped after a certain number of epochs based on validation accuracy.
3. For RL, we stopped after a certain number of PPO iterations based on the reward model score for some KL budget. The KL here is measured from the BC model, and summed over the episode. For the 175B model, we compared a couple of different KL budgets using human evaluations, and for the 760M and 13B models, we chose KL budgets informed by the 175B evaluations.

对于每种训练方法，我们都实施了某种形式的提前停止：
1. 对于 BC，我们根据奖励模型分数（通常会改进超过最小验证损失点）在一定数量的 epoch 后停止。
2. 对于 RM，我们根据验证准确性在一定数量的 epoch 后停止。
3. 对于 RL，我们根据某些 KL 预算的奖励模型分数在一定次数的 PPO 迭代后停止。 这里的 KL 是根据 BC 模型测量的，并在剧集中求和。 对于 175B 模型，我们使用人工评估比较了几个不同的 KL 预算，对于 760M 和 13B 模型，我们选择了 175B 评估提供的 KL 预算。

The points at which we early stopped are given in Table 8.

我们提前停止的点在表8 中给出。

We tuned hyperparameters using similar criteria to early stopping. We used human evaluations sparingly, since they were noisy and expensive, and put less effort into tuning hyperparameters for the 760M and 13B model sizes. As a rule, we found the most important hyperparameter to tune to be the Adam step size multiplier.

我们使用与提前停止类似的标准调整超参数。 我们很少使用人工评估，因为它们嘈杂且昂贵，并且在调整 760M 和 13B 模型大小的超参数方面投入的精力更少。 通常，我们发现要调整的最重要的超参数是 Adam 步长乘数。

For BC and RM, we used Polyak–Ruppert averaging [Polyak and Juditsky, 1992], taking an exponentially-weighted moving average (EMA) of the weights of the model as the final checkpoint. The “EMA decay” hyperparameter refers to the decay of this EMA per gradient step. For RL (but not rejection sampling), we did not use the EMA model for the 760M or 13B reward models, due to a bug.

对于 BC 和 RM，我们使用 Polyak–Ruppert 平均 [Polyak 和 Juditsky，1992]，将模型权重的指数加权移动平均值 (EMA) 作为最终检查点。 “EMA 衰减”超参数指的是每个梯度步骤中该 EMA 的衰减。 对于 RL（但不是拒绝抽样），由于错误，我们没有将 EMA 模型用于 760M 或 13B 奖励模型。

For RL, most PPO hyperparameters did not require tuning, but a few points are worth noting: 
* As discussed in Section 3 of the paper, the reward is the sum of the reward model score at the end of each episode and a KL penalty from the BC model at each token. Even though the reward is part of the environment, we treat the coefficient of this KL penalty as a hyperparameter, called the “KL reward coefficient”. 
* We express hyperparameters such that each timestep corresponds to a single completion (rather than a single token), but we applied PPO clipping and the KL reward at the token level. We also trained token-level value function networks, allowing a token-level baseline to be used for advantage estimation, but we did not use token-level bootstrapping or discount rates. 
* We used separate policy and value function networks for simplicity, although we think that using shared networks is a promising direction for future research. 
* We used 1 epoch, since we were concerned more with compute efficiency than with sample efficiency. 
* Due to GPU memory constraints, we used 16 times as many minibatches per epoch as the default for PPO, but this was easily compensated for by reducing the Adam step size multiplier by a factor of 4. 
* We used the same number of parallel environments and timesteps per rollout as the default for PPO, even though it resulted in slow PPO iterations (lasting multiple hours). This is the easiest way to ensure that PPO performs enough clipping (around 1–2% of tokens).Compared to using fewer timesteps per rollout and fewer minibatches per epoch, we found the KL from the BC model to grow more slowly at the start of training, making training less sensitive to the KL reward coefficient until approaching convergence. This allowed us to replace tuning the KL reward coefficient with early stopping to some extent. 
* We did not use an entropy bonus, which is usually used for exploration. An entropy bonus is equivalent to a KL penalty from the uniform distribution, but the uniform distribution over tokens is somewhat arbitrary – in particular, it is not invariant to “splitting” a single token into two equally-likely indistinguishable tokens. Instead, the KL reward prevents 20 entropy collapse in a more principled way. We still found it useful to measure entropy for monitoring purposes. 
* We happened to use a GAE discount rate of 1 rather than the usual default of 0.999, but we do not expect this to have made much difference, since episodes last for well under 1,000 timesteps. 
* As discussed in Section 3 of the paper, at the end of each episode we inserted additional answering-only episodes using the same references as the previous episode, which is what the “answer phases per browsing phases” hyperparameter refers to. 
* Since some actions (such as quotes and answers) require many more tokens than others, we modified the environment to “chunk” long completions into multiple actions, to improve rollout parallelizability. This is what the “maximum tokens per action” hyperparameter refers to. Note that it has a minor effect on GAE. 

对于 RL，大多数 PPO 超参数不需要调整，但有几点值得注意：
* 如本文第 3 节所述，奖励是每集结束时的奖励模型得分与 BC 模型在每个标记处的 KL 惩罚之和。 尽管奖励是环境的一部分，但我们将此 KL 惩罚的系数视为超参数，称为“KL 奖励系数”。
* 我们表达超参数，使每个时间步长对应一个完成（而不是一个令牌），但我们在令牌级别应用了 PPO 裁剪和 KL 奖励。 我们还训练了令牌级价值函数网络，允许将令牌级基线用于优势估计，但我们没有使用令牌级自举或折扣率。
* 为简单起见，我们使用了单独的策略和价值函数网络，尽管我们认为使用共享网络是未来研究的一个有前途的方向。
* 我们使用了 1 个 epoch，因为我们更关心计算效率而不是样本效率。
* 由于 GPU 内存限制，我们每个时期使用的 minibatches 数量是 PPO 默认值的 16 倍，但这很容易通过将 Adam 步长乘数减少 4 倍来补偿。
* 我们在每次推出时使用与 PPO 默认设置相同数量的并行环境和时间步长，尽管这会导致 PPO 迭代缓慢（持续数小时）。 这是确保 PPO 执行足够裁剪（大约 1-2% 的令牌）的最简单方法。与每次 rollout 使用更少的时间步长和每个 epoch 使用更少的 minibatches 相比，我们发现 BC 模型的 KL 在开始时增长得更慢 训练，使训练对 KL 奖励系数不那么敏感，直到接近收敛。 这使我们能够在某种程度上用提前停止来代替调整 KL 奖励系数。
* 我们没有使用通常用于探索的熵加成。 熵奖励等同于均匀分布的 KL 惩罚，但代币的均匀分布有些随意——特别是，将单个代币“拆分”为两个同样可能无法区分的代币并不是不变的。 相反，KL 奖励以更有原则的方式防止 20 熵崩溃。 我们仍然发现出于监控目的测量熵是有用的。
* 我们碰巧使用了 1 的 GAE 折扣率而不是通常的默认值 0.999，但我们预计这不会产生太大影响，因为剧集持续的时间步长远低于 1,000。
* 如本文第 3 节所述，在每一集的末尾，我们使用与前一集相同的参考插入了额外的仅回答集，这就是“每个浏览阶段的回答阶段”超参数所指的内容。
* 由于某些操作（例如引用和回答）比其他操作需要更多的令牌，因此我们修改了环境以将长完成“分块”为多个操作，以提高推出的并行性。 这就是“每个动作的最大令牌数”超参数所指的。 请注意，它对 GAE 的影响很小。

![Table 5](../images/WebGPT/tab_5.png)<br/>
Table 5: Pre-training Adam step sizes, to which we apply multipliers. These are the same as those given in Brown et al. [2020].

![Table 7](../images/WebGPT/tab_7.png)<br/>

![Table 6](../images/WebGPT/tab_6.png)<br/>
Table 7: Reinforcement learning hyperparameters.

![Table 8](../images/WebGPT/tab_8.png)<br/>
Table 8: Early stopping points.


## F Minimal comparison instructions
As discussed in Section 4, for comparing WebGPT’s answers to the reference answers from the ELI5 dataset, we used a much more minimal set of instructions, for ecological validity. The full instructions consisted of the following text:

Comparing answers (minimal version)

In this task, you’ll be provided with a question and a set of two answers. We’d like you to provide ratings comparing the two answers for the following categories: 
* Accuracy – which answer is more factually accurate? ◦ Please use a search engine to fact-check claims in an answer that aren’t obvious to you. Answers may have subtly incorrect or fabricated information, so be careful! 
* Coherence – which answer is easier to follow? 
* Usefulness overall – all things considered, which answer would be more helpful to the person who asked this question?

FAQ 
* What should I do if there’s a URL in the question or one of the answers? ◦ Please don’t click any URLs and interpret the questions and answers based on their remaining textual content. 
* What should I do if the question doesn’t make any sense, or isn’t a question? ◦ Sometimes you’ll see a statement instead of a question, which you should interpret as “Explain: . . . ”. – E.g. a question titled “Magnets” should be interpreted as “Explain: magnets” or “How do magnets work?” ◦ If the question is ambiguous but has a few reasonable interpretations, stick with the interpretation that you think is most likely. ◦ If the question still doesn’t make sense (e.g. if you’d need to click on a

URL to understand it, or if it’s entirely unclear what the question means), then click the “This question does not make sense” checkbox at the top and submit the task. – This should be rare, so use this sparingly. 
* What should I do if the answer to the question depends on when it was asked? ◦ In this case, please be charitable when judging answers with respect to when the question was asked – an answer is considered accurate if it was accurate at any point within the last 10 years. – E.g. valid answers to the question “Who is the current U.S. president” are Barack Obama, Donald Trump, and Joe Biden. 
* What should I do if I only see one answer? ◦ If you only see one answer, you’ll be asked to provide absolute ratings for that answer (very bad, bad, neutral, good, or very good) instead of comparison ratings. – For the “usefulness overall” category, please calibrate your ratings such that “very bad” indicates an answer that is worse than not having an answer at all (e.g. due to being very misleading), “bad” indicates an answer that’s about as helpful as not having an answer, and higher ratings indicate useful answers with varying degrees of quality. 

## G TriviaQA evaluation
Although WebGPT was trained primarily to perform long-form question-answering, we were interested to see how well it would perform short-form question-answering. To this end, we evaluated WebGPT on TriviaQA [Joshi et al., 2017], a dataset of short-form questions from trivia websites. For this evaluation, we used the WebGPT 175B BC model with a sampling temperature of 0.8 and no rejection sampling.

尽管 WebGPT 主要是为执行长格式问答而训练的，但我们很想知道它在执行短格式问答方面的表现如何。 为此，我们在 TriviaQA [Joshi et al., 2017] 上评估了 WebGPT，这是一个来自琐事网站的简短问题数据集。 对于此评估，我们使用了 WebGPT 175B BC 模型，采样温度为 0.8，没有拒绝采样。

To address the mismatch between WebGPT’s long-form answers and the short-form answers expected by TriviaQA, we fine-tuned GPT-3 175B to answer TriviaQA questions conditioned on the output of WebGPT. Since this is a simple extraction task, and out of concern for test-train overlap [Lewis et al., 2020b], we used only 256 questions for this fine-tuning (with a batch size of 32 and a learning rate of 1.5 × 10−6 ). This was in addition to the 143 TriviaQA demonstrations on which the WebGPT model was trained. As an ablation, we also fine-tuned GPT-3 175B in the same way, but without the WebGPT output.

为了解决 WebGPT 的长格式答案与 TriviaQA 期望的短格式答案之间的不匹配，我们微调了 GPT-3 175B 以回答以 WebGPT 输出为条件的 TriviaQA 问题。 由于这是一个简单的提取任务，并且出于对测试训练重叠的考虑 [Lewis et al., 2020b]，我们仅使用 256 个问题进行微调（批量大小为 32，学习率为 1.5 × 10−6）。 这是对 WebGPT 模型进行训练的 143 个 TriviaQA 演示的补充。 作为消融，我们还以相同的方式微调 GPT-3 175B，但没有 WebGPT 输出。

Our results are shown in Table 9, along with those of the best existing model, UnitedQA [Cheng et al., 2021]. We report results on the TriviaQA development set splits defined in Lewis et al. [2020b]. We perform slightly better than UnitedQA-E on questions with no test-train overlap, and slightly worse on questions with test-train overlap. We hypothesize that this difference is the result of WebGPT being trained on far fewer TriviaQA questions.

我们的结果以及现有最佳模型 UnitedQA [Cheng et al., 2021] 的结果如表9 所示。 我们报告了 Lewis 等人定义的 TriviaQA 开发集拆分的结果。 [2020b]。 我们在没有测试训练重叠的问题上比 UnitedQA-E 表现稍好，在测试训练重叠的问题上表现稍差。 我们假设这种差异是 WebGPT 在更少的 TriviaQA 问题上接受训练的结果。

![Table 9](../images/WebGPT/tab_9.png)<br/>
Table 9: TriviaQA development set accuracy (exact match scores).
表9：TriviaQA 开发集准确性（精确匹配分数）。

Note that we use far more compute than UnitedQA, and also use live access to the web rather than only the corpus provided by Joshi et al. [2017] (although we still censor trivia websites in the same way for this evaluation). On the other hand, WebGPT was trained primarily to perform long-form question-answering, and so the transfer to the short-form setting is notable. 

请注意，我们使用的计算比 UnitedQA 多得多，并且还使用实时访问网络，而不仅仅是 Joshi 等人提供的语料库。 [2017]（尽管我们仍然以同样的方式审查琐事网站以进行此评估）。 另一方面，WebGPT 的训练主要是为了执行长格式问答，因此向短格式设置的转移是值得注意的。

## H Analysis of effect of question stance and reference point bias 问题立场和参考点偏差的影响分析
In this section we investigate the impact of question “stance” (whether the question implicitly supports or refutes some relevant belief) on the model’s accuracy, and on its tendency to support or refute that belief in its answer. We also probe the model’s bias towards “assuming” a certain cultural reference point with an example culturally dependent question (“What does a wedding look like?”).

在本节中，我们研究了问题“立场”（问题是否隐含地支持或反驳某些相关信念）对模型准确性的影响，以及它在其答案中支持或反驳该信念的倾向。 我们还通过一个文化相关问题（“婚礼是什么样的？”）来探讨模型对“假设”某个文化参考点的偏见。

### H.1 Effect of question stance on factual accuracy and answer stance  
We ran a small experiment to investigate the impact of question stance on the model’s answers. Inspired by TruthfulQA [Lin et al., 2021], we chose 10 well-known conspiracy theories and 10 common misconceptions for this experiment. For each conspiracy theory or misconception, we wrote three questions, each taking one of three stances: one expressing skepticism around the implicit belief, one neutral about the implicit belief, and one affirming the implicit belief. This resulted in the 60 questions given in Table 10. We collected answers to these questions for the three compute-efficient WebGPT models (see Section 5.2), and used these answers to look for a couple of different effects. 
* Factual accuracy. First, we examined whether the stance of the question impacts the model’s factual accuracy. To do this, we labelled each answer as accurate or inaccurate, by fact-checking any central or specific claims in the answer, and labeling the answer as inaccurate if a significant number (more than around 25%) of those claims could not be easily verified. Our results are given in Figure 11. We found suggestive evidence that, across model sizes, questions that affirm an implicit belief in a conspiracy or misconception tend to elicit inaccurate answers from the model more often than questions that are framed in a neutral or skeptical way. While our experiment had too small of a sample size for us to draw definitive conclusions, it demonstrates the model’s potential to misinform users who have erroneous beliefs in ways that reinforce those beliefs. 
* Answer stance. Second, we studied whether the model mirrors the stance of the question in the content of its response. To do this, we labelled each answer on whether it explicitly refutes the implicit belief or explicitly affirms the implicit belief. Note that in some cases it is possible for an answer to affirm the belief in the conspiracy theory or misconception while remaining factually accurate, by including appropriate caveats. If an answer initially affirms the belief but then reverses its stance, saying for example “but this is a myth”, then we consider it to have refuted the belief. Our results are given in Figure 12. We found that all the models tended to refute the implicit beliefs more often than they affirmed them, and that this effect increased with model size. However, we did not find any clear evidence that the stance of the question has any effect on this behavior.

我们进行了一个小实验来研究问题立场对模型答案的影响。 受 TruthfulQA [Lin et al., 2021] 的启发，我们选择了 10 个众所周知的阴谋论和 10 个常见的误解来进行这个实验。 对于每个阴谋论或误解，我们写了三个问题，每个问题都采取三种立场之一：一种表达对隐含信念的怀疑，一种对隐含信念持中立态度，一种肯定隐含信念。 这导致了表10 中给出的 60 个问题。我们收集了三个计算高效的 WebGPT 模型（参见第 5.2 节）的这些问题的答案，并使用这些答案来寻找几个不同的效果。
* 事实准确性。 首先，我们检查了问题的立场是否会影响模型的事实准确性。 为此，我们将每个答案标记为准确或不准确，方法是对答案中的任何中心或特定声明进行事实核查，如果这些声明中有很大一部分（超过约 25%）无法轻易识别，则将答案标记为不准确 验证。 我们的结果在图11 中给出。我们发现有启发性的证据表明，在不同的模型规模中，确认对阴谋或误解的隐含信念的问题往往比以中立或怀疑的方式提出的问题更容易从模型中引出不准确的答案 . 虽然我们的实验样本量太小，无法得出明确的结论，但它证明了该模型有可能以强化这些信念的方式误导有错误信念的用户。
* 回答的立场。 其次，我们研究了该模型是否在其回答内容中反映了问题的立场。 为此，我们将每个答案标记为明确反驳隐含信念或明确肯定隐含信念。 请注意，在某些情况下，通过包含适当的警告，答案可以在保持事实准确的同时确认对阴谋论或误解的信念。 如果一个答案最初肯定了这个信念，但后来又改变了立场，例如说“但这是一个神话”，那么我们认为它已经反驳了这个信念。 我们的结果在图12 中给出。我们发现所有模型倾向于反驳隐含信念的频率高于它们肯定隐含信念的频率，并且这种影响随着模型大小的增加而增加。 然而，我们没有发现任何明确的证据表明问题的立场对这种行为有任何影响。

Given the small scale of this experiment, it would be informative to see further research on the effect of question stance on model answers. We remark that humans exhibit sensitivity to the framing of questions [Chong and Druckman, 2007]. In addition to this, it would be useful to study the effects of various other factors, such as the training data collection methodology, the relative degree of skepticism, neutrality or affirmation in the questions, the relative volumes of skeptical or affirming sources on the web, and whether the questions themselves appear in the training data or on the web.

鉴于该实验的规模较小，进一步研究问题立场对模型答案的影响将是有益的。 我们注意到人类对问题的框架表现出敏感性 [Chong 和 Druckman，2007]。 除此之外，研究各种其他因素的影响将很有用，例如训练数据收集方法、问题中的怀疑、中立或肯定的相对程度，网络上怀疑或肯定来源的相对数量 ，以及问题本身是出现在训练数据中还是网络上。

### H.2 Reference point bias
Rather than having a strong stance, some questions may reveal very little information about the user, but the model may nevertheless assume a certain cultural reference point. We refer to this as reference point bias. To probe this phenomenon, we conducted a simple case study, in which we analyzed 64 answers from the WebGPT 175B BC model to the following question: “What does a wedding look like?”.

In response to this question, the model tended to assume a Western, and often specifically an American, point-of-view. Out of the 64 answers, 20 included the word “America” or “American”, and only 4 focused on a specific, named culture other than American: Vietnamese (1); Indian (1); and Croatian (2). While 8 of 64 responses noted that there is no standard wedding, all but one of these still also included at least one detail typical of a Western and often American wedding. And 2 of the these 8 – including the answer with the highest reward model score – noted that there is no standard or typical American wedding. 

![Figure 11](../images/WebGPT/fig_11.png)<br/>
Figure 11: Results of experiment on effect of question stance on factual accuracy.

![Figure 12](../images/WebGPT/fig_12.png)<br/>
Figure 12: Results of experiment on effect of question stance on answer stance.

The assumption of a Western and often American reference point in this case may be influenced by the data the model has seen during pre-training, by Internet search data, by the viewpoints represented by the contractors we worked with, and by our research team. When asked specifically “What does a Vietnamese wedding look like?”, the model usually generates responses pertaining to Vietnamese weddings, but these come up rarely in response to a generic question. When in doubt, the model defaults to assuming a Western or American viewpoint in this case.

Furthermore, we also noticed that the model often makes other assumptions that exclude or erase some identities in response to this question, for example by assuming that a couple consists of a male groom and female bride and assuming that a bride’s father walks her down an aisle. While our experiment was focused on probing its bias towards a Western or American point of view, we encourage further interdisciplinary research in these and other areas of bias. 

![Table 10](../images/WebGPT/tab_10.png)<br/>
Table 10: Questions used to study the effect of question stance on the model’s answers. Each of the top 10 topics refers a well-known conspiracy theory, and each of the bottom 10 topics refers to a common misconception. For each topic, we wrote a question with a skeptical stance, a question with neutral stance, and a question with an affirming stance.

## I Predicting rejection sampling performance
It is helpful to be able to predict human preference of answers produced using rejection sampling (best-of-n). To do this, we evaluate answers using a validation reward model (trained on a separate dataset split), to try to account for the original reward model being overoptimized. For large n, the naive Monte Carlo estimator of the expected validation reward model score requires many model samples to produce accurate estimates. Here we describe an alternative estimator, which produces accurate estimates more efficiently.

Let Q be the distribution of questions, and given a question q, let A (q) be the distribution of answers produced by the model. Given a question q and an answer a (with references), let R train (a ∣ q) be the original reward model score, and let R val (a ∣ q) be the validation reward model score. Let n be the number of answers sampled when rejection sampling (i.e., the n in best-of-n).

To predict the Elo score corresponding to human preference for a given question q, we estimate

Rn pred (q) ∶= EA1,...,An∼A(q) [R val ( argmax a∈{A1,...,An}R train (a ∣ q) ∣ q)] .

To predict the overall Elo score corresponding to human preference, we estimate

EQ∼Q [Rn pred (Q)] .

As shown in Figure 5, this predicts human preference well for n ≤ 64, although we expect it to overestimate human preference for sufficiently large n, as the validation reward model will eventually become overoptimized.

The simplest way to estimate Rn pred (q) for a given question q is with a Monte Carlo estimator, by repeatedly sampling A1, A2, . . . , An ∼ A (q). However, this is very wasteful, since it takes n answers to produce each estimate, and moreover, answers are not re-used for different values of n.

Instead, we sample A1, A2, . . . , AN ∼ A (q) for some N ≥ n, and compute 1(Nn ) ∑ 1≤i1<⋅⋅⋅<in≤N R val (a∈{ argmax

Ai1 ,...,Ain }R train (a ∣ q) ∣ q) , which is an unbiased estimator of Rn pred (q) by linearity of expectation. This can be computed efficiently by sorting A1, A2, . . . , AN by original reward model score to obtain S1, S2, . . . SN with

R train (S1 ∣ q) ≤ ⋅ ⋅ ⋅ ≤ R train (SN ∣ q), and then computing 1(Nn ) ∑ 1≤i1<⋅⋅⋅<in≤N R val (a∈{ argmax

Si1 ,...,Sin }R train (a ∣ q) ∣ q) = N∑i=n (i−1 n−1) (Nn ) R val (Si ∣ q) .

To estimate EQ∼Q [Rn pred (Q)] for n = 1, 2, . . . , nmax, we sample a number of different questions

Q ∼ Q, sample A1, A2, . . . , AN ∼ A (Q) for some N ≥ nmax for each question Q, and for each n take the empirical average of the above estimator over the different questions. 28

## J References for example answer and alternative answers
In Table 2, we gave an answer produced by our 175B best-of-64 model to a randomly-chosen question from the ELI5 test set: “Why did we decide that certain words were "bad" and shouldn’t be used in social settings?”. Here we give the full text of the references for that answer, along with an answer produced by one of our human demonstrators, and the reference answer from the ELI5 dataset. Further samples are available at https://openaipublic.blob.core.windows.net/ webgpt-answer-viewer/index.html.


![Table 11](../images/WebGPT/tab_11.png)<br/>
Table 11: Full text of the references found by our model for the answer given in Table 2.

![Table 12](../images/WebGPT/tab_12.png)<br/>
Table 12: An answer and references produced by one of our human demonstrators to the same question as the one in Table 2.

Answer Swear words, obscenities, and derogatory terms are some categories of words that society considers ‘bad’ and exempt from usage in social settings. In a more definitive approach, courts presume harm from speech in cases involving discrimination or sexual harassment [1]. While the original justification for these obscenity laws was predicated on an unfounded assumption that speech can deprave or corrupt children, and it is important to note that there is little (if any) social science data demonstrating that a word is by itself harmful [1], it is also pertinent to consider the origin of some derogatory words and the context of their usage to understand why they should be unacceptable in social settings. Examples include: -Ape/Monkey: These are considered offensive terms when used to describe a person of color because offensive comparisons of black people to apes date back hundreds of centuries, such as when scientists Josiah C. Nott and George R. Gliddon created the 1854 Types of Mankind (a leading scientific text on race in its time), and they compared black people not to other (white) people but to chimpanzees, gorillas, and orangutans. They are now, justifiably, considered extremely derogatory slang. [2] -Savage: When used as an adjective to describe acts of violence, this word is not considered offensive, but the use of ‘savage’ as a noun to describe people dates back to a time when Christopher

Columbus was enslaving Native Americans and claiming their lands and work for his own. The word became a slur used by white Europeans to describe Native American people. The usage stuck around for hundreds of years, and it’s still a word that many indigenous peoples around the world find offensive. [3]

References [1] The Science of Swearing (www.psychologicalscience.org)

![Table 13](../images/WebGPT/tab_13.png)<br/>
Table 13: The reference answer from the ELI5 dataset for the question given in Table 2. 


## K Comparison dataset release details
We are releasing a dataset of all comparisons that were marked as suitable for reward modeling by the end of the project, totalling 19,578 comparisons. It can be downloaded from https:// openaipublic.blob.core.windows.net/webgpt-answer-viewer/comparisons.jsonl.

Each item in the dataset is a pair of records, each of which has the following fields: 
* Question: The text of the question, together with the name of the dataset from which it was taken and a unique ID. The question is the same for each record in the pair. 
* Quotes: The extracts found during browsing, together with the title of the page on which the extract was found, constructed from the HTML title and domain name of the page. 
* Answer: The final answer composed using the quotes. 
* Tokens: The prefix that would have been given to the model in the final step of the episode, and the completion given by the model or human. The prefix is made up of the question and the quotes, with some truncation, and the completion is simply the answer. Both are tokenized using the GPT-2 tokenizer. The concatenation of the prefix and completion is the input used for reward modeling. 
* Score: The strength of the preference for the answer as a number from −1 to 1. The two scores in each pair sum to zero, and an answer is preferred if and only if its score is positive. For reward modeling, we treat scores of 0 as soft 50% labels, and all other scores as hard labels (using only their sign). 32
