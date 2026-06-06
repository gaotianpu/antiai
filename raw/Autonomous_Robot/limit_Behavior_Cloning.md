# Exploring the Limitations of Behavior Cloning for Autonomous Driving
探索自动驾驶行为克隆的局限性 https://arxiv.org/abs/1904.08980

## Abstract
Driving requires reacting to a wide variety of complex environment conditions and agent behaviors. Explicitly modeling each possible scenario is unrealistic. In contrast, imitation learning can, in theory, leverage data from large fleets of human-driven cars. Behavior cloning in particular has been successfully used to learn simple visuomotor policies end-to-end, but scaling to the full spectrum of driving behaviors remains an unsolved problem. In this paper, we propose a new benchmark to experimentally investigate the scalability and limitations of behavior cloning. We show that behavior cloning leads to state-of-the-art results, including in unseen environments, executing complex lateral and longitudinal maneuvers without these reactions being explicitly programmed. However, we confirm well-known limitations (due to dataset bias and overfitting), new generalization issues (due to dynamic objects and the lack of a causal model), and training instability requiring further research before behavior cloning can graduate to real-world driving. We will release our benchmark and code. 

驾驶需要对各种复杂的环境条件和智能体行为做出反应。 对每个可能的场景进行明确建模是不现实的。 相比之下，理论上，模仿学习可以利用来自大量人类驾驶汽车的数据。 行为克隆尤其已成功用于端到端学习简单的视觉运动策略，但扩展到整个驾驶行为范围仍然是一个未解决的问题。 在本文中，我们提出了一个新的基准来通过实验研究行为克隆的可扩展性和局限性。 我们表明，行为克隆可以产生最先进的结果，包括在看不见的环境中执行复杂的横向和纵向机动，而无需明确编程这些反应。 然而，我们确认了众所周知的局限性(由于数据集偏差和过度拟合)、新的泛化问题(由于动态对象和因果模型的缺乏)以及训练不稳定性需要进一步研究才能将行为克隆升级到真实世界的驾驶 . 我们将发布我们的基准和代码。

Figure 1. Driving scenarios from our new benchmark where the agent needs to react to dynamic changes in the environment, handle clutter (only part of the environment is causally relevant), and predict complex sensorimotor controls (lateral and longitudinal). We show that Behavior Cloning yields state-of-the-art policies in these complex scenarios and investigate its limitations.
图 1. 我们新基准的驾驶场景，其中智能体需要对环境的动态变化做出反应，处理混乱(只有部分环境是因果相关的)，并预测复杂的感觉运动控制(横向和纵向)。 我们展示了行为克隆在这些复杂场景中产生了最先进的策略并研究了它的局限性。

## 1. Introduction
End-to-end behavior cloning for autonomous driving has recently attracted renewed interest [8, 6, 10, 42, 29] as a simple alternative to traditional modular approaches used in industry [11, 22]. In this paradigm, perception and control are learned simultaneously using a deep network. Explicit sub-tasks are not defined, but may be implicitly learned from data. These sensorimotor controllers are typically obtained by imitation learning from human demonstrations [2, 32, 1, 38]. The deep network learns, without being explicitly programmed, to recognize patterns associating sensory input (e.g., a single RGB image) with a desired reaction in terms of vehicle control parameters producing a target maneuver. Behavior cloning can directly learn from large fleets of human-driven vehicles without requiring a fixed ontology and extensive amounts of labeling. Finally, end-to-end imitative systems can be learned off-line in a safe way, in contrast to reinforcement learning approaches that typically require millions of trial and error runs in the target environment [23] or a faithful simulation.

用于自动驾驶的端到端行为克隆最近引起了新的兴趣 [8、6、10、42、29]，作为工业中使用的传统模块化方法的简单替代方法 [11、22]。 在这个范例中，感知和控制是使用深度网络同时学习的。 未定义显式子任务，但可以从数据中隐式学习。 这些感觉运动控制器通常是通过模仿人类示范学习获得的 [2, 32, 1, 38]。 深度网络在没有明确编程的情况下学习识别将感官输入(例如，单个 RGB 图像)与产生目标机动的车辆控制参数方面的期望反应相关联的模式。 行为克隆可以直接从大量的人类驾驶车辆中学习，而不需要固定的本体和大量的标签。 最后，端到端模仿系统可以以安全的方式离线学习，这与通常需要在目标环境 [23] 或忠实模拟中进行数百万次试错运行的强化学习方法形成对比。

End-to-end imitative systems can suffer a domain shift between the off-line training experience and the on-line behavior [34]. This problem, however, can be addressed in practice by data augmentation [6, 10]. Nonetheless, in spite of the early and recent successes of behavior cloning for end-to-end driving [31, 21, 8, 6, 10], it has not yet proved to scale to the full spectrum of driving behaviors, such as reacting to multiple dynamic objects.

端到端模仿系统可能会在离线训练体验和在线行为之间发生域转换 [34]。 然而，这个问题实际上可以通过数据增广 [6、10] 来解决。 尽管如此，尽管用于端到端驾驶的行为克隆在早期和最近取得了成功 [31、21、8、6、10]，但尚未证明它可以扩展到所有驾驶行为，例如反应 到多个动态对象。

In this paper, we propose a new benchmark, called NoCrash, and perform a large scale analysis of end-to-end behavioral cloning systems in complex driving conditions not studied in this context before. We use a high fidelity simulated environment based on the open source CARLA simulator [12] to enable reproducible large scale off-line training and on-line evaluation in over 80 hours of driving under several different conditions. We describe a strong Conditional Imitation Learning baseline, derived from [10], that significantly improves upon state of the art modular [24], affordance based [36], and reinforcement learning [26] approaches, both in terms of generalization performance in training environments and unseen ones.

在本文中，我们提出了一个名为 NoCrash 的新基准，并对复杂驾驶条件下的端到端行为克隆系统进行了大规模分析，此前在此背景下未进行过研究。 我们使用基于开源 CARLA 模拟器 [12] 的高保真模拟环境，以在多种不同条件下进行 80 多个小时的驾驶，从而实现可重现的大规模离线训练和在线评估。 我们描述了一个强大的条件模仿学习基线，该基线源自 [10]，它显著改进了最先进的模块化 [24]、基于可供性的 [36] 和强化学习 [26] 方法，无论是在训练中的泛化性能方面 环境和看不见的。

Despite its positive performance, we identify limitations that prevent behavior cloning from successfully graduating to real-world applications. First, although generalization performance should scale with training data, generalizing to complex conditions is still an open problem with a lot of room for improvement. In particular, we show that no approach reliably handles dense traffic scenes with many dynamic agents. Second, we report generalization issues due to dataset biases and the lack of a causal model. We indeed observe diminishing returns after a certain amount of demonstrations, and even characterize a degradation of performance on unseen environments. Third, we observe a significant variability in generalization performance when varying the initialization or the training sample order, similar to on-policy RL issues [17]. We conduct experiments estimating the impact of ImageNet pre-training and show that it is not able to fully reduce the variance. This suggests the order of training samples matters for off-policy Imitation Learning, similar to the on-policy case [45].

尽管它有积极的表现，但我们发现了阻止行为克隆成功升级到实际应用程序的局限性。 首先，虽然泛化性能应该与训练数据成比例，但泛化到复杂条件仍然是一个有很大改进空间的悬而未决的问题。 特别是，我们表明没有任何方法可以可靠地处理具有许多动态智能体的密集交通场景。 其次，我们报告了由于数据集偏差和缺乏因果模型导致的泛化问题。 我们确实观察到一定数量的演示后收益递减，甚至描述了在看不见的环境中性能下降的特征。 第三，当改变初始化或训练样本顺序时，我们观察到泛化性能的显著变化，类似于 on-policy RL 问题 [17]。 我们进行了实验来评估 ImageNet 预训练的影响，并表明它无法完全减少方差。 这表明训练样本的顺序对于离策略模仿学习很重要，类似于基于策略的情况 [45]。

Our paper is organized as follows. Section 2 describes related work, Section 3 our strong behavior cloning baseline, Section 4 our evalution protocol, including our new NoCrash benchmark, Section 5 our experimental results, and Section 6 our conclusion.

我们的论文的结构安排如下。 第 2 节描述相关工作，第 3 节我们的强大行为克隆基线，第 4 节我们的评估协议，包括我们新的 NoCrash 基准，第 5 节我们的实验结果，第 6 节我们的结论。

## 2. Related Work
Behavior cloning for driving dates back to the work of Pomerleau [31] on lane following, later followed by other approaches [21], including going beyond driving [1, 39]. The distributional shift between the training and testing distributions is the main known limitation of this approach, which might require on-policy data collection [33, 34], obtained by the learning agent. Nonetheless, recent works have proposed effective off-policy solutions, for instance by expanding the space of image/action pairs either using noise [20, 10], extra sensors [6], or modularization [36, 25]. We show, however, that there are other limitations important to consider in complex driving scenarios, in particular dataset bias and high variance, which both harm scaling generalization performance with training data.

驾驶行为克隆可以追溯到 Pomerleau [31] 在车道跟随方面的工作，后来出现了其他方法 [21]，包括超越驾驶 [1, 39]。 训练和测试分布之间的分布转移是这种方法的主要已知限制，这可能需要由学习智能体获得的在线策略数据收集 [33、34]。 尽管如此，最近的工作提出了有效的 off-policy 解决方案，例如通过使用噪声 [20、10]、额外传感器 [6] 或模块化 [36、25] 来扩展图像/动作对的空间。 然而，我们表明，在复杂的驾驶场景中还有其他重要的限制需要考虑，特别是数据集偏差和高方差，它们都会损害训练数据的扩展泛化性能。

Dataset bias is a core problem of real-world machine learning applications [41, 4] that can have dramatic effects in a safety-critical application like autonomous driving. Imitation learning approaches are particularly sensitive to this issue, as the learning objective might be dominated by the main modes in the training data. Going beyond the original CARLA benchmark [12], we use our new NoCrash benchmark to quantitatively assess the magnitude of this problem on generalization performance for more realistic and challenging driving behaviors.

数据集偏差是现实世界机器学习应用程序 [41, 4] 的核心问题，它可以在自动驾驶等安全关键应用程序中产生巨大影响。 模仿学习方法对这个问题特别敏感，因为学习目标可能由训练数据中的主要模式决定。 超越原始的 CARLA 基准 [12]，我们使用我们新的 NoCrash 基准来定量评估这个问题在泛化性能上的严重性，以获得更现实和更具挑战性的驾驶行为。

High variance is a key problem in powerful deep neural networks, and we show that high performance behavior cloning models are particularly suffering from this. This problem is related to sensitivity to both initialization and sampling order [30], reproducibility issues in Reinforcement Learning [17, 28], and the need to move beyond the i.i.d. data assumption towards curriculum learning [5] for sensorimotor control [45, 3].

高方差是强大的深度神经网络中的一个关键问题，我们表明高性能行为克隆模型尤其受此影响。 这个问题与初始化和采样顺序 [30] 的敏感性、强化学习中的可重复性问题 [17、28] 以及超越独立同分布的需要有关。 用于感觉运动控制 [45, 3] 的课程学习 [5] 的数据假设。

Driving benchmarks fall in two main categories: offline datasets, e.g., [13, 35, 43, 16], or on-line environments. We focus here on on-line benchmarks, as visuomotor models performing well in dataset-based evaluations do not necessarily translate to good driving policies [9]. Driving is obviously a safety-critical robotic application. Consequently, for safety and to enable reproducibility, researchers focus on using photo-realistic simulation environments. In particular, the CARLA open-source driving simulator [12] is emerging as a standard platform for driving research, used in [10, 29, 36, 26, 25]. Note, however, that transferring policies from simulation to the real-world is an open problem [27] out of the scope of this paper, although recent works have shown encouraging results [29, 44].

驾驶基准分为两大类：离线数据集，例如 [13、35、43、16]，或在线环境。 我们在这里关注在线基准测试，因为在基于数据集的评估中表现良好的视觉运动模型不一定转化为良好的驾驶策略 [9]。 驾驶显然是一种对安全至关重要的机器人应用。 因此，为了安全和实现可重复性，研究人员专注于使用照片般逼真的模拟环境。 特别是，CARLA 开源驾驶模拟器 [12] 正在成为驾驶研究的标准平台，用于 [10、29、36、26、25]。 但是请注意，将策略从模拟转移到现实世界是一个超出本文范围的开放性问题 [27]，尽管最近的工作已经显示出令人鼓舞的结果 [29、44]。

## 3. A Strong Baseline for Behavior Cloning
In this section, we first describe the behavior cloning framework we use, its limitations, and a robustified baseline that tries to tackle these issues.

### 3.1. Conditional Imitation Learning
Behavior cloning [31, 37, 34, 23] is a form of supervised learning that can learn sensorimotor policies from off-line collected data. The only requirements are pairs of input sensory observations associated with expert actions. We use an expanded formulation for self-driving cars called Conditional Imitation Learning, CIL [10]. It uses a high-level navigational command c that disambiguates imitation around multiple types of intersections. Given an expert policy π∗(x) with access to the environment state x, we can execute this policy to produce a dataset, D = {hoi, ci, aii}Ni=1, where oi are sensor data observations, ci are high-level commands (e.g., take the next right, left, or stay in lane) and ai = π∗(xi) are the resulting vehicle actions (low-level controls). Observations oi = {i, vm} contain a single image i and the ego car speed vm [10] added for the system to properly react to dynamic objects on the road. Without the speed context, the model cannot learn if and when it should accelerate or brake to reach a desired speed or stop.

We want to learn a policy π parametrized by θ to produce similar actions to π∗ based only on observations o and highlevel commands c. The best parameters θ∗ are obtained by minimizing an imitation cost ` : θ∗ = arg min θ X i `  π(oi, ci; θ), ai . (1)

In order to evaluate the performance of the learned policy π(oi, ci; θ) on-line at test time, we assume access to a score function giving a numeric value expressing the performance of the policy π on a given benchmark (cf. section 4).

### 3.2. Limitations
In addition to the distributional shift problem [34], behavior cloning presents some key limitations.

Bias in Naturalistic Driving Datasets. The appeal of behavior cloning lies in its simplicity and theoretical scalability, as it can indeed learn by imitation from large offline collected demonstrations (e.g., using driving logs from manually driven production vehicles). It is, however, susceptible to dataset biases like all learning methods. This is exacerbated in the case of imitation learning of driving policies, as most of real-world driving consists in either a few simple behaviors or a heavy tail of complex reactions to rare events. Consequently, this can result in performance degrading as more data is collected, because the diversity of the dataset does not grow fast enough compared to the main mode of demonstrations. This phenomenon was not clearly measured before. Using our new NoCrash benchmark (section 4), we confirm it may happen in practice.

Causal Confusion. Related to dataset bias, end-to-end behavior cloning can suffer from causal confusion [14]: spurious correlations cannot be distinguished from true causes in observed training demonstration patterns unless an explicit causal model or on-policy demonstrations are used. Our new NoCrash benchmark confirms the theoretical observation and toy experiments of [14] in realistic driving conditions. In particular, we identify a typical failure mode due to a subtle dataset bias: the inertia problem. When the ego vehicle is stopped (e.g., at a red traffic light), the probability it stays static is indeed overwhelming in the training data. This creates a spurious correlation between low speed and no acceleration, inducing excessive stopping and diffi- cult restarting in the imitative policy. Although mediated

Figure 2. Our proposed network architecture, called CILRS, for end-to-end urban driving based on CIL [10]. A ResNet perception module processes an input image to a latent space followed by two prediction heads: one for controls and one for speed. perception approaches that explicitly model causal signals like traffic lights do not suffer from this theoretical limitation, they still under-perform end-to-end learning in unconstrained environments, because not all causes might be modeled (e.g., some potential obstacles) and errors at the perception layer (e.g., missed detections) are irrecoverable.

High variance. With a fixed off-policy training dataset, one would expect CIL to always learn the same policy in different runs of the training phase. However, the cost function is optimized via Stochastic Gradient Descent (SGD), which assumes the data is independent and identically distributed [7]. When training a reactive policy on snapshots of longer human demonstrations included in the training data, the i.i.d. assumption does not hold. Consequently, we might observe a high sensitivity to the initialization and the order in which the samples are seen during training. We confirm this in our experiments, finding an overall high variance due to both initialization and sampling order, following the decomposition in [30]:

Var(π) = ED V arI (π|D) + V arD EI [π|D] , (2) where I denotes the randomness in initialization. Because the policy π is evaluated on-line in simulated environments, we evaluate in practice the variance of the score on the test benchmark, and report results when freezing the initialization and/or varying the sampling order for different training datasets D (including of varying sizes).

### 3.3. Model
In order to explore the aforementioned limitations of behavior cloning, we propose a robustified CIL model designed to improve on [10] while remaining strictly offpolicy. Our network architecture, called CILRS, is shown in Figure 2. We describe our enhancements below.

Deeper Residual Architecture. We use a ResNet34 architecture [15] for the perception backbone P(i). In the presence of large amounts of data, using deeper architectures can be an effective strategy to improve performance [15]. In particular, it can reduce both bias and variance, maintaining in particular a constant variance due to training set sampling with both network width and depth [30]. For end-to-end driving, the choice of architecture has been mostly limited to small networks so far [6, 10, 36] to avoid overfitting on limited datasets. In contrast, we notice that bigger models have better generalization performance on learning reactions to dynamic objects and traffic lights in complex urban environments.

Speed Prediction Regularization. To cope with the inertia problem without an explicit mapping of potential causes or on-policy interventions, we jointly train a sensorimotor controller with a network that predicts the ego vehicle’s speed. Both neural networks share the same representation via our ResNet perception backbone. Intuitively, what happens is that this joint optimization enforces the perception module to have speed related features into the learned representation. This reduces the dependency on input speed as the only way to get dynamics of the scene, leveraging instead visual cues that are predictive of the car’s velocity (e.g., free space, curves, traffic light states, etc).

Other changes. We use L1 as loss function ` instead of the mean squared error (MSE), as it is more correlated to driving performance [9]. As our NoCrash benchmark consists of complex realistic driving conditions in the presence of dynamic agents, we collect demonstrations from an expert game AI using privileged information to drive correctly (i.e. always respecting rules of the road and not crashing into any obstacle). Robustness to heavy noise in the demonstrations is beyond the scope of our work, as we aim to explore limitations of behavior cloning methods in spite of good demonstrations. Finally, we pre-trained our perception backbone on ImageNet to reduce initialization variance and benefit from generic transfer learning, a standard practice in deep learning seldom explored for behavior cloning.

## 4. Evaluation
In this section we discuss the simulated environment we use, CARLA, and review the original CARLA benchmark.

Due to its limitations, we propose a new benchmark, called

NoCrash, that tries to better evaluate driving controllers reaction to dynamic objects. This new benchmark, thanks to its complexity, allows further analysis on limitations of behavior cloning and other policy learning methods.

### 4.1. Simulated Environment
We use the CARLA simulator [12] version 0.8.4. The

CARLA environment is divided in two different towns.

Town 01 contains 2.9 km of drivable roads in a suburban environment. Town 02 is approximately 1.4 km of drivable roads, also in a suburban environment.

The CARLA environment may contain dynamic obstacles that interact with the ego car. Pedestrians, for instance, might cross the road on random occasions without any apparent previous notice. This action forces the ego car to promptly react. The CARLA environment also contains a diversity of car brands that cruise at different speeds. Overall it provides a diverse, photo-realistic, and dynamic environment with challenging driving conditions (cf. Figure 1).

The original CARLA benchmark [12] evaluates driving controllers on several goal directed tasks of increasing dif- ficulty. Three of the tasks consist of navigation in an empty town and one of them in a town with a small number of dynamic objects. Each task is tested in four different conditions of increasingly different from the training environment. The conditions are: same as training, new weather conditions that are derivatives from those seen during training, and a new town that has different buildings and different shadow patterns. Note that the biggest generalization test is the combination of new weather and new town.

The goal directed tasks are evaluated based on success rate. If the agent reaches the goal regardless of what happened during the episode, this episode is considered a success. The collisions and other infractions are considered and the average number of kilometers between infractions is measured. This evaluation induces the benchmark to be mainly focused on problems of a static nature. These problems consider the environmental conditions and the static objects of the world like buildings and trees. Thus, the original CARLA benchmark mostly evaluates skills such as lane keeping and performing 90 degrees turns.

### 4.2. NoCrash Benchmark
We propose a new larger scale CARLA driving benchmark, called NoCrash, designed to test the ability of ego vehicles to handle complex events caused by changing traf- fic conditions (e.g., traffic lights) and dynamic agents in the scene. For this benchmark, we propose different tasks and metrics than the original CARLA benchmark [12] to precisely measure specific reaction patterns that we know good drivers must master in urban conditions.

We propose three different tasks, each one corresponding to 25 goal directed episodes. In each episode, the agent starts at a random position and is directed by a high-level planner into reaching some goal position. The three tasks have the same set of start and end positions, as well as an increasing level of difficulty as follows:
1. Empty Town: no dynamic objects.
2. Regular Traffic: moderate number of cars and pedestrians.
3. Dense Traffic: large number of pedestrians and heavy traffic (dense urban scenario).

Similar to the CARLA Benchmark, NoCrash has six different weather conditions, where four were seen in training and two reserved for testing. It also has two different towns, one that is seen during training, and the other reserved for testing. For more details about the benchmark configuration, please refer to the supplementary material. As mentioned above, the measure of success of an episode should be more representative of the agent capabilities to react to dynamic objects. The original CARLA benchmark [12] has a goal conditioned success rate metric that is computed separately from a kilometers between infractions metric. The latter metric was proposed to be analogous to the one commonly used by real-world driving evaluations where the number of human interventions per kilometer is counted [18]. These interventions usually happen when the safety driver notices some inconsistent behavior that would lead the vehicle to a possibly dangerous state. On a potentially inconsistent behavior, the human intervention will put the vehicle back to a safe state. However, in the CARLA benchmark analysis, when an infraction is made, the episode continues after the infraction, leading to some inaccuracy in infraction counting. An example of inaccuracy includes whether a crash after leaving the road be counted as one or two infractions.

In NoCrash, instead of counting the number of infractions per kilometer, we end the episode as failing when any collision bigger than a fixed magnitude happens. With this limitation, we are setting a lower bound and have a guarantee of acceptable behaviors based on the measured percentage of success. Furthermore, this makes the evaluation even more similar to the km/interventions evaluation used in real world, since a new episode always sends the agent back to a safe starting state. In summary, we consider an episode to be successful if the agent reaches a certain goal under a time limit without colliding with any object. We also care about the ability of the agent to obey traffic rules. In particular, we measure and report the percentage of traffic light violations in Supplementary material. Note that an episode is not terminated when a traffic light violation occurs unless they are followed by a collision.

## 5. Experiments
In this section we detail our protocol for model training and briefly show that it is competitive with the state of the art. We also explore several corner cases to explore the limitations of the behavior cloning approach.

### 5.1. Training Details
First, we collected more than 400 hours of realistic simulated driving data from a single town of the CARLA environment using more than 200 GPU-days. We used an expert driving AI agent that leverages privileged information about the scene to drive naturally and well in complex conditions. After automatically filtering the data for simulation failures, duplicates, and edge cases using simple rules, we built a dataset of 100 hours of driving, called CARLA100.

To enable running a wide range of experiments, we train all methods using a subset of 10 hours of expert demonstrations by default. We also report larger scale training experiments and scalability analyses in Section 5.3 and in supplementary material. We will release the code for our demonstrator and our CARLA100 training dataset for reproducibility. More details about them are given in the supplementary material.

Training controllers on this dataset, we found that augmentation was not as crucial as reported by previous works [10, 25]. The only regularization we found important for performance was using a 50% dropout rate [40] after the last convolutional layer. Any larger dropout led us to under- fitting models. All models were trained using Adam [19] with minibatches of 120 samples and an initial learning rate of 0.0002. At each iteration, a minibatch is sampled randomly from the entire dataset and presented to the network for training. If we detect that the training error has not decreased for over 1, 000 iterations we divide the learning rate by 10. We used a 2 hours validation dataset to discover when to stop the training process. We validate every 20k iterations and if the validation error increases for three iterations we stop the training process and use this checkpoint to test on the benchmarks, both CARLA and NoCrash. We build a validation dataset as described in [9].

### 5.2. Comparison with the state of the art
We compare our results using both the original CARLA benchmark from [12] and our proposed NoCrash benchmark. We compare two versions of our method: “CILRS” (our CIL extension with a ResNet architecture and speed prediction, as described in section 3), and a version without the speed prediction branch noted “CILR”. We compare our method with the original CIL from [10] and three state-ofthe-art approaches: CAL [36], MT [25], and CIRL [26]. In contrast to end-to-end behavior cloning, these methods enforce some modularization that require extra information at training time, such as affordances (CAL), semantic segmentation (MT), or extra on-policy interaction with the environment (CIRL). Our approach only requires a fixed off-policy dataset of demonstrations.

We show results on the original CARLA benchmark [12] in Table 1 and results on our proposed NoCrash benchmark in Table 2. While most methods perform well in most conditions on the original CARLA benchmark, they all perform

Training conditions New town & weather

Task CIL[10] CIRL[26] CAL[36] MT[25] CILR CILRS CIL[10] CIRL[26] CAL[36] MT[25] CILR CILRS

Straight 98 98 100 96 94 96 80 98 94 96 92 96

One Turn 89 97 97 87 92 92 48 80 72 82 92 92

Navigation 86 93 92 81 88 95 44 68 68 78 88 92

Nav. Dynamic 83 82 83 81 85 92 42 62 64 62 82 90

Table 1. Comparison with the state of the art on the original CARLA benchmark. The “CILRS” version corresponds to our CIL-based

ResNet using the speed prediction branch, whereas “CILR” is without this speed prediction. These two models and CIL are the only ones that do not use any extra supervision or online interaction with the environment during training. The table reports the percentage of successfully completed episodes in each condition, selecting the best seed out of five runs.

Training conditions New Town & Weather

Task CIL[10] CAL[36] MT[25] CILR CILRS CIL[10] CAL[36] MT[25] CILR CILRS

Empty 79 ± 1 81 ± 1 84 ± 1 92 ± 1 97 ± 2 24 ± 1 25 ± 3 57 ± 0 66 ± 2 90 ± 2

Regular 60 ± 1 73 ± 2 54 ± 2 72 ± 5 83 ± 0 13 ± 2 14 ± 2 32 ± 2 54 ± 2 56 ± 2

Dense 21 ± 2 42 ± 3 13 ± 4 28 ± 1 42 ± 2 2 ± 0 10 ± 0 14 ± 2 13 ± 4 24 ± 8

Table 2. Results on our NoCrash benchmark. Mean and standard deviation on three runs, as CARLA 0.8.4 has significant non-determinism. significantly worse on NoCrash, especially when trying to generalize to new conditions. This confirms the usefulness of NoCrash in terms of exploring the limitations of driving policy learning due to its more challenging nature.

In addition, our proposed CILRS model significantly improves over the state of the art, e.g., +9% and +26% on

CARLA “Nav. Dynamic” in training and new conditions respectively, +10% and +24% on NoCrash Regular traffic in training and new conditions respectively. The significant improvements in generalization conditions, both w.r.t. CIL and mediated approaches, confirm that our improved endto-end behavior cloning architecture can effectively learn complex general policies from demonstrations alone. Furthermore, our ablative analysis shows that speed prediction is helpful: CILR can indeed be up to −14% worse than

CILRS on NoCrash.

### 5.3. Analysis of Limitations
Although clearly above the state of the art, our improved

CILRS architecture nonetheless sees a strong degradation of performance similar to all other methods in the presence of challenging driving conditions. We investigate how this degradation relates to the limitations of behavior cloning mentioned in Section 3.2 by using the NoCrash benchmark, in particular to better evaluate the interaction of the agents with dynamic objects.

Generalization in the presence of dynamic objects.

Limited generalization was previously reported for end-toend driving approaches [12]. In our experiments, we observed additional, and more prominent, generalization issues when the control policies have to deal with dynamic objects. Table 2 indeed shows a large drop in performance as we change to tasks with more traffic, e.g., −55% and −66% from Empty to Dense traffic in NoCrash training / new conditions respectively. In contrast, results in Empty town only degrade by −7% when changing to a new environment and weather. Therefore, the learned policies have a much harder time dealing robustly with a large number of vehicles and pedestrians. Furthermore, this impacts all policy learning methods, including those using additional supervision or on-policy demonstrations, often even more than our proposed CILRS method.

Driving Dataset Biases. Figure 3 evaluates the effect of the amount of training demonstrations on the learned policy. Here we compare models trained with 2, 10, 50 and 100 hours of demonstrations. The plots show the mean success rate and standard deviation over four different training cycles with different random seeds. Our best results on most of the scenarios were obtained by using only 10 hours of training data, in particular on the “Dense Traffic” tasks and novel conditions such as New Weather and New Town.

These results quantify a limitation described in Section 3.2: the risk of overfitting to data that lacks diversity.

This is here exacerbated by the limited spatial extent and visual variety of our environment, including in terms of dynamic objects. We indeed observed that some types of vehicles tend to elicit better reactions from the policy than others. The more common the vehicle model and color, the better the trained agent reacts to it. This raises ethical challenges in automated driving, requiring further research in fair machine learning for decision-making systems [4].

Causal confusion and the inertia problem. The main problem we observe caused by bias is the inertia problem stemming from causal confusion, as detailed in Section 3.2.

Figure 3. Due to biases in the data, the results may get either saturated or worse with increasing amounts of training data.

Figure 4. The percentage of episodes that failed due to the inertia problem. We can see that by increasing the amount of data, this bias may further degrade the generalization capabilities of the models.

Figure 5. Comparison between the results with and without the speed prediction and different amounts of training demonstrations.

We report the results only for the case were highest generalization is needed (New Weather and Town).

Figure 4 shows the percentage of episodes that failed due to the agent staying still, without any intention to use the throttle, for at least 8 seconds before the timeout. Our results show the percentage of episodes failed due to that inertia problem increases with the amount of data used for training. We proposed to use a speed prediction branch as part of our CILRS model (cf. Figure 2) to mitigate this problem. Figure 5 shows the percentage of successes for the

New Weather & Town conditions on different tasks with and without speed prediction. We observe that the speed prediction branch can substantially improve the success rate thanks to its regularization effect. It is, however, not a final solution to this problem, as we still observe instances of the inertia problem after using this approach.

High Variance. Repeatability of the training process is crucial for enhancing trust in end-to-end models. Unfortunately, we can still see drastic changes in the learned policy performance due to the variance caused by initialization and data sampling (cf. Section 3.2). Figure 6 compares the cause of episode termination for two models where the only difference is the random seed during training. The Model

S1 has a much higher chance of ending episodes due to vehicle collisions. Qualitatively, it seemed to have learned a less general braking policy and was more prone to rear-end collisions with other vehicles. On the other hand, Model S2 is able to complete more episodes and is less likely to fail due to vehicle crashes. However, we can see that it times out more, showing a tendency to stop a lot, even in non threatening situations. This can be seen by analyzing the histograms of the throttle applied by both models during the benchmark, as shown in Figure 7. We can see a tendency

Figure 6. Cause of episode termination on NoCrash for two CILRS models (trained on 10 hours with ImageNet initialization) with identical parameters but different random seeds. The episodes were ran under “New Weather & Town” conditions of the “Dense Traffic” task. for throttles of higher magnitude on Model S1.

Figure 7. Probability distribution of having certain throttle values comparing models with two different random seeds but trained with the same hyper-parameters and data. We can see that S1 (red) is much more likely to have a higher throttle value.

As off-policy imitation learning uses a static dataset for training, this randomness comes from the order in which training data is sampled and the initialization of the random weights. This can possibly define which minima the models converges to. Table 3 quantifies the effect of initialization on the success rate of driving tasks by computing the variance expressed in Equation 2. The expected policy score was computed by averaging twelve different training runs.

We also consider the variance with and without ImageNet initialization. We can see that the success rate can change by up to 42% for tasks with dynamic objects. ImageNet initialization tends to reduce the training variability, mainly due to smaller randomness on initialization but also due to a more stable learned policy.

Task Variance

CILRS

Empty 23%

Regular 26%

Dense 42%

CILRS (ImageNet)

Empty 4%

Regular 12%

Dense 38%

Table 3. Estimated variance of the success rate of CILRS on

NoCrash computed by training 12 times the same model with different random seeds. The variance is reduced by fixing part of the initial weights with ImageNet pre-training.

## 6. Conclusion
Our new driving dataset (CARLA100), benchmark (NoCrash), and end-to-end sensorimotor architecture (CILRS) indicate that behavior cloning on large scale offpolicy demonstration datasets can vastly improve over the state of the art in terms of generalization performance, including mediated perception approaches with additional supervision. This is thanks to using a deeper residual architecture with an additional speed prediction target and good regularization.

Nonetheless, our extensive experimental analysis has shown that some big challenges remain open. First of all, the amount of dynamic objects in the scene directly hurts all policy learning methods, as multi-agent dynamics are not directly captured. Second, the self-supervised nature of behavior cloning enables it to scale to large datasets of demonstrations, but with diminishing returns (or worse) due to driving-specific dataset biases that require explicit treatment, in particular biases that create causal confusion (e.g., the inertia problem). Third, the large variance resulting from initialization and sampling order indicates that running multiple runs on the same off-policy data is key to identify the best possible policies. This is part of the broader deep learning challenges regarding non-convexity and initialization, curriculum learning and training stability.

We will release CARLA100, the code of our demonstrator AI and CILRS model, as well as our NoCrash benchmark to stimulate future research on these topics.

## 7. Acknowledgements
Felipe Codevilla was supported in part by FI grant 2017FI-B1-00162. Antonio M. L´opez and Felipe Codevilla acknowledges the financial support by the Spanish TIN2017-88709-R (MINECO/AEI/FEDER, UE). Antonio M. Lpez also acknowledges the financial support by ICREA under the ICREA Academia Program. As CVC/UAB researchers, Antonio and Felipe also acknowledge the Generalitat de Catalunya CERCA Program and its ACCIO agency. We also thank the generous TRI support in AWS instances to run the additional experiments after Felipe’s internship. Special thanks to Yi Xiao for all the help on making the video.

## References
1. P. Abbeel, A. Coates, M. Quigley, and A. Y. Ng. An application of reinforcement learning to aerobatic helicopter flight. In Proceedings of the 19th International Conference on Neural Information Processing Systems, NIPS’06, pages 1–8, Cambridge, MA, USA, 2006. MIT Press. 1, 2
2. P. Abbeel and A. Y. Ng. Apprenticeship learning via inverse reinforcement learning. In International Conference on Machine Learning (ICML), 2004. 1
3. M. Andrychowicz, F. Wolski, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, O. P. Abbeel, and W. Zaremba. Hindsight experience replay. In Advances in Neural Information Processing Systems, pages 5048–5058, 2017. 2
4. S. Barocas, M. Hardt, and A. Narayanan. Fairness in machine learning. 2, 6
5. Y. Bengio, J. Louradour, R. Collobert, and J. Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48. ACM, 2009. 2
6. M. Bojarski, D. D. Testa, D. Dworakowski, B. Firner, B. Flepp, P. Goyal, L. D. Jackel, M. Monfort, U. Muller, J. Zhang, X. Zhang, J. Zhao, and K. Zieba. End to end learning for self-driving cars. arXiv:1604.07316, 2016. 1, 2, 4
7. L. Bottou and O. Bousquet. The tradeoffs of large scale learning. In Advances in neural information processing systems, pages 161–168, 2008. 3
8. C. Chen, A. Seff, A. Kornhauser, and J. Xiao. Deepdriving: Learning affordance for direct perception in autonomous driving. In Proceedings of the IEEE International Conference on Computer Vision, pages 2722–2730, 2015. 1, 2
9. F. Codevilla, A. M. Lopez, V. Koltun, and A. Dosovitskiy. On offline evaluation of vision-based driving models. In European Conference on Computer Vision (ECCV), pages 236–251, 2018. 2, 4, 5
10. F. Codevilla, M. M¨uller, A. L´opez, V. Koltun, and A. Dosovitskiy. End-to-end driving via conditional imitation learning. In International Conference on Robotics and Automation (ICRA), 2018. 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16
11. E. D. Dickmanns. The development of machine vision for road vehicles in the last decade. In Intelligent Vehicle Symposium, 2002. IEEE, volume 1, pages 268– 281. IEEE, 2002. 1
12. A. Dosovitskiy, G. Ros, F. Codevilla, A. L´opez, and V. Koltun. CARLA: An open urban driving simulator. In Conference on Robot Learning (CoRL), 2017. 2, 4, 5, 6, 14
13. A. Geiger, P. Lenz, and R. Urtasun. Are we ready for autonomous driving? The KITTI vision benchmark suite. In Computer Vision and Pattern Recognition (CVPR), 2012. 2
14. P. Hamm, D. Jayaraman, and S. Levine. Causal confusion in imitation learning. In ”Neural Information Processing Systems Imitation Learning and its Challenges in Robotics Workshop (NeurIPS ILR), 2018. 3
15. K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 4, 14
16. S. Hecker, D. Dai, and L. Van Gool. End-to-end learning of driving models with surround-view cameras and route planners. In The European Conference on Computer Vision (ECCV), September 2018. 2
17. P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger. Deep reinforcement learning that matters. In Thirty-Second AAAI Conference on Artifi- cial Intelligence, 2018. 2
18. N. Kalra and S. M. Paddock. Driving to safety: How many miles of driving would it take to demonstrate autonomous vehicle reliability? Transportation Research Part A: Policy and Practice, 94:182 – 193, 2016. 5
19. D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representation (ICLR), 2015. 5
20. M. Laskey, A. Dragan, J. Lee, K. Goldberg, and R. Fox. Dart: Optimizing noise injection in imitation learning. In Conference on Robot Learning (CoRL), 2017. 2, 12
21. Y. LeCun, U. Muller, J. Ben, E. Cosatto, and B. Flepp. Off-road obstacle avoidance through end-to-end learning. In Neural Information Processing Systems (NIPS), 2005. 2
22. J. Leonard, J. How, S. Teller, M. Berger, S. Campbell, G. Fiore, L. Fletcher, E. Frazzoli, A. Huang, S. Karaman, et al. A perception-driven autonomous urban vehicle. Journal of Field Robotics, 25(10):727–774, 2008. 1
23. S. Levine, P. Pastor, A. Krizhevsky, and D. Quillen. Learning hand-eye coordination for robotic grasping with large-scale data collection. In International Symposium on Experimental Robotics (ISER), 2017. 1, 2
24. L. Li, Z. Liu, O. Ozgner, J. Lian, Y. Zhou, and Y. Zhao. Dense 3D semantic slam of traffic environment based on stereo vision. In Intelligent Vehicles Symposium (IV), 2018. 2
25. Z. Li, T. Motoyoshi, K. Sasaki, T. Ogata, and S. Sugano. Rethinking self-driving: Multi-task knowledge for better generalization and accident explanation ability. arXiv preprint arXiv:1809.11100, 2018. 2, 5, 6, 13
26. X. Liang, T. Wang, L. Yang, and E. Xing. Cirl: Controllable imitative reinforcement learning for visionbased self-driving. In European Conference on Computer Vision (ECCV), pages 584–599, 2018. 2, 5, 6, 13
27. A. Lopez, G. Villalonga, L. Sellart, G. Ros, D. Vzquez, J. Xu, J. Marin, and A. Mozafari. Training my car to see using virtual worlds. 2
28. M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. Hausknecht, and M. Bowling. Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents. Journal of Artificial Intelligence Research, 61:523–562, 2018. 2
29. M. M¨uller, A. Dosovitskiy, B. Ghanem, and V. Koltun. Driving policy transfer via modularity and abstraction. arXiv preprint arXiv:1804.09364, 2018. 1, 2
30. B. Neal, S. Mittal, A. Baratin, V. Tantia, M. Scicluna, S. Lacoste-Julien, and I. Mitliagkas. A modern take on the bias-variance tradeoff in neural networks. arXiv preprint arXiv:1810.08591, 2018. 2, 3, 4
31. D. Pomerleau. ALVINN: An autonomous land vehicle in a neural network. In Neural Information Processing Systems (NIPS), 1988. 2
32. N. D. Ratliff, J. A. Bagnell, and S. S. Srinivasa. Imitation learning for locomotion and manipulation. In International Conference on Humanoid Robots, 2007. 1
33. S. Ross and D. Bagnell. Efficient reductions for imitation learning. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 661–668, 2010. 2
34. S. Ross, G. J. Gordon, and J. A. Bagnell. A reduction of imitation learning and structured prediction to noregret online learning. In AISTATS, 2011. 1, 2, 3
35. E. Santana and G. Hotz. Learning a driving simulator. arXiv:1608.01230, 2016. 2
36. A. Sauer, N. Savinov, and A. Geiger. Conditional affordance learning for driving in urban environments. arXiv preprint arXiv:1806.06498, 2018. 2, 4, 5, 6, 13
37. S. Schaal, A. J. Ijspeert, and A. Billard. Computational approaches to motor learning by imitation. Philosophical Transactions of the Royal Society B, 358(1431), 2003. 2
38. D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, et al. Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587), 2016. 1
39. S. P. Soundararaj, A. K. Sujeeth, and A. Saxena. Autonomous indoor helicopter flight using a single onboard camera. In 2009 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5307–5314. IEEE, 2009. 2
40. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929–1958, 2014. 5
41. A. Torralba and A. Efros. Unbiased look at dataset bias. In Proceedings of the 2011 IEEE Conference on Computer Vision and Pattern Recognition, pages 1521–1528. IEEE Computer Society, 2011. 2
42. Q. Wang, L. Chen, and W. Tian. End-to-end driving simulation via angle branched network. arXiv:1805.07545, 2018. 1
43. H. Xu, Y. Gao, F. Yu, and T. Darrell. End-to-end learning of driving models from large-scale video datasets. In Computer Vision and Pattern Recognition (CVPR), 2017. 2
44. L. Yang, X. Liang, T. Wang, and E. Xing. Real-tovirtual domain unification for end-to-end autonomous driving. In The European Conference on Computer Vision (ECCV), September 2018. 2
45. J. Zhang and K. Cho. Query-efficient imitation learning for end-to-end simulated driving. In AAAI, 2017. 2

## Appendix

### A. CARLA100
Here we describe the content of the CARLA100 dataset.

Note that for training our model we only used RGB sensor data, the ego-vehicle forward speed, the high level turn intentions for the conditional imitation learning and the ego vehicle controls.

#### A.1. Expert Demonstrator
We collect the dataset, here referred as CARLA100, by executing an automated navigation expert in the simulated environment. The expert has access to privileged information about the simulation state, including the exact map of the environment and the exact positions of the ego-car, all other vehicles, and pedestrians.

The path driven by the expert is calculated using a standard planner. This planner uses an A* algorithm to determine the path to reach a certain goal. This path is then converted into waypoints used by a PID controller to generate the throttle, brake, and steering for the expert demonstrator.

The expert drives steadily on the center of the lane, keeping a constant speed of 35 km/h when driving straight and reducing the speed when making turns to about 15 Km/h.

In addition, the expert is programmed to react to visible pedestrians when required to prevent collisions. The expert reduces its speed proportionally to the collision distance when the pedestrian is over 5 meters away and less than 15 meters away, or breaking to full stop when the pedestrian is less than 5 meters away.

The proposed demonstrator also reduces its speed to follow lead cars. The expert stops when the leading vehicle is closer than 5 meters. For our data collection process the expert never performs lane changes or overtakes.

To improve diversity, realism, and increase the number of visited state-action pairs, we add noise to the ego car controls. This reduces the difference between offline training and online testing scenarios [20]. We input noise to the expert demonstrator in a similar way as proposed by [10].

The noise simulates a gradual drift away from the desired trajectory of the experts. However, for training, the drift is not used, but only the reactions performed by the expert to correct the path. The added noise signal is detailed on Section A.3

#### A.2. Content
The dataset collection is divided into goal directed episodes where the expert goes from a start position into a goal position while stopping to avoid collisions with dynamic obstacles. In total, we collected 2373 episodes with different characteristics. The entire dataset was collected on

Town01. Each episode has the following features: 
* Number of Pedestrians: the total number of spawned pedestrians around the town. This number is randomly sampled from the interval [50, 100]. 
* Number of Vehicles: the total number of spawned vehicle around the town. This number is randomly sampled from the interval [30, 70]. 
* Spawned seed for pedestrians and vehicles: the random seed used for the CARLA object spawning process. 
* Weather: the weather used for the episode is sampled from the set: Clear Noon, Clear Noon After Rain, Heavy Rain Noon, Clear Sunset.

Each episode last from 1 to 5 minutes partitioned in simulation steps of 100 ms. For each step, we store data divided into two different categories: sensor data is stored as PNG images, and measurement data is stored as json files.

For the sensor data we have the different camera sensors used: RGB camera, and depth camera, and semantic segmentation pseudo sensor. For each sensor we record data in three positions: aligned with the car center, rotated 30 degrees to the left and rotated 30 degrees to the right.

As measurements, we have data measured from the egovehicle, the world status, and from all the other non player agents. The following data was collected from the egovehicle and the world status: 
* Step Number: the simulation step that starts at zero and is incremented by one for every 100ms in game time. 
* Game Time-stamp: the time that has passed since the simulation has started. 
* Position: the world position of the ego-vehicle. It is expressed as a three dimensional vector [x, y, z] in meters. 
* Orientation: the orientation of the vehicle with respect to the world expressed as Euler Angles (row, pitch and yaw). 
* Acceleration: the acceleration vector of the egovehicle with respect to the world. 
* Forward Speed: the scalar speed of the ego vehicle in the forward direction of movement. 
* Intentions: a signal that is proportional to the effect that the dynamic objects in the scene are having in the ego car actions. We use three different intention signals: stopping for pedestrians, stopping for cars and stopping for traffic lights. For example, an intention of 1 for stopping for pedestrian means that the ego car



Table 4. Comparison with the state of the art on the original CARLA benchmark for the conditions “New Town” and “New Weather”. The table reports the percentage of successfully completed episodes in each condition.


Table 5. Comparison with the state-of-the-art on the NoCrash Benchmark. Here we compare on two extra conditions. New Weather refers to the same town as during training but with new weather conditions. New Town refers to a town not seen during training. totally stopped for a pedestrian that is less than 5 meters away. An intention of the same class of 0.5 means that the expert noticed a pedestrians and has reduced its speed to a certain extent. An intention of 0 means there are no pedestrians nearby in the field of view of the expert. 
* High Level Commands: the high level indication stating what the ego-vehicle should do in the next intersection: go straight, turn left, turn right, or do not care. Each of these commands are encoded as a integer number. 2 is do not care, 3 for turn left, 4 for turn right, 5 for go straight. 
* Waypoints: a set containing the next 10 future positions the vehicle should assume. This is calculated with the path planning algorithm. 
* Steering Angle: the current steering angle of the vehicle’s steering wheel. 
* Throttle: the current pressure on the throttle pedal. 
* Brake: the current pressure on the brake pedal. 
* Hand Brake: if the hand brake is activated not. 
* Steer Noise: the current steering angle of the vehicle considering the noise function. 
* Throttle Noise: the current pressure in the throttle pedal considering the noise function. 
* Brake Noise: the current pressure in the brake pedal considering the noise function. The noise function is described in Section A.3

For each of the non-player agents (pedestrians, vehicles, traffic light), the following information is provided: 
* Unique ID: an unique identifier of this agent. 
* Type: if it is a pedestrian, a vehicle or a traffic light. 
* Position: the world position of the agent. It is expressed as a three dimensional vector [x, y, z] in meters. 
* Orientation: the orientation of the agent with respect to the world. Expressed as Euler angles (row, pitch and yaw). 
* Forward Speed: the scalar speed of the agent in the forward direction of movement. 
* State: only for traffic lights. Contains the state of the traffic light: either red, yellow or green.

#### A.3. Noise Distribution
During training data collection, 20% of the time we injected noise into expert’s steering signal. Namely, at random point in time we added a perturbation to the steering angle provided by the driver. The perturbation is a triangular impulse: it increases linearly, reaches the maximum value and then linearly declines. This simulates smooth drift from the desired trajectory, similar to what might happen with a poorly trained controller. The triangular impulse is parametrized by its starting time t0, duration τ ∈ <+, sign σ ∈ {−1, +1} and intensity γ ∈ <+: sperturb(t) = σγ max  0,  1 − 2(t − t0) τ − 1  . (3)

Every second of driving we started a perturbation with probability pperturb. We used pperturb = 0.1 in our experiments. The sign of each perturbation was sampled at random, the duration was sampled uniformly from 0.5 to 2 seconds, the intensity was fixed to 0.15.

#### A.4. NoCrash Benchmark
The benchmark consist of three different tasks: “Empty”, “Regular” and “Cluttered”. The tasks are better explained on section 3 of the main text. Each task consists of 25 goal directed episodes. In each episode, the agent is guided with a global planner to reach a certain goal position. We consider an episode as a success if the agent reaches a certain goal under a time limit without colliding with any object, static or dynamic. The tuples of start/goal positions are based on the ones used in CARLA CoRL2017 benchmark for the tasks “Navigation” and “Nav. Dynamic”.

However, we removed some start-goal positions that were too close to each other. Figure 8 shows the start-goad positions for both Towns.

The benchmark is executed under four different conditions: 
* Training: The same one as collected on the training data. As mentioned above, we collected training data only in Town01 and using the weather conditions: “Clear Noon”, “Clear Noon After Rain”, “Heavy Rain Noon”, “Clear Sunset’ 
* New weather: The city as in the training data but with two different new weathers, “After Rain Sunset” and “Soft Rain Sunset”. 
* New Town: Same weathers as in “Training” but the tests take place in Town02. 
* New Town & Weather: Same weathers as in “New Weather” but played in Town 02.

### B. Training Details
#### B.1. Architecture
Table 6 details the standard architecture used in the experiments. For the perception module, we also experimented with ResNet 18, ResNet 50 and with the architecture proposed in [10].

#### B.2. Image Input
Starting from a raw 800 × 600 pixels image, we cropped 125 pixels from the top and 90 at the bottom of the image and resized the resulting image to 200 × 88 pixels.

### C. Additional Results
Comparison Further comparisons with the state-of-theart in the CARLA CoRL 2017 benchmark can be seen in Table 4. We can see that the “New Town” condition is harder for some models than the “New Weather & Town” conditions. Yet the proposed methods can still outperform previously proposed methods. One would expect “New Weather & Town” to be the hardest task, but this discrepancy had been previously observed in the literature [12]. module input dimension channels



Table 6. Exact configurations of the architecture used on the experiments.

Changing Architecture In Figure 12, we compare the results of the 8 layer convolutional model used by Codevilla et. al. [10] and several new ResNet based configurations using our new dataset. First, we noticed that the 8 convolutions model obtained worse results than the ones reported as

CIL at Table 1. This happened since we trained the 8 convolutions architecture with the more complex CARLA100 dataset. The model did not have enough capacity to capture the more complex actions and only fitted the several segments where the demonstrator stands still in front of traffic lights. This shows that higher capacity models are able to better learn different sub-tasks. However, the results get worse for the deeper ResNet50 based models, showing that there is still possibility for different types of overfitting.

ImageNet Initialization We also observed a considerable change in the success rate results when not using ImageNet initialization, as show in Figure 11. Without Imagnet initialization, the highest success rates obtained by models trained on 100 hours of demonstrations. However, these later results are still below what can be achieved with less data and

ImageNet pre-training, specially on the dense traffic tasks.

#### C.1. Reacting to Traffic Lights
We show that interesting policies around traffic lights emerged for some of the models we trained. In Table 7 we show the percentage of traffic lights that were crossed on green light for different models. This number is computed for the “Empty Town” task from the dynamic urban scenarios benchmark. The original CIL model trained with older data [10] represents an effective policy that was trained without demonstrations of stopping for red traffic light, so its number can be seen as a lower bound. The 8 convolutions is a model with the same architecture as the CIL model but trained to react to traffic lights with the proposed new dataset. We can see that this model was more reactive to traffic lights, but very poorly. On the other hand, our best 

Figure 8. The start and goal positions for the CARLA100 Benchmark. The start positions are in red and the goal positions are in green.

Same number correspond to matching start-goal positions. (a) Input Image (b) Layer 1 (c) Layer 2

Figure 9. Activation maps showing the increased selectivity for traffic lights in the ResNet34 case (bottom) compared to the standard 8 convolution architecture (top). For the ResNet34, layer 1, refers to the attention maps obtained after a full ResNet block.


Table 7. Percentage of times the agents crossed a traffic light on red (lower is better) in the “Empty” conditions of the NoCrash benchmark. model, having only 47% of traffic light violations, is clearly stopping for a significant amount of red traffic lights. This result is even more expressive considering the version using 100 hours of training data which did only 27% of traffic light violations. However, when we analyze generalization conditions, Tab. 7 bottom, we see there is an ample room for improvement. Regardless, such improvement in this longitudinal controls task is promising for modeling lateral and longitudinal controls jointly end-to-end.

#### C.2. Main Causes of Failure
On Table 8 we show some of our models compared with some of the literature with regard to their cause of failure.

We specify the percentage of episodes that ended due to different causes of crash, due to timeout of the task or if the main cause is that the controller stopped and never resumed moving again (i.e the inertia problem).

Figure 10. Percentage of episodes ended by the “inertia problem” under different conditions. We report the mean and the standard deviation over four different training runs. We compare models with different amounts of training data and without image-net pre-training. We can see that the inertia problem becomes more prominent with more data.

Figure 11. The importance of data and initialization without ImageNet pre-training. We can see that the overall results improve with more data but not significantly. We can also see a case of worse performance when changing the ammount of training data from 50 to 100 hours under the New Weather & Town conditions with Dense Traffic.

Figure 12. Ablative analysis between different architectures. The eight convolutions architecture,“8conv”, proposed by Codevilla [10] obtained poor results on the more complex CARLA100 benchmark. ResNet based deeper architectures, “res18” and “res34”, were able to improve the results. However, when testing ResNet 50 we notice a significant drop in the quality of the results.

Table 8. Analysis of the causes of episode termination for different methods. We show the results for all tasks and weather conditions.

The columns for a single method/task/condition should add up to 1. For each cause of episode termination we highlight the method with higher probability (i.e. worse performance). For the success row, we highlight the method with the best performance. The reported results are the average over three different runs of the benchmark.
