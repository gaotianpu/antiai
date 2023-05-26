# Trust Region Policy Optimization
https://arxiv.org/abs/1502.05477

## 阅读笔记
* https://spinningup.openai.com/en/latest/algorithms/trpo.html

## Abstract
We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified procedure, we develop a practical algorithm, called Trust Region Policy

Optimization (TRPO). This algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks. Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits; and playing Atari games using images of the screen as input. Despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters.

## 1 Introduction
Most algorithms for policy optimization can be classified into three broad categories: (1) policy iteration methods, which alternate between estimating the value function under the current policy and improving the policy (Bertsekas, 2005); (2) policy gradient methods, which use an estimator of the gradient of the expected return (total reward) obtained from sample trajectories (Peters & Schaal, 2008a) (and which, as we later discuss, have a close connection to policy iteration); and (3) derivative-free optimization methods, such as the cross-entropy method (CEM) and covariance matrix adaptation (CMA), which treat the return as a black box function to be optimized in terms of the policy parameters (Szita & L¨orincz, 2006).

General derivative-free stochastic optimization methods such as CEM and CMA are preferred on many problems, because they achieve good results while being simple to understand and implement. For example, while

Proceedings of the 31 st International Conference on Machine

Learning, Lille, France, 2015. JMLR: W&CP volume 37. Copyright 2015 by the author(s).

Tetris is a classic benchmark problem for approximate dynamic programming (ADP) methods, stochastic optimization methods are difficult to beat on this task (Gabillon et al., 2013). For continuous control problems, methods like CMA have been successful at learning control policies for challenging tasks like locomotion when provided with hand-engineered policy classes with low-dimensional parameterizations (Wampler & Popovi´c, 2009). The inability of ADP and gradient-based methods to consistently beat gradient-free random search is unsatisfying, since gradient-based optimization algorithms enjoy much better sample complexity guarantees than gradient-free methods (Nemirovski, 2005). Continuous gradient-based optimization has been very successful at learning function approximators for supervised learning tasks with huge numbers of parameters, and extending their success to reinforcement learning would allow for efficient training of complex and powerful policies.

In this article, we first prove that minimizing a certain surrogate objective function guarantees policy improvement with non-trivial step sizes. Then we make a series of approximations to the theoretically-justified algorithm, yielding a practical algorithm, which we call trust region policy optimization (TRPO). We describe two variants of this algorithm: first, the single-path method, which can be applied in the model-free setting; second, the vine method, which requires the system to be restored to particular states, which is typically only possible in simulation. These algorithms are scalable and can optimize nonlinear policies with tens of thousands of parameters, which have previously posed a major challenge for model-free policy search (Deisenroth et al., 2013). In our experiments, we show that the same TRPO methods can learn complex policies for swimming, hopping, and walking, as well as playing Atari games directly from raw images. 2 Preliminaries

Consider an infinite-horizon discounted Markov decision process (MDP), defined by the tuple (S, A, P, r, ρ0, γ), where S is a finite set of states, A is a finite set of actions,

P : S × A × S → R is the transition probability distriarXiv:1502.05477v5 [cs.LG] 20 Apr 2017

Trust Region Policy Optimization bution, r : S → R is the reward function, ρ0 : S → R is the distribution of the initial state s0, and γ ∈ (0, 1) is the discount factor.

Let π denote a stochastic policy π : S × A → [0, 1], and let η(π) denote its expected discounted reward: η(π) = Es0,a0,... " ∞X t=0 γ t r(st)

# , where
 s0 ∼ ρ0(s0), at ∼ π(at|st), st+1 ∼ P(st+1|st, at).

We will use the following standard definitions of the stateaction value function Qπ, the value function Vπ, and the advantage function Aπ:

Qπ(st, at) = Est+1,at+1,... " ∞X l=0 γ l r(st+l)

# ,
Vπ(st) = Eat,st+1,... " ∞X l=0 γ l r(st+l)

# ,
Aπ(s, a) = Qπ(s, a) − Vπ(s), where at ∼ π(at|st), st+1 ∼ P(st+1|st, at) for t ≥ 0.

The following useful identity expresses the expected return of another policy ˜π in terms of the advantage over π, accumulated over timesteps (see Kakade & Langford (2002) or

Appendix A for proof): η(˜π) = η(π) + Es0,a0,···∼˜π " ∞X t=0 γ tAπ(st, at)

# (1)
 where the notation Es0,a0,···∼˜π [. . . ] indicates that actions are sampled at ∼ ˜π(·|st). Let ρπ be the (unnormalized) discounted visitation frequencies ρπ(s)=P(s0 = s)+γP(s1 = s)+γ 2P(s2 = s)+. . . , where s0 ∼ ρ0 and the actions are chosen according to π.

We can rewrite Equation (1) with a sum over states instead of timesteps: η(˜π) = η(π) + ∞X t=0

X s

P(st = s|˜π)

X a ˜π(a|s)γ tAπ(s, a) = η(π) +X s ∞X t=0 γ tP(st = s|˜π)

X a ˜π(a|s)Aπ(s, a) = η(π) +X s ρ˜π(s)

X a ˜π(a|s)Aπ(s, a). (2)

This equation implies that any policy update π → ˜π that has a nonnegative expected advantage at every state s, i.e., P a ˜π(a|s)Aπ(s, a) ≥ 0, is guaranteed to increase the policy performance η, or leave it constant in the case that the expected advantage is zero everywhere. This implies the classic result that the update performed by exact policy iteration, which uses the deterministic policy ˜π(s) = arg maxa Aπ(s, a), improves the policy if there is at least one state-action pair with a positive advantage value and nonzero state visitation probability, otherwise the algorithm has converged to the optimal policy. However, in the approximate setting, it will typically be unavoidable, due to estimation and approximation error, that there will be some states s for which the expected advantage is negative, that is, P a ˜π(a|s)Aπ(s, a) < 0. The complex dependency of ρ˜π(s) on ˜π makes Equation (2) difficult to optimize directly. Instead, we introduce the following local approximation to η:

Lπ(˜π) = η(π) +X s ρπ(s)

X a ˜π(a|s)Aπ(s, a). (3)

Note that Lπ uses the visitation frequency ρπ rather than ρ˜π, ignoring changes in state visitation density due to changes in the policy. However, if we have a parameterized policy πθ, where πθ(a|s) is a differentiable function of the parameter vector θ, then Lπ matches η to first order (see Kakade & Langford (2002)). That is, for any parameter value θ0,

Lπθ0 (πθ0 ) = η(πθ0 ), ∇θLπθ0 (πθ)   θ=θ0 = ∇θη(πθ)   θ=θ0 . (4)

Equation (4) implies that a sufficiently small step πθ0 → ˜π that improves Lπθold will also improve η, but does not give us any guidance on how big of a step to take.

To address this issue, Kakade & Langford (2002) proposed a policy updating scheme called conservative policy iteration, for which they could provide explicit lower bounds on the improvement of η. To define the conservative policy iteration update, let πold denote the current policy, and let π 0 = arg maxπ0 Lπold (π 0 ). The new policy πnew was defined to be the following mixture: πnew(a|s) = (1 − α)πold(a|s) + απ0 (a|s). (5)

Kakade and Langford derived the following lower bound: η(πnew)≥Lπold (πnew) − 2γ (1 − γ) 2 α 2 where  = max s  

Ea∼π0 (a|s) [Aπ(s, a)] . (6) (We have modified it to make it slightly weaker but simpler.) Note, however, that so far this bound only applies to mixture policies generated by Equation (5). This policy class is unwieldy and restrictive in practice, and it is desirable for a practical policy update scheme to be applicable to all general stochastic policy classes. 3 Monotonic Improvement Guarantee for

General Stochastic Policies

Equation (6), which applies to conservative policy iteration, implies that a policy update that improves the right-hand

Trust Region Policy Optimization side is guaranteed to improve the true performance η. Our principal theoretical result is that the policy improvement bound in Equation (6) can be extended to general stochastic policies, rather than just mixture polices, by replacing α with a distance measure between π and ˜π, and changing the constant  appropriately. Since mixture policies are rarely used in practice, this result is crucial for extending the improvement guarantee to practical problems. The particular distance measure we use is the total variation divergence, which is defined by DT V (p k q) = 1 2

P i |pi − qi | for discrete probability distributions p, q. 1 Define Dmax

TV (π, ˜π) as

DTV max(π, ˜π) = max s

DT V (π(·|s) k ˜π(·|s)). (7)

Theorem 1. Let α = DTV max(πold, πnew). Then the following bound holds: η(πnew) ≥ Lπold (πnew) − 4γ (1 − γ) 2 α 2 where  = max s,a |Aπ(s, a)| (8)

We provide two proofs in the appendix. The first proof extends Kakade and Langford’s result using the fact that the random variables from two distributions with total variation divergence less than α can be coupled, so that they are equal with probability 1 − α. The second proof uses perturbation theory.

Next, we note the following relationship between the total variation divergence and the KL divergence (Pollard (2000), Ch. 3): DT V (p k q) 2 ≤ DKL(p k q). Let

DKL max(π, ˜π) = maxs DKL(π(·|s) k ˜π(·|s)). The following bound then follows directly from Theorem 1: η(˜π) ≥ Lπ(˜π) − CDKL max(π, ˜π), where C = 4γ (1 − γ) 2 . (9)

Algorithm 1 describes an approximate policy iteration scheme based on the policy improvement bound in Equation (9). Note that for now, we assume exact evaluation of the advantage values Aπ.

It follows from Equation (9) that Algorithm 1 is guaranteed to generate a monotonically improving sequence of policies η(π0) ≤ η(π1) ≤ η(π2) ≤ . . . . To see this, let Mi(π) =

Lπi (π) − CDKL max(πi , π). Then η(πi+1) ≥ Mi(πi+1) by Equation (9) η(πi) = Mi(πi), therefore, η(πi+1) − η(πi) ≥ Mi(πi+1) − M(πi). (10)

Thus, by maximizing Mi at each iteration, we guarantee that the true objective η is non-decreasing. This algorithm 1Our result is straightforward to extend to continuous states and actions by replacing the sums with integrals.

Algorithm 1 Policy iteration algorithm guaranteeing nondecreasing expected return η

Initialize π0. for i = 0, 1, 2, . . . until convergence do

Compute all advantage values Aπi (s, a).

Solve the constrained optimization problem πi+1 = arg max π [Lπi (π) − CDKL max(πi , π)] where C = 4γ/(1 − γ) 2 and Lπi (π)=η(πi)+X s ρπi (s)

X a π(a|s)Aπi (s, a) end for is a type of minorization-maximization (MM) algorithm (Hunter & Lange, 2004), which is a class of methods that also includes expectation maximization. In the terminology of MM algorithms, Mi is the surrogate function that minorizes η with equality at πi . This algorithm is also reminiscent of proximal gradient methods and mirror descent.

Trust region policy optimization, which we propose in the following section, is an approximation to Algorithm 1, which uses a constraint on the KL divergence rather than a penalty to robustly allow large updates. 4 Optimization of Parameterized Policies

In the previous section, we considered the policy optimization problem independently of the parameterization of π and under the assumption that the policy can be evaluated at all states. We now describe how to derive a practical algorithm from these theoretical foundations, under finite sample counts and arbitrary parameterizations.

Since we consider parameterized policies πθ(a|s) with parameter vector θ, we will overload our previous notation to use functions of θ rather than π, e.g. η(θ) := η(πθ),

Lθ(θ ˜) := Lπθ (πθ˜), and DKL(θ k θ ˜) := DKL(πθ k πθ˜). We will use θold to denote the previous policy parameters that we want to improve upon.

The preceding section showed that η(θ) ≥ Lθold (θ) −

CDKL max(θold, θ), with equality at θ = θold. Thus, by performing the following maximization, we are guaranteed to improve the true objective η: maximize θ [Lθold (θ) − CDKL max(θold, θ)] .

In practice, if we used the penalty coefficient C recommended by the theory above, the step sizes would be very small. One way to take larger steps in a robust way is to use a constraint on the KL divergence between the new policy and the old policy, i.e., a trust region constraint: maximize θ

Lθold (θ) (11) subject to DKL max(θold, θ) ≤ δ.

Trust Region Policy Optimization

This problem imposes a constraint that the KL divergence is bounded at every point in the state space. While it is motivated by the theory, this problem is impractical to solve due to the large number of constraints. Instead, we can use a heuristic approximation which considers the average KL divergence:

D ρ

KL(θ1, θ2) := Es∼ρ [DKL(πθ1 (·|s) k πθ2 (·|s))] .

We therefore propose solving the following optimization problem to generate a policy update: maximize θ

Lθold (θ) (12) subject to D ρθold

KL (θold, θ) ≤ δ.

Similar policy updates have been proposed in prior work (Bagnell & Schneider, 2003; Peters & Schaal, 2008b; Peters et al., 2010), and we compare our approach to prior methods in Section 7 and in the experiments in Section 8.

Our experiments also show that this type of constrained update has similar empirical performance to the maximum

KL divergence constraint in Equation (11). 5 Sample-Based Estimation of the Objective and Constraint

The previous section proposed a constrained optimization problem on the policy parameters (Equation (12)), which optimizes an estimate of the expected total reward η subject to a constraint on the change in the policy at each update. This section describes how the objective and constraint functions can be approximated using Monte Carlo simulation.

We seek to solve the following optimization problem, obtained by expanding Lθold in Equation (12): maximize θ

X s ρθold (s)

X a πθ(a|s)Aθold (s, a) subject to D ρθold

KL (θold, θ) ≤ δ. (13)

We first replace P s ρθold (s) [. . . ] in the objective by the expectation 1 1−γ

Es∼ρθold [. . . ]. Next, we replace the advantage values Aθold by the Q-values Qθold in Equation (13), which only changes the objective by a constant. Last, we replace the sum over the actions by an importance sampling estimator. Using q to denote the sampling distribution, the contribution of a single sn to the loss function is

X a πθ(a|sn)Aθold (sn, a) = Ea∼q  πθ(a|sn) q(a|sn)

Aθold (sn, a)  .

Our optimization problem in Equation (13) is exactly equivalent to the following one, written in terms of expectations: maximize θ

Es∼ρθold ,a∼q  πθ(a|s) q(a|s)

Qθold (s, a)  (14) subject to Es∼ρθold [DKL(πθold (·|s) k πθ(·|s))] ≤ δ. all state-action pairs used in objective trajectories s ann ρ 0 1 a2 sn rollout set two rollouts using CRN sampling trajectories ρ 0

Figure 1. Left: illustration of single path procedure. Here, we generate a set of trajectories via simulation of the policy and incorporate all state-action pairs (sn, an) into the objective. Right: illustration of vine procedure. We generate a set of “trunk” trajectories, and then generate “branch” rollouts from a subset of the reached states. For each of these states sn, we perform multiple actions (a1 and a2 here) and perform a rollout after each action, using common random numbers (CRN) to reduce the variance.

All that remains is to replace the expectations by sample averages and replace the Q value by an empirical estimate.

The following sections describe two different schemes for performing this estimation.

The first sampling scheme, which we call single path, is the one that is typically used for policy gradient estimation (Bartlett & Baxter, 2011), and is based on sampling individual trajectories. The second scheme, which we call vine, involves constructing a rollout set and then performing multiple actions from each state in the rollout set. This method has mostly been explored in the context of policy iteration methods (Lagoudakis & Parr, 2003; Gabillon et al., 2013).

## 5.1 Single Path
In this estimation procedure, we collect a sequence of states by sampling s0 ∼ ρ0 and then simulating the policy πθold for some number of timesteps to generate a trajectory s0, a0, s1, a1, . . . , sT −1, aT −1, sT . Hence, q(a|s) = πθold (a|s). Qθold (s, a) is computed at each state-action pair (st, at) by taking the discounted sum of future rewards along the trajectory.

## 5.2 Vine
In this estimation procedure, we first sample s0 ∼ ρ0 and simulate the policy πθi to generate a number of trajectories. We then choose a subset of N states along these trajectories, denoted s1, s2, . . . , sN , which we call the “rollout set”. For each state sn in the rollout set, we sample

K actions according to an,k ∼ q(·|sn). Any choice of q(·|sn) with a support that includes the support of πθi (·|sn) will produce a consistent estimator. In practice, we found that q(·|sn) = πθi (·|sn) works well on continuous problems, such as robotic locomotion, while the uniform distribution works well on discrete tasks, such as the Atari games, where it can sometimes achieve better exploration.

For each action an,k sampled at each state sn, we esti-

Trust Region Policy Optimization mate ˆQθi (sn, an,k) by performing a rollout (i.e., a short trajectory) starting with state sn and action an,k. We can greatly reduce the variance of the Q-value differences between rollouts by using the same random number sequence for the noise in each of the K rollouts, i.e., common random numbers. See (Bertsekas, 2005) for additional discussion on Monte Carlo estimation of Q-values and (Ng & Jordan, 2000) for a discussion of common random numbers in reinforcement learning.

In small, finite action spaces, we can generate a rollout for every possible action from a given state. The contribution to Lθold from a single state sn is as follows:

Ln(θ) =

K

X k=1 πθ(ak|sn) ˆQ(sn, ak), (15) where the action space is A = {a1, a2, . . . , aK}. In large or continuous state spaces, we can construct an estimator of the surrogate objective using importance sampling.

The self-normalized estimator (Owen (2013), Chapter 9) of Lθold obtained at a single state sn is

Ln(θ) =

P

K k=1 πθ(an,k|sn) πθold (an,k|sn) ˆQ(sn, an,k)

P

K k=1 πθ(an,k|sn) πθold (an,k|sn) , (16) assuming that we performed K actions an,1, an,2, . . . , an,K from state sn. This self-normalized estimator removes the need to use a baseline for the

Q-values (note that the gradient is unchanged by adding a constant to the Q-values). Averaging over sn ∼ ρ(π), we obtain an estimator for Lθold , as well as its gradient.

The vine and single path methods are illustrated in Figure 1.

We use the term vine, since the trajectories used for sampling can be likened to the stems of vines, which branch at various points (the rollout set) into several short offshoots (the rollout trajectories).

The benefit of the vine method over the single path method that is our local estimate of the objective has much lower variance given the same number of Q-value samples in the surrogate objective. That is, the vine method gives much better estimates of the advantage values. The downside of the vine method is that we must perform far more calls to the simulator for each of these advantage estimates. Furthermore, the vine method requires us to generate multiple trajectories from each state in the rollout set, which limits this algorithm to settings where the system can be reset to an arbitrary state. In contrast, the single path algorithm requires no state resets and can be directly implemented on a physical system (Peters & Schaal, 2008b). 6 Practical Algorithm

Here we present two practical policy optimization algorithm based on the ideas above, which use either the single path or vine sampling scheme from the preceding section.

The algorithms repeatedly perform the following steps:

## 1. Use the single path or vine procedures to collect a set
 of state-action pairs along with Monte Carlo estimates of their Q-values.

## 2. By averaging over samples, construct the estimated
 objective and constraint in Equation (14).

## 3. Approximately solve this constrained optimization
 problem to update the policy’s parameter vector θ.

We use the conjugate gradient algorithm followed by a line search, which is altogether only slightly more expensive than computing the gradient itself. See Appendix C for details.

With regard to (3), we construct the Fisher information matrix (FIM) by analytically computing the Hessian of the KL divergence, rather than using the covariance matrix of the gradients. That is, we estimate Aij as 1

N

P

N n=1 ∂ 2 ∂θi∂θj

DKL(πθold (·|sn) k πθ(·|sn)), rather than 1

N

P

N n=1 ∂ ∂θi log πθ(an|sn) ∂ ∂θj log πθ(an|sn). The analytic estimator integrates over the action at each state sn, and does not depend on the action an that was sampled.

As described in Appendix C, this analytic estimator has computational benefits in the large-scale setting, since it removes the need to store a dense Hessian or all policy gradients from a batch of trajectories. The rate of improvement in the policy is similar to the empirical FIM, as shown in the experiments.

Let us briefly summarize the relationship between the theory from Section 3 and the practical algorithm we have described: 
* The theory justifies optimizing a surrogate objective with a penalty on KL divergence. However, the large penalty coefficient C leads to prohibitively small steps, so we would like to decrease this coefficient.

Empirically, it is hard to robustly choose the penalty coefficient, so we use a hard constraint instead of a penalty, with parameter δ (the bound on KL divergence). 
* The constraint on DKL max(θold, θ) is hard for numerical optimization and estimation, so instead we constrain

DKL(θold, θ). 
* Our theory ignores estimation error for the advantage function. Kakade & Langford (2002) consider this error in their derivation, and the same arguments would hold in the setting of this paper, but we omit them for simplicity. 7 Connections with Prior Work

As mentioned in Section 4, our derivation results in a policy update that is related to several prior methods, providing a unifying perspective on a number of policy update

Trust Region Policy Optimization schemes. The natural policy gradient (Kakade, 2002) can be obtained as a special case of the update in Equation (12) by using a linear approximation to L and a quadratic approximation to the DKL constraint, resulting in the following problem: maximize θ h ∇θLθold (θ)   θ=θold · (θ − θold) i (17) subject to 1 2 (θold − θ)

T A(θold)(θold − θ) ≤ δ, where A(θold)ij = ∂ ∂θi ∂ ∂θj

Es∼ρπ [DKL(π(·|s, θold) k π(·|s, θ))] θ=θold .

The update is θnew = θold + 1 λA(θold) −1∇θL(θ)   θ=θold , where the stepsize 1 λ is typically treated as an algorithm parameter. This differs from our approach, which enforces the constraint at each update. Though this difference might seem subtle, our experiments demonstrate that it significantly improves the algorithm’s performance on larger problems.

We can also obtain the standard policy gradient update by using an ` 2 constraint or penalty: maximize θ h ∇θLθold (θ)   θ=θold · (θ − θold) i (18) subject to 1 2 k θ − θoldk 2 ≤ δ.

The policy iteration update can also be obtained by solving the unconstrained problem maximizeπ Lπold (π), using L as defined in Equation (3).

Several other methods employ an update similar to Equation (12). Relative entropy policy search (REPS) (Peters et al., 2010) constrains the state-action marginals p(s, a), while TRPO constrains the conditionals p(a|s). Unlike

REPS, our approach does not require a costly nonlinear optimization in the inner loop. Levine and Abbeel (2014) also use a KL divergence constraint, but its purpose is to encourage the policy not to stray from regions where the estimated dynamics model is valid, while we do not attempt to estimate the system dynamics explicitly. Pirotta et al. (2013) also build on and generalize Kakade and Langford’s results, and they derive different algorithms from the ones here. 8 Experiments

We designed our experiments to investigate the following questions:

## 1. What are the performance characteristics of the single
 path and vine sampling procedures?

#### 2. TRPO is related to prior methods (e.g. natural policy
 gradient) but makes several changes, most notably by using a fixed KL divergence rather than a fixed penalty coefficient. How does this affect the performance of the algorithm?

Figure 2. 2D robot models used for locomotion experiments.

From left to right: swimmer, hopper, walker. The hopper and walker present a particular challenge, due to underactuation and contact discontinuities.

Control

Standard deviations

Fully connected layer 30 units

Input layer

Mean parameters Sampling 4×4 4×4 4×4 4×4 4×4 4×4 4×4 4×4

Control

Hidden layer 20 units

Conv. layer

Conv. layer

Input layer 16 filters 16 filters

Action probabilities Sampling

Figure 3. Neural networks used for the locomotion task (top) and for playing Atari games (bottom).

## 3. Can TRPO be used to solve challenging large-scale
 problems? How does TRPO compare with other methods when applied to large-scale problems, with regard to final performance, computation time, and sample complexity?

To answer (1) and (2), we compare the performance of the single path and vine variants of TRPO, several ablated variants, and a number of prior policy optimization algorithms. With regard to (3), we show that both the single path and vine algorithm can obtain high-quality locomotion controllers from scratch, which is considered to be a hard problem. We also show that these algorithms produce competitive results when learning policies for playing Atari games from images using convolutional neural networks with tens of thousands of parameters.

## 8.1 Simulated Robotic Locomotion
We conducted the robotic locomotion experiments using the MuJoCo simulator (Todorov et al., 2012). The three simulated robots are shown in Figure 2. The states of the robots are their generalized positions and velocities, and the controls are joint torques. Underactuation, high dimensionality, and non-smooth dynamics due to contacts make these

Joint angles and kinematics Screen input

Trust Region Policy Optimization tasks very challenging. The following models are included in our evaluation:

### 1. Swimmer. 10-dimensional state space, linear reward
 for forward progress and a quadratic penalty on joint effort to produce the reward r(x, u) = vx−10−5k uk 2 .

The swimmer can propel itself forward by making an undulating motion.

### 2. Hopper. 12-dimensional state space, same reward as
 the swimmer, with a bonus of +1 for being in a nonterminal state. We ended the episodes when the hopper fell over, which was defined by thresholds on the torso height and angle.

#### 3. Walker. 18-dimensional state space. For the walker,
 we added a penalty for strong impacts of the feet against the ground to encourage a smooth walk rather than a hopping gait.

We used δ = 0.01 for all experiments. See Table 2 in the

Appendix for more details on the experimental setup and parameters used. We used neural networks to represent the policy, with the architecture shown in Figure 3, and further details provided in Appendix D. To establish a standard baseline, we also included the classic cart-pole balancing problem, based on the formulation from Barto et al. (1983), using a linear policy with six parameters that is easy to optimize with derivative-free black-box optimization methods.

The following algorithms were considered in the comparison: single path TRPO; vine TRPO; cross-entropy method (CEM), a gradient-free method (Szita & L¨orincz, 2006); covariance matrix adaption (CMA), another gradient-free method (Hansen & Ostermeier, 1996); natural gradient, the classic natural policy gradient algorithm (Kakade, 2002), which differs from single path by the use of a fixed penalty coefficient (Lagrange multiplier) instead of the KL divergence constraint; empirical FIM, identical to single path, except that the FIM is estimated using the covariance matrix of the gradients rather than the analytic estimate; max KL, which was only tractable on the cart-pole problem, and uses the maximum KL divergence in Equation (11), rather than the average divergence, allowing us to evaluate the quality of this approximation. The parameters used in the experiments are provided in Appendix E. For the natural gradient method, we swept through the possible values of the stepsize in factors of three, and took the best value according to the final performance.

Learning curves showing the total reward averaged across five runs of each algorithm are shown in Figure 4. Single path and vine TRPO solved all of the problems, yielding the best solutions. Natural gradient performed well on the two easier problems, but was unable to generate hopping and walking gaits that made forward progress. These results provide empirical evidence that constraining the KL divergence is a more robust way to choose step sizes and make fast, consistent progress, compared to using a fixed 0 10 20 30 40 50 number of policy iterations 0 2 4 6 8 10 Cartpole

Vine

Single Path

Natural Gradient

Max KL

Empirical FIM

CEM

CMA

RWR 0 10 20 30 40 50 number of policy iterations

## 0.10
## 0.05
## 0.00
## 0.05
## 0.10
## 0.15
Swimmer

Vine

Single Path

Natural Gradient

Empirical FIM

CEM

CMA

RWR 0 50 100 150 200 number of policy iterations

## 1.0
## 0.5
## 0.0
## 0.5
## 1.0
## 1.5
## 2.0
## 2.5 Hopper
Vine

Single Path

Natural Gradient

CEM

RWR 0 50 100 150 200 number of policy iterations

## 1.0
## 0.5
## 0.0
## 0.5
## 1.0
## 1.5
## 2.0
## 2.5
## 3.0
## 3.5
Walker

Vine

Single Path

Natural Gradient

CEM

RWR

Figure 4. Learning curves for locomotion tasks, averaged across five runs of each algorithm with random initializations. Note that for the hopper and walker, a score of −1 is achievable without any forward velocity, indicating a policy that simply learned balanced standing, but not walking. penalty. CEM and CMA are derivative-free algorithms, hence their sample complexity scales unfavorably with the number of parameters, and they performed poorly on the larger problems. The max KL method learned somewhat more slowly than our final method, due to the more restrictive form of the constraint, but overall the result suggests that the average KL divergence constraint has a similar effect as the theorecally justified maximum KL divergence.

Videos of the policies learned by TRPO may be viewed on the project website: http://sites.google.com/ site/trpopaper/.

Note that TRPO learned all of the gaits with generalpurpose policies and simple reward functions, using minimal prior knowledge. This is in contrast with most prior methods for learning locomotion, which typically rely on hand-architected policy classes that explicitly encode notions of balance and stepping (Tedrake et al., 2004; Geng et al., 2006; Wampler & Popovi´c, 2009).

## 8.2 Playing Games from Images
To evaluate TRPO on a partially observed task with complex observations, we trained policies for playing Atari games, using raw images as input. The games require learning a variety of behaviors, such as dodging bullets and hitting balls with paddles. Aside from the high dimensionality, challenging elements of these games include delayed rewards (no immediate penalty is incurred when a life is lost in Breakout or Space Invaders); complex sequences of behavior (Q*bert requires a character to hop on 21 different platforms); and non-stationary image statistics (Enduro involves a changing and flickering background).

We tested our algorithms on the same seven games reported on in (Mnih et al., 2013) and (Guo et al., 2014), which are reward cost (-velocity + ctrl) reward

Trust Region Policy Optimization

#### B. Rider Breakout Enduro Pong Q*bert Seaquest S. Invaders
Random 354 1.2 0 −20.4 157 110 179

Human (Mnih et al., 2013) 7456 31.0 368 −3.0 18900 28010 3690

Deep Q Learning (Mnih et al., 2013) 4092 168.0 470 20.0 1952 1705 581

UCC-I (Guo et al., 2014) 5702 380 741 21 20025 2995 692

TRPO - single path 1425.2 10.8 534.6 20.9 1973.5 1908.6 568.4

TRPO - vine 859.5 34.2 430.8 20.9 7732.5 788.4 450.2

Table 1. Performance comparison for vision-based RL algorithms on the Atari domain. Our algorithms (bottom rows) were run once on each task, with the same architecture and parameters. Performance varies substantially from run to run (with different random initializations of the policy), but we could not obtain error statistics due to time constraints. made available through the Arcade Learning Environment (Bellemare et al., 2013) The images were preprocessed following the protocol in Mnih et al (2013), and the policy was represented by the convolutional neural network shown in

Figure 3, with two convolutional layers with 16 channels and stride 2, followed by one fully-connected layer with 20 units, yielding 33,500 parameters.

The results of the vine and single path algorithms are summarized in Table 1, which also includes an expert human performance and two recent methods: deep Q-learning (Mnih et al., 2013), and a combination of Monte-Carlo Tree

Search with supervised training (Guo et al., 2014), called

UCC-I. The 500 iterations of our algorithm took about 30 hours (with slight variation between games) on a 16-core computer. While our method only outperformed the prior methods on some of the games, it consistently achieved reasonable scores. Unlike the prior methods, our approach was not designed specifically for this task. The ability to apply the same policy search method to methods as diverse as robotic locomotion and image-based game playing demonstrates the generality of TRPO. 9 Discussion

We proposed and analyzed trust region methods for optimizing stochastic control policies. We proved monotonic improvement for an algorithm that repeatedly optimizes a local approximation to the expected return of the policy with a KL divergence penalty, and we showed that an approximation to this method that incorporates a KL divergence constraint achieves good empirical results on a range of challenging policy learning tasks, outperforming prior methods. Our analysis also provides a perspective that unifies policy gradient and policy iteration methods, and shows them to be special limiting cases of an algorithm that optimizes a certain objective subject to a trust region constraint.

In the domain of robotic locomotion, we successfully learned controllers for swimming, walking and hopping in a physics simulator, using general purpose neural networks and minimally informative rewards. To our knowledge, no prior work has learned controllers from scratch for all of these tasks, using a generic policy search method and non-engineered, general-purpose policy representations. In the game-playing domain, we learned convolutional neural network policies that used raw images as inputs. This requires optimizing extremely high-dimensional policies, and only two prior methods report successful results on this task.

Since the method we proposed is scalable and has strong theoretical foundations, we hope that it will serve as a jumping-off point for future work on training large, rich function approximators for a range of challenging problems. At the intersection of the two experimental domains we explored, there is the possibility of learning robotic control policies that use vision and raw sensory data as input, providing a unified scheme for training robotic controllers that perform both perception and control. The use of more sophisticated policies, including recurrent policies with hidden state, could further make it possible to roll state estimation and control into the same policy in the partiallyobserved setting. By combining our method with model learning, it would also be possible to substantially reduce its sample complexity, making it applicable to real-world settings where samples are expensive.

Acknowledgements

We thank Emo Todorov and Yuval Tassa for providing the MuJoCo simulator; Bruno Scherrer, Tom Erez, Greg

Wayne, and the anonymous ICML reviewers for insightful comments, and Vitchyr Pong and Shane Gu for pointing our errors in a previous version of the manuscript. This research was funded in part by the Office of Naval Research through a Young Investigator Award and under grant number N00014-11-1-0688, DARPA through a Young Faculty

Award, by the Army Research Office through the MAST program.

## References
 Bagnell, J. A. and Schneider, J. Covariant policy search. IJCAI, 2003. Bartlett, P. L. and Baxter, J. Infinite-horizon policy-gradient estimation. arXiv preprint arXiv:1106.0665, 2011. Barto, A., Sutton, R., and Anderson, C. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE Transactions on Systems, Man and Cybernetics, (5):834–846, 1983. Trust Region Policy Optimization Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47:253– 279, jun 2013. Bertsekas, D. Dynamic programming and optimal control, volume 1. 2005. Deisenroth, M., Neumann, G., and Peters, J. A survey on policy search for robotics. Foundations and Trends in Robotics, 2(1- 2):1–142, 2013. Gabillon, Victor, Ghavamzadeh, Mohammad, and Scherrer, Bruno. Approximate dynamic programming finally performs well in the game of Tetris. In Advances in Neural Information Processing Systems, 2013. Geng, T., Porr, B., and W¨org¨otter, F. Fast biped walking with a reflexive controller and realtime policy searching. In Advances in Neural Information Processing Systems (NIPS), 2006. Guo, X., Singh, S., Lee, H., Lewis, R. L., and Wang, X. Deep learning for real-time atari game play using offline MonteCarlo tree search planning. In Advances in Neural Information Processing Systems, pp. 3338–3346, 2014. Hansen, Nikolaus and Ostermeier, Andreas. Adapting arbitrary normal mutation distributions in evolution strategies: The covariance matrix adaptation. In Evolutionary Computation, 1996., Proceedings of IEEE International Conference on, pp. 312–317. IEEE, 1996. Hunter, David R and Lange, Kenneth. A tutorial on MM algorithms. The American Statistician, 58(1):30–37, 2004. Kakade, Sham. A natural policy gradient. In Advances in Neural Information Processing Systems, pp. 1057–1063. MIT Press, 2002. Kakade, Sham and Langford, John. Approximately optimal approximate reinforcement learning. In ICML, volume 2, pp. 267–274, 2002. Lagoudakis, Michail G and Parr, Ronald. Reinforcement learning as classification: Leveraging modern classifiers. In ICML, volume 3, pp. 424–431, 2003. Levin, D. A., Peres, Y., and Wilmer, E. L. Markov chains and mixing times. American Mathematical Society, 2009. Levine, Sergey and Abbeel, Pieter. Learning neural network policies with guided policy search under unknown dynamics. In Advances in Neural Information Processing Systems, pp. 1071–1079, 2014. Martens, J. and Sutskever, I. Training deep and recurrent networks with hessian-free optimization. In Neural Networks: Tricks of the Trade, pp. 479–535. Springer, 2012. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., and Riedmiller, M. Playing Atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013. Nemirovski, Arkadi. Efficient methods in convex programming. 2005. Ng, A. Y. and Jordan, M. PEGASUS: A policy search method for large mdps and pomdps. In Uncertainty in artificial intelligence (UAI), 2000. Owen, Art B. Monte Carlo theory, methods and examples. 2013. Pascanu, Razvan and Bengio, Yoshua. Revisiting natural gradient for deep networks. arXiv preprint arXiv:1301.3584, 2013. Peters, J. and Schaal, S. Reinforcement learning of motor skills with policy gradients. Neural Networks, 21(4):682–697, 2008a. Peters, J., M¨ulling, K., and Alt¨un, Y. Relative entropy policy search. In AAAI Conference on Artificial Intelligence, 2010. Peters, Jan and Schaal, Stefan. Natural actor-critic. Neurocomputing, 71(7):1180–1190, 2008b. Pirotta, Matteo, Restelli, Marcello, Pecorino, Alessio, and Calandriello, Daniele. Safe policy iteration. In Proceedings of The 30th International Conference on Machine Learning, pp. 307– 315, 2013. Pollard, David. Asymptopia: an exposition of statistical asymptotic theory. 2000. URL http://www.stat.yale.edu/ ˜pollard/Books/Asymptopia. Szita, Istv´an and L¨orincz, Andr´as. Learning tetris using the noisy cross-entropy method. Neural computation, 18(12): 2936–2941, 2006. Tedrake, R., Zhang, T., and Seung, H. Stochastic policy gradient reinforcement learning on a simple 3d biped. In IEEE/RSJ International Conference on Intelligent Robots and Systems, 2004. Todorov, Emanuel, Erez, Tom, and Tassa, Yuval. MuJoCo: A physics engine for model-based control. In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on, pp. 5026–5033. IEEE, 2012. Wampler, Kevin and Popovi´c, Zoran. Optimal gait and form for animal locomotion. In ACM Transactions on Graphics (TOG), volume 28, pp. 60. ACM, 2009. Wright, Stephen J and Nocedal, Jorge. Numerical optimization, volume 2. Springer New York, 1999. Trust Region Policy Optimization A Proof of Policy Improvement Bound This proof (of Theorem 1) uses techniques from the proof of Theorem 4.1 in (Kakade & Langford, 2002), adapting them to the more general setting considered in this paper. An informal overview is as follows. Our proof relies on the notion of coupling, where we jointly define the policies π and π 0 so that they choose the same action with high probability = (1 − α). Surrogate loss Lπ(˜π) accounts for the the advantage of ˜π the first time that it disagrees with π, but not subsequent disagreements. Hence, the error in Lπ is due to two or more disagreements between π and ˜π, hence, we get an O(α 2 ) correction term, where α is the probability of disagreement. We start out with a lemma from Kakade & Langford (2002) that shows that the difference in policy performance η(˜π)−η(π) can be decomposed as a sum of per-timestep advantages. Lemma 1. Given two policies π, ˜π, η(˜π) = η(π)+Eτ∼˜π " ∞X t=0 γ tAπ(st, at)

#
 (19) This expectation is taken over trajectories τ := (s0, a0, s1, a0, . . .), and the notation Eτ∼˜π [. . . ] indicates that actions are sampled from ˜π to generate τ . Proof. First note that Aπ(s, a) = Es 0 ∼P (s 0 |s,a)
r(s) + γVπ(s 0 ) − Vπ(s)]. Therefore, Eτ|˜π " ∞X t=0 γ tAπ(st, at)

# (20)
 = Eτ|˜π " ∞X t=0 γ t (r(st) + γVπ(st+1) − Vπ(st))# (21) = Eτ|˜π " −Vπ(s0) + ∞X t=0 γ t r(st)

# (22)
 = −Es0
Vπ(s0). + Eτ|˜π " ∞X t=0 γ t r(st)

# (23)
 = −η(π) + η(˜π) (24) Rearranging, the result follows. Define ¯A(s) to be the expected advantage of ˜π over π at state s: ¯A(s) = Ea∼˜π(·|s)
Aπ(s, a). . (25) Now Lemma 1 can be written as follows: η(˜π) = η(π) + Eτ∼˜π " ∞X t=0 γ t ¯A(st)

# (26)
 Note that Lπ can be written as Lπ(˜π) = η(π) + Eτ∼π " ∞X t=0 γ t ¯A(st)

# (27)
 The difference in these equations is whether the states are sampled using π or ˜π. To bound the difference between η(˜π) and Lπ(˜π), we will bound the difference arising from each timestep. To do this, we first need to introduce a measure of how much π and ˜π agree. Specifically, we’ll couple the policies, so that they define a joint distribution over pairs of actions. Definition 1. (π, ˜π) is an α-coupled policy pair if it defines a joint distribution (a, ˜a)|s, such that P(a 6 = ˜a|s) ≤ α for all s. π and ˜π will denote the marginal distributions of a and ˜a, respectively. Trust Region Policy Optimization Computationally, α-coupling means that if we randomly choose a seed for our random number generator, and then we sample from each of π and ˜π after setting that seed, the results will agree for at least fraction 1 − α of seeds. Lemma 2. Given that π, ˜π are α-coupled policies, for all s,   ¯A(s)  ≤ 2α max s,a |Aπ(s, a)| (28) Proof. ¯A(s) = E˜a∼˜π [Aπ(s, ˜a)] = E(a,˜a)∼(π,˜π)
Aπ(s, ˜a) − Aπ(s, a). since Ea∼π Aπ(s, a). = 0 (29) = P(a 6 = ˜a|s)E(a,˜a)∼(π,˜π)|a6=˜a [Aπ(s, ˜a) − Aπ(s, a)] (30) | ¯A(s)| ≤ α · 2 max s,a |Aπ(s, a)| (31) Lemma 3. Let (π, ˜π) be an α-coupled policy pair. Then   Est∼˜π  ¯A(st) − Est∼π  ¯A(st)   ≤ 2α max s ¯A(s) ≤ 4α(1 − (1 − α) t ) max s |Aπ(s, a)| (32) Proof. Given the coupled policy pair (π, ˜π), we can also obtain a coupling over the trajectory distributions produced by π and ˜π, respectively. Namely, we have pairs of trajectories τ, τ˜, where τ is obtained by taking actions from π, and τ˜ is obtained by taking actions from ˜π, where the same random seed is used to generate both trajectories. We will consider the advantage of ˜π over π at timestep t, and decompose this expectation based on whether π agrees with ˜π at all timesteps i < t. Let nt denote the number of times that ai 6 = ˜ai for i < t, i.e., the number of times that π and ˜π disagree before timestep t. Est∼˜π  ¯A(st) = P(nt = 0)Est∼˜π|nt=0  ¯A(st) + P(nt > 0)Est∼˜π|nt>0  ¯A(st) (33) The expectation decomposes similarly for actions are sampled using π: Est∼π  ¯A(st) = P(nt = 0)Est∼π|nt=0  ¯A(st) + P(nt > 0)Est∼π|nt>0  ¯A(st) (34) Note that the nt = 0 terms are equal: Est∼˜π|nt=0  ¯A(st) = Est∼π|nt=0  ¯A(st) , (35) because nt = 0 indicates that π and ˜π agreed on all timesteps less than t. Subtracting Equations (33) and (34), we get Est∼˜π  ¯A(st) − Est∼π  ¯A(st) = P(nt > 0) Est∼˜π|nt>0  ¯A(st) − Est∼π|nt>0  ¯A(st)  (36) By definition of α, P(π, ˜π agree at timestep i) ≥ 1 − α, so P(nt = 0) ≥ (1 − α) t , and P(nt > 0) ≤ 1 − (1 − α) t (37) Next, note that   Est∼˜π|nt>0  ¯A(st) − Est∼π|nt>0  ¯A(st)   ≤ Est∼˜π|nt>0  ¯A(st)   + Est∼π|nt>0  ¯A(st)   (38) ≤ 4α max s,a |Aπ(s, a)| (39) Where the second inequality follows from Lemma 3. Plugging Equation (37) and Equation (39) into Equation (36), we get   Est∼˜π  ¯A(st) − Est∼π  ¯A(st)   ≤ 4α(1 − (1 − α) t ) max s,a |Aπ(s, a)| (40) Trust Region Policy Optimization The preceding Lemma bounds the difference in expected advantage at each timestep t. We can sum over time to bound the difference between η(˜π) and Lπ(˜π). Subtracting Equation (26) and Equation (27), and defining  = maxs,a |Aπ(s, a)|, |η(˜π) − Lπ(˜π)| = ∞X t=0 γ t Eτ∼˜π  ¯A(st) − Eτ∼π  ¯A(st)   (41) ≤ ∞X t=0 γ t · 4α(1 − (1 − α) t ) (42) = 4α  1 1 − γ − 1 1 − γ(1 − α)  (43) = 4α 2γ (1 − γ)(1 − γ(1 − α)) (44) ≤ 4α 2γ (1 − γ) 2 (45) Last, to replace α by the total variation divergence, we need to use the correspondence between TV divergence and coupled random variables: Suppose pX and pY are distributions with DT V (pX k pY ) = α. Then there exists a joint distribution (X, Y ) whose marginals are pX, pY , for which X = Y with probability 1 − α. See (Levin et al., 2009), Proposition 4.7. It follows that if we have two policies π and ˜π such that maxs DT V (π(·|s) k ˜π(·|s)) ≤ α, then we can define an α-coupled policy pair (π, ˜π) with appropriate marginals. Taking α = maxs DT V (π(·|s) k ˜π(·|s)) ≤ α in Equation (45), Theorem 1 follows. B Perturbation Theory Proof of Policy Improvement Bound We also provide an alternative proof of Theorem 1 using perturbation theory. Proof. Let G = (1+γPπ+(γPπ) 2+. . .) = (1−γPπ) −1 , and similarly Let ˜G = (1+γP˜π+(γP˜π) 2+. . .) = (1−γP˜π) −1 . We will use the convention that ρ (a density on state space) is a vector and r (a reward function on state space) is a dual vector (i.e., linear functional on vectors), thus rρ is a scalar meaning the expected reward under density ρ. Note that η(π) = rGρ0, and η(˜π) = c ˜Gρ0. Let ∆ = P˜π − Pπ. We want to bound η(˜π) − η(π) = r( ˜G − G)ρ0. We start with some standard perturbation theory manipulations. G −1 − ˜G −1 = (1 − γPπ) − (1 − γP˜π) = γ∆. (46) Left multiply by G and right multiply by ˜G. ˜G − G = γG∆ ˜G ˜G = G + γG∆ ˜G (47) Substituting the right-hand side into ˜G gives ˜G = G + γG∆G + γ 2G∆G∆ ˜G (48) So we have η(˜π) − η(π) = r( ˜G − G)ρ = γrG∆Gρ0 + γ 2 rG∆G∆ ˜Gρ0 (49) Let us first consider the leading term γrG∆Gρ0. Note that rG = v, i.e., the infinite-horizon state-value function. Also note that Gρ0 = ρπ. Thus we can write γcG∆Gρ0 = γv∆ρπ. We will show that this expression equals the expected Trust Region Policy Optimization advantage Lπ(˜π) − Lπ(π). Lπ(˜π) − Lπ(π) = X s ρπ(s) X a (˜π(a|s) − π(a|s))Aπ(s, a) = X s ρπ(s) X a  πθ(a|s) − πθ˜(a|s)  " r(s) +X s 0 p(s 0 |s, a)γv(s 0 ) − v(s)

#
 = X s ρπ(s) X s 0 X a (π(a|s) − ˜π(a|s)) p(s 0 |s, a)γv(s 0 ) = X s ρπ(s) X s 0 (pπ(s 0 |s) − p˜π(s 0 |s))γv(s 0 ) = γv∆ρπ (50) Next let us bound the O(∆2 ) term γ 2 rG∆G∆ ˜Gρ. First we consider the product γrG∆ = γv∆. Consider the component s of this dual vector. |(γv∆)s| =     X a (˜π(s, a) − π(s, a))Qπ(s, a)     =     X a (˜π(s, a) − π(s, a))Aπ(s, a)     ≤ X a |˜π(s, a) − π(s, a)| · max a |Aπ(s, a)| ≤ 2α (51) where the last line used the definition of the total-variation divergence, and the definition of  = maxs,a |Aπ(s, a)|. We bound the other portion G∆ ˜Gρ using the ` 1 operator norm k Ak 1 = sup ρ  k Aρk 1 k ρk 1  (52) where we have that k Gk 1 = k ˜Gk 1 = 1/(1 − γ) and k ∆k 1 = 2α. That gives k G∆ ˜Gρk 1 ≤ kGk 1k ∆k 1k ˜Gk 1k ρk 1 = 1 1 − γ · 2α · 1 1 − γ · 1 (53) So we have that γ 2   rG∆G∆ ˜Gρ ≤ γk γrG∆k ∞k G∆ ˜Gρk 1 ≤ γk v∆k ∞k G∆ ˜Gρk 1 ≤ γ · 2α · 2α (1 − γ) 2 = 4γ (1 − γ) 2 α 2 (54) C Efficiently Solving the Trust-Region Constrained Optimization Problem This section describes how to efficiently approximately solve the following constrained optimization problem, which we must solve at each iteration of TRPO: maximize L(θ) subject to DKL(θold, θ) ≤ δ. (55) Trust Region Policy Optimization The method we will describe involves two steps: (1) compute a search direction, using a linear approximation to objective and quadratic approximation to the constraint; and (2) perform a line search in that direction, ensuring that we improve the nonlinear objective while satisfying the nonlinear constraint. The search direction is computed by approximately solving the equation Ax = g, where A is the Fisher information matrix, i.e., the quadratic approximation to the KL divergence constraint: DKL(θold, θ) ≈ 1 2 (θ −θold) T A(θ −θold), where Aij = ∂ ∂θi ∂ ∂θj DKL(θold, θ). In large-scale problems, it is prohibitively costly (with respect to computation and memory) to form the full matrix A (or A−1 ). However, the conjugate gradient algorithm allows us to approximately solve the equation Ax = b without forming this full matrix, when we merely have access to a function that computes matrix-vector products y → Ay. Appendix C.1 describes the most efficient way to compute matrix-vector products with the Fisher information matrix. For additional exposition on the use of Hessian-vector products for optimizing neural network objectives, see (Martens & Sutskever, 2012) and (Pascanu & Bengio, 2013). Having computed the search direction s ≈ A−1 g, we next need to compute the maximal step length β such that θ + βs will satisfy the KL divergence constraint. To do this, let δ = DKL ≈ 1 2 (βs) T A(βs) = 1 2 β 2 s T As. From this, we obtain β = p 2δ/sT As, where δ is the desired KL divergence. The term s T As can be computed through a single Hessian vector product, and it is also an intermediate result produced by the conjugate gradient algorithm. Last, we use a line search to ensure improvement of the surrogate objective and satisfaction of the KL divergence constraint, both of which are nonlinear in the parameter vector θ (and thus depart from the linear and quadratic approximations used to compute the step). We perform the line search on the objective Lθold (θ) − X [DKL(θold, θ) ≤ δ], where X [. . . ] equals zero when its argument is true and +∞ when it is false. Starting with the maximal value of the step length β computed in the previous paragraph, we shrink β exponentially until the objective improves. Without this line search, the algorithm occasionally computes large steps that cause a catastrophic degradation of performance. C.1 Computing the Fisher-Vector Product Here we will describe how to compute the matrix-vector product between the averaged Fisher information matrix and arbitrary vectors. This matrix-vector product enables us to perform the conjugate gradient algorithm. Suppose that the parameterized policy maps from the input x to “distribution parameter” vector µθ(x), which parameterizes the distribution π(u|x). Now the KL divergence for a given input x can be written as follows: DKL(πθold (·|x) k πθ(·|x)) = kl(µθ(x), µold) (56) where kl is the KL divergence between the distributions corresponding to the two mean parameter vectors. Differentiating kl twice with respect to θ, we obtain ∂µa(x) ∂θi ∂µb(x) ∂θj kl00ab(µθ(x), µold) + ∂ 2µa(x) ∂θi∂θj kl0a (µθ(x), µold) (57) where the primes (0 ) indicate differentiation with respect to the first argument, and there is an implied summation over indices a, b. The second term vanishes, leaving just the first term. Let J := ∂µa(x) ∂θi (the Jacobian), then the Fisher information matrix can be written in matrix form as J TMJ, where M = kl00ab(µθ(x), µold) is the Fisher information matrix of the distribution in terms of the mean parameter µ (as opposed to the parameter θ). This has a simple form for most parameterized distributions of interest. The Fisher-vector product can now be written as a function y → J TMJy. Multiplication by J T and J can be performed by most automatic differentiation and neural network packages (multiplication by J T is the well-known backprop operation), and the operation for multiplication by M can be derived for the distribution of interest. Note that this Fisher-vector product is straightforward to average over a set of datapoints, i.e., inputs x to µ. One could alternatively use a generic method for calculating Hessian-vector products using reverse mode automatic differentiation ((Wright & Nocedal, 1999), chapter 8), computing the Hessian of DKL with respect to θ. This method would be slightly less efficient as it does not exploit the fact that the second derivatives of µ(x) (i.e., the second term in Equation (57)) can be ignored, but may be substantially easier to implement. We have described a procedure for computing the Fisher-vector product y → Ay, where the Fisher information matrix is averaged over a set of inputs to the function µ. Computing the Fisher-vector product is typically about as expensive as computing the gradient of an objective that depends on µ(x) (Wright & Nocedal, 1999). Furthermore, we need to compute Trust Region Policy Optimization k of these Fisher-vector products per gradient, where k is the number of iterations of the conjugate gradient algorithm we perform. We found k = 10 to be quite effective, and using higher k did not result in faster policy improvement. Hence, a na¨ıve implementation would spend more than 90% of the computational effort on these Fisher-vector products. However, we can greatly reduce this burden by subsampling the data for the computation of Fisher-vector product. Since the Fisher information matrix merely acts as a metric, it can be computed on a subset of the data without severely degrading the quality of the final step. Hence, we can compute it on 10% of the data, and the total cost of Hessian-vector products will be about the same as computing the gradient. With this optimization, the computation of a natural gradient step A−1 g does not incur a significant extra computational cost beyond computing the gradient g. D Approximating Factored Policies with Neural Networks The policy, which is a conditional probability distribution πθ(a|s), can be parameterized with a neural network. This neural network maps (deterministically) from the state vector s to a vector µ, which specifies a distribution over action space. Then we can compute the likelihood p(a|µ) and sample a ∼ p(a|µ). For our experiments with continuous state and action spaces, we used a Gaussian distribution, where the covariance matrix was diagonal and independent of the state. A neural network with several fully-connected (dense) layers maps from the input features to the mean of a Gaussian distribution. A separate set of parameters specifies the log standard deviation of each element. More concretely, the parameters include a set of weights and biases for the neural network computing the mean, {Wi , bi} L i=1, and a vector r (log standard deviation) with the same dimension as a. Then, the policy is defined by the normal distribution N  mean = NeuralNet  s; {Wi , bi} L i=1 ,stdev = exp(r)  . Here, µ = [mean,stdev]. For the experiments with discrete actions (Atari), we use a factored discrete action space, where each factor is parameterized as a categorical distribution. That is, the action consists of a tuple (a1, a2, . . . , aK) of integers ak ∈ {1, 2, . . . , Nk}, and each of these components is assumed to have a categorical distribution, which is specified by a vector µk =
p1, p2, . . . , pNk ]. Hence, µ is defined to be the concatenation of the factors’ parameters: µ = [µ1, µ2, . . . , µK] and has dimension dim µ = P K k=1 Nk. The components of µ are computed by taking applying a neural network to the input s and then applying the softmax operator to each slice, yielding normalized probabilities for each factor. E Experiment Parameters Swimmer Hopper Walker State space dim. 10 12 20 Control space dim. 2 3 6 Total num. policy params 364 4806 8206 Sim. steps per iter. 50K 1M 1M Policy iter. 200 200 200 Stepsize (DKL) 0.01 0.01 0.01 Hidden layer size 30 50 50 Discount (γ) 0.99 0.99 0.99 Vine: rollout length 50 100 100 Vine: rollouts per state 4 4 4 Vine: Q-values per batch 500 2500 2500 Vine: num. rollouts for sampling 16 16 16 Vine: len. rollouts for sampling 1000 1000 1000 Vine: computation time (minutes) 2 14 40 SP: num. path 50 1000 10000 SP: path len. 1000 1000 1000 SP: computation time 5 35 100 Table 2. Parameters for continuous control tasks, vine and single path (SP) algorithms. Trust Region Policy Optimization All games Total num. policy params 33500 Vine: Sim. steps per iter. 400K SP: Sim. steps per iter. 100K Policy iter. 500 Stepsize (DKL) 0.01 Discount (γ) 0.99 Vine: rollouts per state ≈ 4 Vine: computation time ≈ 30 hrs SP: computation time ≈ 30 hrs Table 3. Parameters used for Atari domain. F Learning Curves for the Atari Domain 0 100 200 300 400 500 number of policy iterations 1600 1400 1200 1000 800 600 400 beam rider single path vine 0 100 200 300 400 500 number of policy iterations 45 40 35 30 25 20 15 10 5 0 breakout single path vine 0 100 200 300 400 500 number of policy iterations 600 500 400 300 200 100 0 100 enduro single path vine 0 100 200 300 400 500 number of policy iterations 30 20 10 0 10 20 30 pong single path vine 0 100 200 300 400 500 number of policy iterations 8000 7000 6000 5000 4000 3000 2000 1000 0 qbert single path vine 0 100 200 300 400 500 number of policy iterations 2000 1500 1000 500 0 500 seaquest single path vine 0 100 200 300 400 500 number of policy iterations 600 500 400 300 200 100 space invaders single path vine Figure 5. Learning curves for the Atari domain. For historical reasons, the plots show cost = negative reward. cost
