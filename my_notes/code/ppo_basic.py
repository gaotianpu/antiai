# from : https://zhuanlan.zhihu.com/p/605387491
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

# Categorical, 创建一个由probs或logits（但不是两者都有）参数化的分类分布。

# 定义神经网络模型
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

# 定义 PPO 算法
class PPO:
    def __init__(self, env_name, gamma, eps_clip, k_epochs, lr):
        self.env = gym.make(env_name)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.lr = lr
        
        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
        
    def update(self, memory):
        states, actions, log_probs_old, returns, advantages = memory
        for _ in range(self.k_epochs):
            # 计算损失函数  截断概率比？
            probs = self.policy(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值函数损失
            value = self.policy(torch.from_numpy(states).float())
            value_loss = F.mse_loss(value.squeeze(), torch.tensor(returns))
            
            # 进行梯度下降
            self.optimizer.zero_grad()
            loss = actor_loss + 0.5 * value_loss
            loss.backward()
            self.optimizer.step()
            
    def train(self, num_episodes, max_steps):
        for i_episode in range(num_episodes):
            state = self.env.reset() #状态
            rewards = [] #奖励
            log_probs_old = []
            states = []
            actions = []
            for t in range(max_steps):
                action, log_prob = self.select_action(state) 
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                log_probs_old.append(log_prob)
                states.append(state)
                actions.append(action)
                if done:
                    break
                    
            # 计算折扣回报和优势函数
            returns = []
            discounted_reward = 0
            for reward in reversed(rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                returns.insert(0, discounted_reward）


# 构建智能体和环境
class Agent:
    def __init__(self, n_states, n_actions):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_states, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_actions)
        )
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = torch.softmax(self.model(state), dim=1)
        action = np.random.choice(len(action_probs[0]), p=action_probs.detach().numpy()[0])
        return action

# 构建环境    
class Environment:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        
    def reset(self):
        self.state = np.zeros(self.n_states)
        self.state[0] = 1  # 将智能体放在起始位置
        return self.state
    
    def step(self, action):
        if action == 0: #上一步
            self.state[0] -= 1
        elif action == 1: #下一步
            self.state[0] += 1
        else:  #？
            self.state[1] += 1 #?
            
        reward = 0
        done = False
        if self.state[0] == 0 and self.state[1] == 0:  # 智能体到达目标位置
            reward = 1
            done = True
        return self.state, reward, done

# 定义 RLHF 算法
class RLHF:
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
        
    def train(self, num_episodes, human_feedback_fn):
        optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=0.001)
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.agent.act(state)
                state_next, reward, done = self.env.step(action)
                # 获取人类专家反馈
                human_feedback = human_feedback_fn(state, action, state_next, reward)
                human_reward = torch.tensor(human_feedback)
                # 计算损失函数
                action_probs = torch.softmax(self.agent.model(torch.from_numpy(state).float()), dim=1)
                dist = torch.distributions.Categorical(probs=action_probs)
                log_prob = dist.log_prob(torch.tensor(action))
                ratio = torch.exp(log_prob - torch.log(human_reward))
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                loss = -torch.min(ratio * human_reward, clipped_ratio * human_reward).mean()
                # 进行近端优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state = state_next
                
# 创建环境、智能体和 RLHF 实例，并开始训练
env = Environment(n_states=2, n_actions=3)
agent = Agent(n_states=2, n_actions=3)
rlhf = RLHF(agent=agent, environment=env)
rlhf.train(num_episodes=100, human_feedback_fn=lambda s,a,sn,r: 1）