# 方策勾配法を使って、カートポールに取り組む
# 参考：
#   https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
#   https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/ch09/simple_pg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import gymnasium
from utils import plot_total_reward


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=0)
        return x


class Agent:
    def __init__(self):
        self.lr = 0.0002
        
        self.memory = []
        self.pi = Policy()
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)
    
    def get_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.pi(state)
        
        # 行動の確率のリストに対してカテゴリ分布を作成する
        m = Categorical(probs)
        
        # 分布を使ってアクションをサンプリングする
        action = m.sample()
        
        return action.item(), m.log_prob(action)
    
    def add(self, prob):
        self.memory.append(prob)
    
    def update(self, total_reward):
        G = total_reward
        loss = 0
        
        for prob in self.memory:
            loss += -prob * G
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.memory = []


if __name__ == '__main__':
    seed = 0
    trials = 100
    episodes = 5000
    
    torch.manual_seed(seed)
    env = gymnasium.make('CartPole-v1')
    mean_reward_history = np.zeros(episodes)

    for trial in range(1, trials + 1):
        agent = Agent()
        reward_history = []
        
        for episode in range(1, episodes + 1):
            total_reward = 0
            state, _ = env.reset(seed=seed)
            done = False
            truncated = False
            
            while not done and not truncated:
                action, prob = agent.get_action(state)
                agent.add(prob)
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
            
            agent.update(total_reward)
            reward_history.append(total_reward)
            
            print('\rtrial: {}, episode: {}, reward: {}'.format(trial, episode, total_reward), end='')
        
        mean_reward_history += (np.array(reward_history) - mean_reward_history) / trial
    
    plot_total_reward(mean_reward_history)