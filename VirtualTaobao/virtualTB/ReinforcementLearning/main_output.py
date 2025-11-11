import gym
import math
import torch
import random
import virtualTB
import time, sys
import configparser
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import wrappers
from ddpg import DDPG
from copy import deepcopy
from collections import namedtuple
import csv
import os

FLOAT = torch.FloatTensor
LONG = torch.LongTensor

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

env = gym.make('VirtualTB-v0')

env.seed(0)
np.random.seed(0)
torch.manual_seed(0)

agent = DDPG(gamma = 0.95, tau = 0.001, hidden_size = 128,
                    num_inputs = env.observation_space.shape[0], action_space = env.action_space)

memory = ReplayMemory(1000000)

ounoise = OUNoise(env.action_space.shape[0])
param_noise = None

rewards = []
total_numsteps = 0
updates = 0

# 新增：创建保存目录
os.makedirs("logs", exist_ok=True)
os.makedirs("trajectories", exist_ok=True)

log_file_path = "logs/training_log_ddpg.txt"
traj_file_path = "trajectories/trajectory_ddpg.csv"

# 新增：打开日志文件用于写入（追加模式）
log_file = open(log_file_path, "a")

# 新增：准备轨迹CSV文件，写入表头
with open(traj_file_path, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    # state和action维度可能很大，这里写成字符串存储，reward是标量
    csv_writer.writerow(['episode', 'step', 'state', 'action', 'reward'])

for i_episode in range(100000):
    state = torch.Tensor([env.reset()])

    episode_reward = 0
    episode_step = 0
    episode_trajectory = []  # 新增：保存当前episode轨迹

    while True:
        action = agent.select_action(state, ounoise, param_noise)
        print(action)
        next_state, reward, done, _ = env.step(action.numpy()[0])
        total_numsteps += 1
        episode_reward += reward
        episode_step += 1

        action_tensor = torch.Tensor(action)
        mask = torch.Tensor([not done])
        next_state_tensor = torch.Tensor([next_state])
        reward_tensor = torch.Tensor([reward])

        memory.push(state, action_tensor, mask, next_state_tensor, reward_tensor)

        # 新增：保存当前step轨迹数据，转换tensor为列表方便写CSV
        episode_trajectory.append((
            i_episode,
            episode_step,
            state.numpy().tolist()[0],    # state是1xN，取第0个元素
            action_tensor.numpy().tolist(),
            reward
        ))

        state = next_state_tensor

        if len(memory) > 128:
            for _ in range(5):
                transitions = memory.sample(128)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_parameters(batch)

                updates += 1
        if done:
            break

    rewards.append(episode_reward)

    # 新增：每个episode结束后，把轨迹写入CSV文件（追加）
    with open(traj_file_path, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        for row in episode_trajectory:
            # state和action写成字符串，防止CSV格式错乱
            csv_writer.writerow([
                row[0],
                row[1],
                str(row[2]),
                str(row[3]),
                row[4]
            ])

    if i_episode % 10 == 0:
        episode_reward = 0
        episode_step = 0
        for i in range(50):
            state = torch.Tensor([env.reset()])
            while True:
                action = agent.select_action(state)

                next_state, reward, done, info = env.step(action.numpy()[0])
                episode_reward += reward
                episode_step += 1

                next_state = torch.Tensor([next_state])

                state = next_state
                if done:
                    break

        # 新增：把打印信息写入文件并打印到屏幕
        log_str = "Episode: {}, total numsteps: {}, average reward: {:.4f}, CTR: {:.4f}".format(
            i_episode, episode_step, episode_reward / 50, episode_reward / episode_step / 10
        )
        print(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()

env.close()
log_file.close()  # 关闭日志文件

