import csv
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import namedtuple
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
        
class TrajectoryDataset(Dataset):
    def __init__(self, csv_path):
        self.data = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                state = np.array(eval(row['state']), dtype=np.float32)
                action = np.array(eval(row['action']), dtype=np.float32)
                reward = float(row['reward'])
                done = float(row.get('done', 0))  # 如果csv有done字段就用，否则0
                self.data.append((state, action, reward, done))

        self.samples = []
        for i in range(len(self.data) - 1):
            s, a, r, d = self.data[i]
            s_next = self.data[i + 1][0]
            self.samples.append({
                'state': torch.tensor(s),
                # 'action': torch.tensor(a),
                'action': torch.tensor(a, dtype=torch.float32).squeeze(0),  # from [1, D] -> [D]
                'reward': torch.tensor(r).unsqueeze(0),
                'next_state': torch.tensor(s_next),
                'done': torch.tensor(d).unsqueeze(0)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = torch.stack([b[key] for b in batch])
    return batch_dict


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class VNet(nn.Module):
    def __init__(self, state_dim):
        super(VNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.net(state)


class IQL(nn.Module):
    def __init__(self, state_dim, action_dim, device='cuda'):
        super(IQL, self).__init__()
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.v_net = VNet(state_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=5e-4)

        self.discount = 0.99
        self.expectile = 0.7

    def expectile_loss(self, diff):
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * diff ** 2).mean()

    def update(self, batch):
        # state = batch['state'].to(self.device)
        # action = batch['action'].to(self.device)
        # reward = batch['reward'].to(self.device)
        # next_state = batch['next_state'].to(self.device)
        # done = batch['done'].to(self.device)
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        done = torch.cat(batch.mask)
        next_state = torch.cat(batch.next_state)

        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)
        if action.dim() == 3 and action.size(1) == 1:
            action = action.squeeze(1)

        with torch.no_grad():
            v_next = self.v_net(next_state)
            target_q = reward + (1 - done) * self.discount * v_next
        
        v = self.v_net(state)  # 这里需要梯度！
        diff = target_q - v
        v_loss = self.expectile_loss(diff)


        q = self.critic(state, action)
        critic_loss = F.mse_loss(q, target_q.detach())

        q_val = self.critic(state, self.actor(state))
        v_val = self.v_net(state)
        adv = q_val - v_val
        exp_adv = torch.exp(adv.detach().clamp(max=20))  # 防止爆炸

        actor_loss = (exp_adv * (self.actor(state) - action).pow(2).sum(dim=-1)).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return v_loss.item(), critic_loss.item(), actor_loss.item()

def load_iql_model(iql_agent, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    iql_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    iql_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    iql_agent.v_net.load_state_dict(checkpoint['v_net_state_dict'])
    iql_agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    iql_agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    iql_agent.v_optimizer.load_state_dict(checkpoint['v_optimizer_state_dict'])
    print(f"Loaded model from {checkpoint_path} at epoch {checkpoint['epoch']}")

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     csv_path = "trajectories/trajectory.csv"

#     dataset = TrajectoryDataset(csv_path)
#     dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

#     state_dim = dataset[0]['state'].shape[0]
#     action_dim = dataset[0]['action'].shape[0]
#     print(state_dim,action_dim)
#     iql_agent = IQL(state_dim, action_dim, device=device)

#     # num_epochs = 50
#     # for epoch in range(num_epochs):
#     #     total_v_loss = 0
#     #     total_c_loss = 0
#     #     total_a_loss = 0
#     #     batches = 0
#     #     for batch in dataloader:
#     #         # print(batch)
#     #         v_loss, c_loss, a_loss = iql_agent.update(batch)
#     #         total_v_loss += v_loss
#     #         total_c_loss += c_loss
#     #         total_a_loss += a_loss
#     #         batches += 1
#     #     print(f"Epoch {epoch+1}/{num_epochs} | V_loss: {total_v_loss/batches:.4f} | Critic_loss: {total_c_loss/batches:.4f} | Actor_loss: {total_a_loss/batches:.4f}")
#     num_epochs = 50
#     save_interval = 10  # 每10个epoch保存一次模型
#     save_path = "./iql_model.pth"

#     for epoch in range(num_epochs):
#         total_v_loss = 0
#         total_c_loss = 0
#         total_a_loss = 0
#         batches = 0
#         for batch in dataloader:
#             v_loss, c_loss, a_loss = iql_agent.update(batch)
#             total_v_loss += v_loss
#             total_c_loss += c_loss
#             total_a_loss += a_loss
#             batches += 1
#         print(f"Epoch {epoch+1}/{num_epochs} | V_loss: {total_v_loss/batches:.4f} | Critic_loss: {total_c_loss/batches:.4f} | Actor_loss: {total_a_loss/batches:.4f}")

#         # 定期保存模型
#         if (epoch + 1) % save_interval == 0:
#             torch.save({
#                 'actor_state_dict': iql_agent.actor.state_dict(),
#                 'critic_state_dict': iql_agent.critic.state_dict(),
#                 'v_net_state_dict': iql_agent.v_net.state_dict(),
#                 'actor_optimizer_state_dict': iql_agent.actor_optimizer.state_dict(),
#                 'critic_optimizer_state_dict': iql_agent.critic_optimizer.state_dict(),
#                 'v_optimizer_state_dict': iql_agent.v_optimizer.state_dict(),
#                 'epoch': epoch + 1
#             }, save_path)
#             print(f"Model saved at epoch {epoch+1} to {save_path}")

#     # 训练结束后也保存一次
#     torch.save({
#         'actor_state_dict': iql_agent.actor.state_dict(),
#         'critic_state_dict': iql_agent.critic.state_dict(),
#         'v_net_state_dict': iql_agent.v_net.state_dict(),
#         'actor_optimizer_state_dict': iql_agent.actor_optimizer.state_dict(),
#         'critic_optimizer_state_dict': iql_agent.critic_optimizer.state_dict(),
#         'v_optimizer_state_dict': iql_agent.v_optimizer.state_dict(),
#         'epoch': num_epochs
#     }, save_path)
#     print(f"Final model saved to {save_path}")
import gym  # 确保已经安装并导入gym
import virtualTB
import csv

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "trajectories/trajectory.csv"

    dataset = TrajectoryDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    state_dim = dataset[0]['state'].shape[0]
    action_dim = dataset[0]['action'].shape[0]
    print("State dim:", state_dim, "Action dim:", action_dim)

    iql_agent = IQL(state_dim, action_dim, device=device)

    # 加载最后保存的模型
    checkpoint_path = "./iql_model.pth"
    load_iql_model(iql_agent, checkpoint_path, device)

    # 初始化环境
    env = gym.make('VirtualTB-v0')
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    memory = ReplayMemory(1000000)
    # 这里要写你的ReplayMemory和Transition数据结构定义，以及agent.select_action等函数
    # 假设你有agent封装了iql_agent.actor，并有select_action方法

    traj_file_path = "trajectories/trajectory_iql.csv"  # 保存轨迹的csv
    log_file = open("logs/training_log_iql.txt", "a")

    num_episodes = 100000
    for i_episode in range(num_episodes):
        state = torch.Tensor([env.reset()]).to(device)

        episode_reward = 0
        episode_step = 0
        episode_trajectory = []

        done = False
        while not done:
            # 这里你要实现select_action方法，或者直接调用actor网络输出动作
            # 假设actor网络输出动作张量，转换为numpy并给环境
            with torch.no_grad():
                action = iql_agent.actor(state).to(device)
            # 可能需要根据环境action space做映射，假设直接用action.numpy()[0]
            action_np = action.cpu().numpy()[0]
            print(action_np.shape)
            next_state_raw, reward, done, _ = env.step(action_np)
            # next_state_raw, reward, done, _ = env.step(action.numpy()[0])
            episode_reward += reward
            episode_step += 1

            next_state = torch.Tensor([next_state_raw]).to(device)
            reward_tensor = torch.Tensor([reward]).to(device)
            done_tensor = torch.Tensor([done]).to(device)

            # 这里要存经验到memory，假设你有定义ReplayMemory和push方法
            memory.push(state, action, done_tensor, next_state, reward_tensor)

            # 保存轨迹（转换tensor到列表或标量）
            episode_trajectory.append((
                i_episode,
                episode_step,
                state.cpu().numpy().tolist()[0],
                action.cpu().numpy().tolist(),
                reward
            ))

            state = next_state

            # 这里可以调用更新参数的逻辑，比如每步采样训练
            if len(memory) > 128:
                for _ in range(5):
                    batch = memory.sample(128)
                    batch = Transition(*zip(*batch))
                    iql_agent.update(batch)

        # 结束一个episode，写轨迹到csv
        with open(traj_file_path, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            for row in episode_trajectory:
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
                state = torch.Tensor([env.reset()]).to(device)
                while True:
                    with torch.no_grad():
                        action = iql_agent.actor(state).to(device)
                        
                    action_np = action.cpu().numpy()[0]
                    next_state_raw, reward, done, _ = env.step(action_np)
                    # next_state, reward, done, info = env.step(action.numpy()[0])
                    episode_reward += reward
                    episode_step += 1
    
                    next_state = torch.Tensor([next_state_raw]).to(device)
    
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
    log_file.close()


if __name__ == "__main__":
    main()
