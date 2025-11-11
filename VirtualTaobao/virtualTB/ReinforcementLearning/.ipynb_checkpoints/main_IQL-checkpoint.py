import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "trajectories/trajectory.csv"

    dataset = TrajectoryDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    state_dim = dataset[0]['state'].shape[0]
    action_dim = dataset[0]['action'].shape[0]
    print(state_dim,action_dim)
    iql_agent = IQL(state_dim, action_dim, device=device)

    # num_epochs = 50
    # for epoch in range(num_epochs):
    #     total_v_loss = 0
    #     total_c_loss = 0
    #     total_a_loss = 0
    #     batches = 0
    #     for batch in dataloader:
    #         # print(batch)
    #         v_loss, c_loss, a_loss = iql_agent.update(batch)
    #         total_v_loss += v_loss
    #         total_c_loss += c_loss
    #         total_a_loss += a_loss
    #         batches += 1
    #     print(f"Epoch {epoch+1}/{num_epochs} | V_loss: {total_v_loss/batches:.4f} | Critic_loss: {total_c_loss/batches:.4f} | Actor_loss: {total_a_loss/batches:.4f}")
    num_epochs = 50
    save_interval = 10  # 每10个epoch保存一次模型
    save_path = "./iql_model.pth"

    for epoch in range(num_epochs):
        total_v_loss = 0
        total_c_loss = 0
        total_a_loss = 0
        batches = 0
        for batch in dataloader:
            v_loss, c_loss, a_loss = iql_agent.update(batch)
            total_v_loss += v_loss
            total_c_loss += c_loss
            total_a_loss += a_loss
            batches += 1
        print(f"Epoch {epoch+1}/{num_epochs} | V_loss: {total_v_loss/batches:.4f} | Critic_loss: {total_c_loss/batches:.4f} | Actor_loss: {total_a_loss/batches:.4f}")

        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'actor_state_dict': iql_agent.actor.state_dict(),
                'critic_state_dict': iql_agent.critic.state_dict(),
                'v_net_state_dict': iql_agent.v_net.state_dict(),
                'actor_optimizer_state_dict': iql_agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': iql_agent.critic_optimizer.state_dict(),
                'v_optimizer_state_dict': iql_agent.v_optimizer.state_dict(),
                'epoch': epoch + 1
            }, save_path)
            print(f"Model saved at epoch {epoch+1} to {save_path}")

    # 训练结束后也保存一次
    torch.save({
        'actor_state_dict': iql_agent.actor.state_dict(),
        'critic_state_dict': iql_agent.critic.state_dict(),
        'v_net_state_dict': iql_agent.v_net.state_dict(),
        'actor_optimizer_state_dict': iql_agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': iql_agent.critic_optimizer.state_dict(),
        'v_optimizer_state_dict': iql_agent.v_optimizer.state_dict(),
        'epoch': num_epochs
    }, save_path)
    print(f"Final model saved to {save_path}")

if __name__ == "__main__":
    main()
