import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        try:
            state = batch['state'].to(self.device)
            action = batch['action'].to(self.device)
            reward = batch['reward'].to(self.device)
            next_state = batch['next_state'].to(self.device)
            done = batch['done'].to(self.device)
        except:
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

def save_iql_model(iql_agent, epoch, save_path):
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

