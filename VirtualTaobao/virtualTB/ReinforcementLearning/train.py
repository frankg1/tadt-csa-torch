import csv
import torch
import random
import gym  # 确保已经安装并导入gym
import virtualTB
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from collections import namedtuple
FLOAT = torch.FloatTensor
LONG = torch.LongTensor
import os
from  IQL import *
from TADT_CSA import *
# from DT4Rec import *
# from DT4IER import *
# from CDT4Rec import *
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
        
class TransitionDataset(Dataset):
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

class TrajectoryDataset(Dataset):
    def __init__(self, csv_path, max_seq_len=20):
        self.max_seq_len = max_seq_len
        self.trajectories = []

        print(f"Loading CSV from {csv_path}")
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            data = []
            for row in reader:
                state = np.array(eval(row['state']), dtype=np.float32)
                action = np.array(eval(row['action']), dtype=np.float32)
                reward = float(row['reward'])
                data.append((state, action, reward))

        print(f"Loaded {len(data)} transitions. Grouping into trajectories...")

        # Group transitions into trajectories of length max_seq_len
        num_trajs = len(data) // max_seq_len
        for i in range(num_trajs):
            traj = data[i * max_seq_len : (i + 1) * max_seq_len]
            states, actions, rewards = zip(*traj)

            self.trajectories.append({
                'state': torch.tensor(states, dtype=torch.float),            # shape: [max_seq_len, state_dim]
                'action': torch.tensor(actions, dtype=torch.float),        # shape: [max_seq_len, action_dim]
                'reward': torch.tensor(rewards, dtype=torch.float),        # shape: [max_seq_len]
                'length': max_seq_len                                       # all are full length
            })
            # print(states[0].shape)

        print(f"Constructed {len(self.trajectories)} trajectories of length {max_seq_len}")
        # 统计信息
        self._compute_statistics()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]
    
    def _compute_statistics(self):
        """计算数据集统计信息"""
        lengths = [t['length'] for t in self.trajectories]
        rewards = []
        actions = []
        
        for t in self.trajectories:
            rewards.extend(t['reward'][:t['length']])
            actions.extend(t['action'][:t['length']])
        # print(actions[0].shape)
        actions_tensor = torch.stack([torch.tensor(a).squeeze(0) for a in actions])  # shape: [N, 27]
        
        self.stats = {
            'num_trajectories': len(self.trajectories),
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'num_actions': len(set(actions)),
            # 'action_range': (min(actions), max(actions)),
            # 'action_range': (actions_tensor.min().item(), actions_tensor.max().item()),
            'action_range': (1,len(set(actions))),
            'positive_rate': np.mean(rewards),
            'total_interactions': len(rewards)
        }
        
        print("Dataset Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
            
def collate_fn_1(batch):
    """数据批处理函数"""
    batch_size = len(batch)
    
    # 收集所有数据
    obs_list = [item['state'] for item in batch]
    action_list = [item['action'] for item in batch]
    reward_list = [item['reward'] for item in batch]
    length_list = [item['length'] for item in batch]
    
    # 堆叠成张量
    obs = torch.stack(obs_list)  # [B, L, obs_len]
    action = torch.stack(action_list)  # [B, L]
    reward = torch.stack(reward_list)  # [B, L]
    length = torch.tensor(length_list, dtype=torch.long)  # [B]
    
    return {
        'state': obs,
        'action': action,
        'reward': reward,
        'length': length
    }
from ActionModel import *
def run_item_prediction(obs_seq, act_seq):
    model = ActionModel()
    model.load()
    model.eval()  # 关闭dropout等训练行为
    B,L,_ = obs_seq.shape
    # 构造假数据
    # user = torch.randn(1, 88)       # 1个用户特征向量
    # page = torch.randn(1, 1)        # 当前页面（或其他信息）
    # weight = torch.randn(1, 27)     # 权重或嵌入信息
    users = obs_seq[:,:,:88].cpu()
    pages = torch.ones((B, L, 1)).cpu()
    weights = act_seq.squeeze(2).cpu()
    # 预测动作
    actions = torch.zeros((B,L,2))
    
    # for b in range(B):
    for l in range(L):
        with torch.no_grad():
            # print(users[:,l,:].shape)
            # print(pages[:,l,:].shape)
            # print(weights[:,l,:].shape)
            actions[:,l,:] = model.predict(users[:,l,:], pages[:,l,:], weights[:,l,:])

    # print("输入向量维度：", user.shape[1] + page.shape[1] + weight.shape[1])
    # print("预测动作：", action)  # shape: [1, 2]，内容为 [a_id, b_id]
    actions_combined = actions[:, :, 0] * 100 + actions[:, :, 1]  # [B, L]
    actions_combined = actions_combined.squeeze(-1)  # 变成 [B, L, 1]
    # print(actions_combined)
    return actions_combined.long()

Trajectory = namedtuple('Trajectory', ('state', 'action', 'reward', 'length'))

class ReplayMemoryT(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a trajectory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Trajectory(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="IQL", help="模型名字")
    parser.add_argument('--train_with_trajectory', action='store_true', help='开启 trajectory 训练')
    parser.add_argument('--no_train_with_trajectory', dest='train_with_trajectory', action='store_false')
    
    parser.set_defaults(train_with_trajectory=True)
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--item_emb_dim', type=int, default=27, help='Item embedding dimension')
    parser.add_argument('--codebook_size', type=int, default=64, help='Codebook size')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for codebook')
    parser.add_argument('--max_seq_len', type=int, default=20, help='Maximum sequence length')
    parser.add_argument('--pos_encoding_type', type=str, default='learned', 
                       choices=['sin', 'learned'], help='Position encoding type')

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--context_length', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--item_num', type=int, default=5000)
    parser.add_argument('--data_dir_prefix', type=str, default='./data/')
    
    args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    csv_path = "trajectories/trajectory-ddpg.csv"#"trajectories/trajectory.csv"

    dataset = TrajectoryDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_1)
     # 获取数据集统计信息
    obs_len = 20  # 固定state长度
    obs_dim = 20  # state维度
    state_dim = dataset[0]['state'].shape[1] # 91
    action_dim = dataset[0]['action'].shape[2]
    print(obs_dim,action_dim)
    item_num = dataset.stats['num_actions']
    max_sign = max(dataset.stats['action_range']) + 1
    
    print(f"Dataset stats: {dataset.stats}")
    print(f"Model config: obs_len={obs_len}, obs_dim={obs_dim}, item_num={item_num}, max_sign={max_sign}")
    
    if args.model_name in ["IQL"]:
        dataset = TransitionDataset(csv_path)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    os.makedirs("models", exist_ok=True)

    # 初始化模型
    if args.model_name == "IQL":
        agent = IQL(state_dim, action_dim, device=device)
        save_path = "models/iql_model.pth"
    elif args.model_name == "TADT-CSA":
        # from TADT_CSA import *
        # 创建模型
        min_rtg_value = 1.0
        max_rtg_value = 10000.0
        min_adv_value = -5000.0
        max_adv_value = 0.0
        
        agent = TADT_CSA(
            obs_len=obs_len,
            obs_dim=obs_dim, 
            item_num=item_num,
            codebook_size=args.codebook_size,
            alpha=args.alpha,
            embed_dim=args.embed_dim,
            item_emb_dim=args.item_emb_dim,
            max_sign=max_sign,
            max_seq_len=args.max_seq_len,
            pos_encoding_type=args.pos_encoding_type,
            min_rtg_value=min_rtg_value, max_rtg_value=max_rtg_value, min_adv_value=min_adv_value, max_adv_value=max_adv_value
        ).to(device)
        save_path = "models"
        
        # 创建优化器
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # 创建损失函数
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.MSELoss()
        
        # 创建item embedding矩阵（随机初始化）
        item_emb = torch.randn(item_num, args.item_emb_dim).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in agent.parameters()):,}")
        print(f"Starting training for {args.num_epochs} epochs...")
        
    elif args.model_name == "DT4Rec":
        # from DT4Rec import *
        B, L = 32, 20
        state_dim, act_dim, obs_len, item_num, max_sign = 64, 64, 20, 100, 100
        item_emb_dim = 64
        learning_rate = 5e-3#1e-4
        num_epochs = 100
        rtg_bucket_num = 100
        min_rtg_value = 0#1.0
        max_rtg_value = 100#10000.0
        min_adv_value = -50#-5000.0
        max_adv_value = 0.0

        conf = GPTConfig(item_num, 3 * L, n_layer=3, n_head=4, n_embd=64, model_type=args.model_type, max_timestep=20,
                 rtg_bucket_number=rtg_bucket_num, min_rtg_value=min_rtg_value, max_rtg_value=max_rtg_value, min_adv_value=min_adv_value, max_adv_value=max_adv_value)
        agent = DT4Rec(conf).to(device)
        # optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        save_path = "models"
        
        # 创建优化器
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        # 创建损失函数
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.MSELoss()
        
        # 创建item embedding矩阵（随机初始化）
        item_emb = torch.randn(item_num, args.item_emb_dim).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in agent.parameters()):,}")
        print(f"Starting training for {args.num_epochs} epochs...")
    elif args.model_name == "DT4IER":
        # from DT4IER import *
        B, L = 32, 20
        state_dim, act_dim, obs_len, item_num, max_sign = 64, 64, 20, 100, 100
        item_emb_dim = 64
        learning_rate = 1e-4
        num_epochs = 100
        rtg_bucket_num = 100
        min_rtg_value = 1.0
        max_rtg_value = 10000.0
        min_adv_value = -5000.0
        max_adv_value = 0.0
        
        conf = GPTConfig(item_num, 3 * L, n_layer=3, n_head=4, n_embd=64, model_type=args.model_type, max_timestep=20,
                         rtg_bucket_number=rtg_bucket_num, min_rtg_value=min_rtg_value, max_rtg_value=max_rtg_value, min_adv_value=min_adv_value, max_adv_value=max_adv_value)
        agent = DT4IER(conf).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    elif args.model_name == "CDT4Rec":
        B, L = 32, 20
        state_dim, act_dim, obs_len, item_num, max_sign = 64, 64, 20, 100, 100
        item_emb_dim = 64
        learning_rate = 1e-4
        num_epochs = 100
        agent = CDT4Rec(state_dim=state_dim, act_dim=act_dim, item_num=item_num, hidden_size=64, item_emb_dim=64, max_length=L,
                        n_layer=3, n_head=4, activation_function='gelu').to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid model name")
    if args.train_with_trajectory:
        num_epochs = 50
        save_interval = 10  # 每10个epoch保存一次模型
        

        for epoch in range(num_epochs):
            if args.model_name == 'IQL':
                total_v_loss = 0
                total_c_loss = 0
                total_a_loss = 0
            elif args.model_name == "TADT-CSA":
                agent.train()
                epoch_losses = {
                    'total_loss': 0.0,
                    'action_loss': 0.0,
                    'rtg_loss': 0.0,
                    'adv_loss': 0.0,
                    'reward_loss': 0.0,
                    'transition_loss': 0.0,
                    'quantile_loss': 0.0,
                    'entropy_loss': 0.0
                }
            elif args.model_name == "DT4Rec":
                agent.train()
                total_loss = 0
                mse_loss = 0
                contra_loss = 0
            elif args.model_name == "DT4IER":
                agent.train()
            elif args.model_name == "CDT4Rec":
                agent.train()
            batches = 0
            # print("start learning from trajectories")
            for batch in dataloader:
                if args.model_name == "IQL":
                    v_loss, c_loss, a_loss = agent.update(batch)
                    total_v_loss += v_loss
                    total_c_loss += c_loss
                    total_a_loss += a_loss
                elif args.model_name == "TADT-CSA":
                    # 移动数据到设备
                    obs_seq = batch['state'].to(device)  # [B, L, obs_len]
                    act_id_seq = batch['action'].to(device)  # [B, L]
                    reward_seq = batch['reward'].to(device) / 10  # [B, L]
                    seq_lengths = batch['length'].to(device)  # [B]
                    # print("act_id_seq.shape:", act_id_seq.shape)
                    # act_id_seq = run_item_prediction(obs_seq, act_id_seq)
                    # print("act_id_seq.shape:", act_id_seq.shape)
                    B, L = obs_seq.shape[:2]
                    
                    # 计算RTG和advantage
                    rtg_seq, adv_seq = compute_rtg_and_advantage_trend_from_rewards(reward_seq)
                    
                    # 生成各种索引和mask
                    type_idx = generate_type_index(B, L, device=device)
                    time_idx = generate_time_index(B, L, device=device)
                    tgt_mask = generate_causal_mask(3 * L, device=device)
                    key_padding_mask = generate_padding_mask( seq_lengths.unsqueeze(1),  L, device=device)
                    # 前向传播
                    (action_preds, return_preds, reward_preds, cur_emb, next_emb, cur_act_emb, codebook_id, cur_codebook_id,
                     nxt_codebook_id, weights, s_t_seq) = agent(obs_seq, act_id_seq, rtg_seq, adv_seq, type_idx,
                                                                time_idx, tgt_mask=tgt_mask, key_padding_mask=key_padding_mask)
                    
                    # 准备ground truth
                    gt_actions = act_id_seq.clone().detach().view(-1, 27)
                    # print("action_preds:", action_preds.shape)
                    # print("gt_actions:", gt_actions.shape)
                    gt_rtgs = rtg_seq.clone().detach()
                    gt_advs = adv_seq.clone().detach()
                    
                    # 分割return predictions
                    rtg_preds, adv_preds = torch.split(return_preds, 1, dim=-1)
                    rtg_preds, adv_preds = rtg_preds.squeeze(-1), adv_preds.squeeze(-1)
                    
                    # 创建loss mask
                    loss_mask = (torch.arange(L, device=device).expand(B, L) < seq_lengths.unsqueeze(1))
                    loss_mask_pre = loss_mask[:, :-1]
                    loss_mask_flat = loss_mask.reshape(-1)
                    loss_mask_pre_flat = loss_mask_pre.reshape(-1)
                    
                    # 计算各种损失
                    action_preds_flat = action_preds.view(-1, action_preds.size(-1))
                    valid_action_preds_flat = action_preds_flat[loss_mask_flat]
                    valid_gt_actions = gt_actions[loss_mask_flat]
                    
                    action_loss = loss_fn(valid_action_preds_flat, valid_gt_actions)
                    rtg_loss = (loss_mask * (rtg_preds - gt_rtgs) ** 2).sum() / loss_mask.sum()
                    adv_loss = (loss_mask * (adv_preds - gt_advs) ** 2).sum() / loss_mask.sum()
                    reward_loss = (loss_mask * (reward_preds - reward_seq) ** 2).sum() / loss_mask.sum()
                    transition_loss = contrastive_transition_loss(agent.trans_predictor, cur_emb, cur_act_emb, next_emb, cur_codebook_id, nxt_codebook_id, loss_mask_pre_flat)
                    # print("act_id_seq.shape:", act_id_seq.shape)
                    act_id_seq = run_item_prediction(obs_seq, act_id_seq).to(device)
                    # print("act_id_seq.shape:", act_id_seq.shape)
                    quantile_loss = quantile_ranking_loss(action_preds, rtg_seq, codebook_id, act_id_seq, loss_mask)
                    
                    valid_weights = weights[loss_mask_flat]
                    entropy_loss = torch.mean(torch.sum(valid_weights * torch.log(valid_weights + 1e-8), dim=-1))
                    
                    total_loss = action_loss + rtg_loss + adv_loss + quantile_loss + reward_loss + transition_loss + entropy_loss
                    
                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    # 更新损失统计
                    epoch_losses['total_loss'] += total_loss.item()
                    epoch_losses['action_loss'] += action_loss.item()
                    epoch_losses['rtg_loss'] += rtg_loss.item()
                    epoch_losses['adv_loss'] += adv_loss.item()
                    epoch_losses['reward_loss'] += reward_loss.item()
                    epoch_losses['transition_loss'] += transition_loss.item()
                    epoch_losses['quantile_loss'] += quantile_loss.item()
                    epoch_losses['entropy_loss'] += entropy_loss.item()
                elif args.model_name == 'DT4Rec':
                    # 移动数据到设备
                    obs_seq = batch['state'].to(device)  # [B, L, obs_len]
                    act_id_seq = batch['action'].to(device)  # [B, L]
                    reward_seq = batch['reward'].to(device) / 10  # [B, L]
                    seq_lengths = batch['length'].to(device)  # [B]
                    # print("act_id_seq.shape:", act_id_seq.shape)
                    # act_id_seq = run_item_prediction(obs_seq, act_id_seq)
                    # print("act_id_seq.shape:", act_id_seq.shape)
                    B, L = obs_seq.shape[:2]
                    # print(seq_lengths)
                    # 计算RTG
                    rtg_seq = compute_rtg_from_rewards(reward_seq)
                    # print("rtg_seq.shape:",rtg_seq.shape)
                    # 生成attention mask
                    attention_mask = generate_attention_mask(seq_lengths, L,B, device=obs_seq.device)
                    # print(attention_mask.device)
                    # _, mse_loss, contra_loss = agent(obs_seq, act_id_seq, seq_lengths, rtg_seq, item_emb, attention_mask)
                    _, mse_loss, contra_loss = agent(obs_seq, act_id_seq, seq_lengths, rtg_seq, attention_mask)
                    total_loss = mse_loss + contra_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                elif args.model_name == "DT4IER":
                    obs_seq = batch['state'].to(device)  # [B, L, obs_len]
                    act_id_seq = batch['action'].to(device)  # [B, L]
                    reward_gt = batch['reward'].to(device) / 10  # [B, L]
                    seq_lengths = batch['length'].to(device)  # [B]
                    B, L = obs_seq.shape[:2]
                    # 计算RTG
                    rtg_seq = compute_rtg_from_rewards(reward_gt)
                    # 生成attention mask
                    attention_mask = generate_attention_mask(seq_lengths, L,B, device=obs_seq.device)
                    _, ce_loss, contra_loss, reward_loss = agent(obs_seq, act_id_seq, seq_lengths, rtg_seq, reward_gt, attention_mask)
                    total_loss = ce_loss + contra_loss + reward_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                elif args.model_name == "CDT4Rec":
                    obs_seq = batch['state'].to(device)  # [B, L, obs_len]
                    act_id_seq = batch['action'].to(device)  # [B, L]
                    reward_gt = batch['reward'].to(device) / 10  # [B, L]
                    seq_lengths = batch['length'].to(device)  # [B]
                    B, L = obs_seq.shape[:2]
                    # 计算RTG
                    rtg_seq = compute_rtg_from_rewards(reward_gt)
                    # 生成attention mask
                    attention_mask = generate_attention_mask(seq_lengths, L,B, device=obs_seq.device)
                    # print(rtg_seq)
                    state_preds, action_preds, return_preds = agent(obs_seq, act_id_seq, rtg_seq, attention_mask=attention_mask)
                
                    gt_actions = act_id_seq.squeeze(2).clone().detach().view(-1, 27)
                    gt_rtgs = rtg_seq.clone().detach()
                
                    action_preds_flat = action_preds.view(-1, action_preds.size(-1))
                    attention_mask_flat = attention_mask.reshape(-1)
                    # action loss
                    valid_action_preds_flat = action_preds_flat[attention_mask_flat]
                    valid_gt_actions = gt_actions[attention_mask_flat]
                    # print(valid_action_preds_flat.shape)
                    # print(valid_gt_actions.shape)
                    action_loss = loss_fn(valid_action_preds_flat, valid_gt_actions)
                    # return loss
                    rtg_loss = (attention_mask * (return_preds - gt_rtgs) ** 2).sum() / attention_mask.sum()  # [B, L]
                    # total loss
                    total_loss = action_loss + rtg_loss
                
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                
                batches += 1
        
            if args.model_name == "IQL":
                print(f"Epoch {epoch+1}/{num_epochs} | V_loss: {total_v_loss/batches:.4f} | Critic_loss: {total_c_loss/batches:.4f} | Actor_loss: {total_a_loss/batches:.4f}")
            elif args.model_name == "TADT-CSA":
                # 计算epoch平均损失
                num_batches = len(dataloader)
                for key in epoch_losses:
                    epoch_losses[key] /= num_batches
                
                # 打印epoch结果
                print(f"\nEpoch {epoch}/{args.num_epochs} Summary:")
                print(f"  Total Loss: {epoch_losses['total_loss']:.4f}")
                print(f"  Action Loss: {epoch_losses['action_loss']:.4f}")
                print(f"  RTG Loss: {epoch_losses['rtg_loss']:.4f}")
                print(f"  Advantage Loss: {epoch_losses['adv_loss']:.4f}")
                print(f"  Reward Loss: {epoch_losses['reward_loss']:.4f}")
                print(f"  Transition Loss: {epoch_losses['transition_loss']:.4f}")
                print(f"  Quantile Loss: {epoch_losses['quantile_loss']:.4f}")
                print(f"  Entropy Loss: {epoch_losses['entropy_loss']:.4f}")
            elif args.model_name == "DT4Rec":
                print(f"Epoch {epoch} | Total Loss: {total_loss.item():.4f} | MSE Loss: {mse_loss.item():.4f} | "
                          f"Contra Loss: {contra_loss.item():.4f}")
            elif args.model_name == "DT4IER":
                 print(f"Epoch {epoch} | Total Loss: {total_loss.item():.4f} | CE Loss: {ce_loss.item():.4f} | "
          f"Contra Loss: {contra_loss.item():.4f} | Reward Loss: {reward_loss.item():.4f}")
            elif args.model_name == "CDT4Rec":
                print(f"Epoch {epoch} | Total Loss: {total_loss.item():.4f} | CE Loss: {action_loss.item():.4f} | "
          f"Return Loss: {rtg_loss.item():.4f}")
            # 定期保存模型
            if (epoch + 1) % save_interval == 0:
                if args.model_name == "IQL":
                    save_iql_model(agent, epoch, save_path)
                elif args.model_name == "TADT-CSA":
                    save_tadt_csa_model(agent, optimizer, epoch, save_path)
                

        # 训练结束后也保存一次
        if args.model_name == "IQL":
            save_iql_model(agent, epoch, save_path)
        elif args.model_name == "TADT-CSA":
            save_tadt_csa_model(agent, optimizer, epoch, save_path)


    # 加载最后保存的模型
    if args.model_name == "IQL":
        checkpoint_path = "models/iql_model.pth"
        load_iql_model(agent, checkpoint_path, device)
    elif args.model_name == "TADT-CSA":
        print("have not load any params")
    elif args.model_name == "DT4Rec":
        print("have not load any params")
    elif args.model_name == "DT4IER":
        print("have not load any params")
    elif args.model_name == "CDT4Rec":
        print("have not load any params")
    else:
        raise ValueError("Invalid model name")

    # 初始化环境
    env = gym.make('VirtualTB-v0')
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if args.model_name == "IQL":
        memory = ReplayMemory(1000000)
        traj_file_path = "trajectories/trajectory_iql.csv"  # 保存轨迹的csv
        log_file = open("logs/training_log_iql.txt", "a")
    elif args.model_name == "TADT-CSA":
        memory = ReplayMemoryT(capacity=1000000)
        traj_file_path = "trajectories/trajectory_tadt_csa.csv"  # 保存轨迹的csv
        log_file = open("logs/training_log_tadt_csa.txt", "a")
    elif args.model_name == "DT4Rec":
        memory = ReplayMemoryT(capacity=1000000)
        traj_file_path = "trajectories/trajectory_dt4rec.csv"  # 保存轨迹的csv
        log_file = open("logs/training_log_dt4rec.txt", "a")
    elif args.model_name == "DT4IER":
        memory = ReplayMemoryT(capacity=1000000)
        traj_file_path = "trajectories/trajectory_dt4ier.csv"  # 保存轨迹的csv
        log_file = open("logs/training_log_dt4ier.txt", "a")
    elif args.model_name == "CDT4Rec":
        memory = ReplayMemoryT(capacity=1000000)
        traj_file_path = "trajectories/trajectory_cdt4rec.csv"  # 保存轨迹的csv
        log_file = open("logs/training_log_cdt4rec.txt", "a")
    

    num_episodes = 100000
    for i_episode in range(num_episodes):
        state = torch.Tensor([env.reset()]).to(device)

        episode_reward = 0
        episode_step = 0
        steps = 0
        max_reward = 0
        min_reward = 1
        max_rtg = 0
        min_rtg = 1
        max_adv = 0
        min_adv = 1
        episode_trajectory = []
        states = []
        actions = []
        rewards = []
        # steps = []
        state_list = torch.stack([state]).to(device)
        action_list = torch.stack([torch.rand(27)]).unsqueeze(0).to(device) #None  # 空 tensor，shape: [1, 1]
        rtg_list = torch.tensor([[[100.0]]], dtype=torch.float32).to(device)  # shape: [1, 1]
        rtg_list1 = torch.tensor([[[100.0]]], dtype=torch.float32).to(device)  # shape: [1, 1]
        # print(state_list.shape)
        # print(rtg_list.shape)
        done = False
        while not done:
            # 这里你要实现select_action方法，或者直接调用actor网络输出动作
            # 假设actor网络输出动作张量，转换为numpy并给环境
            if args.model_name == "IQL":
                with torch.no_grad():
                    action = agent.actor(state).to(device)
                action_np = action.cpu().numpy()[0]
            elif args.model_name == "TADT-CSA":
                rtg_seq, adv_seq = compute_rtg_and_advantage_trend_from_rewards(rtg_list.squeeze(-1))
                action = agent.predict_single_step(state_list, action_list, rtg_seq, adv_seq)
                action_np = action.detach().cpu().squeeze(0).numpy()
            elif args.model_name == "DT4Rec":
                # print("action_list.shape:",action_list)
                action = agent.predict_single_step(state_list, action_list, rtg_list)
                action_np = action.detach().cpu().squeeze(0).numpy()
            elif args.model_name == "DT4IER":
                # print(state_list.shape)
                # action = agent.predict_single_step(state_list, action_list, torch.tensor(steps+1).to(device), rtg_list)
                rtg_combined = torch.cat([rtg_list, rtg_list1], dim=-1)
                action = agent.predict_single_step(state_list, action_list, rtg_combined)
            # 可能需要根据环境action space做映射，假设直接用action.numpy()[0]

                # action_np = action.cpu().numpy()[0]
                action_np = action.detach().cpu().squeeze(0).numpy()
            elif args.model_name == "CDT4Rec":
                # print("action_list.shape:",action_list)
                action = agent.predict_single_step(state_list, action_list, rtg_list)
                action_np = action.detach().cpu().squeeze(0).numpy()
            # print(action_np.shape)
            next_state_raw, reward, done, _ = env.step(action_np)
            reward = reward/10
            max_reward = max(max_reward, reward)
            min_reward = min(min_reward, reward)
            # max_rtg = max(max_rtg,rtg_seq.max().item())
            # min_rtg = min(min_rtg,rtg_seq.min().item())
            # max_adv = max(max_adv,adv_seq.max().item())
            # min_adv = min(min_adv,adv_seq.min().item())
            episode_reward += reward
            episode_step += 1
            steps += 1
            next_state = torch.Tensor([next_state_raw]).to(device)
            reward_tensor = torch.Tensor([reward]).to(device)
            done_tensor = torch.Tensor([done]).to(device)
            
            # print(state_list.shape)
            # print(next_state.shape)
            
            new_value = ((rtg_list[0, -1, 0] - reward)/0.9).unsqueeze(0).unsqueeze(0).unsqueeze(1)
            new_value1 = ((rtg_list1[0, -1, 0] - reward)/0.6).unsqueeze(0).unsqueeze(0).unsqueeze(1)
            # if action_list is not None:
            #     state_list = torch.cat([state_list, next_state.unsqueeze(1)], dim=1)
            #     action_list = torch.cat([action_list, action.unsqueeze(1)], dim=1)
            #     rtg_list = torch.cat([rtg_list, new_value], dim=1)
            # else:
            #     state_list = torch.stack([state]).to(device)
            #     action_list = torch.stack([action]).to(device)
            #     rtg_list = new_value.to(device)
            state_list = torch.cat([state_list, next_state.unsqueeze(1)], dim=1)
            # print(action_list.shape)
            # print(action.shape)
            action_list = torch.cat([action_list, action.unsqueeze(1)], dim=1)
            # action_list = torch.cat([action_list, action], dim=0)
            rtg_list = torch.cat([rtg_list, new_value], dim=1)
            rtg_list1 = torch.cat([rtg_list1, new_value1], dim=1)
            states.append(state.cpu().numpy().tolist()[0])
            actions.append(action.detach().cpu().numpy().tolist())
            rewards.append(reward)
            # 这里要存经验到memory，假设你有定义ReplayMemory和push方法
            if args.model_name == "IQL":
                memory.push(state, action, done_tensor, next_state, reward_tensor)
            else:
                if steps == 20:
                    states_seq = torch.tensor(states)    # shape [seq_len, ...]
                    actions_seq = torch.tensor(actions)  # shape [seq_len, ...]
                    rewards_seq = torch.tensor(rewards, dtype=torch.float32)
                    steps_seq = torch.tensor([steps])
                    # print(steps_seq)
                    memory.push(states_seq, actions_seq, rewards_seq, steps_seq)
                    states = []
                    actions = []
                    rewards = []
                    steps = 0
                    new_value = (rtg_list[0, -1, 0] - reward).unsqueeze(0).unsqueeze(0).unsqueeze(1)
                    state_list = torch.stack([state]).to(device)
                    action_list = torch.stack([action]).to(device)
                    rtg_list = new_value.to(device)
                # else:
                #     states.append(state.cpu().numpy().tolist()[0])
                #     actions.append(action.detach().cpu().numpy().tolist())
                #     rewards.append(reward)
                    # steps.append(episode_step)
                    
            # 保存轨迹（转换tensor到列表或标量）
            episode_trajectory.append((
                i_episode,
                episode_step,
                state.cpu().numpy().tolist()[0],
                action.detach().cpu().numpy().tolist(),
                reward
            ))

            state = next_state

            # 这里可以调用更新参数的逻辑，比如每步采样训练
        if len(memory) > 32:
            for _ in range(5):
                batch = memory.sample(32)
                batch = Trajectory(*zip(*batch))
                if args.model_name == "IQL":
                    agent.update(batch)
                elif args.model_name == "TADT-CSA":
                    obs_seq = torch.stack(batch.state).to(device)  # [B, L, obs_len]
                    act_id_seq = torch.stack(batch.action).to(device)  # [B, L]
                    reward_seq = torch.stack(batch.reward).to(device)  # [B, L]
                    seq_lengths = torch.stack(batch.length).squeeze(1).to(device)  # [B]
                    # seq_lengths = torch.ones_like(seq_lengths)
                    
                    B, L = obs_seq.shape[:2]
                    
                    # 计算RTG和advantage
                    rtg_seq, adv_seq = compute_rtg_and_advantage_trend_from_rewards(reward_seq)
                    
                    # 生成各种索引和mask
                    type_idx = generate_type_index(B, L, device=device)
                    time_idx = generate_time_index(B, L, device=device)
                    tgt_mask = generate_causal_mask(3 * L, device=device)
                    key_padding_mask = generate_padding_mask( seq_lengths.unsqueeze(1),  L, device=device)
                    # 前向传播
                    (action_preds, return_preds, reward_preds, cur_emb, next_emb, cur_act_emb, codebook_id, cur_codebook_id,
                     nxt_codebook_id, weights, s_t_seq) = agent(obs_seq, act_id_seq, rtg_seq, adv_seq, type_idx,
                                                                time_idx, tgt_mask=tgt_mask, key_padding_mask=key_padding_mask)
                    
                    # 准备ground truth
                    gt_actions = act_id_seq.clone().detach().view(-1, 27)
                    # print("action_preds:", action_preds.shape)
                    # print("gt_actions:", gt_actions.shape)
                    gt_rtgs = rtg_seq.clone().detach()
                    gt_advs = adv_seq.clone().detach()
                    
                    # 分割return predictions
                    rtg_preds, adv_preds = torch.split(return_preds, 1, dim=-1)
                    rtg_preds, adv_preds = rtg_preds.squeeze(-1), adv_preds.squeeze(-1)
                    
                    # 创建loss mask
                    loss_mask = (torch.arange(L, device=device).expand(B, L) < seq_lengths.unsqueeze(1))
                    loss_mask_pre = loss_mask[:, :-1]
                    loss_mask_flat = loss_mask.reshape(-1)
                    loss_mask_pre_flat = loss_mask_pre.reshape(-1)
                    
                    # 计算各种损失
                    action_preds_flat = action_preds.view(-1, action_preds.size(-1))
                    valid_action_preds_flat = action_preds_flat[loss_mask_flat]
                    valid_gt_actions = gt_actions[loss_mask_flat]
                    
                    action_loss = loss_fn(valid_action_preds_flat, valid_gt_actions)
                    rtg_loss = (loss_mask * (rtg_preds - gt_rtgs) ** 2).sum() / loss_mask.sum()
                    adv_loss = (loss_mask * (adv_preds - gt_advs) ** 2).sum() / loss_mask.sum()
                    reward_loss = (loss_mask * (reward_preds - reward_seq) ** 2).sum() / loss_mask.sum()
                    transition_loss = contrastive_transition_loss(agent.trans_predictor, cur_emb, cur_act_emb, next_emb, cur_codebook_id, nxt_codebook_id, loss_mask_pre_flat)
                    # print("act_id_seq.shape:", act_id_seq.shape)
                    act_id_seq = run_item_prediction(obs_seq, act_id_seq).to(device)
                    # print("act_id_seq.shape:", act_id_seq.shape)
                    quantile_loss = quantile_ranking_loss(action_preds, rtg_seq, codebook_id, act_id_seq, loss_mask)
                    
                    valid_weights = weights[loss_mask_flat]
                    entropy_loss = torch.mean(torch.sum(valid_weights * torch.log(valid_weights + 1e-8), dim=-1))
                    
                    total_loss = action_loss + rtg_loss + adv_loss + quantile_loss + reward_loss + transition_loss + entropy_loss
                    
                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                elif args.model_name == "DT4Rec":
                    # print(batch)
                    obs_seq = torch.stack(batch.state).to(device)  # [B, L, obs_len]
                    act_id_seq = torch.stack(batch.action).to(device)  # [B, L]
                    reward_seq = torch.stack(batch.reward).to(device)  # [B, L]
                    seq_lengths = torch.stack(batch.length).squeeze(1).to(device)  # [B]
                    # seq_lengths = torch.ones_like(seq_lengths)
                    
                    B, L = obs_seq.shape[:2]
                    # print(seq_lengths)
                    # 计算RTG
                    rtg_seq = compute_rtg_from_rewards(reward_seq)
                    # print("rtg_seq.shape:",rtg_seq.shape)
                    # 生成attention mask
                    attention_mask = generate_attention_mask(seq_lengths, L,B, device=obs_seq.device)
                    # print(attention_mask.device)
                    # _, mse_loss, contra_loss = agent(obs_seq, act_id_seq, seq_lengths, rtg_seq, item_emb, attention_mask)
                    _, mse_loss, contra_loss = agent(obs_seq, act_id_seq, seq_lengths, rtg_seq, attention_mask)
                    total_loss = mse_loss + contra_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                elif args.model_name == "DT4IER":
                    obs_seq = torch.stack(batch.state).to(device)  # [B, L, obs_len]
                    act_id_seq = torch.stack(batch.action).to(device)  # [B, L]
                    reward_gt = torch.stack(batch.reward).to(device)  # [B, L]
                    seq_lengths = torch.stack(batch.length).squeeze(1).to(device)  # [B]
                    B, L = obs_seq.shape[:2]
                    # 计算RTG
                    rtg_seq = compute_rtg_from_rewards(reward_gt)
                    # 生成attention mask
                    attention_mask = generate_attention_mask(seq_lengths, L,B, device=obs_seq.device)
                    _, ce_loss, contra_loss, reward_loss = agent(obs_seq, act_id_seq, seq_lengths, rtg_seq, reward_gt, attention_mask)
                    total_loss = ce_loss + contra_loss + reward_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                elif args.model_name == "CDT4Rec":
                    obs_seq = torch.stack(batch.state).to(device)  # [B, L, obs_len]
                    act_id_seq = torch.stack(batch.action).to(device)  # [B, L]
                    reward_gt = torch.stack(batch.reward).to(device)  # [B, L]
                    seq_lengths = torch.stack(batch.length).squeeze(1).to(device)  # [B]
                    B, L = obs_seq.shape[:2]
                    # 计算RTG
                    rtg_seq = compute_rtg_from_rewards(reward_gt)
                    # 生成attention mask
                    attention_mask = generate_attention_mask(seq_lengths, L,B, device=obs_seq.device)
                    # print(rtg_seq)
                    state_preds, action_preds, return_preds = agent(obs_seq, act_id_seq, rtg_seq, attention_mask=attention_mask)
                
                    gt_actions = act_id_seq.squeeze(2).clone().detach().view(-1, 27)
                    gt_rtgs = rtg_seq.clone().detach()
                
                    action_preds_flat = action_preds.view(-1, action_preds.size(-1))
                    attention_mask_flat = attention_mask.reshape(-1)
                    # action loss
                    valid_action_preds_flat = action_preds_flat[attention_mask_flat]
                    valid_gt_actions = gt_actions[attention_mask_flat]
                    # print(valid_action_preds_flat.shape)
                    # print(valid_gt_actions.shape)
                    action_loss = loss_fn(valid_action_preds_flat, valid_gt_actions)
                    # return loss
                    rtg_loss = (attention_mask * (return_preds - gt_rtgs) ** 2).sum() / attention_mask.sum()  # [B, L]
                    # total loss
                    total_loss = action_loss + rtg_loss
                
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                if args.model_name == "IQL":
                    print(f"Epoch {epoch+1}/{num_epochs} | V_loss: {total_v_loss/batches:.4f} | Critic_loss: {total_c_loss/batches:.4f} | Actor_loss: {total_a_loss/batches:.4f}")
                elif args.model_name == "TADT-CSA":
                    # 计算epoch平均损失
                    num_batches = len(dataloader)
                    for key in epoch_losses:
                        epoch_losses[key] /= num_batches
                    
                    # 打印epoch结果
                    print(f"\nEpoch {i_episode}/{10000} Summary:")
                    print(f"  Total Loss: {total_loss:.4f}")
                    # print(f"  Action Loss: {epoch_losses['action_loss']:.4f}")
                    # print(f"  RTG Loss: {epoch_losses['rtg_loss']:.4f}")
                    # print(f"  Advantage Loss: {epoch_losses['adv_loss']:.4f}")
                    # print(f"  Reward Loss: {epoch_losses['reward_loss']:.4f}")
                    # print(f"  Transition Loss: {epoch_losses['transition_loss']:.4f}")
                    # print(f"  Quantile Loss: {epoch_losses['quantile_loss']:.4f}")
                    # print(f"  Entropy Loss: {epoch_losses['entropy_loss']:.4f}")
                elif args.model_name == "DT4Rec":
                    print(f"Epoch {i_episode} | Total Loss: {total_loss.item():.4f} | MSE Loss: {mse_loss.item():.4f} | "
                              f"Contra Loss: {contra_loss.item():.4f}")
                elif args.model_name == "DT4IER":
                     print(f"Epoch {i_episode} | Total Loss: {total_loss.item():.4f} | CE Loss: {ce_loss.item():.4f} | "
              f"Contra Loss: {contra_loss.item():.4f} | Reward Loss: {reward_loss.item():.4f}")
                elif args.model_name == "CDT4Rec":
                    print(f"Epoch {i_episode} | Total Loss: {total_loss.item():.4f} | CE Loss: {action_loss.item():.4f} | "
              f"Return Loss: {rtg_loss.item():.4f}")
        # print("Episode: {}, total numsteps: {}, average reward: {:.4f}, CTR: {:.4f}, max_reward:{:.4f}, min_reward:{:.4f}, max_rtg:{:.4f}, min_rtg:{:.4f}, max_adv:{:.4f}, min_adv:{:.4f}".format(
        #         i_episode, episode_step, episode_reward / 50, episode_reward / episode_step / 10, max_reward, min_reward,max_rtg, min_rtg,max_adv,min_adv))
        # print("Episode: {}, total numsteps: {}, average reward: {:.4f}, CTR: {:.4f}, max_reward:{:.4f}, min_reward:{:.4f}".format(
        #         i_episode, episode_step, episode_reward / 50, episode_reward / episode_step / 10, max_reward, min_reward))
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
            for i in range(50): #50个测试episode
                state = torch.Tensor([env.reset()]).to(device)
                steps = 0
                state_list = torch.stack([state]).to(device)
                action_list = None  # 空 tensor，shape: [1, 1]
                rtg_list = torch.tensor([[[1000.0]]], dtype=torch.float32).to(device)  # shape: [1, 1]
                while True:
                    if args.model_name == "IQL":
                        with torch.no_grad():
                            action = agent.actor(state).to(device)
                        action_np = action.cpu().numpy()[0]
                    elif args.model_name == "TADT-CSA":
                        rtg_seq, adv_seq = compute_rtg_and_advantage_trend_from_rewards(rtg_list.squeeze(-1))
                        action = agent.predict_single_step(state_list, action_list, rtg_seq, adv_seq)
                        action_np = action.detach().cpu().squeeze(0).numpy()
                    elif args.model_name == "DT4Rec" or args.model_name == "CDT4Rec":
                        # print("action_list.shape:",action_list)
                        action = agent.predict_single_step(state_list, action_list, rtg_list)
                    # 可能需要根据环境action space做映射，假设直接用action.numpy()[0]
        
                        # action_np = action.cpu().numpy()[0]
                        action_np = action.detach().cpu().squeeze(0).numpy()
                    elif args.model_name == "DT4IER":
                        # print(state_list.shape)
                        # action = agent.predict_single_step(state_list, action_list, torch.tensor(steps+1).to(device), rtg_list)
                        action = agent.predict_single_step(state_list, action_list, rtg_list)
                        action_np = action.detach().cpu().squeeze(0).numpy()
                    # action_np = action.cpu().numpy()[0]
                    next_state, reward, done, info = env.step(action_np)
                    episode_reward += reward
                    episode_step += 1
                    steps += 1
                    next_state = torch.Tensor([next_state]).to(device)
                    new_value = (rtg_list[0, -1, 0] - reward).unsqueeze(0).unsqueeze(0).unsqueeze(1)
                    if action_list is not None:
                        state_list = torch.cat([state_list, next_state.unsqueeze(1)], dim=1)
                        action_list = torch.cat([action_list, action.unsqueeze(1)], dim=1)
                        rtg_list = torch.cat([rtg_list, new_value], dim=1)
                    else:
                        state_list = torch.stack([state]).to(device)
                        action_list = torch.stack([action]).to(device)
                        rtg_list = new_value.to(device)
                    if steps == 20:
                        steps = 0
                        new_value = (rtg_list[0, -1, 0] - reward).unsqueeze(0).unsqueeze(0).unsqueeze(1)
                        state_list = torch.stack([state]).to(device)
                        action_list = torch.stack([action]).to(device)
                        rtg_list = new_value.to(device)
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
