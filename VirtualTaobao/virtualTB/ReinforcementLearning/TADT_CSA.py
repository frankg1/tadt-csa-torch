import os
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse


def generate_causal_mask(seq_len, device):
    # PyTorch Transformer expects mask shape [L, L], with True values being masked positions
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def generate_padding_mask(lengths, max_seq_len, device):
    # 创建范围序列 [0, 1, 2, ..., L-1]
    range_row = torch.arange(max_seq_len, device=device).unsqueeze(0).repeat(lengths.shape[0], 1)  # [B, L]
    # lengths expand成 [B, L]
    lengths_expanded = lengths.repeat(1, max_seq_len)  # [B, L]
    # mask = True 表示padding位置（idx >= lengths）
    padding_mask = range_row >= lengths_expanded
    return padding_mask


def generate_type_index(batch_size, seq_len, device):
    type_idx = torch.arange(0, 3, device=device)  # [0, 1, 2]
    # 横向repeat
    type_idx_repeated = type_idx.repeat(seq_len)  # [1,2,3,1,2,3,...], 长度 = 3*L
    # 添加batch维度，纵向repeat B次
    type_idx_repeated = type_idx_repeated.unsqueeze(0).repeat(batch_size, 1)  # [B, 3*K]
    return type_idx_repeated


def generate_time_index(batch_size, seq_len, device, type_num=3):
    time_idx = torch.arange(0, seq_len, device=device)  # [0, 2, ..., 99]
    # 每个元素repeat M次
    time_idx_repeated = time_idx.repeat_interleave(type_num)  # [0,0,0, 1,1,1, ..., 99,99,99]
    # 添加batch维度，纵向repeat B次
    time_idx_repeated = time_idx_repeated.unsqueeze(0).repeat(batch_size, 1)  # [B, T*M]
    return time_idx_repeated


def compute_rtg_and_advantage_trend_from_rewards(reward_seq, gamma=0.9):
    """
    padding的位置reward必须为0，否则影响RTG计算。
    根据 reward 序列计算 advantage trend。

    输入：
        reward_seq: [B, L]  奖励序列
        gamma: float 衰减因子（0~1）

    返回：
        adv_trend: [B, L] 每个时刻的 advantage trend
    """
    B, L = reward_seq.shape
    device = reward_seq.device
    # Step 1: 计算 Return-to-Go
    rtg = torch.zeros_like(reward_seq).to(device)
    for t in reversed(range(L)):
        if t == L - 1:
            rtg[:, t] = reward_seq[:, t]
        else:
            rtg[:, t] = reward_seq[:, t] + gamma * rtg[:, t + 1]  # G_t = r_t + γ G_{t+1}
    # Step 2: 计算 advantage trend（带衰减的差分和）
    adv_trend = torch.zeros_like(reward_seq)
    for t in range(1, L):
        # 差分序列：G_0, G_1, ..., G_t 的差分
        diffs = rtg[:, 1:t+1] - rtg[:, :t]   # G_t - G_{t-1}, [B, t]
        gammas = gamma ** torch.arange(t - 1, -1, -1).float().to(device)  # [0, ..., t-1]
        weighted_diffs = diffs * gammas.view(1, -1)  # gamma ^ (t - i)
        adv_trend[:, t] = weighted_diffs.sum(dim=1)
    return rtg, adv_trend  # [B, L]


def quantile_ranking_loss(action_preds, rtg_seq, codebook_id, act_id_seq, loss_mask, quantile=0.8, delta=0.1):
    """
    action_preds: [B, L, act_dim] 动作预测
    rtg_seq: [B, L] Return-To-Go
    codebook_id: [B*L, 1] codebook聚类id，flatten后
    act_id_seq: [B, L] 原动作ID序列
    loss_mask: [B, L] mask矩阵适配变长序列，1表示参与计算，0表示不参与计算

    返回：单步的ranking loss标量
    """
    B, L = rtg_seq.shape
    device = rtg_seq.device
    total_loss = torch.tensor(0.0, device=device)
    total_pairs = 0
    # 预处理：将codebook_id恢复为[B, L]
    codebook_id_mat = codebook_id.view(B, L)
    # 遍历时间步 t=1~L-1，因为要对比前面子轨迹到 t-1
    for t in range(0, L):
        # Batch所有样本都被mask
        if loss_mask[:, t].sum() == 0:
            break
        # 取前t步的codebook id 和前t - 1步的动作，用于判断子轨迹是否完全相同, 构造group key
        prefix_codebook = codebook_id_mat[:, :t+1].tolist()  # 每个样本的前缀codebook id
        prefix_action = act_id_seq[:, :t].tolist()       # 每个样本的前缀动作
        groups = defaultdict(list)
        # 构建group
        for i in range(B):
            if loss_mask[i, t]:
                key = (tuple(prefix_codebook[i]), tuple(prefix_action[i]))
                # print(key)
                groups[key].append(i)

        # 对每个组内部两两比较
        for group_indices in groups.values():
            # group样本少于2个
            if len(group_indices) < 2:
                continue
            # 获取group内所有样本的RTG和action scores
            group_rtg = rtg_seq[group_indices, t]  # [N]
            group_scores = action_preds[group_indices, t]  # [N, act_dim]
            group_actions = act_id_seq[group_indices, t]  # [N]
            # print(group_actions)
            # print(group_rtg)
            # 获取每个样本的当前动作得分
            # selected_scores = torch.gather(
            #     group_scores,
            #     dim=1,
            #     index=group_actions.unsqueeze(1)
            # ).squeeze(1)  # [N]
            selected_scores = group_scores

            # 计算分位数阈值
            N = len(group_indices)
            k = int(N * quantile)
            if k == 0 or k == N:
                continue
            # 获取top-K样本索引
            pos_indices = torch.topk(group_rtg, k, dim=0).indices
            pos_mask = torch.zeros_like(group_rtg, dtype=torch.bool)
            pos_mask[pos_indices] = True
            all_indices = torch.arange(N, device=group_rtg.device).reshape(-1, 1)
            neg_indices = all_indices[~pos_mask]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            # 向量化计算正负样本得分差异
            pos_scores = selected_scores[pos_indices]  # [P]
            neg_scores = selected_scores[neg_indices]  # [N-P]

            # 扩展维度进行广播计算
            pos_scores_expanded = pos_scores.view(-1, 1)  # [P, 1]
            neg_scores_expanded = neg_scores.view(1, -1)  # [1, N-P]

            # 计算所有正负对的得分差异
            score_diff = pos_scores_expanded - neg_scores_expanded - delta  # [P, N-P]

            # 计算loss（使用margin-based sigmoid loss）
            loss_matrix = -F.logsigmoid(score_diff)  # [P, N-P]

            # 平均每个正样本与负样本对的loss
            group_loss = loss_matrix.sum()
            total_loss += group_loss
            total_pairs += len(pos_indices) * len(neg_indices)
    if total_pairs > 0:
        return total_loss / total_pairs
    else:
        return torch.tensor(0.0, device=device)


def contrastive_transition_loss(trans_predictor, cur_emb, act_emb, next_emb, cur_codebook_id, nxt_codebook_id, loss_mask,
                                K=5, tau=1.0):
    """
    对比学习 transition loss，使用 InfoNCE + 合法性掩码

    trans_predictor: 预测下一状态embedding的网络，输出logits（未sigmoid）
    cur_emb: [B * (L - 1), D] 当前状态embedding
    act_emb: [B * (L - 1), act_dim] 动作embedding
    next_emb: [B * (L - 1), D] 真实下一状态embedding（正样本）
    cur_codebook_id: [B * (L - 1), 1] 当前codebook_id，用于负采样
    nxt_codebook_id: [B * (L - 1), 1] 下一状态codebook_id，用于负采样
    loss_mask: [B * (L - 1)] mask矩阵适配变长序列，1表示参与计算，0表示不参与计算
    K: 每个样本负采样数量

    返回：loss标量
    """
    device = cur_emb.device
    # 选出合法的数据
    cur_emb = cur_emb[loss_mask]
    act_emb = act_emb[loss_mask]
    next_emb = next_emb[loss_mask]
    cur_codebook_id = cur_codebook_id[loss_mask]
    nxt_codebook_id = nxt_codebook_id[loss_mask]
    N = cur_emb.shape[0]

    # Step 1: 构造负样本池，随机采样 N * K 个负样本索引
    neg_indices = torch.randint(0, N, (N, K), device=device)  # [N, K]

    # 获取负样本的 embedding 和 codebook id
    neg_next_emb = next_emb[neg_indices]  # [N, K, D]
    neg_codebook_id = nxt_codebook_id[neg_indices].squeeze(-1)  # [N, K]

    # Step 2: 构造合法性 mask（屏蔽当前样本的 cur/nxt id）
    cur_id_exp = cur_codebook_id.expand(-1, K)  # [N, K]
    nxt_id_exp = nxt_codebook_id.expand(-1, K)  # [N, K]
    valid_mask = ((neg_codebook_id != cur_id_exp) &
                  (neg_codebook_id != nxt_id_exp))  # [N, K]

    # Step 3: 正样本 logits
    pos_logits = trans_predictor(cur_emb, next_emb, act_emb)  # [N, 1]

    # Step 4: 负样本 logits
    cur_emb_exp = cur_emb.unsqueeze(1).expand(-1, K, -1).reshape(N * K, -1)  # [N*K, D]
    act_emb_exp = act_emb.unsqueeze(1).expand(-1, K, -1).reshape(N * K, -1)  # [N*K, A]
    neg_next_emb_flat = neg_next_emb.reshape(N * K, -1)  # [N*K, D]

    neg_logits = trans_predictor(cur_emb_exp, neg_next_emb_flat, act_emb_exp).squeeze()  # [N*K]
    neg_logits = neg_logits.view(N, K)  # reshape 回 [N, K]

    # Step 5: 构造 InfoNCE logits 和 Mask
    all_logits = torch.cat([pos_logits, neg_logits], dim=1)  # [N, 1+K]
    all_logits /= tau

    # 将非法负样本的 logit 设为非常小的值（类似 -inf）
    minus_inf = torch.tensor(-1e9, device=device)
    valid_mask_with_pos = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=device), valid_mask], dim=1)  # [N, 1+K]
    masked_logits = torch.where(valid_mask_with_pos, all_logits, minus_inf)

    # 交叉熵，正样本 index 都是 0
    labels = torch.zeros(N, dtype=torch.long, device=device)
    loss = F.cross_entropy(masked_logits, labels)
    return loss


# ---- 数据加载和预处理 ----
class TrajectoryDataset(Dataset):
    """轨迹数据集类"""
    def __init__(self, data_path, max_seq_len=30, max_obs_len=20):
        self.max_seq_len = max_seq_len
        self.max_obs_len = max_obs_len
        
        print(f"Loading trajectories from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.trajectories = json.load(f)
        
        # 转换为列表格式
        self.trajectory_list = []
        for user_id, trajectory in self.trajectories.items():
            self.trajectory_list.append({
                'user_id': user_id,
                'obs': trajectory['obs'],
                'action': trajectory['action'],
                'reward': trajectory['reward'],
                'length': trajectory['length']
            })
        
        print(f"Loaded {len(self.trajectory_list)} trajectories")
        
        # 统计信息
        self._compute_statistics()
    
    def _compute_statistics(self):
        """计算数据集统计信息"""
        lengths = [t['length'] for t in self.trajectory_list]
        rewards = []
        actions = []
        
        for t in self.trajectory_list:
            rewards.extend(t['reward'][:t['length']])
            actions.extend(t['action'][:t['length']])
        
        self.stats = {
            'num_trajectories': len(self.trajectory_list),
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'num_actions': len(set(actions)),
            'action_range': (min(actions), max(actions)),
            'positive_rate': np.mean(rewards),
            'total_interactions': len(rewards)
        }
        
        print("Dataset Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
    
    def __len__(self):
        return len(self.trajectory_list)
    
    def __getitem__(self, idx):
        trajectory = self.trajectory_list[idx]
        
        # 获取实际长度
        actual_length = min(trajectory['length'], self.max_seq_len)
        
        # 处理obs: [max_seq_len, max_obs_len]
        obs = torch.tensor(trajectory['obs'][:self.max_seq_len], dtype=torch.float)
        
        # 处理action: [max_seq_len] - 确保action ID在有效范围内
        action = torch.tensor(trajectory['action'][:self.max_seq_len], dtype=torch.long)
        # 确保action ID在[0, item_num-1]范围内
        action = torch.clamp(action, 0, self.stats['num_actions'] - 1)
        
        # 处理reward: [max_seq_len]
        reward = torch.tensor(trajectory['reward'][:self.max_seq_len], dtype=torch.float)
        
        return {
            'obs': obs,
            'action': action,
            'reward': reward,
            'length': actual_length,
            'user_id': trajectory['user_id']
        }


def load_dataset(dataset_name, data_dir="/home/gaoxiang12/aaai/live-rl/TADT-CSA/dt_format_datasets/"):
    """加载指定数据集"""
    dataset_map = {
        'kuairand': 'kuairand_trajectories.json',
        'ml': 'ml_trajectories.json', 
        'retailrocket': 'retailrocket_trajectories.json'
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_map.keys())}")
    
    data_path = os.path.join(data_dir, dataset_map[dataset_name])
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    return TrajectoryDataset(data_path)


def collate_fn(batch):
    """数据批处理函数"""
    batch_size = len(batch)
    
    # 收集所有数据
    obs_list = [item['obs'] for item in batch]
    action_list = [item['action'] for item in batch]
    reward_list = [item['reward'] for item in batch]
    length_list = [item['length'] for item in batch]
    
    # 堆叠成张量
    obs = torch.stack(obs_list)  # [B, L, obs_len]
    action = torch.stack(action_list)  # [B, L]
    reward = torch.stack(reward_list)  # [B, L]
    length = torch.tensor(length_list, dtype=torch.long)  # [B]
    
    return {
        'obs': obs,
        'action': action,
        'reward': reward,
        'length': length
    }


# ---- 1. 编码器 ----
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, max_sign=10000, add_emb=False):
        super().__init__()
        if add_emb:
            self.emb = nn.Embedding(max_sign, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

    def forward(self, x, add_emb=False):
        # x: [N*L, obs_len]
        assert len(x.shape) == 2
        if add_emb:
            x = self.emb(x)    # [N*L, obs_len, D]
            x = x.mean(dim=1)  # avg pooling, [N*L, D]
        return self.net(x)


class DiscreteEncoder(nn.Module):
    def __init__(self, embed_dim, bucket_number=100, min_value=0.0, max_value=100.0):
        super(DiscreteEncoder, self).__init__()
        self.bucket_number = bucket_number
        self.min_value = min_value
        assert bucket_number > 0
        self.step = (max_value - min_value) / self.bucket_number
        self.embedding_table = nn.Embedding(bucket_number, embed_dim)

    def forward(self, x):
        if len(x.shape) == 2 and x.shape[1] == 1:
            x = x.squeeze(dim=-1)
        index = ((x - self.min_value) / self.step).floor().long()
        index = index.clamp(0, self.bucket_number - 1)
        x_embed = self.embedding_table(index)
        return x_embed


class PositionalEncoding(nn.Module):
    """经典正余弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [N, L, D]
        x = x + self.pe[:, :x.size(1)]
        return x


class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x: [N, L, D]
        if x.shape[1] != self.pos_embedding.shape[1]:
            x = x + self.pos_embedding[:,:x.shape[1],:]
        else:
            x = x + self.pos_embedding
        return x


class TokenTypeEncoding(nn.Module):
    """可学习token类型编码"""
    def __init__(self, d_model, type_num=3):
        super().__init__()
        self.type_embedding = nn.Embedding(type_num, d_model)

    def forward(self, x, type_idx):
        # x: [N, L, D], type_idx: [B, L]
        # x = x + self.type_embedding(type_idx)
        if x.shape[1] != self.type_embedding(type_idx).shape[1]:
            x = x + self.type_embedding(type_idx)[:,:x.shape[1],:]
        else:
            x = x + self.type_embedding(type_idx)
        return x


class TokenTimeEncoding(nn.Module):
    """可学习token时间编码"""

    def __init__(self, d_model, max_seq_len=100):
        super().__init__()
        self.time_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x, time_idx):
        # x: [N, L, D], type_idx: [B, L]
        # x = x + self.time_embedding(time_idx)
        if x.shape[1] != self.time_embedding(time_idx).shape[1]:
            x = x + self.time_embedding(time_idx)[:,:x.shape[1],:]
        else:
            xx = x + self.time_embedding(time_idx)
        return x


# ---- 2. Advantage-guided Codebook Module ----
class CodebookModule(nn.Module):
    def __init__(self, codebook_size, embed_dim, temperature=1.0, alpha=0.5):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, embed_dim))
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, state_emb, adv_emb, B, L):
        """
            state_emb: [B*L, D] 当前时刻 s_t embedding
            adv_emb: [B*L, D] 当前时刻 advantage embedding，但我们要用 adv_{t-1} 来计算相似度
            B, L: batch size 和序列长度，用于做时间维度reshape和shift
        """
        device = state_emb.device
        D = state_emb.shape[-1]
        # state_emb: [B*L, D]  adv_emb: [B*L, D]
        # reshape为[B, L, D]
        adv_emb_seq = adv_emb.view(B, L, D)
        zero_vec = torch.zeros(B, 1, D, device=device)
        adv_emb_shift = torch.cat([zero_vec, adv_emb_seq[:, 1:, :]], dim=1)  # [B, L, D]
        # reshape回[B*L, D]
        adv_emb_shift = adv_emb_shift.reshape(B * L, D)
        # 计算相似度
        sim_state = F.cosine_similarity(state_emb.unsqueeze(1), self.codebook.unsqueeze(0), dim=-1)  # [B*L, K]
        sim_adv = F.cosine_similarity(adv_emb.unsqueeze(1), self.codebook.unsqueeze(0), dim=-1)  # [B*L, K]
        sim = self.alpha * sim_state + (1.0 - self.alpha) * sim_adv
        weights_hard = F.gumbel_softmax(sim, tau=self.temperature, hard=True)  # hard assignment
        weights_soft = F.gumbel_softmax(sim, tau=self.temperature, hard=False) # [B*L, K]
        indices = torch.argmax(weights_hard, dim=-1).unsqueeze(-1)
        z_q = torch.matmul(weights_hard, self.codebook)  # [B*L, D]
        return z_q, weights_soft, indices


# ---- 3. 奖励预测器 ----
class RewardPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)


# ---- 4. 对比状态转移预测器 ----
class ContrastiveTransitionPredictor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * state_dim + action_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)   # 输出logit
        )

    def forward(self, cur_state, next_state, action):
        x = torch.cat([cur_state, next_state, action], dim=-1)
        return self.model(x)


class CausalDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, **kwargs):
        # masked self-attention only
        # print(tgt_mask.shape)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# ---- 5. Decision Transformer（Causal Transformer） ----
class DecisionTransformer(nn.Module):
    def __init__(self, embed_dim, item_dim, n_layers=3, n_heads=4):
        super().__init__()
        encoder_layer = CausalDecoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)
        self.action_predictor = nn.Linear(embed_dim, item_dim)
        self.return_predictor = nn.Linear(embed_dim, 2)

    def forward(self, token_seq, item_emb_mat, tgt_mask=None, key_padding_mask=None):
        # token_seq: [B, 3L, D], item_emb_mat: [N, D]
        # 注意这里的memory是encoder序列的embedding,适用于encoder-decoder架构
        # print(tgt_mask.shape)
        h = self.transformer(token_seq, None, tgt_mask=tgt_mask, tgt_key_padding_mask=key_padding_mask)
        action_tokens = h[:, 1::3, :]  # S tokens: 1,4,7,...
        return_tokens = h[:, 2::3, :]  # A tokens: 2,5,8,...
        # 预测动作和回报
        action_pred_emb = self.action_predictor(action_tokens)  # 预测动作的embedding, [B, L, D]
        action_preds = action_pred_emb
        return_preds = self.return_predictor(return_tokens)  # 预测优势和奖励
        return action_preds, return_preds


# ---- 6. 总模型结构 ----
class TADT_CSA(nn.Module):
    def __init__(self, obs_len, obs_dim, item_num, codebook_size, alpha=0.5, embed_dim=64, item_emb_dim=64,
                 rtg_bucket_num=100, adv_bucket_num=100, min_rtg_value=0.0, max_rtg_value=10.0, min_adv_value=-5.0,
                 max_adv_value=5.0, max_sign=10000, max_seq_len=300, pos_encoding_type='learned'):
        super().__init__()
        self.obs_len = obs_len
        self.obs_dim = obs_dim
        self.act_dim = item_num
        self.embed_dim = embed_dim
        self.state_encoder = Encoder(91, embed_dim, max_sign=max_sign)
        self.rtg_encoder = DiscreteEncoder(embed_dim, bucket_number=rtg_bucket_num, min_value=min_rtg_value,
                                           max_value=max_rtg_value)
        self.adv_encoder = DiscreteEncoder(embed_dim, bucket_number=adv_bucket_num, min_value=min_adv_value,
                                           max_value=max_adv_value)
        self.return_encoder = nn.Sequential(
            nn.Linear(2*embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.act_encoder = nn.Sequential(
            nn.Linear(27, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.hidden_size = embed_dim
        self.codebook = CodebookModule(codebook_size, embed_dim, alpha=alpha)
        self.reward_predictor = RewardPredictor(2 * embed_dim)
        self.trans_predictor = ContrastiveTransitionPredictor(embed_dim, embed_dim)
        if pos_encoding_type == 'sin':
            self.pos_encoder = PositionalEncoding(embed_dim, 3 * max_seq_len)
        elif pos_encoding_type == 'learned':
            self.pos_encoder = LearnablePositionalEncoding(embed_dim, 3 * max_seq_len)
        else:
            raise ValueError("pos_encoding_type must be 'sin' or 'learned'")
        self.type_encoder = TokenTypeEncoding(embed_dim, type_num=3)
        self.time_encoder = TokenTimeEncoding(embed_dim, max_seq_len=max_seq_len)
        self.decision_transformer = DecisionTransformer(embed_dim, item_emb_dim)

    def forward(self, obs_seq, act_id_seq, rtg_seq, adv_seq, type_idx, time_idx, tgt_mask=None, key_padding_mask=None):
        B, L, _ = obs_seq.shape
        # 编码 obs 和 advantage
        obs_flat = obs_seq.view(-1, _)             # [B*L, obs_len]
        act_id_seq = act_id_seq.squeeze(2)
        act_flat = act_id_seq.view(-1, 27)                        # [B*L]
        adv_flat = adv_seq.view(-1, 1)                        # [B*L, 1]
        rtg_flat = rtg_seq.view(-1, 1)                        # [B*L, 1]
        # print(obs_flat.shape)
        obs_emb = self.state_encoder(obs_flat, add_emb=False)  # [B*L, embed_dim]
        adv_emb = self.adv_encoder(adv_flat)                  # [B*L, D]
        rtg_emb = self.rtg_encoder(rtg_flat)                  # [B*L, D]
        act_emb = self.act_encoder(act_flat)                  # [B*L, D]
        return_emb = torch.cat([adv_emb, rtg_emb], dim=-1)    # [B*L, 2*D]
        return_seq_emb = self.return_encoder(return_emb)      # [B*L, D]
        # codebook量化
        s_t, weights, codebook_id = self.codebook(obs_emb, adv_emb, B, L)        # [B*L, D]
        s_t_seq = s_t.view(B, L, -1)
        return_seq_emb = return_seq_emb.view(B, L, -1)
        act_seq_emb = act_emb.view(B, L, -1)  # [B, L, D]
        # 构造 transformer token sequence: [R_1, S_1, A_1, R_2, S_2, A_2,  ...]
        token_seq = torch.stack([return_seq_emb, s_t_seq, act_seq_emb], dim=2).reshape(B, 3 * L, -1)
        # 添加位置编码
        token_seq = self.pos_encoder(token_seq)  # [N, 3L, D]
        # 添加type编码
        token_seq = self.type_encoder(token_seq, type_idx)
        # 添加时间编码
        token_seq = self.time_encoder(token_seq, time_idx)
        stacked_key_padding_mask = key_padding_mask.repeat_interleave(3, dim=1)  # [B, 3L]
        # transformer 输出每一步 a_t 的预测值
        # print(tgt_mask.shape)
        # print(stacked_key_padding_mask.shape)
        action_preds, return_preds = self.decision_transformer(token_seq, None, tgt_mask, stacked_key_padding_mask)  # [B, L, act_dim]
        # 奖励预测
        reward_preds = self.reward_predictor(s_t, act_emb).reshape(B, L)
        # 对比状态预测
        cur_emb = s_t_seq[:, :-1, :].reshape(-1, self.embed_dim)  # [B*(L-1), emb_dim]
        next_emb = s_t_seq[:, 1:, :].reshape(-1, self.embed_dim)  # [B*(L-1), emb_dim]
        cur_act_emb = act_seq_emb[:, :-1, :].reshape(-1, self.embed_dim)  # [B*(L-1), emb_dim]
        cur_codebook_id = codebook_id.view(B, L)[:, :-1].reshape(-1, 1)  # [B*(L-1), 1]
        nxt_codebook_id = codebook_id.view(B, L)[:, 1:].reshape(-1, 1)  # [B*(L-1), 1]
        return (action_preds, return_preds, reward_preds, cur_emb, next_emb, cur_act_emb, codebook_id, cur_codebook_id,
                nxt_codebook_id, weights, s_t_seq)

    def predict_single_step(self, states, actions, rtgs, advs, attention_mask=None):
        # state: [B, L, D_s], action: [B, L - 1, D_a], rtgs: [B, L, D_r]
        device = states.device
        B, L, _ = states.shape
        type_idx = generate_type_index(B, L, device=device)
        time_idx = generate_time_index(B, L, device=device)
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((B, L), dtype=torch.float32).to(device)
        # 编码 obs 和 advantage
        obs_flat = states.view(-1, _)  # [B*L, obs_len]
        if actions is not None:
            act_seq = actions.squeeze(2)
            act_flat = act_seq.view(-1, 27)  # [B*L]
        adv_flat = advs.view(-1, 1)  # [B*L, 1]
        rtg_flat = rtgs.view(-1, 1)  # [B*L, 1]
        # print(obs_flat.shape)
        obs_emb = self.state_encoder(obs_flat, add_emb=False)  # [B*L, embed_dim]
        adv_emb = self.adv_encoder(adv_flat)  # [B*L, D]
        rtg_emb = self.rtg_encoder(rtg_flat)  # [B*L, D]
        if actions is not None:
            act_emb = self.act_encoder(act_flat)  # [B*L, D]
        return_emb = torch.cat([adv_emb, rtg_emb], dim=-1)  # [B*L, 2*D]
        return_seq_emb = self.return_encoder(return_emb)  # [B*L, D]
        # codebook量化
        s_t, weights, codebook_id = self.codebook(obs_emb, adv_emb, B, L)  # [B*L, D]
        s_t_seq = s_t.view(B, L, -1)
        return_seq_emb = return_seq_emb.view(B, L, -1)
        if actions is not None:
            act_seq_emb = act_emb.view(B, L, -1)  # [B, L, D]
        # 构造 transformer token sequence: [R_1, S_1, A_1, R_2, S_2, A_2,  ...]
        if actions is not None:
            token_seq = torch.stack(
                (return_seq_emb, s_t_seq, act_seq_emb), dim=1
            ).permute(0, 2, 1, 3).reshape(B, 3 * L, self.hidden_size)
            stacked_attention_mask = attention_mask.repeat_interleave(3, dim=1)  # [B, 3L]
            tgt_mask = generate_causal_mask(3 * L, device=device)
        else:
            token_seq = torch.stack(
                (return_seq_emb, s_t_seq), dim=1
            ).permute(0, 2, 1, 3).reshape(B, 2 * L, self.hidden_size)
            stacked_attention_mask = attention_mask.repeat_interleave(2, dim=1)  # [B, 2L]
            tgt_mask = generate_causal_mask(2 * L, device=device)
        # 添加位置编码
        token_seq = self.pos_encoder(token_seq)  # [N, 3L, D]
        # 添加type编码
        token_seq = self.type_encoder(token_seq, type_idx)
        # 添加时间编码
        token_seq = self.time_encoder(token_seq, time_idx)
        # transformer 输出每一步 a_t 的预测值
        action_preds, return_preds = self.decision_transformer(token_seq, None, tgt_mask,
                                                               stacked_attention_mask)  # [B, L, act_dim]
        return action_preds[:, -1]


def save_tadt_csa_model(model, optimizer, epoch, save_path):
    model_path = os.path.join(save_path, f"tadt_csa_epoch{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': epoch_losses['total_loss'],
    }, model_path)
    print(f"Model saved to {model_path}")