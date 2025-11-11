import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from utils import compute_metrics
from tqdm import tqdm
import json as pyjson
from torch.utils.tensorboard import SummaryWriter

# 参考代码来源
# https://github.com/kesenzhao/DT4Rec/blob/main/mingpt/model_seq.py

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
        self._compute_statistics()

    def _compute_statistics(self):
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
            'num_actions': max(actions),
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
        actual_length = min(trajectory['length'], self.max_seq_len)
        obs = torch.tensor(trajectory['obs'][:self.max_seq_len], dtype=torch.long)
        action = torch.tensor(trajectory['action'][:self.max_seq_len], dtype=torch.long)
        reward = torch.tensor(trajectory['reward'][:self.max_seq_len], dtype=torch.float)
        return {
            'obs': obs,
            'action': action,
            'reward': reward,
            'length': actual_length,
            'user_id': trajectory['user_id']
        }

def collate_fn(batch):
    obs_list = [item['obs'] for item in batch]
    action_list = [item['action'] for item in batch]
    reward_list = [item['reward'] for item in batch]
    length_list = [item['length'] for item in batch]
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

# --- 数据集路径适配 ---
def get_dataset_path(dataset, data_dir, split):
    base = {
        'kuairand': 'kuairand_trajectories',
        'ml': 'ml_trajectories',
        'retailrocket': 'retailrocket_trajectories',
        'netflix': 'netflix_trajectories'
    }[dataset]
    return os.path.join(data_dir, f"{base}_{split}.json")


def compute_rtg_from_rewards(reward_seq, gamma_short=0.6, gamma_long=0.9):
    B, L = reward_seq.shape
    device = reward_seq.device
    rtg_short = torch.zeros_like(reward_seq).to(device)
    rtg_long = torch.zeros_like(reward_seq).to(device)
    for t in reversed(range(L)):
        if t == L - 1:
            rtg_short[:, t] = reward_seq[:, t]
            rtg_long[:, t] = reward_seq[:, t]
        else:
            rtg_short[:, t] = reward_seq[:, t] + gamma_short * rtg_short[:, t + 1]
            rtg_long[:, t] = reward_seq[:, t] + gamma_long * rtg_long[:, t + 1]
    rtg = torch.stack([rtg_short, rtg_long], dim=2)
    return rtg


def generate_attention_mask(lengths, max_seq_len, device):
    range_row = torch.arange(max_seq_len, device=device).unsqueeze(0).repeat(lengths.shape[0], 1)
    lengths_expanded = lengths.repeat(1, max_seq_len)
    padding_mask = range_row < lengths_expanded
    return padding_mask


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                             .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, attn_mask, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # query_mask & key mask & causal mask
        final_mask = attn_mask[:, None, :, None] & attn_mask[:, None, None, :] & self.mask[:, :, :T, :T].bool()
        # final_mask shape: [B, 1, T, T]
        att = att.masked_fill(~final_mask, -1e9)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class RewardWeightNet(nn.Module):
    def __init__(self, input_size):
        super(RewardWeightNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Still outputting a 2D vector for weights

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1)  # Apply softmax to the output
        return x


class RewardValueNet(nn.Module):
    def __init__(self, n_emb, bucket_number=100, min_value=0.0, max_value=100.0):
        super(RewardValueNet, self).__init__()
        self.bucket_number = bucket_number
        self.min_value = min_value
        self.step = (max_value - min_value) / self.bucket_number
        assert n_emb % 2 == 0
        self.embedding_table = nn.Embedding(bucket_number, n_emb // 2)
        self.weight_layer = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        B, L = x.shape[:2]
        index = ((x - self.min_value) / self.step).floor().long()
        index = index.clamp(0, self.bucket_number - 1)
        x_embed = self.embedding_table(index)
        weight = self.weight_layer(x).unsqueeze(dim=-1)
        output = (weight * x_embed).reshape(B, L, -1)
        return output


# Define the custom loss function
class BalancedRewardLoss(nn.Module):
    def __init__(self, lambda_term=1.0, regularization_factor=0):
        super().__init__()
        self.lambda_term = lambda_term
        self.regularization_factor = regularization_factor

    def forward(self, predictions, targets):
        ws, wl = predictions[:, 0].unsqueeze(dim=1), predictions[:, 1].unsqueeze(dim=1)
        Rs, Rl = targets[:, 0], targets[:, 1]
        balanced_metric = ws * Rs + wl * Rl
        penalty = torch.square(ws * Rs - wl * Rl)
        loss = -balanced_metric + self.lambda_term * penalty
        return torch.mean(loss)


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# @save
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0.0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class Seq2SeqDecoder(nn.Module):
    """用于序列到序列学习的循环神经网络解码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0.0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, logits_new, item_embedding):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = logits_new.unsqueeze(0).repeat(X.shape[0], 1, 1)
        # context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context)
        output_emb = output.permute(1, 0, 2)
        action_preds = torch.matmul(output_emb, item_embedding.T)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return action_preds, state, output_emb


# @save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


# @save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_mean的形状：(batch_size,)
    def forward(self, pred, label, valid_mask):
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * valid_mask).sum(dim=1)
        return weighted_loss


def InfoNCE(pred_seq, pos_seq, reward_gt, mask, tau=0.07, num_negatives=8):
    # pred_seq:  [batch_size, len, D]
    # pos_seq:   [batch_size, len, D]
    # reward_gt: [batch_size, 1]
    batch_size, rec_item_num, emb_dim = pred_seq.shape
    device = pred_seq.device

    pos_sim = F.cosine_similarity(pred_seq, pos_seq, dim=-1).unsqueeze(-1)  # [B, len, 1]
    neg_mask = reward_gt < 0.6  # shape: [B]
    neg_pool = pos_seq[neg_mask]  # shape: [B_neg, len, D]
    if len(neg_pool) == 0:
        return torch.tensor(0.0, device=device)
    # Flatten neg_pool 到 [B_neg * L, D]
    neg_candidates = neg_pool.reshape(-1, emb_dim)
    total_neg = neg_candidates.shape[0]

    # 随机采样索引 [B, L, num_negatives]
    rand_indices = torch.randint(0, total_neg, (batch_size, rec_item_num, num_negatives), device=device)
    neg_samples = neg_candidates[rand_indices]  # shape [B, len, num_negatives, D]
    pred_expand = pred_seq.unsqueeze(2)  # [B, len, 1, D]
    neg_sim = F.cosine_similarity(pred_expand, neg_samples, dim=-1)  # [B, len, num_negatives]

    logits = torch.cat([pos_sim, neg_sim], dim=-1).view(-1, 1 + num_negatives)  # [B, len, 1+num_negatives]
    logits = logits / tau  # 温度缩放
    labels = torch.zeros(batch_size, rec_item_num, dtype=torch.long, device=device).view(-1)  # 正样本在第 0 位
    criterion = nn.CrossEntropyLoss(reduction='none')
    origin_loss = criterion(logits, labels).reshape(batch_size, rec_item_num)
    loss = (mask * origin_loss).sum(dim=1)
    return loss


class DT4IER(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.embedding_dim = config.n_embd
        self.state_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                                            0.2)
        self.action_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                                             0.2)
        self.reward_net = RewardWeightNet(config.n_embd)
        self.decoder = Seq2SeqDecoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                                      0.2)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.ret_emb = RewardValueNet(config.n_embd, bucket_number=config.rtg_bucket_number, min_value=config.min_rtg_value,
                                      max_value=config.max_rtg_value)

        # my state_embedding
        self.state_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        self.item_emb_layer = nn.Embedding(config.vocab_size, config.n_embd)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        for i in (param_dict.keys() - union_params):
            no_decay.add(str(i))
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, y_len, rtgs, reward_gt, attention_mask):
        # 这里面每个state的action是一个item id的序列，维度[B, L, K]
        # y_len 是action序列的长度，因为原始action是一个item list且做了padding，y_len是为了防止取到padding的0的位置
        device = rtgs.device
        B, L = states.shape[0], states.shape[1]
        state_embeddings = torch.zeros([B, L, self.embedding_dim], device=device)

        for i in range(L):
            states_seq = states[:, i, :].type(torch.long)
            output, state = self.state_encoder(states_seq)
            context = state.permute(1, 0, 2)
            state_embeddings[:, i, :] = context[:, -1, :]
        user_features = state_embeddings.mean(dim=1)
        reward_weight = self.reward_net(user_features)   # [batch_size, 2]
        # rebalance the rtg
        rtgs = reward_weight.unsqueeze(dim=1) * rtgs
        action_embeddings = torch.zeros([B, L, self.embedding_dim], device=device)
        if len(actions.size()) == 2:
            actions = actions.unsqueeze(dim=-1)
        state_allstep = []
        for i in range(L):
            action_seq = actions[:, i, :].type(torch.long)
            output, state = self.action_encoder(action_seq)
            context = state.permute(1, 0, 2)
            action_embeddings[:, i, :] = context[:, -1, :]
            state_allstep.append(state)

        if actions is not None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((B, 3 * L, self.config.n_embd), dtype=torch.float32, device=device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((B, 2 * L, self.config.n_embd), dtype=torch.float32, device=device)
            token_embeddings[:, ::2, :] = rtg_embeddings
            token_embeddings[:, 1::2, :] = state_embeddings
        elif actions is not None and self.model_type == 'naive':
            token_embeddings = torch.zeros((B, 2 * L, self.config.n_embd), dtype=torch.float32, device=device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings[:, -states.shape[1]:, :]
        elif actions is None and self.model_type == 'naive':
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        # 统一位置编码逻辑，彻底消除shape不一致
        position_embeddings = self.pos_emb[:, :token_embeddings.size(1), :].to(device)  # [1, seq_len, D]
        x = self.drop(token_embeddings + position_embeddings)
        attention_mask = attention_mask.repeat_interleave(3, dim=1)  # [B, 3L]
        for block in self.blocks:
            x = block(x, attention_mask)
        logits_emb = x

        if actions is not None and self.model_type == 'reward_conditioned':
            logits_emb = logits_emb[:, 1::3, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits_emb = logits_emb[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits_emb = logits_emb[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits_emb = logits_emb  # for completeness
        else:
            raise NotImplementedError()
        # logits_emb shape: [9 * batch_size, T, n_embd]

        # if we are given some desired targets also calculate the loss
        loss_func = MaskedSoftmaxCELoss()
        reward_loss_func = BalancedRewardLoss()
        ce_loss_list = []
        contra_loss_list = []
        action_logit_list = []

        for i in range(actions.shape[1]):
            logits_new = logits_emb[:, i, :]
            targets_seq = actions[:, i, :].type(torch.long)
            pos_seq = actions[:, i, :].type(torch.long)
            bos = torch.tensor([0] * B, device=device).reshape(-1, 1)
            if targets_seq.shape[1] > 1:
                dec_input = torch.cat([bos, targets_seq[:, :-1]], dim=1)
            else:
                dec_input = bos
            logits_new_pos = logits_new[:actions.shape[0]]
            Y_hat, _, Y_emb = self.decoder(dec_input, logits_new_pos, self.item_emb_layer.weight)
            action_logit_list.append(Y_hat)
            mask = attention_mask[:, i].unsqueeze(dim=-1).to(device)
            loss_ce = loss_func(Y_hat, targets_seq, mask)
            pos_seq_emb = self.action_encoder.embedding(pos_seq)
            loss_contra = InfoNCE(Y_emb, pos_seq_emb, reward_gt[:, i], mask)
            ce_loss_list.append(loss_ce)
            contra_loss_list.append(loss_contra)
        stack_ce_loss = torch.stack(ce_loss_list).permute(0, 1)   # [B, L]
        stack_contra_loss = torch.stack(contra_loss_list).permute(0, 1)  # [B, L]
        ce_loss = stack_ce_loss.mean()
        contra_loss = stack_contra_loss.mean()
        reward_loss = reward_loss_func(reward_weight, rtgs)
        stack_action_logits = torch.stack(action_logit_list, dim=1).squeeze(dim=2)  # [B, L, item_num], rec_item_num=1
        return stack_action_logits, ce_loss, contra_loss, reward_loss


def bleu_seq(y_pred, y):
    score_sum = 0
    for i in range(y_pred.shape[0]):
        for j in range(y.shape[0]):
            if y_pred[i] == y[j]:
                score_sum += 1
                break
    score = score_sum / y_pred.shape[0]
    return score


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--item_num', type=int, default=5000)
parser.add_argument('--data_dir_prefix', type=str, default='../dt_format_datasets/')
# 新增数据集参数
parser.add_argument('--dataset', type=str, default='kuairand', 
                   choices=['kuairand', 'ml', 'retailrocket', 'netflix'],
                   help='Dataset to train on')
parser.add_argument('--data_dir', type=str, 
                   default='../dt_format_datasets/',
                   help='Directory containing dataset files')
# 新增 device 参数
parser.add_argument('--device', type=str, default=None, help='Device to use: cpu, cuda, cuda:0, cuda:1, etc.')
args = parser.parse_args()

# 加载训练集
train_path = get_dataset_path(args.dataset, args.data_dir, 'train')
train_dataset = TrajectoryDataset(train_path)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)
# 加载验证集
val_path = get_dataset_path(args.dataset, args.data_dir, 'val')
val_dataset = TrajectoryDataset(val_path)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

obs_len = 20
item_num = train_dataset.stats['num_actions']
max_sign = max(train_dataset.stats['action_range']) + 1
item_emb_dim = 64
learning_rate = 5e-3
num_epochs = args.epochs

# 方案B: item_emb和vocab_size都用item_num+1，action id不变
conf = GPTConfig(item_num+1, 3 * args.context_length, n_layer=3, n_head=4, n_embd=64, model_type=args.model_type, max_timestep=100,
                 rtg_bucket_number=100, min_rtg_value=0.0, max_rtg_value=10.0)
model = DT4IER(conf)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard writer
log_dir = f"runs/dt4ier_{args.dataset}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# 新增：根据参数选择device，支持cuda:0等
import os
if args.device is not None:
    if args.device.startswith('cuda:'):
        # 只设置当前指定的GPU可见
        gpu_idx = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
        device = torch.device('cuda:0')
    else:
        device = torch.device(args.device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

for epoch in range(1, num_epochs + 1):
    model.train()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        obs_seq = batch['obs'].to(device)  # [B, L, obs_len]
        act_id_seq = batch['action'].to(device)  # [B, L]
        reward_gt = batch['reward'].to(device)  # [B, L]
        seq_lengths = batch['length'].unsqueeze(1).to(device)  # [B, 1]
        # 计算RTG
        rtg_seq = compute_rtg_from_rewards(reward_gt)
        # 生成attention mask
        attention_mask = generate_attention_mask(seq_lengths, obs_seq.shape[1], device=obs_seq.device)
        outputs = model(obs_seq, act_id_seq, seq_lengths, rtg_seq, reward_gt, attention_mask)
        _, ce_loss, contra_loss, reward_loss = outputs
        total_loss = ce_loss + reward_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    # TensorBoard logging for training loss
    writer.add_scalar('Loss/Total', total_loss.item(), epoch)
    writer.add_scalar('Loss/CE', ce_loss.item(), epoch)
    writer.add_scalar('Loss/Contra', contra_loss.item(), epoch)
    writer.add_scalar('Loss/Reward', reward_loss.item(), epoch)
    pbar.set_postfix({
        'Total': f"{total_loss.item():.4f}",
        'CE': f"{ce_loss.item():.4f}",
        'Contra': f"{contra_loss.item():.4f}",
        'Reward': f"{reward_loss.item():.4f}"
    })
    print(f"Epoch {epoch} | Total Loss: {total_loss.item():.4f} | CE Loss: {ce_loss.item():.4f} | "
          f"Contra Loss: {contra_loss.item():.4f} | Reward Loss: {reward_loss.item():.4f}")
    # --- 验证集评估 ---
    model.eval()
    val_eval_predictions = []
    val_eval_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            obs_seq = batch['obs'].to(device)
            act_id_seq = batch['action'].to(device)
            reward_gt = batch['reward'].to(device)
            seq_lengths = batch['length'].unsqueeze(1).to(device)
            rtg_seq = compute_rtg_from_rewards(reward_gt)
            attention_mask = generate_attention_mask(seq_lengths, obs_seq.shape[1], device=obs_seq.device)
            outputs = model(obs_seq, act_id_seq, seq_lengths, rtg_seq, reward_gt, attention_mask)
            logits = outputs[0]
            B, L = act_id_seq.shape
            last_indices = batch['length'] - 1
            batch_indices = torch.arange(B, device=device)
            valid_mask = batch['length'] > 0
            if valid_mask.sum() > 0:
                last_action_preds = logits[batch_indices, last_indices]
                last_gt_actions = act_id_seq[batch_indices, last_indices]
                val_eval_predictions.append(last_action_preds[valid_mask].detach().cpu().numpy())
                val_eval_labels.append(last_gt_actions[valid_mask].unsqueeze(1).detach().cpu().numpy())
    if len(val_eval_predictions) > 0:
        import numpy as np
        val_eval_predictions_np = np.concatenate(val_eval_predictions, axis=0)
        val_eval_labels_np = np.concatenate(val_eval_labels, axis=0)
        metrics = compute_metrics(val_eval_predictions_np, val_eval_labels_np)
        print(f"[Val][Epoch {epoch}] Metrics: {metrics}")
        # 写入日志
        log_data = {'epoch': epoch, 'metrics': metrics}
        log_filename = f'dt4ier_{args.dataset}_eval.log'
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(pyjson.dumps(log_data, ensure_ascii=False) + '\n')
        # TensorBoard logging for metrics
        for k, v in metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

# Save model weights at the end of training
model_save_path = os.path.join(log_dir, f"dt4ier_{args.dataset}_final.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
writer.close()