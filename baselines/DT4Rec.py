import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json as pyjson
import sys
sys.path.append(os.path.dirname(__file__))
from utils import compute_metrics
import os
from torch.utils.tensorboard import SummaryWriter

# 参考代码来源
# https://github.com/kesenzhao/DT4Rec/blob/main/mingpt/model_seq.py

def compute_rtg_from_rewards(reward_seq, gamma=0.9):
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
    rtg = torch.zeros_like(reward_seq).to(device)
    for t in reversed(range(L)):
        if t == L - 1:
            rtg[:, t] = reward_seq[:, t]
        else:
            rtg[:, t] = reward_seq[:, t] + gamma * rtg[:, t + 1]  # G_t = r_t + γ G_{t+1}
    return rtg  # [B, L]


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
        final_mask = attn_mask[:, None, :, None] & attn_mask[:, None, None, :] & self.mask[:, :, :T, :T].bool()
        att = att.masked_fill(~final_mask, -1e9)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Autodis(nn.Module):
    def __init__(self, config, bucket_number, min_value=0.0, max_value=100.0):
        super().__init__()
        bucket_number = bucket_number
        min_value = min_value
        max_value = max_value
        self.bucket_value = (torch.torch.linspace(min_value, max_value, bucket_number).reshape(bucket_number, 1)
                             .type(torch.float32))
        self.bucket = nn.Sequential(nn.Linear(1, config.n_embd))
        self.ret_emb_score = nn.Sequential(nn.Linear(1, bucket_number, bias=False), nn.LeakyReLU())
        self.res = nn.Linear(bucket_number, bucket_number, bias=False)
        self.temp = nn.Sequential(
            nn.Linear(1, bucket_number, bias=False),
            nn.LeakyReLU(),
            nn.Linear(bucket_number, bucket_number, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(dim=-1)
        Meta_emb = self.bucket(self.bucket_value.to(x.device))
        x = self.ret_emb_score(x)  # [batch_size, timestep, bucket_value]
        x = x + self.res(x)
        max_value, _ = torch.max(x, dim=2, keepdim=True)
        x = torch.exp(x - max_value)
        soft_sum = torch.sum(x, dim=2).unsqueeze(2)
        x = x / (1e-8 + soft_sum)
        x = torch.einsum('nck,km->ncm', [x, Meta_emb])
        return x


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
                          dropout=dropout, batch_first=True)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        output, state = self.rnn(X)
        # output的形状:(batch_size,num_steps,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class Seq2SeqDecoder(nn.Module):
    """用于序列到序列学习的循环神经网络解码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0.0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout, batch_first=True)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, logits_new, item_embedding):
        # X：(batch_size,num_steps,embed_size)
        # logits_new: (batch_size, hidden_dim)
        # state: [num_layers, batch_size, hidden_dim]
        X = self.embedding(X)
        # 广播context，使其具有与X相同的num_steps
        context = logits_new.unsqueeze(1).repeat(1, X.shape[1], 1)
        X_and_context = torch.cat([X, context], 2)
        output, state = self.rnn(X_and_context)
        action_preds = torch.matmul(output, item_embedding.T)
        # output的形状:(batch_size,num_steps,vocab_size)
        return action_preds, state, output


# @save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_mean的形状：(batch_size, 1)
    def forward(self, pred, label, valid_mask):
        self.reduction = 'none'
        if pred.size(1) == 1:
            pred, label = pred.squeeze(1), label.squeeze(1)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred, label).unsqueeze(1)  # [B, 1]
        weighted_loss = (unweighted_loss * valid_mask)  # [B, ]
        return weighted_loss


def InfoNCE(pred_seq, pos_seq, neg_seq, mask, tau=0.07):
    B, rec_item_num, _ = pred_seq.shape
    device = pred_seq.device
    score_neg = torch.zeros([8, B, rec_item_num], device=device)
    for i in range(8):
        score_neg[i, :, :] = torch.cosine_similarity(pred_seq, neg_seq[i * B:(i + 1) * B], dim=-1)
    l_neg = torch.zeros([B, 8], device=device)
    for i in range(B):
        l_neg[i] = torch.mean(score_neg[:, i], dim=1)
    pos_score_step = torch.cosine_similarity(pred_seq, pos_seq, dim=-1)
    l_pos = torch.zeros([B, 1], device=device)
    for i in range(B):
        pos_score_batch = torch.mean(pos_score_step[i])
        l_pos[i] = pos_score_batch
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= tau
    # logits shape: [B, 9], 其中第0列是正样本
    # label是第0个位置
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    origin_loss = criterion(logits, labels).unsqueeze(dim=-1)
    loss = (mask * origin_loss).sum(dim=1)
    return loss


class DT4Rec(nn.Module):
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
        self.state_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, config.encoder_layer_num,
                                            0.2)
        self.action_encoder = Seq2SeqEncoder(config.vocab_size, config.n_embd, config.n_embd, config.decoder_layer_num,
                                             0.2)
        self.decoder = Seq2SeqDecoder(config.vocab_size, config.n_embd, config.n_embd, 2,
                                      0.2)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.ret_emb = Autodis(config, config.rtg_bucket_number, min_value=config.min_rtg_value, max_value=config.max_rtg_value)

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

    # state, action, and return
    def forward(self, states, actions, y_len, rtgs, attention_mask):
        # 这里面每个state的action是一个item id的序列，维度[B, L, K]
        # y_len 是action序列的长度，因为原始action是一个item list且做了padding，y_len是为了防止取到padding的0的位置
        device = rtgs.device
        B, L = states.shape[0], states.shape[1]
        state_embeddings = torch.zeros([B, L, self.embedding_dim], device=device)

        for i in range(L):
            states_seq = states[:, i, :].type(torch.long)
            output, state = self.state_encoder(states_seq)
            context = state.permute(1, 0, 2)  # (batch_size, num_layers, num_hiddens)
            state_embeddings[:, i, :] = context[:, -1, :]

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
        # rtg 负采样
        rtg_neg = torch.zeros([8 * B, rtgs.shape[1]], device=device)
        for i in range(8):
            for j in range(B):
                rtg_neg[i * B + j, :-1] = rtgs[j, 1:]
                rtg_neg[i * B + j, -1] = rtgs[j, -1]

        if actions is not None and self.model_type == 'reward_conditioned':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((B, 3 * L, self.config.n_embd), dtype=torch.float32, device=device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':  # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((B, 2 * L, self.config.n_embd), dtype=torch.float32, device=device)
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            token_embeddings = torch.zeros((B, 2 * L, self.config.n_embd), dtype=torch.float32, device=device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
        elif actions is None and self.model_type == 'naive':  # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        # 在第0维上重复batch次
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)  # batch_size, traj_length, n_embd
        if y_len.dim() == 1:
            y_len = y_len.unsqueeze(1)
        indices = torch.repeat_interleave(y_len, self.config.n_embd, dim=-1).unsqueeze(1)
        # gather的步骤是取出真实长度对应的那个pos embedding
        gather_pos_emb = torch.gather(all_global_pos_emb, 1, indices)
        position_embeddings = gather_pos_emb + self.pos_emb
        token_neg_embeddings = torch.repeat_interleave(token_embeddings, 8, dim=0)
        rtg_neg_embeddings = self.ret_emb(rtg_neg.type(torch.float32))
        token_neg_embeddings[:, ::3, :] = rtg_neg_embeddings
        token_all = torch.cat((token_embeddings, token_neg_embeddings), dim=0)  # [9B, 3L, emb_dim]
        position_all = torch.repeat_interleave(position_embeddings, 9, dim=0)   # [9B, 3L, emb_dim]
        x = self.drop(token_all + position_all)
        attention_mask = attention_mask.repeat_interleave(3, dim=1)  # [B, 3L]
        attention_mask_expand = attention_mask.unsqueeze(1).repeat(1, 9, 1)  #[B, 9, 3L]
        attention_mask_expand = attention_mask_expand.permute(1, 0, 2).reshape(9 * B, -1)  #[9B, 3L]
        for block in self.blocks:
            x = block(x, attention_mask_expand)
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
        ce_loss_list = []
        contra_loss_list = []
        action_logit_list = []

        for i in range(actions.shape[1]):
            logits_new = logits_emb[:, i, :]   # [9B, emb_dim]
            targets_seq = actions[:, i, :].type(torch.long)   # [B, rec_item_num]
            pos_seq = actions[:, i, :].type(torch.long)
            # item编号从1开始，这样0就是mock的bos_id
            bos = torch.tensor([0] * B, device=device).reshape(-1, 1)
            # 序列预测，给mock的第一步，和1 ~ K -1步，concat到一起去预测第1 ~ K步
            if targets_seq.shape[1] > 1:
                dec_input = torch.cat([bos, targets_seq[:, :-1]], dim=1)   # [B, rec_item_num - 1]
            else:
                dec_input = bos  # [B, 1]
            logits_new_pos = logits_new[:actions.shape[0]]  # [B, emb_dim]
            Y_hat, _, Y_emb = self.decoder(dec_input, logits_new_pos, self.item_emb_layer.weight)
            # Y_hat: [batch_size, rec_item_num, all_item_num]的概率分布
            # Y_emb: [batch_size, rec_item_num, n_emb]的embedding
            action_logit_list.append(Y_hat)

            dec_input_neg = torch.repeat_interleave(dec_input, 8, 0)
            state_neg = torch.repeat_interleave(state_allstep[i], 8, 1)
            logits_new_neg = logits_new[actions.shape[0]:]
            Y_hat_all, _, neg_seq_emb = self.decoder(dec_input_neg, logits_new_neg, self.item_emb_layer.weight)

            mask = attention_mask[:, i].unsqueeze(dim=-1)
            loss_ce = loss_func(Y_hat, targets_seq, mask)
            pos_seq_emb = self.action_encoder.embedding(pos_seq)
            loss_contra = InfoNCE(Y_emb, pos_seq_emb, neg_seq_emb, mask)
            ce_loss_list.append(loss_ce)
            contra_loss_list.append(loss_contra)
        stack_ce_loss = torch.stack(ce_loss_list).transpose(0, 1)   # [B, L]
        stack_contra_loss = torch.stack(contra_loss_list).transpose(0, 1)  # [B, L]
        ce_loss = stack_ce_loss.mean()
        contra_loss = stack_contra_loss.mean()
        stack_action_logits = torch.stack(action_logit_list, dim=1).squeeze(dim=2)  # [B, L, item_num], rec_item_num=1
        return stack_action_logits, ce_loss, contra_loss


# ---- 数据加载和预处理（抄DT4IER） ----
class TrajectoryDataset(Dataset):
    def __init__(self, data_path, max_seq_len=30, max_obs_len=20):
        self.max_seq_len = max_seq_len
        self.max_obs_len = max_obs_len
        print(f"Loading trajectories from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.trajectories = json.load(f)
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
    obs = torch.stack(obs_list)
    action = torch.stack(action_list)
    reward = torch.stack(reward_list)
    length = torch.tensor(length_list, dtype=torch.long)
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


# --- argparse 增加数据集参数 ---
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
parser.add_argument('--dataset', type=str, default='kuairand',
                   choices=['kuairand', 'ml', 'retailrocket', 'netflix'],
                   help='Dataset to train on')
parser.add_argument('--data_dir', type=str,
                   default='../dt_format_datasets/',
                   help='Directory containing dataset files')
# 新增 device 参数
parser.add_argument('--device', type=str, default=None, help='Device to use: cpu, cuda, cuda:0, cuda:1, etc.')
args = parser.parse_args()

# --- 设置device ---
import os
if args.device is not None:
    if args.device.startswith('cuda:'):
        gpu_idx = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
        device = torch.device('cuda:0')
    else:
        device = torch.device(args.device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 加载数据集 ---
train_path = get_dataset_path(args.dataset, args.data_dir, 'train')
train_dataset = TrajectoryDataset(train_path)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)
val_path = get_dataset_path(args.dataset, args.data_dir, 'val')
val_dataset = TrajectoryDataset(val_path)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0
)

# --- 训练参数和模型初始化 ---
obs_len = 20
item_num = train_dataset.stats['num_actions']
max_sign = max(train_dataset.stats['action_range']) + 1
item_emb_dim = 64
learning_rate = 5e-3 #1e-4
num_epochs = args.epochs
rtg_bucket_num = 100
min_rtg_value = 0.0
max_rtg_value = 10.0

# vocab_size = item_num+1，保证embedding和label一致
conf = GPTConfig(item_num+1, 3 * args.context_length, n_layer=3, n_head=4, n_embd=64, model_type=args.model_type, max_timestep=3 * args.context_length,
                 rtg_bucket_number=rtg_bucket_num, min_rtg_value=min_rtg_value, max_rtg_value=max_rtg_value,
                 encoder_layer_num=2, decoder_layer_num=2)
model = DT4Rec(conf)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TensorBoard writer
log_dir = f"runs/dt4rec_{args.dataset}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# --- 训练与验证主循环 ---
for epoch in range(1, num_epochs + 1):
    model.train()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        obs_seq = batch['obs'].to(device)
        act_id_seq = batch['action'].to(device)
        reward_gt = batch['reward'].to(device)
        seq_lengths = batch['length'].unsqueeze(1).to(device)
        rtg_seq = compute_rtg_from_rewards(reward_gt)
        attention_mask = generate_attention_mask(seq_lengths, obs_seq.shape[1], device=obs_seq.device)
        outputs = model(obs_seq, act_id_seq, seq_lengths, rtg_seq, attention_mask)
        _, ce_loss, contra_loss = outputs
        total_loss = ce_loss  +  contra_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    # TensorBoard logging for training loss
    writer.add_scalar('Loss/Total', total_loss.item(), epoch)
    writer.add_scalar('Loss/CE', ce_loss.item(), epoch)
    writer.add_scalar('Loss/Contra', contra_loss.item(), epoch)
    pbar.set_postfix({
        'Total': f"{total_loss.item():.4f}",
        'CE': f"{ce_loss.item():.4f}",
        'Contra': f"{contra_loss.item():.4f}"
    })
    print(f"Epoch {epoch} | Total Loss: {total_loss.item():.4f} | CE Loss: {ce_loss.item():.4f} | "
          f"Contra Loss: {contra_loss.item():.4f}")
    # --- 验证集评估 ---
    model.eval()
    val_eval_predictions = []
    val_eval_labels = []
    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            obs_seq = batch['obs'].to(device)
            act_id_seq = batch['action'].to(device)
            reward_gt = batch['reward'].to(device)
            seq_lengths = batch['length'].unsqueeze(1).to(device)
            rtg_seq = compute_rtg_from_rewards(reward_gt)
            attention_mask = generate_attention_mask(seq_lengths, obs_seq.shape[1], device=obs_seq.device)
            outputs = model(obs_seq, act_id_seq, seq_lengths, rtg_seq, attention_mask)
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
        log_filename = f'dt4rec_{args.dataset}_eval.log'
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(pyjson.dumps(log_data, ensure_ascii=False) + '\n')
        # TensorBoard logging for metrics
        for k, v in metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

# Save model weights at the end of training
model_save_path = os.path.join(log_dir, f"dt4rec_{args.dataset}_final.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
writer.close()