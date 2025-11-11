import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 参考代码来源
# https://github.com/kesenzhao/DT4Rec/blob/main/mingpt/model_seq.py


def compute_rtg_from_rewards(reward_seq, gamma_short=0.6, gamma_long=0.9):
    """
    padding的位置reward必须为0，否则影响RTG计算。
    根据 reward 序列计算 advantage trend。

    输入：
        reward_seq: [B, L]  奖励序列
        gamma: float 衰减因子（0~1）

    返回：
        adv_trend: [B, L, 2] 每个时刻的 advantage trend
    """
    B, L = reward_seq.shape
    device = reward_seq.device
    rtg_short = torch.zeros_like(reward_seq).to(device)
    rtg_long = torch.zeros_like(reward_seq).to(device)
    for t in reversed(range(L)):
        if t == L - 1:
            rtg_short[:, t] = reward_seq[:, t]
            rtg_long[:, t] = reward_seq[:, t]
        else:
            rtg_short[:, t] = reward_seq[:, t] + gamma_short * rtg_short[:, t + 1]  # G_t = r_t + γ G_{t+1}
            rtg_long[:, t] = reward_seq[:, t] + gamma_long * rtg_long[:, t + 1]  # G_t = r_t + γ G_{t+1}
    rtg = torch.stack([rtg_short, rtg_long], dim=2)
    return rtg  # [B, L, 2]


def generate_attention_mask(lengths, max_seq_len, B, device):
    # 创建范围序列 [0, 1, 2, ..., L-1]
    range_row = torch.arange(max_seq_len, device=device).unsqueeze(0).repeat(B, 1)  # [B, L]
    # lengths expand成 [B, L]
    lengths_expanded = lengths.unsqueeze(1).repeat(1, max_seq_len)  # [B, L]
    # mask = True 表示参与attention计算的位置（idx < lengths）
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

    def forward(self, x, attention_mask):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


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
        # assert len(x.shape) == 2
        if add_emb:
            x = self.emb(x)    # [N*L, obs_len, D]
            x = x.mean(dim=1)  # avg pooling, [N*L, D]
        return self.net(x)


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
        #self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, logits_new, state=None):
        context = logits_new.unsqueeze(dim=0)  # [rec_item_num, batch_size, emb_dim]
        if state is None:
            output, state = self.rnn(context)
        else:
            output, state = self.rnn(context, state)
        output_emb = output.permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output_emb, state


# @save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


# class MaskedMSELoss(nn.MSELoss):
#     """带遮蔽的MSE损失函数"""

#     # pred的形状：(batch_size,num_steps,vocab_size)
#     # label的形状：(batch_size,num_steps)
#     # valid_mean的形状：(batch_size,)
#     def forward(self, pred, label, valid_mask):
#         self.reduction = 'none'
#         # print(pred.permute(0, 2, 1).shape, label.shape)
#         unweighted_loss = super(MaskedMSELoss, self).forward(pred, label)
#         weighted_loss = (unweighted_loss * valid_mask)
#         return weighted_loss
class MaskedMSELoss(nn.Module):
    """
    带 mask 的 MSE Loss
    pred: (batch_size, seq_len, feature_dim)
    label: same as pred
    valid_mask: (batch_size, seq_len)  → 1 表示有效，0 表示 padding
    """
    def __init__(self, reduction='none'):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, label, valid_mask):
        if pred.shape[1] == 1:
            label = label.unsqueeze(dim=1)
        loss = self.mse(pred, label)  # [B, T, D]
        # Expand mask to match pred shape
        mask = valid_mask.unsqueeze(-1).float()  # [B, T, 1]
        loss = loss * mask  # [B, T, D]

        if self.reduction == 'mean':
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def InfoNCE(pred_seq, pos_seq, reward_gt, mask, tau=0.07, num_negatives=8):
    # pred_seq:  [batch_size, len, D]
    # pos_seq:   [batch_size, len, D]
    # reward_gt: [batch_size, 1]
    # print("pred_seq shape: ", pred_seq.shape, ", pos_seq shape: ", pos_seq.shape, ", reward_gt shape: ", reward_gt.shape)
    B, rec_item_num, _ = pred_seq.shape
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
    rand_indices = torch.randint(0, total_neg, (B, rec_item_num, num_negatives), device=device)
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
        self.state_encoder = Encoder(91, config.n_embd)
        self.action_encoder = nn.Sequential(
            nn.Linear(27, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        self.reward_net = RewardWeightNet(config.n_embd)
        # self.decoder = Seq2SeqDecoder(config.vocab_size, config.n_embd, config.n_embd, 2,
        #                               0.2)
        self.decoder = Seq2SeqDecoder(config.vocab_size, config.n_embd, 27, 2,
                                      0.2)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.ret_emb = RewardValueNet(config.n_embd, bucket_number=config.rtg_bucket_number, min_value=config.min_rtg_value,
                                      max_value=config.max_rtg_value)
        self.linear = nn.Sequential(
            nn.Linear(config.n_embd, 27)
        )
        # my state_embedding
        self.state_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

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
    def forward(self, states, actions, y_len, rtgs, reward_gt, attention_mask):
        # 这里面每个state的action是一个item id的序列，维度[B, L, K]
        # y_len 是action序列的长度，因为原始action是一个item list且做了padding，y_len是为了防止取到padding的0的位置
        device = rtgs.device
        B, L = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)
        reward_weight = self.reward_net(state_embeddings)   # [batch_size, 2]
        # rebalance the rtg
        # print(reward_weight.shape)
        # print(rtgs.shape)
        # rtgs = reward_weight.unsqueeze(dim=1) * rtgs
        if rtgs.dim() < reward_weight.dim():
            rtgs = reward_weight * rtgs.unsqueeze(-1)
        else:
            rtgs = reward_weight * rtgs
        action_embeddings = self.action_encoder(actions.squeeze(2))

        if actions is not None and self.model_type == 'reward_conditioned':
            rtgs = rtgs.to(device).float()
            rtg_embeddings = self.ret_emb(rtgs)
            token_embeddings = torch.zeros((B, 3 * L, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':  # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((B, 2 * L, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            token_embeddings = torch.zeros((B, 2 * L, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
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
        position_embeddings = torch.gather(all_global_pos_emb, 1, indices) + self.pos_emb
        x = self.drop(token_embeddings + position_embeddings)
        attention_mask = attention_mask.repeat_interleave(3, dim=1)  # [B, 3L]
        for block in self.blocks:
            x = block(x, attention_mask)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = x[:, 1::3, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = x[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = x[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = x  # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss_func = MaskedMSELoss()
        reward_loss_func = BalancedRewardLoss()
        mse_loss_list = []
        contra_loss_list = []
        action_logit_list = []

        for i in range(actions.shape[1]):
            logits_new = logits[:, i, :]
            targets_seq = actions[:, i, :]
            pos_seq = actions[:, i, :].unsqueeze(dim=1)
            # item编号从1开始，这样0就是mock的bos_id
            bos = torch.tensor([0] * B).to(device).reshape(-1, 1)
            # 序列预测，给mock的第一步，和1 ~ K -1步，concat到一起去预测第1 ~ K步
            if targets_seq.shape[1] > 1:
                dec_input = torch.cat([bos, targets_seq[:, :-1]], dim=1)
            else:
                dec_input = bos
            logits_new_pos = logits_new[:actions.shape[0]]
            Y_emb, _ = self.decoder(logits_new_pos)
            action_logit_list.append(Y_emb)
            # Y_hat: [batch_size, rec_item_num, all_item_num]的概率分布
            # Y_emb: [batch_size, rec_item_num, n_emb]的embedding

            mask = attention_mask[:, i].unsqueeze(dim=-1)
            loss_mse = loss_func(Y_emb, targets_seq, mask)
            # print("Y_emb shape", Y_emb.shape, ", target_seq shape: ", targets_seq.shape, ", mask shape: ", mask.shape, ", loss mse shape: ", loss_mse.shape)
            loss_mse = torch.sum(loss_mse, dim=-1)
            # print("loss_mse shape: ", loss_mse.shape)
            pos_seq_emb = pos_seq
            loss_contra = InfoNCE(Y_emb, pos_seq_emb, reward_gt[:, i].unsqueeze(dim=1), mask)
            
            mse_loss_list.append(loss_mse)
            contra_loss_list.append(loss_contra)
        # print("stack mse loss shape: ", torch.stack(mse_loss_list).shape)
        stack_mse_loss = torch.stack(mse_loss_list).squeeze().permute(1, 0)   # [B, L]
        stack_contra_loss = torch.stack(contra_loss_list).permute(1, 0)  # [B, L]
        mse_loss = stack_mse_loss.mean()
        contra_loss = stack_contra_loss.mean()
        reward_loss = reward_loss_func(reward_weight, rtgs)
        stack_action_logits = torch.stack(action_logit_list, dim=1).squeeze(dim=2)
        return stack_action_logits, mse_loss, contra_loss, reward_loss

    def predict_single_step(self, states, actions, rtgs):
        # 这里面每个state的action是一个item id的序列，维度[B, L, K]
        # y_len 是action序列的长度，因为原始action是一个item list且做了padding，y_len是为了防止取到padding的0的位置
        device = rtgs.device
        B, L = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)
        reward_weight = self.reward_net(state_embeddings)  # [batch_size, 2]
        # rebalance the rtg
        # rtgs = reward_weight.unsqueeze(dim=1) * rtgs
        rtgs = reward_weight * rtgs
        # print(rtgs.shape)
        # rtgs = reward_weight * rtgs
        if actions is not None:
            # print(actions.shape)
            action_embeddings = self.action_encoder(actions.squeeze(2))
            # print(action_embeddings.shape)
        if actions is not None and self.model_type == 'reward_conditioned':
            rtgs = rtgs.to(device).float()
            rtg_embeddings = self.ret_emb(rtgs)
            # print(rtg_embeddings.shape)
            
            token_embeddings = torch.zeros((B, 3 * L, self.config.n_embd), dtype=torch.float32,
                                           device=state_embeddings.device)
            # print(token_embeddings.shape)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':  # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((B, 2 * L, self.config.n_embd), dtype=torch.float32,
                                           device=state_embeddings.device)
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            token_embeddings = torch.zeros((B, 2 * L, self.config.n_embd), dtype=torch.float32,
                                           device=state_embeddings.device)
            token_embeddings[:, ::2, :] = state_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
        elif actions is None and self.model_type == 'naive':  # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        # 在第0维上重复batch次
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size,
                                                     dim=0)  # batch_size, traj_length, n_embd
        indices = torch.full((1, 1, self.config.n_embd), L, dtype=torch.long).to(device)
        # gather的步骤是取出真实长度对应的那个pos embedding
        position_embeddings = torch.gather(all_global_pos_emb, 1, indices) + torch.gather(self.pos_emb, 1, indices)
        x = self.drop(token_embeddings + position_embeddings)
        attention_mask = torch.ones((B, x.shape[1])).bool().to(device)  # [B, 3L]
        for block in self.blocks:
            x = block(x, attention_mask)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = x[:, 1::3, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = x[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = x[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = x  # for completeness
        else:
            raise NotImplementedError()
        return logits[:, -1]