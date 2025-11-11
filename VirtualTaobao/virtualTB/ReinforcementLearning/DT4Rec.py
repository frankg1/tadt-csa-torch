import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        device = next(self.bucket.parameters()).device
        self.bucket_value = self.bucket_value.to(device)
        Meta_emb = self.bucket(self.bucket_value)
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

    def forward(self, X, logits_new, state=None):
        # X：(batch_size,num_steps,embed_size)
        # logits_new: (batch_size, hidden_dim)
        # state: [num_layers, batch_size, hidden_dim]
        X = self.embedding(X)
        # 广播context，使其具有与X相同的num_steps
        context = logits_new.unsqueeze(1).repeat(1, X.shape[1], 1)
        X_and_context = torch.cat([X, context], 2)
        if state is None:
            output, state = self.rnn(X_and_context)
        else:
            output, state = self.rnn(X_and_context, state)
        # output的形状:(batch_size,num_steps,vocab_size)
        return output, state


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
        self.min_rtg_value = config.min_rtg_value
        self.max_rtg_value = config.max_rtg_value


        self.state_encoder = Encoder(91, config.n_embd)
        self.action_encoder = nn.Sequential(
            nn.Linear(27, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        self.decoder = Seq2SeqDecoder(config.vocab_size, config.n_embd, 27, 2,
                                      0.2)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.ret_emb = Autodis(config, config.rtg_bucket_number, min_value=config.min_rtg_value, max_value=config.max_rtg_value)
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
    def forward(self, states, actions, y_len, rtgs, attention_mask):
        # 这里面每个state的action是一个环境给定的embedding，维度[B, L, D]
        # y_len 是action序列的长度，因为原始action是一个item list且做了padding，y_len是为了防止取到padding的0的位置
        device = rtgs.device
        B, L = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)
        action_embeddings = self.action_encoder(actions.squeeze(2))  # actions [batch_size,seq_len,item_num,emb_dim]
        # rtg 负采样
        rtg_neg = torch.zeros([8 * B, rtgs.shape[1]], device=device)
        for i in range(8):
            for j in range(B):
                rtg_neg[i * B + j, :-1] = rtgs[j, 1:]
                rtg_neg[i * B + j, -1] = rtgs[j, -1]

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
        token_neg_embeddings = torch.repeat_interleave(token_embeddings, 8, dim=0)
        rtg_neg_embeddings = self.ret_emb(rtg_neg.type(torch.float32))
        token_neg_embeddings[:, ::3, :] = rtg_neg_embeddings
        token_all = torch.cat((token_embeddings, token_neg_embeddings), dim=0)  # [9B, L, emb_dim]
        position_all = torch.repeat_interleave(position_embeddings, 9, dim=0)   # [9B, L, emb_dim]
        x = self.drop(token_all + position_all)
        attention_mask = attention_mask.repeat_interleave(3, dim=1)  # [B, 3L]
        attention_mask_expand = attention_mask.unsqueeze(1).repeat(1, 9, 1)  # [B, 9, 3L]
        attention_mask_expand = attention_mask_expand.permute(1, 0, 2).reshape(9 * B, -1)  # [9B, 3L]
        for block in self.blocks:
            x = block(x, attention_mask_expand)
        logits = self.linear(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits  # for completeness
        else:
            raise NotImplementedError()
        # logits shape: [9 * batch_size, T, n_emb]
        # if we are given some desired targets also calculate the loss
        loss_func = MaskedMSELoss()
        mse_loss_list = []
        contra_loss_list = []
        action_logit_list = []

        for i in range(actions.shape[1]):
            logits_new = logits[:, i, :]
            targets_seq = actions[:, i, :]
            pos_seq = actions[:, i, :]
            # item编号从1开始，这样0就是mock的bos_id
            bos = torch.tensor([0] * B).to(device).reshape(-1, 1)
            # 序列预测，给mock的第一步，和1 ~ K -1步，concat到一起去预测第1 ~ K步
            if targets_seq.shape[1] > 1:
                dec_input = torch.cat([bos, targets_seq[:, :-1]], dim=1)
            else:
                dec_input = bos
            logits_new_pos = logits_new[:actions.shape[0]]

            Y_emb, _ = self.decoder(dec_input, logits_new_pos)
            action_logit_list.append(Y_emb)

            # Y_emb: [batch_size, rec_item_num, n_emb]的embedding

            dec_input_neg = torch.repeat_interleave(dec_input, 8, 0)
            logits_new_neg = logits_new[actions.shape[0]:]
            neg_seq_emb, _ = self.decoder(dec_input_neg, logits_new_neg)

            mask = attention_mask[:, i].unsqueeze(dim=-1)
            # print(Y_emb.shape, targets_seq.shape, mask.shape)
            loss_mse = loss_func(Y_emb, targets_seq, mask)
            loss_mse = torch.sum(loss_mse, dim=-1)
            pos_seq_emb = pos_seq
            loss_contra = InfoNCE(Y_emb, pos_seq_emb, neg_seq_emb, mask)
            
            mse_loss_list.append(loss_mse)
            contra_loss_list.append(loss_contra)
        # print(torch.stack(mse_loss_list).shape)
        stack_mse_loss = torch.stack(mse_loss_list).squeeze().permute(1, 0)   # [B, L]
        stack_contra_loss = torch.stack(contra_loss_list).permute(1, 0)  # [B, L]
        mse_loss = stack_mse_loss.mean()
        contra_loss = stack_contra_loss.mean()
        stack_action_logits = torch.stack(action_logit_list, dim=1).squeeze(dim=2)
        return stack_action_logits, mse_loss, contra_loss

    def predict_single_step(self, states, actions, rtgs):
        # state: [B, L, D_s], action: [B, L - 1, D_a], rtgs: [B, L, D_r]
        device = rtgs.device
        B, L = states.shape[0], states.shape[1]
        # state_embeddings = states
        # action_embeddings = actions
        state_embeddings = self.state_encoder(states)
        # actions = actions.squeeze(2)
        if actions is not None:
            action_embeddings = self.action_encoder(actions)

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
            token_embeddings = torch.zeros((B, 2 * L - 1, self.config.n_embd), dtype=torch.float32, device=device)
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
        logits = self.linear(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits  # for completeness
        else:
            raise NotImplementedError()

        return logits[:, -1]

        # logits_new = logits[:, -1, :]
        # targets_seq = actions[:, -1, :]
        # pos_seq = actions[:, -1, :]
        # # item编号从1开始，这样0就是mock的bos_id
        # bos = torch.tensor([0] * B).reshape(-1, 1).to(device)
        # # 序列预测，给mock的第一步，和1 ~ K -1步，concat到一起去预测第1 ~ K步
        # # print(actions.shape)
        # if targets_seq.shape[1] > 1:
        #     dec_input = torch.cat([bos, targets_seq[:, :-1]], dim=1)
        # else:
        #     dec_input = bos
        # logits_new_pos = logits_new[:actions.shape[0]]
        # # print("logits_new_pos.shape:", logits_new_pos.shape)
        # Y_emb, _ = self.decoder(dec_input, state_allstep[-1].permute(1, 0, 2), logits_new_pos)
        # # Y_emb: [batch_size, rec_item_num, n_emb]的embedding
        # return Y_emb

