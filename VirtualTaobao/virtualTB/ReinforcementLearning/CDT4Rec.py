import torch
import torch.nn as nn
import transformers
from gpt2 import GPT2Model


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


def generate_causal_mask(seq_len, device):
    # PyTorch Transformer expects mask shape [L, L], with True values being masked positions
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask


def generate_attention_mask(lengths, max_seq_len, B, device):
    # 创建范围序列 [0, 1, 2, ..., L-1]
    range_row = torch.arange(max_seq_len, device=device).unsqueeze(0).repeat(B, 1)  # [B, L]
    # lengths expand成 [B, L]
    lengths_expanded = lengths.unsqueeze(1).repeat(1, max_seq_len)  # [B, L]
    # mask = True 表示参与attention计算的位置（idx < lengths）
    padding_mask = range_row < lengths_expanded
    return padding_mask


class CDT4Rec(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            item_num,
            hidden_size=64,
            item_emb_dim=64,
            max_sign=10000,
            max_length=None,
            action_tanh=True,
            n_layer=3,
            n_head=4,
            activation_function="gelu"):
        super().__init__()
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_positions=max_length,
            n_layer=n_layer,
            n_head=n_head,
            activation_function=activation_function
        )
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # 默认自带causal mask所以不用自己传入
        self.transformer = GPT2Model(config)
        self.hidden_size = hidden_size
        self.embed_timestep = nn.Embedding(3 * max_length, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Sequential(
            # nn.Embedding(max_sign, state_dim),
            nn.Linear(91, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, hidden_size)
        )
        self.embed_action = nn.Sequential(
            # nn.Embedding(item_num, act_dim),
            nn.Linear(27, state_dim),
            nn.GELU(),
            nn.Linear(act_dim, hidden_size)
        )
        self.embed_ln = nn.LayerNorm(hidden_size)
        # note: we don't predict states or returns for the paper
        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.elu = nn.ELU()
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, item_emb_dim)] + ([nn.Tanh()] if action_tanh else []))
        # )
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, 27)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.linear4 = torch.nn.Linear(hidden_size + hidden_size, hidden_size)
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, returns_to_go, causal_mask=None, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if causal_mask is None:
            causal_mask = generate_causal_mask(3 * seq_length, device=states.device)
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        # print(returns_to_go.dtype)
        # print(returns_to_go.dim())
        if returns_to_go.dim() == 2:
            returns_to_go = returns_to_go.unsqueeze(dim=-1)
        # embed each modality with a different head
        # print(states.device)
        state_embeddings = self.embed_state(states)  # [B, L, fea_num, D]
        # state_embeddings = state_embeddings.mean(dim=2)  # [B, L, D]
        action_embeddings = self.embed_action(actions.squeeze(2))
        returns_embeddings = self.embed_return(returns_to_go)
        timesteps = torch.arange(seq_length, device=states.device).expand(batch_size, seq_length)
        time_embeddings = self.embed_timestep(timesteps)
        # print(action_embeddings.shape)
        # print(time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        attention_mask = attention_mask.repeat_interleave(3, dim=1)  # [B, 3L]
        final_mask = attention_mask[:, :, None] & attention_mask[:, None, :] & causal_mask[None, :, :]

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=final_mask,
        )
        x = transformer_outputs['last_hidden_state']
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        #CAUSAL
        state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        y = torch.cat((x[:, 2], x[:, 1]), dim=-1)
        y = self.linear4(y)
        return_preds = self.predict_return(y[:, 2])  # predict next return given state and action
        return state_preds, action_preds, return_preds

    def predict_single_step(self, states, actions, rtgs, attention_mask=None):
        # state: [B, L, D_s], action: [B, L - 1, D_a], rtgs: [B, L, D_r]
        device = rtgs.device
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)
        returns_to_go = rtgs#.unsqueeze(dim=-1)
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)  # [B, L, fea_num, D]
        # state_embeddings = state_embeddings.mean(dim=2)  # [B, L, D]
        action_embeddings = None
        if actions is not None:
            action_embeddings = self.embed_action(actions.squeeze(2))
        returns_embeddings = self.embed_return(returns_to_go)
        timesteps = torch.arange(seq_length, device=states.device).expand(batch_size, seq_length)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        if actions is not None:
            action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if actions is not None:
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
            # to make the attention mask fit the stacked inputs, have to stack it as well
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)
        else:
            # print(returns_embeddings.shape)
            # print(state_embeddings.shape)
            # print(attention_mask.shape)
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2 * seq_length, self.hidden_size)
            # print(stacked_inputs.shape)
            # to make the attention mask fit the stacked inputs, have to stack it as well
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 2 * seq_length)

        stacked_inputs = self.embed_ln(stacked_inputs)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        # print(stacked_inputs.shape)
        # print(stacked_attention_mask.shape)
        if stacked_attention_mask.dim() == 2: 
            stacked_attention_mask = stacked_attention_mask[:, None] * stacked_attention_mask[:, :, None]  # [1, 2, 2]

            # 添加维度使其匹配形状 [batch, 1, seq_len, seq_len]
            # stacked_attention_mask = stacked_attention_mask[:, None, :, :]  # [1, 1, 2, 2]
        # print(stacked_attention_mask.shape)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        if actions is not None:
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)
        # CAUSAL
        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        return action_preds[:, -1]