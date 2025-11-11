import torch
import torch.nn as nn
import os
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json as pyjson
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))
from utils import compute_metrics
from torch.utils.tensorboard import SummaryWriter

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
    rtg = torch.zeros_like(reward_seq)
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


def generate_attention_mask(lengths, max_seq_len, device):
    range_row = torch.arange(max_seq_len, device=device).unsqueeze(0).repeat(lengths.shape[0], 1)
    lengths_expanded = lengths.repeat(1, max_seq_len)
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
        # 修改embed_state以处理[B, L, 20]的输入
        self.item_embedding = nn.Embedding(max_sign, state_dim)
        self.state_projection = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Sequential(
            nn.Embedding(item_num, act_dim),
            nn.Linear(act_dim, hidden_size)
        )
        self.embed_ln = nn.LayerNorm(hidden_size)
        # note: we don't predict states or returns for the paper
        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.elu = nn.ELU()
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, item_emb_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.linear4 = torch.nn.Linear(hidden_size + hidden_size, hidden_size)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        self.item_emb_layer = nn.Embedding(item_num, item_emb_dim)

    def forward(self, states, actions, returns_to_go, causal_mask=None, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if causal_mask is None:
            causal_mask = generate_causal_mask(3 * seq_length, device=states.device)
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        returns_to_go = returns_to_go.unsqueeze(dim=-1)
        # embed each modality with a different head
        # states: [B, L, 20] -> item_embeddings: [B, L, 20, state_dim]
        item_embeddings = self.item_embedding(states)  # [B, L, 20, state_dim]
        # 对item embeddings取平均，将20个item embedding合并成一个state embedding
        state_embeddings = item_embeddings.mean(dim=2)  # [B, L, state_dim]
        # 投影到hidden_size
        state_embeddings = self.state_projection(state_embeddings)  # [B, L, hidden_size]
        
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        timesteps = torch.arange(seq_length, device=states.device).expand(batch_size, seq_length)
        time_embeddings = self.embed_timestep(timesteps)
        

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
        item_emb_mat = self.item_emb_layer.weight
        action_preds = torch.matmul(action_preds, item_emb_mat.T)
        y = torch.cat((x[:, 2], x[:, 1]), dim=-1)
        y = self.linear4(y)
        return_preds = self.predict_return(y[:, 2])  # predict next return given state and action
        return state_preds, action_preds, return_preds


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
    
    # 检查是否存在分离的train/val文件
    split_path = os.path.join(data_dir, f"{base}_{split}.json")
    if os.path.exists(split_path):
        return split_path
    
    # 如果不存在分离文件，使用单个文件
    single_path = os.path.join(data_dir, f"{base}.json")
    if os.path.exists(single_path):
        print(f"Warning: Using single file {single_path} for both train and val")
        return single_path
    
    raise FileNotFoundError(f"Dataset file not found: {split_path} or {single_path}")

# --- argparse 增加数据集参数 ---
import argparse
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
parser.add_argument('--data_dir_prefix', type=str, default='./data/')
parser.add_argument('--dataset', type=str, default='kuairand', 
                   choices=['kuairand', 'ml', 'retailrocket', 'netflix'],
                   help='Dataset to train on')
parser.add_argument('--data_dir', type=str, 
                   default='../dt_format_datasets/',
                   help='Directory containing dataset files')
parser.add_argument('--device', type=str, default='cuda:0',
                   choices=['cuda:0', 'cuda:1', 'cpu'],
                   help='Device to run on (cuda:0, cuda:1, or cpu)')
parser.add_argument('--use_amp', action='store_true',
                   help='Use automatic mixed precision for faster training')
args = parser.parse_args()

# --- 设置设备 ---
device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    def print_gpu_memory():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        cached = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
else:
    def print_gpu_memory():
        pass

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

# 计算状态和动作的词汇表大小
# 使用更安全的方法，基于已有的统计信息
print("Computing vocabulary sizes...")

# 对于动作，使用已有的统计信息
max_action_id = train_dataset.stats['num_actions'] + 1

# 对于状态，需要检查obs的实际范围
obs_values = []
for t in train_dataset.trajectory_list:
    if 'obs' in t and isinstance(t['obs'], list):
        valid_length = min(t['length'], len(t['obs']))
        for i in range(valid_length):
            obs_item = t['obs'][i]
            if isinstance(obs_item, list):
                # 如果obs_item是列表，展平所有元素
                for sub_item in obs_item:
                    if isinstance(sub_item, (int, float)):
                        obs_values.append(sub_item)
            elif isinstance(obs_item, (int, float)):
                obs_values.append(obs_item)


if not obs_values:
    # 如果没有obs数据，使用action的范围作为默认值
    print("Warning: No obs data found, using action range for state vocabulary")
    max_obs_id = max_action_id
else:
    max_obs_id = max(obs_values) + 1

# 添加更详细的调试信息
if len(train_dataset.trajectory_list) > 0:
    sample_traj = train_dataset.trajectory_list[0]
    print(f"Debug - Sample trajectory keys: {list(sample_traj.keys())}")
    if 'obs' in sample_traj:
        print(f"Debug - Sample obs type: {type(sample_traj['obs'])}, length: {len(sample_traj['obs'])}")
        if len(sample_traj['obs']) > 0:
            print(f"Debug - Sample obs[0] type: {type(sample_traj['obs'][0])}")
            if isinstance(sample_traj['obs'][0], list):
                print(f"Debug - Sample obs[0] length: {len(sample_traj['obs'][0])}")
                if len(sample_traj['obs'][0]) > 0:
                    print(f"Debug - Sample obs[0][0] type: {type(sample_traj['obs'][0][0])}")

print(f"State vocabulary size: {max_obs_id}")
print(f"Action vocabulary size: {max_action_id}")
if len(obs_values) > 0:
    print(f"State range: {min(obs_values)} to {max(obs_values)}")
print(f"Action range: {train_dataset.stats['action_range']}")

item_emb_dim = 64
learning_rate = 5e-3
num_epochs = args.epochs

# vocab_size = item_num+1，保证embedding和label一致
state_dim = 64
act_dim = 64
L = args.context_length
from gpt2 import GPT2Model
import transformers
model = CDT4Rec(state_dim=state_dim, act_dim=act_dim, item_num=max_action_id, hidden_size=64, item_emb_dim=64, max_sign=max_obs_id, max_length=L,
                n_layer=3, n_head=4, activation_function='gelu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# TensorBoard writer
log_dir = f"runs/cdt4rec_{args.dataset}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# Initialize mixed precision training if enabled
scaler = None
if args.use_amp and device.type == 'cuda':
    scaler = torch.cuda.amp.GradScaler()
    print("Using Automatic Mixed Precision (AMP)")

# --- 训练与验证主循环 ---
for epoch in range(1, num_epochs + 1):
    model.train()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    
    # Clear GPU cache at the beginning of each epoch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print_gpu_memory()
    for batch in pbar:
        obs_seq = batch['obs']
        act_id_seq = batch['action']
        reward_gt = batch['reward']
        seq_lengths = batch['length'].unsqueeze(1)
        # Move data to device
        obs_seq = obs_seq.to(device)
        act_id_seq = act_id_seq.to(device)
        reward_gt = reward_gt.to(device)
        seq_lengths = seq_lengths.to(device)

        rtg_seq = compute_rtg_from_rewards(reward_gt)
        causal_mask = generate_causal_mask(3 * L, device=device)
        attention_mask = generate_attention_mask(seq_lengths, obs_seq.shape[1], device=device)
        state_preds, action_preds, return_preds = model(obs_seq, act_id_seq, rtg_seq, causal_mask=causal_mask, attention_mask=attention_mask)
        gt_actions = act_id_seq.clone().detach().view(-1)
        gt_rtgs = rtg_seq.clone().detach()
        action_preds_flat = action_preds.view(-1, action_preds.size(-1))
        attention_mask_flat = attention_mask.reshape(-1)
        valid_action_preds_flat = action_preds_flat[attention_mask_flat]
        valid_gt_actions = gt_actions[attention_mask_flat]
        action_loss = loss_fn(valid_action_preds_flat, valid_gt_actions)
        rtg_loss = (attention_mask * (return_preds - gt_rtgs) ** 2).sum() / attention_mask.sum()
        total_loss = action_loss + rtg_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Clear gradients to save memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    # TensorBoard logging for training loss
    writer.add_scalar('Loss/Total', total_loss.item(), epoch)
    writer.add_scalar('Loss/CE', action_loss.item(), epoch)
    writer.add_scalar('Loss/Return', rtg_loss.item(), epoch)
    pbar.set_postfix({
        'Total': f"{total_loss.item():.4f}",
        'CE': f"{action_loss.item():.4f}",
        'Return': f"{rtg_loss.item():.4f}"
    })
    print(f"Epoch {epoch} | Total Loss: {total_loss.item():.4f} | CE Loss: {action_loss.item():.4f} | "
          f"Return Loss: {rtg_loss.item():.4f}")
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
            causal_mask = generate_causal_mask(3 * L, device=device)
            attention_mask = generate_attention_mask(seq_lengths, obs_seq.shape[1], device=device)
            # Forward pass
            state_preds, action_preds, return_preds = model(obs_seq, act_id_seq, rtg_seq, causal_mask=causal_mask, attention_mask=attention_mask)
            B, L_val = act_id_seq.shape
            last_indices = batch['length'] - 1
            batch_indices = torch.arange(B, device=device)
            valid_mask = batch['length'] > 0
            if valid_mask.sum() > 0:
                last_action_preds = action_preds[batch_indices, last_indices]
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
        log_filename = f'cdt4rec_{args.dataset}_eval.log'
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(pyjson.dumps(log_data, ensure_ascii=False) + '\n')
        # TensorBoard logging for metrics
        for k, v in metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

# Save model weights at the end of training
model_save_path = os.path.join(log_dir, f"cdt4rec_{args.dataset}_final.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
writer.close()