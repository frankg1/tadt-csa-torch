import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 参数
DATA_DIR = '/home/gaoxiang12/datasets/download/training_set'
ITEM_ID_FILE = 'item_ids.json'
ITEM_EMB_FILE = 'item_embeddings.json'
EMB_DIM = 64
MAX_ITEMS = 10000
BATCH_SIZE = 8192
EPOCHS = 1
LR = 1e-3

print("=== Netflix CTR MLP Training ===")
print(f"Data directory: {DATA_DIR}")
print(f"Max items: {MAX_ITEMS}")
print(f"Embedding dimension: {EMB_DIM}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LR}")

# 1. 直接从原始数据读取样本 (user_id, item_id, ctr)
print("\n1. Loading and processing raw data...")
samples = []
user_set = set()
item_count = {}

# 获取所有txt文件
txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
print(f"Found {len(txt_files)} movie files")

for fname in tqdm(txt_files, desc="Processing movie files"):
    movie_id = int(fname.split('_')[1].split('.')[0])
    file_path = os.path.join(DATA_DIR, fname)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            user_id, rating, date = parts
            user_id = int(user_id)
            rating = int(rating)
            ctr = 1 if rating == 5 else 0
            samples.append((user_id, movie_id, ctr))
            user_set.add(user_id)
            item_count[movie_id] = item_count.get(movie_id, 0) + 1

print(f"Total samples loaded: {len(samples):,}")
print(f"Unique users: {len(user_set):,}")
print(f"Unique movies: {len(item_count):,}")

# 2. 选取出现次数最多的前10000个item
print("\n2. Selecting top items by frequency...")
item_ids = sorted(item_count, key=item_count.get, reverse=True)[:MAX_ITEMS]
# 映射到[1, 10000]区间
item2idx = {i: idx + 1 for idx, i in enumerate(item_ids)}  # 从1开始映射
user_ids = sorted(user_set)
user2idx = {u: i for i, u in enumerate(user_ids)}

print(f"Selected top {len(item_ids)} items")
print(f"Item frequency range: {item_count[item_ids[0]]} - {item_count[item_ids[-1]]}")
print(f"Item ID mapping range: 1 - {MAX_ITEMS}")

# 3. 过滤只保留前10000个item的样本
print("\n3. Filtering samples to selected items...")
original_sample_count = len(samples)
samples = [s for s in samples if s[1] in item2idx]
filtered_sample_count = len(samples)

print(f"Filtered from {original_sample_count:,} to {filtered_sample_count:,} samples")
print(f"Filtering ratio: {filtered_sample_count/original_sample_count:.4f}")

# 4. 构建Dataset
print("\n4. Creating dataset and dataloader...")
class CTRDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        user, item, ctr = self.samples[idx]
        return user2idx[user], item2idx[item], ctr

dataset = CTRDataset(samples)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset size: {len(dataset):,}")
print(f"Number of batches: {len(dataloader)}")

# 5. 构建MLP模型
print("\n5. Building MLP model...")
class CTRMLP(nn.Module):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        # item embedding需要处理[1, num_items]的索引，所以size要+1
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)  # 0作为padding
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x), i

model = CTRMLP(len(user2idx), len(item2idx), EMB_DIM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model created with {len(user2idx):,} users and {len(item2idx):,} items")
print(f"Item embedding size: {len(item2idx) + 1} (to handle indices 1-{MAX_ITEMS})")
print(f"Using device: {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# 6. 训练
print("\n6. Starting training...")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batch_count = 0
    
    # 使用tqdm显示每个epoch的进度
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    
    for user_idx, item_idx, ctr in pbar:
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        ctr = ctr.float().to(device).unsqueeze(1)
        
        pred, _ = model(user_idx, item_idx)
        loss = loss_fn(pred, ctr)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * user_idx.size(0)
        batch_count += 1
        
        # 更新进度条
        avg_loss = total_loss / (batch_count * BATCH_SIZE)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    epoch_loss = total_loss / len(dataset)
    print(f'Epoch {epoch+1}/{EPOCHS} completed, Average Loss: {epoch_loss:.4f}')

# 7. 保存item id和item embedding
print("\n7. Saving results...")
# 保存映射后的item_id列表 [1, 10000]
item_id_list = [i for i in range(1, MAX_ITEMS + 1)]  # 1, 2, 3, ..., 10000
with open(ITEM_ID_FILE, 'w') as f:
    json.dump(item_id_list, f)

print(f"Extracting item embeddings...")
item_emb_matrix = model.item_emb.weight.data.cpu().numpy()
# 注意：embedding矩阵的索引0是padding，实际item从索引1开始
# 保存映射后的item_id [1, 10000] 作为key
item_emb_dict = {str(mapped_id): item_emb_matrix[mapped_id].tolist() for mapped_id in tqdm(range(1, MAX_ITEMS + 1), desc="Creating embedding dict")}

with open(ITEM_EMB_FILE, 'w') as f:
    json.dump(item_emb_dict, f)

# 保存原始movie_id到映射item_id的映射关系
original_to_mapped = {original_id: mapped_id for original_id, mapped_id in item2idx.items()}
mapping_file = 'original_to_mapped_mapping.json'
with open(mapping_file, 'w') as f:
    json.dump(original_to_mapped, f)

print(f"✓ Saved {len(item_id_list):,} mapped item ids (1-{MAX_ITEMS}) to {ITEM_ID_FILE}")
print(f"✓ Saved item embeddings to {ITEM_EMB_FILE}")
print(f"✓ Saved original to mapped mapping to {mapping_file}")
print(f"✓ Embedding shape: {item_emb_matrix.shape}")
print(f"✓ Item ID range in saved data: 1 - {MAX_ITEMS}")
print(f"✓ Original movie ID range: {min(item2idx.keys())} - {max(item2idx.keys())}")

print("\n=== Training completed successfully! ===")