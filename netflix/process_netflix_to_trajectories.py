import os
import pickle
import json
from collections import defaultdict

DATA_DIR = '/home/gaoxiang12/datasets/download/training_set'
OUTPUT_FILE = 'netflix_trajectorys.json'
MAPPING_FILE = 'original_to_mapped_mapping.json'  # 新增
MAX_TRAJ_LEN = 30
MAX_OBS_LEN = 20

# 0. 加载 movie_id -> mapped_id 映射
if not os.path.exists(MAPPING_FILE):
    raise FileNotFoundError(f"映射文件 {MAPPING_FILE} 不存在，请先运行 ctr_mlp_train.py 生成！")
with open(MAPPING_FILE, 'r') as f:
    original_to_mapped = json.load(f)
# 转为 int key/int value
original_to_mapped = {int(k): int(v) for k, v in original_to_mapped.items()}

# 1. 读取所有评分数据，格式：(user_id, movie_id, rating, date)
user_records = defaultdict(list)  # user_id -> list of (date, movie_id, rating)

for fname in os.listdir(DATA_DIR):
    if not fname.endswith('.txt'):
        continue
    movie_id = int(fname.split('_')[1].split('.')[0])
    with open(os.path.join(DATA_DIR, fname), 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip first line (movie_id:)
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            user_id, rating, date = parts
            user_records[user_id].append((date, movie_id, int(rating)))

# 2. 按用户聚合并排序
user_trajectories = {}
for user_id, records in user_records.items():
    # 按时间排序
    records.sort(key=lambda x: x[0])
    obs, action, reward = [], [], []
    prev_items = []
    for i, (date, movie_id, rating) in enumerate(records[-MAX_TRAJ_LEN:]):
        # obs: 该时刻前最多20个交互item id（映射）
        obs_i = prev_items[-MAX_OBS_LEN:] if len(prev_items) >= MAX_OBS_LEN else [0]*(MAX_OBS_LEN-len(prev_items)) + prev_items
        # obs_i 映射
        obs_i_mapped = [original_to_mapped.get(mid, 0) for mid in obs_i]
        obs.append(obs_i_mapped if len(obs_i_mapped)==MAX_OBS_LEN else obs_i_mapped[-MAX_OBS_LEN:])
        # action 映射
        action.append(original_to_mapped.get(movie_id, 0))
        reward.append((rating-1)/4)
        prev_items.append(movie_id)
    length = len(action)
    # 补0到MAX_TRAJ_LEN
    while len(obs) < MAX_TRAJ_LEN:
        obs.append([0]*MAX_OBS_LEN)
        action.append(0)
        reward.append(0.0)
    user_trajectories[user_id] = {
        'obs': obs,
        'action': action,
        'reward': reward,
        'length': length
    }

# 3. 保存为json
with open(OUTPUT_FILE, 'w') as f:
    json.dump(user_trajectories, f)

print(f'Done! Saved to {OUTPUT_FILE}')