import pandas as pd
import numpy as np
from collections import defaultdict
import json

# 读取数据
df = pd.read_csv('final_ratings_remapped.csv')

# 按用户分组并按时间排序
df = df.sort_values(['userId', 'timestamp'])

# 初始化存储结构
user_trajectories = defaultdict(lambda: {
    "obs": [],
    "action": [],
    "reward": [],
    "length": 0
})

# 滑动窗口大小
STATE_WINDOW = 20
MAX_TRAJ_LENGTH = 30

# 为每个用户构建轨迹
for user_id, group in df.groupby('userId'):
    movie_sequence = group['movieId'].tolist()
    rating_sequence = group['rating'].tolist()
    
    # 初始化状态序列
    states = []
    state_window = [0] * STATE_WINDOW
    
    # 构建每个时间步的状态
    for i in range(min(len(movie_sequence), MAX_TRAJ_LENGTH)):
        # 更新状态窗口
        if i > 0:
            if len(state_window) >= STATE_WINDOW:
                state_window.pop(0)
            state_window.append(movie_sequence[i-1])
        
        # 确保状态窗口长度为20
        current_state = state_window.copy()
        if len(current_state) < STATE_WINDOW:
            current_state = current_state + [0] * (STATE_WINDOW - len(current_state))
        
        states.append(current_state)
        user_trajectories[user_id]["action"].append(movie_sequence[i])
        user_trajectories[user_id]["reward"].append(rating_sequence[i])
    
    user_trajectories[user_id]["obs"] = states
    user_trajectories[user_id]["length"] = len(states)

# 转换为普通字典并保存
user_trajectories = dict(user_trajectories)
with open('user_trajectories.json', 'w') as f:
    json.dump(user_trajectories, f, indent=2)

print("轨迹数据已保存为user_trajectories.json")