import pandas as pd
import numpy as np

# 1. 读取数据
df = pd.read_csv('events.csv')

# 2. 筛选至少有30条记录的用户
user_counts = df['visitorid'].value_counts()
valid_users = user_counts[user_counts >= 30].index
df_filtered = df[df['visitorid'].isin(valid_users)]

# 3. 对每个用户随机采样30条记录
df_sampled = df_filtered.groupby('visitorid').apply(
    lambda x: x.sample(n=30, random_state=42)
).reset_index(drop=True)

# 4. 逐步累积用户直到itemid接近10,000
unique_items = set()
selected_users = []
target_items = 10000

# 按visitorid排序后逐步添加
sorted_users = np.sort(df_sampled['visitorid'].unique())

for user in sorted_users:
    user_items = df_sampled[df_sampled['visitorid'] == user]['itemid'].unique()
    new_items = set(user_items) - unique_items
    
    if len(unique_items) + len(new_items) <= target_items:
        unique_items.update(new_items)
        selected_users.append(user)
    else:
        break

# 5. 筛选最终用户数据
df_final = df_sampled[df_sampled['visitorid'].isin(selected_users)]

# 6. 新增rating列
df_final['rating'] = np.where(df_final['event'] == 'view', 0, 1)

# 7. 按visitorid排序并保存
df_final = df_final.sort_values(['visitorid','timestamp'])
df_final.to_csv('processed_events.csv', index=False)

# 输出统计信息
print(f"最终用户数: {len(selected_users)}")
print(f"独特itemid数量: {len(unique_items)}")
print(f"文件已保存为 processed_events.csv")


df = pd.read_csv('processed_events.csv')

# 2. 创建映射字典
# visitorid 映射（按原始ID排序后映射）
unique_visitors = sorted(df['visitorid'].unique())
visitor_map = {old_id: new_id for new_id, old_id in enumerate(unique_visitors, 1)}

# itemid 映射（按原始ID排序后映射）
unique_items = sorted(df['itemid'].unique())
item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items, 1)}

# 3. 应用映射
df['visitorid'] = df['visitorid'].map(visitor_map)
df['itemid'] = df['itemid'].map(item_map)

# 4. 保存映射字典（可选）
pd.DataFrame.from_dict(visitor_map, orient='index', columns=['new_id'])\
    .reset_index().rename(columns={'index':'original_id'})\
    .to_csv('visitorid_mapping.csv', index=False)

pd.DataFrame.from_dict(item_map, orient='index', columns=['new_id'])\
    .reset_index().rename(columns={'index':'original_id'})\
    .to_csv('itemid_mapping.csv', index=False)

# 5. 保存最终文件
df.to_csv('final_processed_events.csv', index=False)

# 输出统计信息
print(f"映射完成！")
print(f"原始visitorid数量: {len(unique_visitors)}，映射后范围: 1-{len(unique_visitors)}")
print(f"原始itemid数量: {len(unique_items)}，映射后范围: 1-{len(unique_items)}")
print(f"最终文件已保存为 final_processed_events.csv")
print(f"映射字典已保存为 visitorid_mapping.csv 和 itemid_mapping.csv")


# 读取数据
df = pd.read_csv('final_processed_events.csv')

# 按visitorid和timestamp排序
df = df.sort_values(['visitorid', 'timestamp'])

# 初始化新列
for i in range(1, 11):
    df[f'last_view{i}_itemid'] = 0

# 为每个visitorid处理历史记录
for visitor_id, group in df.groupby('visitorid'):
    view_history = []  # 保存浏览历史
    
    for idx, row in group.iterrows():
        # 先设置当前记录的last_view（不包含当前itemid）
        last_views = view_history[-10:] if len(view_history) >= 10 else [0]*(10-len(view_history)) + view_history
        for i in range(1, 11):
            df.at[idx, f'last_view{i}_itemid'] = last_views[-i] if i <= len(last_views) else 0
        
        # 后更新历史记录（如果是view事件且不包含当前itemid）
        if row['event'] == 'view':
            view_history.append(row['itemid'])

# 保存结果
df.to_csv('final_events_with_history.csv', index=False)

print("处理完成！已添加last_view1_itemid~last_view10_itemid列（不包含当前itemid）")
print(f"结果已保存为 final_events_with_history.csv")