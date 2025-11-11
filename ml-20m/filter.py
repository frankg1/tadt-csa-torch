import pandas as pd
import numpy as np

#读取数据
df = pd.read_csv('ratings.csv')

# 1. 筛选出至少有30条记录的用户
user_counts = df['userId'].value_counts()
valid_users = user_counts[user_counts >= 30].index
df_filtered = df[df['userId'].isin(valid_users)]

# 2. 对每个用户随机采样30条记录
df_sampled = df_filtered.groupby('userId').apply(
    lambda x: x.sample(n=30, random_state=42)  # 固定随机种子，确保可复现
).reset_index(drop=True)

# 3. 逐步累积用户，直到独特 movieId 接近 10,000
unique_movies = set()
selected_users = []
target_movies = 10000
# tolerance = 0.05  # 允许5%的误差（即9500-10500之间）

# 按 userId 排序，逐步添加用户
sorted_users = np.sort(df_sampled['userId'].unique())

for user in sorted_users:
    user_movies = df_sampled[df_sampled['userId'] == user]['movieId'].unique()
    new_movies = set(user_movies) - unique_movies
    
    # 如果添加该用户不会让 movieId 数量远超目标，则保留
    if len(unique_movies) + len(new_movies) <= target_movies:
        unique_movies.update(new_movies)
        selected_users.append(user)
    else:
        break  # 如果超出目标，停止添加

print(f"最终用户数: {len(selected_users)}")
print(f"独特 movieId 数量: {len(unique_movies)}")

# 4. 筛选最终数据并保存
filtered_df = df_sampled[df_sampled['userId'].isin(selected_users)]
filtered_df['rating'] = filtered_df['rating'].apply(lambda x: 1 if x >= 5.0 else 0)
filtered_df = filtered_df.sort_values(['userId','timestamp'])

# 保存结果
filtered_df.to_csv('filtered_ratings_10k_movies.csv', index=False)

# 1. 读取 genome-scores.csv 并处理数据
genome_scores = pd.read_csv('genome-scores.csv')

# 按 movieId 分组，对每个 movieId 按 relevance 降序排序，并取前5个 tagId
top_tags = (
    genome_scores
    .sort_values(['movieId', 'relevance'], ascending=[True, False])
    .groupby('movieId')
    .head(5)
)

# 将前5个 tagId 转换为 tag1~tag5 的格式
top_tags_pivot = (
    top_tags
    .assign(rank=lambda x: x.groupby('movieId').cumcount() + 1)  # 给每个 movieId 的 tag 排名1~5
    .pivot(index='movieId', columns='rank', values='tagId')
    .rename(columns={1: 'tag1', 2: 'tag2', 3: 'tag3', 4: 'tag4', 5: 'tag5'})
    .reset_index()
)

# 2. 读取之前生成的评分文件
ratings = pd.read_csv('filtered_ratings_10k_movies.csv')

# 3. 合并数据
ratings_with_tags = ratings.merge(top_tags_pivot, on='movieId', how='left')

# 4. 保存结果
ratings_with_tags.to_csv('ratings_with_top5_tags.csv', index=False)

print("处理完成！已添加 tag1~tag5 列到文件中。")

df = pd.read_csv('ratings_with_top5_tags.csv')

# 2. 创建映射字典
# userId 映射
unique_users = df['userId'].unique()
user_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_users), 1)}

# movieId 映射
unique_movies = df['movieId'].unique()
movie_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_movies), 1)}

# 3. 应用映射
df['userId'] = df['userId'].map(user_map)
df['movieId'] = df['movieId'].map(movie_map)

# 4. 保存映射字典（可选，供后续参考）
pd.DataFrame(user_map.items(), columns=['original_id', 'new_id']).to_csv('user_id_mapping.csv', index=False)
pd.DataFrame(movie_map.items(), columns=['original_id', 'new_id']).to_csv('movie_id_mapping.csv', index=False)

# 5. 保存最终文件
df.to_csv('final_ratings_remapped.csv', index=False)

print("处理完成！")
print(f"原始用户数: {len(unique_users)}，映射后用户ID范围: 1-{len(unique_users)}")
print(f"原始电影数: {len(unique_movies)}，映射后电影ID范围: 1-{len(unique_movies)}")