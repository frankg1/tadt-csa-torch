import pandas as pd
import numpy as np
import pickle
import os
import json
from collections import defaultdict
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from config import (
    get_data_file_path, get_output_file_path, validate_data_path,
    DATA_CONFIG, OUTPUT_CONFIG
)


class TrajectoryProcessor:
    """
    轨迹处理器 - 将推荐数据转换为指定格式
    轨迹长度最大30，obs每个位置是20长度的item_id列表
    """

    def __init__(self, max_trajectory_length=30, max_obs_length=20):
        """
        初始化轨迹处理器
        Args:
            max_trajectory_length: 最大轨迹长度
            max_obs_length: obs中每个位置的最大长度
        """
        self.max_trajectory_length = max_trajectory_length
        self.max_obs_length = max_obs_length
        
        # 10000个item的集合
        self.filtered_video_ids = set()
        self.video_id_to_embedding_idx = {}

    def load_filtered_video_ids(self):
        """加载CTR训练产出的映射文件"""
        print("Loading CTR training output files...")
        
        # 加载映射的item IDs
        item_ids_path = get_output_file_path('item_ids_file')
        if os.path.exists(item_ids_path):
            mapped_item_ids = torch.load(item_ids_path).tolist()
            print(f"Loaded {len(mapped_item_ids)} mapped item IDs")
            print(f"Mapped ID range: {min(mapped_item_ids)} - {max(mapped_item_ids)}")
        else:
            raise FileNotFoundError(f"Mapped item IDs file not found: {item_ids_path}")
        
        # 加载原始到映射的映射关系
        id_mapping_path = get_output_file_path('id_mapping_file')
        if os.path.exists(id_mapping_path):
            with open(id_mapping_path, 'rb') as f:
                self.original_to_mapped = pickle.load(f)
            print(f"Loaded {len(self.original_to_mapped)} original-to-mapped mappings")
        else:
            raise FileNotFoundError(f"ID mapping file not found: {id_mapping_path}")
        
        # 限制item_id的数量和数值大小都小于等于10000
        max_items = 10000
        filtered_mapped_ids = [vid for vid in mapped_item_ids if vid <= max_items]
        
        if len(filtered_mapped_ids) > max_items:
            filtered_mapped_ids = filtered_mapped_ids[:max_items]
            print(f"Warning: Limited to first {max_items} items")
        
        # 创建映射的video_id集合和索引映射
        self.filtered_video_ids = set(filtered_mapped_ids)
        self.video_id_to_embedding_idx = {vid: idx for idx, vid in enumerate(filtered_mapped_ids)}
        
        # 创建映射到原始的映射关系（用于调试）
        self.mapped_to_original = {mapped_id: original_id for original_id, mapped_id in self.original_to_mapped.items() if mapped_id in self.filtered_video_ids}
        
        print(f"Filtered video IDs (mapped): {len(self.filtered_video_ids)}")
        print(f"Video ID mapping range: {min(self.filtered_video_ids)} - {max(self.filtered_video_ids)}")
        print(f"Video ID values <= 10000: {max(self.filtered_video_ids) <= 10000}")
        print(f"Original video ID range: {min(self.original_to_mapped.keys())} - {max(self.original_to_mapped.keys())}")

    def load_data(self):
        """加载数据"""
        print("Loading data for trajectory processing...")

        # 验证数据路径
        validate_data_path()

        # 加载CTR训练产出的映射文件
        self.load_filtered_video_ids()

        # 加载交互数据
        print("Loading interaction data...")
        interactions = pd.read_csv(get_data_file_path('train_interactions'))
        print(f"Loaded {len(interactions)} interactions")

        # 将原始video_id映射到新的ID范围
        print("Mapping original video IDs to mapped IDs...")
        interactions['mapped_video_id'] = interactions['video_id'].map(self.original_to_mapped)
        
        # 过滤掉无法映射的记录（不在训练数据中的video_id）
        original_count = len(interactions)
        interactions = interactions.dropna(subset=['mapped_video_id'])
        interactions['mapped_video_id'] = interactions['mapped_video_id'].astype(int)
        
        # 进一步过滤，只保留映射ID在10000以内的记录
        interactions = interactions[interactions['mapped_video_id'].isin(self.filtered_video_ids)]
        filtered_count = len(interactions)
        print(f"Filtered from {original_count} to {filtered_count} interactions")
        print(f"Filtering ratio: {filtered_count/original_count:.4f}")
        
        # 验证映射后的ID范围
        mapped_ids = interactions['mapped_video_id'].unique()
        print(f"Mapped video ID range in interactions: {mapped_ids.min()} - {mapped_ids.max()}")
        print(f"Unique mapped video IDs in interactions: {len(mapped_ids)}")
        print(f"All mapped IDs <= 10000: {mapped_ids.max() <= 10000}")

        return interactions

    def create_ctr_labels(self, interactions):
        """创建CTR标签"""
        interactions['reward'] = (
                (interactions['is_click'] == 1) |
                (interactions['is_like'] == 1) |
                (interactions['is_follow'] == 1) |
                (interactions['is_comment'] == 1) |
                (interactions['is_forward'] == 1)
        ).astype(int)
        return interactions

    def build_trajectories(self, interactions):
        """构建用户轨迹"""
        print("Building user trajectories...")

        # 按用户ID和时间排序
        interactions = interactions.sort_values(['user_id', 'time_ms'])

        # 按用户分组
        user_groups = interactions.groupby('user_id')

        trajectories = {}

        for user_id, user_data in tqdm(user_groups, desc="Processing users"):
            if len(user_data) < 2:  # 至少需要2个交互
                continue

            # 构建轨迹
            trajectory = self._build_single_trajectory(user_data)
            
            if trajectory is not None:
                trajectories[str(user_id)] = trajectory

        print(f"Built {len(trajectories)} valid trajectories")
        return trajectories

    def _build_single_trajectory(self, user_data):
        """构建单个用户轨迹"""
        # 限制轨迹长度
        if len(user_data) > self.max_trajectory_length:
            user_data = user_data.head(self.max_trajectory_length)

        actual_length = len(user_data)
        
        # 初始化轨迹数据
        obs = []
        actions = []
        rewards = []
        
        # 历史交互记录
        history_items = []

        for idx, row in user_data.iterrows():
            # 当前action和reward（使用映射后的video_id）
            action = int(row['mapped_video_id'])
            reward = int(row['reward'])
            
            # 构建当前时刻的obs（历史交互的item_id列表）
            current_obs = history_items.copy()
            
            # 如果历史不足max_obs_length，用0填充
            if len(current_obs) < self.max_obs_length:
                padding = [0] * (self.max_obs_length - len(current_obs))
                current_obs.extend(padding)
            else:
                # 如果超过max_obs_length，只保留最近的
                current_obs = current_obs[-self.max_obs_length:]
            
            obs.append(current_obs)
            actions.append(action)
            rewards.append(reward)
            
            # 更新历史记录
            history_items.append(action)
            # 只保留最近的max_obs_length个
            if len(history_items) > self.max_obs_length:
                history_items = history_items[-self.max_obs_length:]

        # 如果轨迹长度不足max_trajectory_length，进行填充
        if actual_length < self.max_trajectory_length:
            # 计算需要填充的长度
            padding_length = self.max_trajectory_length - actual_length
            
            # 填充obs（每个位置都是20个0）
            padding_obs = [[0] * self.max_obs_length for _ in range(padding_length)]
            obs.extend(padding_obs)
            
            # 填充actions和rewards
            actions.extend([0] * padding_length)
            rewards.extend([0] * padding_length)

        return {
            'obs': obs,           # 30 * 20
            'action': actions,    # 30
            'reward': rewards,    # 30
            'length': actual_length  # 实际序列长度
        }

    def save_trajectories(self, trajectories, output_path=None):
        """保存轨迹数据为JSON格式"""
        if output_path is None:
            output_path = get_output_file_path('model_dir') + '/kuairand_trajectories.json'

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 只保存轨迹数据，不包含其他映射信息
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trajectories, f, ensure_ascii=False, indent=2)

        print(f"Trajectories saved to {output_path}")
        print(f"Total trajectories: {len(trajectories)}")

        # 统计信息
        self._print_statistics(trajectories)

    def _print_statistics(self, trajectories):
        """打印统计信息"""
        total_interactions = 0
        total_rewards = []
        trajectory_lengths = []
        
        for user_id, trajectory in trajectories.items():
            # 使用length字段
            actual_length = trajectory['length']
            trajectory_lengths.append(actual_length)
            
            # 统计reward
            rewards = trajectory['reward'][:actual_length]
            total_rewards.extend(rewards)
            total_interactions += actual_length

        print(f"Average trajectory length: {np.mean(trajectory_lengths):.2f}")
        print(f"Min trajectory length: {np.min(trajectory_lengths)}")
        print(f"Max trajectory length: {np.max(trajectory_lengths)}")
        print(f"Total interactions: {total_interactions}")
        print(f"Positive rate: {np.mean(total_rewards):.4f}")
        print(f"Trajectory format: {self.max_trajectory_length} * {self.max_obs_length}")

    def get_trajectory_statistics(self, trajectories):
        """获取轨迹统计信息"""
        total_interactions = 0
        total_rewards = []
        trajectory_lengths = []
        
        for user_id, trajectory in trajectories.items():
            actual_length = trajectory['length']
            trajectory_lengths.append(actual_length)
            
            rewards = trajectory['reward'][:actual_length]
            total_rewards.extend(rewards)
            total_interactions += actual_length

        stats = {
            'total_trajectories': len(trajectories),
            'total_interactions': total_interactions,
            'avg_trajectory_length': np.mean(trajectory_lengths),
            'min_trajectory_length': np.min(trajectory_lengths),
            'max_trajectory_length': np.max(trajectory_lengths),
            'filtered_video_count': len(self.filtered_video_ids),
            'mapped_video_id_range': f"{min(self.filtered_video_ids)} - {max(self.filtered_video_ids)}",
            'original_video_id_range': f"{min(self.original_to_mapped.keys())} - {max(self.original_to_mapped.keys())}",
            'positive_rate': np.mean(total_rewards),
            'total_positive': int(np.sum(total_rewards))
        }

        return stats


def main():
    """主函数"""
    print("Starting filtered trajectory processing...")

    # 创建轨迹处理器
    processor = TrajectoryProcessor(
        max_trajectory_length=30,
        max_obs_length=20
    )

    # 加载数据
    interactions = processor.load_data()

    # 创建CTR标签
    interactions = processor.create_ctr_labels(interactions)

    # 构建轨迹
    trajectories = processor.build_trajectories(interactions)

    # 保存轨迹
    processor.save_trajectories(trajectories)

    # 打印统计信息
    stats = processor.get_trajectory_statistics(trajectories)
    print("\nFiltered Trajectory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 显示示例轨迹
    print("\nExample trajectory:")
    if len(trajectories) > 0:
        user_id = list(trajectories.keys())[0]
        trajectory = trajectories[user_id]
        print(f"User ID: {user_id}")
        print(f"Obs shape: {len(trajectory['obs'])} x {len(trajectory['obs'][0])}")
        print(f"Action length: {len(trajectory['action'])}")
        print(f"Reward length: {len(trajectory['reward'])}")
        print(f"Actual length: {trajectory['length']}")
        print(f"First 3 obs: {trajectory['obs'][:3]}")
        print(f"First 5 actions: {trajectory['action'][:5]}")
        print(f"First 5 rewards: {trajectory['reward'][:5]}")
        print(f"Last 5 actions (including padding): {trajectory['action'][-5:]}")

    print("\nFiltered trajectory processing completed!")


if __name__ == "__main__":
    main() 