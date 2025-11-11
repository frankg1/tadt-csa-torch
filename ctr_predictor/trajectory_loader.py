import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from config import get_output_file_path


class TrajectoryDataset(Dataset):
    """
    轨迹数据集 - 为Decision Transformer提供数据接口
    """

    def __init__(self, trajectories, max_length=200, normalize_rewards=True):
        """
        初始化轨迹数据集
        Args:
            trajectories: 轨迹列表
            max_length: 最大序列长度，默认200
            normalize_rewards: 是否标准化reward
        """
        self.trajectories = trajectories
        self.max_length = max_length
        self.normalize_rewards = normalize_rewards

        # 计算reward统计信息用于标准化
        if normalize_rewards:
            all_rewards = []
            for traj in trajectories:
                all_rewards.extend(traj['reward_seq'])
            self.reward_mean = np.mean(all_rewards)
            self.reward_std = np.std(all_rewards) + 1e-8
        else:
            self.reward_mean = 0.0
            self.reward_std = 1.0

        # 计算状态维度
        if len(trajectories) > 0:
            self.state_dim = trajectories[0]['obs_seq'].shape[1]
        else:
            self.state_dim = 0

        print(f"Dataset initialized with {len(trajectories)} trajectories")
        print(f"State dimension: {self.state_dim}")
        print(f"Max sequence length: {self.max_length}")
        print(f"Reward normalization: mean={self.reward_mean:.4f}, std={self.reward_std:.4f}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """获取单个轨迹"""
        trajectory = self.trajectories[idx]

        # 提取序列数据
        obs_seq = trajectory['obs_seq'].copy()
        action_seq = trajectory['action_seq'].copy()
        reward_seq = trajectory['reward_seq'].copy()

        # 标准化reward
        if self.normalize_rewards:
            reward_seq = (reward_seq - self.reward_mean) / self.reward_std

        # 处理序列长度
        current_length = len(obs_seq)

        if current_length < self.max_length:
            # 填充到最大长度
            padding_length = self.max_length - current_length

            # 填充obs_seq（用0填充）
            obs_padding = np.zeros((padding_length, self.state_dim), dtype=np.int64)
            obs_seq = np.concatenate([obs_seq, obs_padding], axis=0)

            # 填充action_seq（用0填充）
            action_padding = np.zeros(padding_length, dtype=np.int64)
            action_seq = np.concatenate([action_seq, action_padding])

            # 填充reward_seq（用0填充）
            reward_padding = np.zeros(padding_length, dtype=np.float32)
            reward_seq = np.concatenate([reward_seq, reward_padding])

        elif current_length > self.max_length:
            # 截断到最大长度
            obs_seq = obs_seq[:self.max_length]
            action_seq = action_seq[:self.max_length]
            reward_seq = reward_seq[:self.max_length]
            current_length = self.max_length

        # 转换为tensor
        obs_seq = torch.LongTensor(obs_seq)  # 整数编码
        action_seq = torch.LongTensor(action_seq)
        reward_seq = torch.FloatTensor(reward_seq)

        return {
            'obs_seq': obs_seq,
            'action_seq': action_seq,
            'reward_seq': reward_seq,
            'trajectory_length': current_length,
            'user_id': trajectory['user_id']
        }

    def get_statistics(self):
        """获取数据集统计信息"""
        lengths = [len(traj['obs_seq']) for traj in self.trajectories]
        rewards = []
        for traj in self.trajectories:
            rewards.extend(traj['reward_seq'])

        return {
            'num_trajectories': len(self.trajectories),
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'state_dim': self.state_dim,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'positive_rate': np.mean(rewards)
        }


class TrajectoryDataLoader:
    """
    轨迹数据加载器 - 提供批处理功能
    """

    def __init__(self, trajectories, batch_size=32, max_length=200,
                 shuffle=True, normalize_rewards=True):
        """
        初始化数据加载器
        Args:
            trajectories: 轨迹列表
            batch_size: 批次大小
            max_length: 最大序列长度，默认200
            shuffle: 是否打乱数据
            normalize_rewards: 是否标准化reward
        """
        self.dataset = TrajectoryDataset(trajectories, max_length, normalize_rewards)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length

        # 创建DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=0  # 避免多进程问题
        )

    def _collate_fn(self, batch):
        """自定义批处理函数，处理固定长度序列"""
        batch_size = len(batch)

        # 所有序列都已经填充到相同长度，直接堆叠
        obs_batch = torch.stack([item['obs_seq'] for item in batch])
        action_batch = torch.stack([item['action_seq'] for item in batch])
        reward_batch = torch.stack([item['reward_seq'] for item in batch])
        length_batch = torch.LongTensor([item['trajectory_length'] for item in batch])
        user_id_batch = [item['user_id'] for item in batch]

        return {
            'obs_seq': obs_batch,  # [batch_size, max_length, state_dim]
            'action_seq': action_batch,  # [batch_size, max_length]
            'reward_seq': reward_batch,  # [batch_size, max_length]
            'trajectory_lengths': length_batch,  # [batch_size]
            'user_ids': user_id_batch
        }

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def get_statistics(self):
        """获取数据统计信息"""
        return self.dataset.get_statistics()


def load_trajectories(file_path=None):
    """
    加载轨迹数据
    Args:
        file_path: 轨迹文件路径，None表示使用默认路径
    Returns:
        trajectories: 轨迹列表
        encoders: 编码器字典
    """
    if file_path is None:
        file_path = get_output_file_path('trajectories_file')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    print(f"Loading trajectories from {file_path}")
    with open(file_path, 'rb') as f:
        save_data = pickle.load(f)

    # 兼容旧格式
    if isinstance(save_data, list):
        trajectories = save_data
        encoders = {}
    else:
        trajectories = save_data['trajectories']
        encoders = {
            'user_encoders': save_data.get('user_encoders', {}),
            'video_encoders': save_data.get('video_encoders', {})
        }

    print(f"Loaded {len(trajectories)} trajectories")
    if encoders:
        print(f"Loaded {len(encoders.get('user_encoders', {}))} user encoders")
        print(f"Loaded {len(encoders.get('video_encoders', {}))} video encoders")

    return trajectories, encoders


def create_trajectory_dataloader(trajectories=None, batch_size=32, max_length=200,
                                 shuffle=True, normalize_rewards=True):
    """
    创建轨迹数据加载器
    Args:
        trajectories: 轨迹列表，None表示从默认文件加载
        batch_size: 批次大小
        max_length: 最大序列长度，默认200
        shuffle: 是否打乱数据
        normalize_rewards: 是否标准化reward
    Returns:
        dataloader: 数据加载器
    """
    if trajectories is None:
        trajectories, _ = load_trajectories()

    return TrajectoryDataLoader(
        trajectories=trajectories,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=shuffle,
        normalize_rewards=normalize_rewards
    )


def split_trajectories(trajectories, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    分割轨迹数据为训练、验证、测试集
    Args:
        trajectories: 轨迹列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    Returns:
        train_trajectories, val_trajectories, test_trajectories
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # 随机打乱
    np.random.shuffle(trajectories)

    # 计算分割点
    n = len(trajectories)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # 分割数据
    train_trajectories = trajectories[:train_end]
    val_trajectories = trajectories[train_end:val_end]
    test_trajectories = trajectories[val_end:]

    print(f"Split trajectories:")
    print(f"  Train: {len(train_trajectories)}")
    print(f"  Val: {len(val_trajectories)}")
    print(f"  Test: {len(test_trajectories)}")

    return train_trajectories, val_trajectories, test_trajectories


def main():
    """测试函数"""
    # 加载轨迹数据
    trajectories, encoders = load_trajectories()

    # 创建数据加载器
    dataloader = create_trajectory_dataloader(
        trajectories=trajectories,
        batch_size=4,
        max_length=200,
        shuffle=True,
        normalize_rewards=True
    )

    # 测试数据加载
    print("\nTesting data loading...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  obs_seq shape: {batch['obs_seq'].shape}")
        print(f"  action_seq shape: {batch['action_seq'].shape}")
        print(f"  reward_seq shape: {batch['reward_seq'].shape}")
        print(f"  trajectory_lengths: {batch['trajectory_lengths']}")
        print(f"  user_ids: {batch['user_ids']}")

        # 检查数据类型
        print(f"  obs_seq dtype: {batch['obs_seq'].dtype}")
        print(f"  action_seq dtype: {batch['action_seq'].dtype}")
        print(f"  reward_seq dtype: {batch['reward_seq'].dtype}")

        if i >= 2:  # 只测试前3个批次
            break

    # 打印统计信息
    stats = dataloader.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 