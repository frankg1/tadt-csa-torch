import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入配置
from config import (
    get_data_file_path, get_output_file_path, validate_data_path,
    MODEL_CONFIG, DATA_CONFIG, OUTPUT_CONFIG
)


class MLPModel(nn.Module):
    def __init__(self, feature_sizes, embedding_size=8, hidden_dims=[128, 64], dropout=0.2):
        """
        简单的MLP模型
        Args:
            feature_sizes: dict, 每个特征域的取值数量
            embedding_size: int, embedding维度
            hidden_dims: list, 深度网络的隐藏层维度
            dropout: float, dropout比例
        """
        super(MLPModel, self).__init__()
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.num_fields = len(feature_sizes)
        
        # Embedding层 - 为video_id特殊处理，映射到[1, 10000]
        self.embeddings = nn.ModuleDict()
        for field, size in feature_sizes.items():
            if field == 'video_id':
                # video_id需要映射到[1, 10000]，所以embedding size要+1，0作为padding
                self.embeddings[field] = nn.Embedding(size + 1, embedding_size, padding_idx=0)
            else:
                self.embeddings[field] = nn.Embedding(size, embedding_size)
        
        # 深度网络部分
        deep_input_dim = self.num_fields * embedding_size
        deep_layers = []
        input_dim = deep_input_dim
        
        for hidden_dim in hidden_dims:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        deep_layers.append(nn.Linear(input_dim, 1))
        self.deep_network = nn.Sequential(*deep_layers)
        
        # 输出层
        self.output = nn.Sigmoid()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: dict, 包含各个特征域的输入
        """
        embeddings_list = []
        
        for field, indices in x.items():
            # 收集embedding
            embeddings_list.append(self.embeddings[field](indices))
        
        # 拼接所有embedding
        embeddings = torch.stack(embeddings_list, dim=1)  # [batch_size, num_fields, embedding_size]
        deep_input = embeddings.view(embeddings.size(0), -1)  # flatten
        
        # 通过深度网络
        deep_output = self.deep_network(deep_input)
        
        return self.output(deep_output).squeeze()
    
    def get_item_embeddings(self, item_indices):
        """获取item embeddings"""
        if 'video_id' in self.embeddings:
            return self.embeddings['video_id'](item_indices)
        else:
            raise ValueError("video_id not found in embeddings")


class CTRDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_sizes = {}
        
    def load_data(self):
        """加载数据"""
        print("Loading data...")
        
        # 验证数据路径
        validate_data_path()
        
        # 加载交互数据
        interactions = pd.read_csv(get_data_file_path('train_interactions'))
        print(f"Loaded {len(interactions)} interactions")
        
        # 加载用户特征
        user_features = pd.read_csv(get_data_file_path('user_features'))
        print(f"Loaded {len(user_features)} user features")
        
        # 加载视频基础特征
        video_basic = pd.read_csv(get_data_file_path('video_basic'))
        print(f"Loaded {len(video_basic)} video basic features")
        
        # 加载视频统计特征
        if DATA_CONFIG['enable_sampling']:
            print("Sampling video statistics features...")
            video_stats = pd.read_csv(get_data_file_path('video_stats'), nrows=DATA_CONFIG['video_stats_sample_size'])
            print(f"Sampled {len(video_stats)} video statistics")
        else:
            print("Loading full video statistics features...")
            video_stats = pd.read_csv(get_data_file_path('video_stats'))
            print(f"Loaded {len(video_stats)} video statistics")
        
        return interactions, user_features, video_basic, video_stats
    
    def create_ctr_labels(self, interactions):
        """创建CTR标签：点击、点赞、关注、评论、转发中任一为正样本"""
        interactions['label'] = (
            (interactions['is_click'] == 1) |
            (interactions['is_like'] == 1) |
            (interactions['is_follow'] == 1) |
            (interactions['is_comment'] == 1) |
            (interactions['is_forward'] == 1)
        ).astype(int)
        return interactions
    
    def merge_features(self, interactions, user_features, video_basic, video_stats):
        """合并特征"""
        print("Merging features...")
        
        # 合并用户特征
        data = interactions.merge(user_features, on='user_id', how='left')
        
        # 合并视频基础特征
        data = data.merge(video_basic, on='video_id', how='left')
        
        # 合并视频统计特征
        data = data.merge(video_stats, on='video_id', how='left')
        
        return data
    
    def select_features(self, data):
        """选择特征"""
        # 分类特征
        categorical_features = [
            'user_id', 'video_id', 'user_active_degree',
            'is_live_streamer', 'is_video_author', 'video_type',
            'upload_type', 'music_type', 'tag',
            'follow_user_num_range', 'fans_user_num_range',
            'friend_user_num_range', 'register_days_range'
        ]
        
        # 数值特征
        numerical_features = [
            'time_ms', 'play_time_ms', 'duration_ms',
            'follow_user_num', 'fans_user_num', 'friend_user_num',
            'register_days', 'video_duration',
            'server_width', 'server_height'
        ]
        
        # 添加onehot特征
        onehot_features = [f'onehot_feat{i}' for i in range(18)]
        numerical_features.extend(onehot_features)
        
        # 如果存在视频统计特征，添加一些关键指标
        stats_features = [
            'show_cnt', 'play_cnt', 'like_cnt', 'comment_cnt',
            'follow_cnt', 'share_cnt'
        ]
        for feat in stats_features:
            if feat in data.columns:
                numerical_features.append(feat)
        
        # 过滤存在的特征
        categorical_features = [f for f in categorical_features if f in data.columns]
        numerical_features = [f for f in numerical_features if f in data.columns]
        
        return categorical_features, numerical_features
    
    def process_features(self, data, categorical_features, numerical_features, is_training=True):
        """处理特征"""
        processed_data = data.copy()
        
        # 处理分类特征
        for feature in categorical_features:
            if feature in processed_data.columns:
                # 填充缺失值
                processed_data[feature] = processed_data[feature].fillna('unknown')
                
                if is_training:
                    # 训练时创建编码器
                    le = LabelEncoder()
                    processed_data[feature] = le.fit_transform(processed_data[feature].astype(str))
                    
                    # 特殊处理video_id：映射到[1, 10000]
                    if feature == 'video_id':
                        processed_data[feature] = processed_data[feature] + 1  # 从1开始
                    
                    self.label_encoders[feature] = le
                    self.feature_sizes[feature] = len(le.classes_)
                else:
                    # 测试时使用已有编码器
                    le = self.label_encoders[feature]
                    # 处理未见过的类别
                    mask = processed_data[feature].astype(str).isin(le.classes_)
                    processed_data.loc[~mask, feature] = 'unknown'
                    processed_data[feature] = le.transform(processed_data[feature].astype(str))
                    
                    # 特殊处理video_id：映射到[1, 10000]
                    if feature == 'video_id':
                        processed_data[feature] = processed_data[feature] + 1  # 从1开始
        
        # 处理数值特征
        for feature in numerical_features:
            if feature in processed_data.columns:
                # 填充缺失值
                processed_data[feature] = processed_data[feature].fillna(0)
                
                if is_training:
                    # 训练时创建标准化器
                    scaler = StandardScaler()
                    processed_data[feature] = scaler.fit_transform(processed_data[[feature]])
                    self.scalers[feature] = scaler
                else:
                    # 测试时使用已有标准化器
                    scaler = self.scalers[feature]
                    processed_data[feature] = scaler.transform(processed_data[[feature]])
        
        return processed_data
    
    def prepare_model_input(self, data, categorical_features):
        """准备模型输入"""
        model_input = {}
        for feature in categorical_features:
            if feature in data.columns:
                model_input[feature] = torch.LongTensor(data[feature].values)
        return model_input


def train_mlp_model():
    """训练MLP模型"""
    # 初始化数据处理器
    processor = CTRDataProcessor()
    
    # 加载数据
    interactions, user_features, video_basic, video_stats = processor.load_data()
    
    # 创建CTR标签
    interactions = processor.create_ctr_labels(interactions)
    
    # 合并特征
    data = processor.merge_features(interactions, user_features, video_basic, video_stats)
    
    # 选择特征
    categorical_features, numerical_features = processor.select_features(data)
    print(f"Selected {len(categorical_features)} categorical features and {len(numerical_features)} numerical features")
    
    # 处理特征
    processed_data = processor.process_features(data, categorical_features, numerical_features, is_training=True)
    
    # 添加调试信息：查看实际的video_id数量
    if 'video_id' in processor.feature_sizes:
        print(f"Actual unique video_ids in training data: {processor.feature_sizes['video_id']}")
        print(f"Video ID mapping range: 1 - {processor.feature_sizes['video_id']}")
        print(f"Original video features file size: {len(video_basic)}")
        print(f"Original video stats file size: {len(video_stats)}")
    
    # 准备训练数据
    model_input = processor.prepare_model_input(processed_data, categorical_features)
    labels = torch.FloatTensor(processed_data['label'].values)
    
    print(f"Training data shape: {len(processed_data)}")
    print(f"Positive samples: {labels.sum().item()}")
    print(f"CTR: {labels.mean().item():.4f}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Embedding dimension: {MODEL_CONFIG['embedding_size']}")
    
    model = MLPModel(
        feature_sizes=processor.feature_sizes,
        embedding_size=MODEL_CONFIG['embedding_size'],
        hidden_dims=MODEL_CONFIG['hidden_dims'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    # 移动数据到设备
    for key in model_input:
        model_input[key] = model_input[key].to(device)
    labels = labels.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), 
                          lr=MODEL_CONFIG['learning_rate'], 
                          weight_decay=MODEL_CONFIG['weight_decay'])
    criterion = nn.BCELoss()
    
    # 训练参数
    batch_size = MODEL_CONFIG['batch_size']
    epochs = MODEL_CONFIG['epochs']
    
    print(f"Starting training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # 创建批次索引
        indices = torch.randperm(len(labels))
        num_batches = len(indices) // batch_size
        
        progress_bar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{epochs}')
        
        for i in progress_bar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # 准备批次数据
            batch_input = {}
            for key, values in model_input.items():
                batch_input[key] = values[batch_indices]
            batch_labels = labels[batch_indices]
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 收集预测结果用于AUC计算
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(batch_labels.detach().cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # 计算epoch的AUC
        epoch_auc = roc_auc_score(all_labels, all_preds)
        avg_loss = total_loss / num_batches
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, AUC: {epoch_auc:.4f}')
    
    print("Training completed!")
    
    # 保存模型
    os.makedirs(get_output_file_path('model_dir'), exist_ok=True)
    torch.save(model.state_dict(), get_output_file_path('model_file'))
    print(f"Model saved to {get_output_file_path('model_file')}")
    
    # 保存数据处理器
    with open(get_output_file_path('processor_file'), 'wb') as f:
        pickle.dump(processor, f)
    print(f"Data processor saved to {get_output_file_path('processor_file')}")
    
    # 保存item embeddings
    model.eval()
    with torch.no_grad():
        if 'video_id' in processor.feature_sizes:
            # 生成[1, 10000]的video_id索引
            video_ids = torch.arange(1, processor.feature_sizes['video_id'] + 1).to(device)
            item_embeddings = model.get_item_embeddings(video_ids)
            
            # 保存映射的item IDs (1到max_id)
            mapped_item_ids = list(range(1, processor.feature_sizes['video_id'] + 1))
            torch.save(torch.tensor(mapped_item_ids), get_output_file_path('item_ids_file'))
            
            # 保存原始到映射的映射关系
            original_to_mapped = {}
            le = processor.label_encoders['video_id']
            for i, original_id in enumerate(le.classes_):
                mapped_id = i + 1  # 映射到[1, max_id]
                original_to_mapped[int(original_id)] = mapped_id
            
            with open(get_output_file_path('id_mapping_file'), 'wb') as f:
                pickle.dump(original_to_mapped, f)
            
            # 保存embeddings
            torch.save(item_embeddings.cpu(), get_output_file_path('item_embeddings_file'))
            
            print(f"Item embeddings saved to {get_output_file_path('item_embeddings_file')}")
            print(f"Item IDs saved to {get_output_file_path('item_ids_file')}")
            print(f"ID mapping saved to {get_output_file_path('id_mapping_file')}")
            print(f"Item embeddings shape: {item_embeddings.shape}")
            print(f"Item ID range: 1 - {processor.feature_sizes['video_id']}")
            print(f"Total mapped items: {len(mapped_item_ids)}")
            print(f"Total original items: {len(original_to_mapped)}")
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        final_preds = []
        final_labels = []
        
        for i in range(0, len(labels), batch_size):
            end_idx = min(i + batch_size, len(labels))
            batch_input = {}
            for key, values in model_input.items():
                batch_input[key] = values[i:end_idx]
            batch_labels = labels[i:end_idx]
            
            outputs = model(batch_input)
            final_preds.extend(outputs.cpu().numpy())
            final_labels.extend(batch_labels.cpu().numpy())
        
        final_auc = roc_auc_score(final_labels, final_preds)
        print(f"Final AUC: {final_auc:.4f}")
    
    return model, processor


if __name__ == "__main__":
    # 训练模型
    model, processor = train_mlp_model()
    print("All done!") 