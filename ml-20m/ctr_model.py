import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import os
import ast
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import json

# 配置文件
MODEL_CONFIG = {
    'embedding_size': 64,
    'hidden_dims': [128, 64],
    'dropout': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 1024,
    'epochs': 10
}

DATA_CONFIG = {
    'data_path': 'final_ratings_remapped.csv',
    'test_size': 0.2,
    'random_state': 42
}

OUTPUT_CONFIG = {
    'model_dir': 'model',
    'model_file': 'ctr_model.pth',
    'processor_file': 'data_processor.pkl',
    'movie_embedding_mapping': 'movieid_embedding_kv.pkl'  # 修改为直接保存映射
}

# 确保输出目录存在
os.makedirs(OUTPUT_CONFIG['model_dir'], exist_ok=True)

class MLPModel(nn.Module):
    def __init__(self, feature_sizes, embedding_size=16, hidden_dims=[128, 64], dropout=0.2):
        super(MLPModel, self).__init__()
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.num_fields = len(feature_sizes)
        
        # Embedding层 - 每个特征一个嵌入层
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(size, embedding_size)
            for field, size in feature_sizes.items()
        })
        
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
        
        # 初始化权重
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
        embeddings_list = []
        
        for field, indices in x.items():
            # 收集每个特征的embedding
            embeddings_list.append(self.embeddings[field](indices))
        
        # 拼接所有embedding
        embeddings = torch.stack(embeddings_list, dim=1)
        deep_input = embeddings.view(embeddings.size(0), -1)
        
        # 通过深度网络
        deep_output = self.deep_network(deep_input)
        
        return self.output(deep_output).squeeze()
    
    def get_item_embeddings(self, item_indices):
        """获取movieId的embedding"""
        return self.embeddings['movieId'](item_indices)


class CTRDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.feature_sizes = {}
        self.unique_movie_ids = None
    
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print("Loading data...")
        data = pd.read_csv(DATA_CONFIG['data_path'])
        print(f"Loaded {len(data)} records")
        
        # 保存原始movieId列表
        self.unique_movie_ids = data['movieId'].unique().tolist()

        return data
    
    def process_features(self, data):
        """处理特征"""
        processed_data = data.copy()
        
        # 特征列表
        features = ['userId', 'movieId'] + [f'tag{i}' for i in range(1,6)]
        
        # 处理每个特征
        for feature in features:
            # 创建标签编码器
            le = LabelEncoder()
            processed_data[feature] = le.fit_transform(processed_data[feature].astype(str))
            
            # 保存编码器和特征大小
            self.label_encoders[feature] = le
            self.feature_sizes[feature] = len(le.classes_)
        
        return processed_data
    
    def prepare_model_input(self, data):
        """准备模型输入"""
        model_input = {}
        features = ['userId', 'movieId'] + [f'tag{i}' for i in range(1,6)]
        
        for feature in features:
            model_input[feature] = torch.LongTensor(data[feature].values)
        
        return model_input


def train_ctr_model():
    """训练CTR模型"""
    # 初始化数据处理器
    processor = CTRDataProcessor()
    
    # 加载并预处理数据
    data = processor.load_and_preprocess_data()
    
    # 处理特征
    processed_data = processor.process_features(data)
    
    # 划分训练集和测试集
    train_data, test_data = train_test_split(
        processed_data, 
        test_size=DATA_CONFIG['test_size'], 
        random_state=DATA_CONFIG['random_state']
    )
    
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    print(f"Unique movie IDs: {len(processor.unique_movie_ids)}")
    
    # 准备模型输入
    train_input = processor.prepare_model_input(train_data)
    test_input = processor.prepare_model_input(test_data)
    
    # 标签
    train_labels = torch.FloatTensor(train_data['rating'].values)
    test_labels = torch.FloatTensor(test_data['rating'].values)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MLPModel(
        feature_sizes=processor.feature_sizes,
        embedding_size=MODEL_CONFIG['embedding_size'],
        hidden_dims=MODEL_CONFIG['hidden_dims'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    # 移动数据到设备
    for key in train_input:
        train_input[key] = train_input[key].to(device)
        test_input[key] = test_input[key].to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), 
                          lr=MODEL_CONFIG['learning_rate'], 
                          weight_decay=MODEL_CONFIG['weight_decay'])
    criterion = nn.BCELoss()
    
    # 训练参数
    batch_size = MODEL_CONFIG['batch_size']
    epochs = MODEL_CONFIG['epochs']
    
    print(f"Starting training for {epochs} epochs...")
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # 创建批次索引
        num_samples = len(train_data)
        indices = torch.randperm(num_samples)
        num_batches = num_samples // batch_size
        
        progress_bar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{epochs}')
        
        for i in progress_bar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # 准备批次数据
            batch_input = {}
            for key, values in train_input.items():
                batch_input[key] = values[batch_indices]
            batch_labels = train_labels[batch_indices]
            
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
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Train AUC: {epoch_auc:.4f}')
        
        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            test_preds = []
            test_labels_list = []
            
            for i in range(0, len(test_data), batch_size):
                end_idx = min(i + batch_size, len(test_data))
                batch_input = {}
                for key, values in test_input.items():
                    batch_input[key] = values[i:end_idx]
                batch_labels = test_labels[i:end_idx]
                
                outputs = model(batch_input)
                test_preds.extend(outputs.cpu().numpy())
                test_labels_list.extend(batch_labels.cpu().numpy())
            
            test_auc = roc_auc_score(test_labels_list, test_preds)
            print(f'Epoch {epoch+1}/{epochs}, Test AUC: {test_auc:.4f}')
        
        model.train()
    
    print("Training completed!")
    
    # 保存模型
    model_path = os.path.join(OUTPUT_CONFIG['model_dir'], OUTPUT_CONFIG['model_file'])
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 保存数据处理器
    processor_path = os.path.join(OUTPUT_CONFIG['model_dir'], OUTPUT_CONFIG['processor_file'])
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    print(f"Data processor saved to {processor_path}")
    
    # 保存movieId到embedding的映射
    model.eval()
    with torch.no_grad():
        # 创建movieId到embedding的映射字典
        movie_embedding_dict = {}
        
        # 获取movieId编码器
        movie_encoder = processor.label_encoders['movieId']
        
        # 处理所有原始movieId
        for movie_id in processor.unique_movie_ids:
            # 将原始movieId编码
            try:
                encoded_id = movie_encoder.transform([str(movie_id)])[0]
            except ValueError:
                # 如果movieId不在编码器中，跳过
                continue
                
            # 获取embedding
            encoded_tensor = torch.LongTensor([encoded_id]).to(device)
            embedding = model.get_item_embeddings(encoded_tensor)[0].cpu().numpy().tolist()
            
            # 保存到字典
            movie_embedding_dict[movie_id] = embedding
        
        print(f"总共有{len(movie_embedding_dict.keys())}个item")
        # 保存映射字典
        mapping_path = os.path.join(OUTPUT_CONFIG['model_dir'], OUTPUT_CONFIG['movie_embedding_mapping'])
        with open(mapping_path, 'wb') as f:
            pickle.dump(movie_embedding_dict, f)
        json_path = os.path.join(OUTPUT_CONFIG['model_dir'], 'movie_embedding_kv.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(movie_embedding_dict, f, indent=2) 
        print(f"Movie ID to embedding mapping saved to {mapping_path}")
        print(f"Saved embeddings for {len(movie_embedding_dict)} movies")
    
    # 最终测试集评估
    model.eval()
    with torch.no_grad():
        test_preds = []
        test_labels_list = []
        
        for i in range(0, len(test_data), batch_size):
            end_idx = min(i + batch_size, len(test_data))
            batch_input = {}
            for key, values in test_input.items():
                batch_input[key] = values[i:end_idx]
            batch_labels = test_labels[i:end_idx]
            
            outputs = model(batch_input)
            test_preds.extend(outputs.cpu().numpy())
            test_labels_list.extend(batch_labels.cpu().numpy())
        
        test_auc = roc_auc_score(test_labels_list, test_preds)
        print(f"Final Test AUC: {test_auc:.4f}")
    
    return model, processor


if __name__ == "__main__":
    # 训练模型
    model, processor = train_ctr_model()
    print("CTR model training completed!")