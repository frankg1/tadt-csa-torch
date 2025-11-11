import pandas as pd
import torch
import pickle
import numpy as np
import os
from ctr_model import DeepFM, CTRDataProcessor
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 导入配置
from config import (
    get_data_file_path, get_output_file_path,
    MODEL_CONFIG, DATA_CONFIG, OUTPUT_CONFIG
)


class CTRPredictor:
    def __init__(self, model_path=None, processor_path=None):
        """
        CTR预测器
        Args:
            model_path: 模型权重文件路径，如果为None则使用默认配置
            processor_path: 数据处理器文件路径，如果为None则使用默认配置
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用配置文件中的默认路径
        if model_path is None:
            model_path = get_output_file_path('model_file')
        if processor_path is None:
            processor_path = get_output_file_path('processor_file')
        
        # 加载数据处理器
        with open(processor_path, 'rb') as f:
            self.processor = pickle.load(f)
        
        # 创建模型
        self.model = DeepFM(
            feature_sizes=self.processor.feature_sizes,
            embedding_size=MODEL_CONFIG['embedding_size'],
            hidden_dims=MODEL_CONFIG['hidden_dims'],
            dropout=MODEL_CONFIG['dropout']
        ).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Feature sizes: {len(self.processor.feature_sizes)}")
    
    def load_test_data(self, data_path=None):
        """加载测试数据，如果没有指定则使用另一个时间段的数据"""
        if data_path is None:
            # 使用另一个时间段的数据作为测试集
            test_interactions = pd.read_csv(get_data_file_path('test_interactions'))
            print(f"Loaded {len(test_interactions)} test interactions")
        else:
            test_interactions = pd.read_csv(data_path)
        
        # 加载其他特征文件
        user_features = pd.read_csv(get_data_file_path('user_features'))
        video_basic = pd.read_csv(get_data_file_path('video_basic'))
        
        # 加载视频统计特征
        if DATA_CONFIG['enable_sampling']:
            video_stats = pd.read_csv(get_data_file_path('video_stats'), nrows=DATA_CONFIG['video_stats_sample_size'])
        else:
            video_stats = pd.read_csv(get_data_file_path('video_stats'))
        
        return test_interactions, user_features, video_basic, video_stats
    
    def predict(self, test_data_path=None, batch_size=1024):
        """
        进行CTR预测
        Args:
            test_data_path: 测试数据路径，如果为None则使用默认测试集
            batch_size: 批次大小
        Returns:
            predictions: 预测结果
            labels: 真实标签（如果存在）
        """
        # 加载测试数据
        test_interactions, user_features, video_basic, video_stats = self.load_test_data(test_data_path)
        
        # 创建标签
        test_interactions = self.processor.create_ctr_labels(test_interactions)
        
        # 合并特征
        test_data = self.processor.merge_features(test_interactions, user_features, video_basic, video_stats)
        
        # 选择特征
        categorical_features, numerical_features = self.processor.select_features(test_data)
        
        # 处理特征（使用训练时的编码器）
        processed_data = self.processor.process_features(
            test_data, categorical_features, numerical_features, is_training=False
        )
        
        # 准备模型输入
        model_input = self.processor.prepare_model_input(processed_data, categorical_features)
        labels = processed_data['label'].values
        
        # 移动到设备
        for key in model_input:
            model_input[key] = model_input[key].to(self.device)
        
        # 预测
        predictions = []
        with torch.no_grad():
            for i in range(0, len(labels), batch_size):
                end_idx = min(i + batch_size, len(labels))
                batch_input = {}
                for key, values in model_input.items():
                    batch_input[key] = values[i:end_idx]
                
                batch_preds = self.model(batch_input)
                predictions.extend(batch_preds.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # 计算AUC
        if len(np.unique(labels)) > 1:  # 确保有正负样本
            auc = roc_auc_score(labels, predictions)
            print(f"Test AUC: {auc:.4f}")
        else:
            print("Unable to calculate AUC (only one class in labels)")
        
        print(f"Prediction stats:")
        print(f"  Mean prediction: {predictions.mean():.4f}")
        print(f"  Std prediction: {predictions.std():.4f}")
        print(f"  Min prediction: {predictions.min():.4f}")
        print(f"  Max prediction: {predictions.max():.4f}")
        
        return predictions, labels
    
    def get_item_embeddings(self, save_path=None):
        """
        获取所有item的embeddings
        Args:
            save_path: 保存路径，如果为None则使用默认配置
        Returns:
            embeddings: item embeddings数组
        """
        if save_path is None:
            save_path = get_output_file_path('extracted_embeddings_file')
        
        with torch.no_grad():
            if 'video_id' in self.processor.feature_sizes:
                video_ids = torch.arange(self.processor.feature_sizes['video_id']).to(self.device)
                embeddings = self.model.get_item_embeddings(video_ids)
                embeddings_np = embeddings.cpu().numpy()
                
                # 保存embeddings
                np.save(save_path, embeddings_np)
                print(f"Item embeddings saved to {save_path}")
                print(f"Shape: {embeddings_np.shape}")
                
                return embeddings_np
            else:
                raise ValueError("video_id not found in feature sizes")
    
    def predict_for_user_item_pairs(self, user_ids, video_ids):
        """
        为指定的用户-物品对进行预测
        Args:
            user_ids: 用户ID列表
            video_ids: 视频ID列表
        Returns:
            predictions: 预测分数
        """
        if len(user_ids) != len(video_ids):
            raise ValueError("user_ids and video_ids must have the same length")
        
        # 创建预测数据
        predict_data = pd.DataFrame({
            'user_id': user_ids,
            'video_id': video_ids
        })
        
        # 对于缺失的特征，填充默认值
        for feature in self.processor.feature_sizes.keys():
            if feature not in predict_data.columns:
                if feature in ['user_id', 'video_id']:
                    continue
                predict_data[feature] = 'unknown'  # 分类特征默认值
        
        # 处理特征
        model_input = {}
        for feature in self.processor.feature_sizes.keys():
            if feature in predict_data.columns:
                if feature in self.processor.label_encoders:
                    le = self.processor.label_encoders[feature]
                    # 处理未见过的值
                    values = predict_data[feature].astype(str)
                    mask = values.isin(le.classes_)
                    values[~mask] = 'unknown'
                    encoded_values = le.transform(values)
                    model_input[feature] = torch.LongTensor(encoded_values).to(self.device)
        
        # 预测
        with torch.no_grad():
            predictions = self.model(model_input)
            return predictions.cpu().numpy()


def main():
    """主函数，演示如何使用预测器"""
    try:
        # 创建预测器
        predictor = CTRPredictor()
        
        # 进行预测
        print("Starting prediction...")
        predictions, labels = predictor.predict()
        
        # 获取item embeddings
        print("\nExtracting item embeddings...")
        item_embeddings = predictor.get_item_embeddings()
        
        # 示例：为特定用户-物品对预测
        print("\nExample: Predicting for specific user-item pairs...")
        sample_user_ids = [0, 1, 2, 3, 4]
        sample_video_ids = [100, 200, 300, 400, 500]
        sample_predictions = predictor.predict_for_user_item_pairs(
            sample_user_ids, sample_video_ids
        )
        
        for i, (uid, vid, pred) in enumerate(zip(sample_user_ids, sample_video_ids, sample_predictions)):
            print(f"User {uid}, Video {vid}: {pred:.4f}")
        
        print("\nPrediction completed!")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Make sure you have trained the model first by running: python ctr_model.py")


if __name__ == "__main__":
    main() 