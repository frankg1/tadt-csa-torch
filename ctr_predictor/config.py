"""
配置文件 - 集中管理所有参数
"""
import os

# 数据路径配置
DATA_PATH = "/home/gaoxiang12/datasets/kuairand/KuaiRand-Pure/data"

# 数据文件名配置
DATA_FILES = {
    'train_interactions': 'log_standard_4_22_to_5_08_pure.csv',
    'test_interactions': 'log_standard_4_08_to_4_21_pure.csv',
    'user_features': 'user_features_pure.csv',
    'video_basic': 'video_features_basic_pure.csv',
    'video_stats': 'video_features_statistic_pure.csv'
}

# 模型配置
MODEL_CONFIG = {
    'embedding_size': 64,
    'hidden_dims': [256, 128, 64],
    'dropout': 0.3,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 4096,
    'epochs': 10
}

# 数据处理配置
DATA_CONFIG = {
    # 数据抽样设置
    'enable_sampling': False,  # 是否启用抽样
    # - False: 加载完整数据集（推荐服务器使用，需要更多内存）
    # - True: 抽样加载（适合本地测试，节省内存）
    'video_stats_sample_size': 50000,  # 抽样数量（仅在enable_sampling=True时生效）
    
    # 其他配置
    'test_size': 0.2,  # 如果需要划分测试集的比例
    'random_seed': 42
}

# 输出路径配置
OUTPUT_CONFIG = {
    'model_dir': 'models',
    'model_file': 'deepfm_model.pth',
    'processor_file': 'data_processor.pkl',
    'item_embeddings_file': 'item_embeddings.pth',
    'item_ids_file': 'item_ids.pth',
    'id_mapping_file': 'id_mapping.pkl',
    'extracted_embeddings_file': 'extracted_item_embeddings.npy',
    'trajectories_file': 'trajectories.pkl',
    'filtered_trajectories_file': 'filtered_trajectories.pkl'
}

# 获取完整路径的辅助函数
def get_data_file_path(file_key):
    """获取数据文件的完整路径"""
    if file_key not in DATA_FILES:
        raise ValueError(f"Unknown data file key: {file_key}")
    return os.path.join(DATA_PATH, DATA_FILES[file_key])

def get_output_file_path(file_key):
    """获取输出文件的完整路径"""
    if file_key not in OUTPUT_CONFIG:
        raise ValueError(f"Unknown output file key: {file_key}")
    
    if file_key == 'model_dir':
        return OUTPUT_CONFIG[file_key]
    else:
        return os.path.join(OUTPUT_CONFIG['model_dir'], OUTPUT_CONFIG[file_key])

# 验证数据路径是否存在
def validate_data_path():
    """验证数据路径和文件是否存在"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data path does not exist: {DATA_PATH}")
    
    missing_files = []
    for file_key, filename in DATA_FILES.items():
        file_path = os.path.join(DATA_PATH, filename)
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Warning: The following data files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("Please check your data path configuration.")
    else:
        print("✓ All data files found successfully!")

if __name__ == "__main__":
    # 验证配置
    print("Data Path Configuration:")
    print(f"  DATA_PATH: {DATA_PATH}")
    print("\nData Files:")
    for key, filename in DATA_FILES.items():
        print(f"  {key}: {get_data_file_path(key)}")
    
    print("\nValidating data files...")
    validate_data_path() 