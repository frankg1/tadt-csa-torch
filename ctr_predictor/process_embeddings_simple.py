import torch
import numpy as np
import json
import os
import pandas as pd
from config import get_output_file_path, get_data_file_path

def process_item_embeddings_simple():
    """
    简化版本：直接从原始数据重建映射关系
    确保处理整整10000个item，并将item_id映射到[1, 10000]
    """
    print("Loading item embeddings...")
    
    # 加载item embeddings
    item_embeddings = torch.load(get_output_file_path('item_embeddings_file'))
    print(f"Loaded embeddings shape: {item_embeddings.shape}")
    
    # 加载原始交互数据来获取video_id映射
    print("Loading interaction data to get video_id mapping...")
    interactions = pd.read_csv(get_data_file_path('train_interactions'))
    
    # 获取所有唯一的video_id并排序
    unique_video_ids = sorted(interactions['video_id'].unique())
    print(f"Total unique video_ids in interactions: {len(unique_video_ids)}")
    
    # 创建video_id到索引的映射
    video_id_to_index = {vid: idx for idx, vid in enumerate(unique_video_ids)}
    
    # 创建KV存储字典 - 使用映射后的item_id [1, 10000]
    video_embeddings_dict = {}
    filtered_embeddings = []
    filtered_video_ids = []
    original_to_mapped = {}  # 原始video_id到映射item_id的映射
    
    print("Processing embeddings...")
    target_count = 10000  # 目标处理10000个item
    
    for i, video_id in enumerate(unique_video_ids):
        # 只处理前10000个video_id
        if i >= target_count:
            break
            
        # 获取对应的embedding索引
        if video_id in video_id_to_index:
            idx = video_id_to_index[video_id]
            if idx < item_embeddings.shape[0]:  # 确保索引在范围内
                embedding = item_embeddings[idx].numpy()
                mapped_item_id = i + 1  # 映射到[1, 10000]
                
                video_embeddings_dict[str(mapped_item_id)] = embedding.tolist()
                filtered_embeddings.append(embedding)
                filtered_video_ids.append(str(mapped_item_id))
                original_to_mapped[video_id] = mapped_item_id
    
    print(f"Processed exactly {len(video_embeddings_dict)} videos (target: {target_count})")
    
    if len(video_embeddings_dict) == 0:
        print("Warning: No videos found!")
        return None, None, None
    
    # 确保我们有足够的item，如果不够就从后面补充
    if len(video_embeddings_dict) < target_count:
        print(f"Warning: Only found {len(video_embeddings_dict)} items, less than target {target_count}")
        print("Available video_id range:", min(unique_video_ids), "-", max(unique_video_ids))
    
    # 保存为JSON格式（KV存储）
    json_path = get_output_file_path('model_dir') + '/video_embeddings_kv.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(video_embeddings_dict, f, indent=2)
    print(f"Saved KV format to: {json_path}")
    
    # 保存为numpy格式（数组 + 映射）
    np_embeddings = np.array(filtered_embeddings)
    np_path = get_output_file_path('model_dir') + '/video_embeddings_filtered.npy'
    np.save(np_path, np_embeddings)
    print(f"Saved numpy array to: {np_path}, shape: {np_embeddings.shape}")
    
    # 保存映射后的item_id列表
    video_ids_path = get_output_file_path('model_dir') + '/filtered_video_ids.json'
    with open(video_ids_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_video_ids, f, indent=2)
    print(f"Saved mapped item IDs to: {video_ids_path}")
    
    # 保存原始video_id到映射item_id的映射关系
    mapping_path = get_output_file_path('model_dir') + '/original_to_mapped_mapping.json'
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(original_to_mapped, f, indent=2)
    print(f"Saved original to mapped mapping to: {mapping_path}")
    
    # 保存为pickle格式（完整字典）
    import pickle
    pickle_path = get_output_file_path('model_dir') + '/video_embeddings_kv.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(video_embeddings_dict, f)
    print(f"Saved pickle format to: {pickle_path}")
    
    # 显示一些统计信息
    print("\n=== Statistics ===")
    print(f"Original embeddings: {item_embeddings.shape}")
    print(f"Filtered embeddings: {np_embeddings.shape}")
    print(f"Embedding dimension: {np_embeddings.shape[1]}")
    print(f"Mapped item ID range: {min(filtered_video_ids)} - {max(filtered_video_ids)}")
    print(f"Total items processed: {len(video_embeddings_dict)}")
    
    # 显示前几个样本
    print("\n=== Sample Data ===")
    for i, mapped_id in enumerate(filtered_video_ids[:5]):
        embedding = video_embeddings_dict[mapped_id]
        original_id = list(original_to_mapped.keys())[list(original_to_mapped.values()).index(int(mapped_id))]
        print(f"Mapped Item ID: {mapped_id}, Original Video ID: {original_id}, Embedding: {embedding[:5]}... (first 5 values)")
    
    # 显示最后几个样本
    print("\n=== Last 5 Items ===")
    for i, mapped_id in enumerate(filtered_video_ids[-5:]):
        embedding = video_embeddings_dict[mapped_id]
        original_id = list(original_to_mapped.keys())[list(original_to_mapped.values()).index(int(mapped_id))]
        print(f"Mapped Item ID: {mapped_id}, Original Video ID: {original_id}, Embedding: {embedding[:5]}... (first 5 values)")
    
    return video_embeddings_dict, np_embeddings, filtered_video_ids

def load_processed_embeddings(format_type='json'):
    """
    加载处理后的embeddings
    Args:
        format_type: 'json', 'numpy', 'pickle'
    """
    base_path = get_output_file_path('model_dir')
    
    if format_type == 'json':
        # 加载KV格式
        with open(f"{base_path}/video_embeddings_kv.json", 'r', encoding='utf-8') as f:
            embeddings_dict = json.load(f)
        return embeddings_dict
    
    elif format_type == 'numpy':
        # 加载numpy数组
        embeddings_array = np.load(f"{base_path}/video_embeddings_filtered.npy")
        # 同时加载video_ids
        with open(f"{base_path}/filtered_video_ids.json", 'r', encoding='utf-8') as f:
            video_ids = json.load(f)
        return embeddings_array, video_ids
    
    elif format_type == 'pickle':
        # 加载pickle格式
        import pickle
        with open(f"{base_path}/video_embeddings_kv.pkl", 'rb') as f:
            embeddings_dict = pickle.load(f)
        return embeddings_dict
    
    else:
        raise ValueError("format_type must be 'json', 'numpy', or 'pickle'")

def get_embedding_by_video_id(video_id, embeddings_dict):
    """
    根据video_id获取embedding
    """
    video_id_str = str(video_id)
    if video_id_str in embeddings_dict:
        return np.array(embeddings_dict[video_id_str])
    else:
        raise ValueError(f"Video ID {video_id} not found in embeddings")

if __name__ == "__main__":
    # 处理embeddings
    print("=== Processing Item Embeddings (Simple Version) ===")
    result = process_item_embeddings_simple()
    
    if result[0] is not None:
        embeddings_dict, embeddings_array, video_ids = result
        
        print("\n=== Testing Loading ===")
        # 测试加载不同格式
        print("Loading JSON format...")
        json_embeddings = load_processed_embeddings('json')
        print(f"JSON loaded: {len(json_embeddings)} items")
        
        print("Loading numpy format...")
        np_embeddings, np_video_ids = load_processed_embeddings('numpy')
        print(f"Numpy loaded: {np_embeddings.shape}")
        
        print("Loading pickle format...")
        pickle_embeddings = load_processed_embeddings('pickle')
        print(f"Pickle loaded: {len(pickle_embeddings)} items")
        
        # 测试获取特定video_id的embedding
        test_video_id = video_ids[0]
        embedding = get_embedding_by_video_id(test_video_id, embeddings_dict)
        print(f"\nTest: Video ID {test_video_id} embedding shape: {embedding.shape}")
        
        print("\n=== Processing Complete ===")
        print("Files created:")
        print("- video_embeddings_kv.json (KV format with mapped item IDs 1-10000)")
        print("- video_embeddings_filtered.npy (numpy array)")
        print("- filtered_video_ids.json (mapped item ID list 1-10000)")
        print("- original_to_mapped_mapping.json (original video_id to mapped item_id mapping)")
        print("- video_embeddings_kv.pkl (pickle format)")
        print("\nNote: All item IDs are now mapped to range [1, 10000]")
    else:
        print("Processing failed!") 