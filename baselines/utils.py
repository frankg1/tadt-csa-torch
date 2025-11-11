import numpy as np
import torch


def compute_metrics(predictions, labels, ks=[1, 5, 10], num_neg_samples=100):
    """
    输入:
        predictions: numpy.array, 形状 (B, N) N 是 item 总数
        labels: numpy.array, 形状 (B, 1) item 索引从 1 开始
        ks: 需要计算的 K 值列表，如 [1, 5, 10]
        num_neg_samples: 随机选取的负样本数量（默认 100）
    返回:
        metrics: 包含 NDCG@K、Recall@K 和 MRR 的字典
    """
    predictions_np = predictions if isinstance(predictions, np.ndarray) else np.array(predictions)
    labels_np = labels if isinstance(labels, np.ndarray) else np.array(labels)

    batch_size = predictions_np.shape[0]
    num_items = predictions_np.shape[1]  # 总 item 数 N

    metrics = {f"Recall@{k}": 0.0 for k in ks}
    metrics.update({f"NDCG@{k}": 0.0 for k in ks})
    mrr = 0.0

    for i in range(batch_size):
        pred_i = predictions_np[i]  # 当前轨迹的预测分数 (N,)
        label_i = labels_np[i][0]  # 当前轨迹的真实 item 索引（0-based）

        # 随机选取负样本（不包括 label_i），如果负样本数量不足则使用所有可用样本
        all_indices = np.arange(num_items)
        available_neg_indices = np.setdiff1d(all_indices, [label_i])
        if len(available_neg_indices) <= num_neg_samples:
            neg_indices = available_neg_indices
        else:
            neg_indices = np.random.choice(
                available_neg_indices,
                size=num_neg_samples,
                replace=False
            )

        # 构造候选集（负样本 + 1 正样本）
        candidate_indices = np.concatenate([neg_indices, [label_i]])
        candidate_scores = pred_i[candidate_indices]

        # 按分数降序排序，获取排名 (0-indexed)
        ranked_indices = candidate_indices[np.argsort(candidate_scores)[::-1]]
        rank = np.where(ranked_indices == label_i)[0][0]

        # 计算指标
        mrr += 1.0 / (rank + 1)
        for k in ks:
            if rank < k:
                metrics[f"Recall@{k}"] += 1.0
                metrics[f"NDCG@{k}"] += 1.0 / np.log2(rank + 2)

    # 归一化（除以 batch_size）
    for k in ks:
        metrics[f"Recall@{k}"] /= batch_size
        metrics[f"NDCG@{k}"] /= batch_size
    metrics["MRR"] = mrr / batch_size

    return metrics

# def to_numpy(data):
#     """将任意类型的数据转换为 NumPy 数组"""
#     if isinstance(data, torch.Tensor):
#         return data.detach().cpu().numpy()  # 处理 PyTorch 张量（自动处理 GPU）
#     elif isinstance(data, np.ndarray):
#         return data  # 已经是 NumPy 数组，直接返回
#     elif isinstance(data, (list, tuple)):
#         if len(data) > 0 and hasattr(data[0], 'requires_grad'):  # 检查是否是 Tensor
#             return np.array([x.detach().numpy() if hasattr(x, 'requires_grad') else x for x in data])
#         else:
#             return np.array(data)  # 普通 list/tuple 直接转换
#     else:
#         raise TypeError(f"不支持的数据类型: {type(data)}")

# def compute_metrics(predictions, labels, ks=[1, 5, 10]):
#     """
#     输入:
#         predictions: numpy.array, 形状 (B, N) N是item总数
#         labels: numpy.array,, 形状 (B, 1) item 索引从 1 开始
#         ks: 需要计算的 K 值列表，如 [1, 5, 10]
#     返回:
#         metrics: 包含 NDCG@K、Recall@K 和 MRR 的字典
#     """
#     # 统一转换为 NumPy 数组
#     predictions_np = to_numpy(predictions)
#     labels_np = to_numpy(labels)

#     batch_size = predictions_np.shape[0]
#     metrics = {f"Recall@{k}": 0.0 for k in ks}
#     metrics.update({f"NDCG@{k}": 0.0 for k in ks})
#     mrr = 0.0

#     for i in range(batch_size):
#         pred_i = predictions_np[i]  # 当前轨迹的预测分数 (N,)
#         label_i = labels_np[i][0] - 1  # 当前轨迹的真实 item 索引

#         # 按分数降序排序，获取排名 (0-indexed)
#         ranked_indices = np.argsort(pred_i)[::-1]
#         rank = np.where(ranked_indices == label_i)[0][0]

#         # 计算指标
#         mrr += 1.0 / (rank + 1)
#         for k in ks:
#             if rank < k:
#                 metrics[f"Recall@{k}"] += 1.0
#                 metrics[f"NDCG@{k}"] += 1.0 / np.log2(rank + 2)

#     # 归一化（除以 batch_size）
#     for k in ks:
#         metrics[f"Recall@{k}"] /= batch_size
#         metrics[f"NDCG@{k}"] /= batch_size
#     metrics["MRR"] = mrr / batch_size

#     return metrics


# def compute_metrics(predictions, labels, ks=[1, 5, 10]):
#     """
#     输入:
#         predictions: numpy.array, 形状 (B, N) N是item总数
#         labels: numpy.array,, 形状 (B, 1) item 索引从 1 开始
#         ks: 需要计算的 K 值列表，如 [1, 5, 10]
#     返回:
#         metrics: 包含 NDCG@K、Recall@K 和 MRR 的字典
#     """
#     # 统一转换为 NumPy 数组
#     predictions_np = to_numpy(predictions)
#     labels_np = to_numpy(labels)

#     batch_size = predictions_np.shape[0]
#     metrics = {f"Recall@{k}": 0.0 for k in ks}
#     metrics.update({f"NDCG@{k}": 0.0 for k in ks})
#     mrr = 0.0

#     for i in range(batch_size):
#         pred_i = predictions_np[i]  # 当前轨迹的预测分数 (N,)
#         label_i = labels_np[i][0] - 1  # 当前轨迹的真实 item 索引

#         # 按分数降序排序，获取排名 (0-indexed)
#         ranked_indices = np.argsort(pred_i)[::-1]
#         rank = np.where(ranked_indices == label_i)[0][0]

#         # 计算指标
#         mrr += 1.0 / (rank + 1)
#         for k in ks:
#             if rank < k:
#                 metrics[f"Recall@{k}"] += 1.0
#                 metrics[f"NDCG@{k}"] += 1.0 / np.log2(rank + 2)

#     # 归一化（除以 batch_size）
#     for k in ks:
#         metrics[f"Recall@{k}"] /= batch_size
#         metrics[f"NDCG@{k}"] /= batch_size
#     metrics["MRR"] = mrr / batch_size

#     return metrics

if __name__ == '__main__':
    P = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]]
    L = [[3], [1], [4]]
    metrics = compute_metrics(np.array(P), np.array(L))
    print(metrics)
    # {'Recall@1': 0.3333333333333333, 'NDCG@1': 0.3333333333333333, 'Recall@5': 0.0, 'NDCG@5': 0.0, 'Recall@10': 0.0, 'NDCG@10': 0.0, 'MRR': 0.3333333333333333}
    P = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]]
    L = [[1], [4], [2]]
    metrics = compute_metrics(np.array(P), np.array(L))
    print(metrics)
    