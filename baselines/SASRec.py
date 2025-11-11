import os
import sys
import copy
import random
import time
import torch
import argparse
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from utils import compute_metrics

# copy from https://github.com/pmixer/SASRec.pytorch/blob/main/python/model.py

# test 300 u-i pairs Result: time: 258.169451(s), valid (NDCG@10: 0.5578, HR@10: 0.7000), test (NDCG@10: 0.7000, HR@10: 0.7000)
#Evaluatingepoch:1000, time: 309.335965(s), valid (NDCG@10: 0.5000, HR@10: 0.5000), test (NDCG@10: 0.7000, HR@10: 0.7000)

def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
# def evaluate(model, dataset, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0

#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:

#         if len(train[u]) < 1 or len(test[u]) < 1: continue

#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [test[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)

#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx
# ]])
#         predictions = predictions[0] # - for 1st argsort DESC

#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1

#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()

#     return NDCG / valid_user, HT / valid_user

# def evaluate(model, dataset, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
#     # 定义要计算的K值列表
#     ks = [1, 5, 10]  # 可以根据需要修改这个列表
    
#     # 初始化指标字典
#     metrics = {
#         f"NDCG@{k}": 0.0 for k in ks
#     }
#     metrics.update({
#         f"Recall@{k}": 0.0 for k in ks
#     })
#     valid_user = 0.0
    
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
    
#     for u in users:
#         if len(train[u]) < 1 or len(test[u]) < 1: 
#             continue

#         # 构建用户序列（包含验证项）
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]  # 验证项
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: 
#                 break
        
#         # 准备候选物品（1个正样本 + 100个负样本）
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [test[u][0]]  # 测试正样本
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: 
#                 t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)
        
#         # 模型预测
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions = predictions[0]  # 获取预测分数
        
#         # 获取正样本的排名（0-indexed）
#         rank = predictions.argsort().argsort()[0].item()
#         valid_user += 1
        
#         # 计算每个K值的指标
#         for k in ks:
#             # Recall@K (命中率)
#             if rank < k:
#                 metrics[f"Recall@{k}"] += 1
            
#             # NDCG@K
#             if rank < k:
#                 metrics[f"NDCG@{k}"] += 1 / np.log2(rank + 2)
        
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
    
#     # 计算平均指标
#     for k in ks:
#         metrics[f"Recall@{k}"] /= valid_user
#         metrics[f"NDCG@{k}"] /= valid_user
    
#     return metrics

# evaluate on val set
# def evaluate_valid(model, dataset, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: continue

#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break

#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [valid[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)

#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions = predictions[0]

#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1

#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()

#     return NDCG / valid_user, HT / valid_user

# def evaluate_valid(model, dataset, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
#     # 定义要计算的K值列表
#     ks = [1, 5, 10]  # 可以根据需要修改这个列表
    
#     # 初始化指标字典
#     metrics = {
#         f"NDCG@{k}": 0.0 for k in ks
#     }
#     metrics.update({
#         f"Recall@{k}": 0.0 for k in ks
#     })
#     valid_user = 0.0
    
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
    
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: 
#             continue

#         # 构建用户序列
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: 
#                 break
        
#         # 准备候选物品（1个正样本 + 100个负样本）
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [valid[u][0]]  # 正样本
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: 
#                 t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)
        
#         # 模型预测
#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions = predictions[0]  # 获取预测分数
        
#         # 获取正样本的排名（0-indexed）
#         rank = predictions.argsort().argsort()[0].item()
#         valid_user += 1
        
#         # 计算每个K值的指标
#         for k in ks:
#             # Recall@K (命中率)
#             if rank < k:
#                 metrics[f"Recall@{k}"] += 1
            
#             # NDCG@K
#             if rank < k:
#                 metrics[f"NDCG@{k}"] += 1 / np.log2(rank + 2)
        
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
    
#     # 计算平均指标
#     for k in ks:
#         metrics[f"Recall@{k}"] /= valid_user
#         metrics[f"NDCG@{k}"] /= valid_user
    
#     return metrics
def get_seqs(dataset):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    users = range(1, usernum + 1)
    val_labels = {}
    val_seqs = {}
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue
        seq = train[u][-args.maxlen:]
        #seq += valid[u]
        seq = [0] *(args.maxlen - len(seq)) + seq
        val_seqs[u] = seq
        val_labels[u] = valid[u]

    test_labels = {}
    test_seqs = {}
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue
        seq = train[u][-args.maxlen:]
        #seq += valid[u]
        seq = [0] *(args.maxlen - len(seq)) + seq
        test_seqs[u] = seq
        test_labels[u] = test[u]
    return val_seqs, val_labels, test_seqs, test_labels
    
def evaluate(model, dataset, test_seqs, test_labels, args, ks = [1, 5, 10]):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    users = range(1, usernum + 1)
    item_idx = range(0, itemnum+1)
    predictions = []
    labels = []
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue
        seq = test_seqs[u]
        labels.append(test_labels[u])
        pred = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions.append(pred.detach().cpu().numpy())
    predictions = np.array(predictions)
    labels = np.array(labels)
  
    #print(predictions.shape, labels.shape)
    #assert 0
    predictions = predictions.squeeze(axis=1)    
    metrics = compute_metrics(predictions, labels, ks)
    
    return metrics

def evaluate_valid(model, dataset, val_seqs, val_labels, args, ks = [1, 5, 10]):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    users = range(1, usernum + 1)
    item_idx = range(0, itemnum+1)
    predictions = []
    labels = []
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue
        seq = val_seqs[u]
        labels.append(val_labels[u])
        pred = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions.append(pred.detach().cpu().numpy())
    predictions = np.array(predictions)
    labels = np.array(labels)
  
    #print(predictions.shape, labels.shape)
    #assert 0
    predictions = predictions.squeeze(axis=1)    
    metrics = compute_metrics(predictions, labels, ks)
    
    return metrics

# def evaluate(model, dataset, args, ks = [1, 5, 10]):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
    
    
#     # 初始化指标字典
    
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
    
#     # 遍历每个用户
#     predictions, labels = [], []
#     for u in users:
#         if len(train[u]) < 1 or len(test[u]) < 1: 
#             continue

#         # 构建用户序列（包含验证项）
#         # seq = np.zeros([args.maxlen], dtype=np.int32)
#         # idx = args.maxlen - 1
#         # seq[idx] = valid[u][0]  # 验证项
#         # idx -= 1
#         # for i in reversed(train[u]):
#         #     seq[idx] = i
#         #     idx -= 1
#         #     if idx == -1: 
#         #         break
#         # labels.append(test[u])
#         seq = train[u][-args.maxlen:]
#         seq += valid[u]
#         seq = [0] *(args.maxlen - len(seq)) + seq
        
#         item_idx = range(0, itemnum+1)

#         labels.append(test[u])
#         # 模型预测
#         pred = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions.append(pred.detach().cpu().numpy())
#     predictions = np.array(predictions)
#     labels = np.array(labels)
  
#     #print(predictions.shape, labels.shape)
#     #assert 0
#     predictions = predictions.squeeze(axis=1)    
#     metrics = compute_metrics(predictions, labels, ks)
    
#     return metrics


# def evaluate_valid(model, dataset, args, ks = [1, 5, 10]):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
#     if usernum > 10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
    
#     predictions, labels = [], []
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: 
#             continue

#         # 构建用户序列
#         # seq = np.zeros([args.maxlen], dtype=np.int32)
#         # idx = args.maxlen - 1
#         # for i in reversed(train[u]):
#         #     seq[idx] = i
#         #     idx -= 1
#         #     if idx == -1: 
#         #         break
#         # labels.append(valid[u])

#         train[u] = train[u][-args.maxlen:]
#         seq = [0] *(args.maxlen - len(train[u])) + train[u]
#         labels.append(valid[u])
        
#         item_idx = range(0, itemnum+1)
        
#         # 模型预测
#         pred = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions.append(pred.detach().cpu().numpy())
#     predictions = np.array(predictions)
#     labels = np.array(labels)
  
#     #print(predictions.shape, labels.shape)
#     #assert 0
#     predictions = predictions.squeeze(axis=1) 
    
#     metrics = compute_metrics(predictions, labels, ks)
#     return metrics

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
    
    def predict_all_positions(self, user_ids, log_seqs, item_indices):
        # 获取整个序列的隐藏状态 [B, T, D]
        log_feats = self.log2feats(log_seqs)  # [batch_size, seq_len, hidden_units]
        
        # 获取候选物品嵌入
        # 处理不同维度的 item_indices（一维或二维）
        if isinstance(item_indices, list) or (isinstance(item_indices, torch.Tensor) and item_indices.dim() == 1):
            # 一维：所有用户使用相同的候选物品集
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # [C, D]
            item_embs = item_embs.unsqueeze(0)  # [1, C, D]
        else:
            # 二维：每个用户有自己的候选集 [B, C]
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # [B, C, D]
        
        # 计算所有位置和候选物品的得分
        # [B, T, D] @ [B, D, C] -> [B, T, C] 或 [B, T, D] @ [1, D, C] -> [B, T, C]
        logits = torch.matmul(
            log_feats,  # [B, T, D]
            item_embs.transpose(1, 2)  # 转置后 [B, D, C] 或 [1, D, C]
        )  # 结果形状 [B, T, C]
        
        return logits


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ml-20m', type=str)
parser.add_argument('--train_dir', default='result', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)

args = parser.parse_args()
args.dataset += '_ui'

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    u2i_index, i2u_index = build_index(args.dataset)

    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    val_seqs, val_labels, test_seqs, test_labels = get_seqs(dataset) 
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    model.eval()
    #t_test = evaluate(model, dataset, args)
    t_val = evaluate_valid(model, dataset, val_seqs, val_labels, args)#evaluate_valid(model, dataset, args)
    #print('test (NDCG@10: %.4f, ReCall@10: %.4f)' % (t_test['NDCG@10'], t_test['Recall@10']))
    print('val (NDCG@10: %.4f, ReCall@10: %.4f)' % (t_val['NDCG@10'], t_val['Recall@10']))
    model.train()  # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb

            pdb.set_trace()

    # if args.inference_only:
    #     model.eval()
    #     t_test = evaluate(model, dataset, args)
    #     print('test (NDCG@10: %.4f, ReCall@10: %.4f)' % (t_test['NDCG@10'], t_test['Recall@10']))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        total_loss = 0.0
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            total_loss += loss.item()
            # print("loss in epoch {} iteration {}: {}".format(epoch, step,
            #                                                  loss.item()))  # expected 0.4~0.6 after init few epochs
        print("loss in epoch {} : {}".format(epoch, total_loss/num_batch))
        # if epoch % 20 == 0:
        #     model.eval()
        #     t1 = time.time() - t0
        #     T += t1
        #     print('Evaluating', end='')
        #     t_test = evaluate(model, dataset, args)
        #     t_valid = evaluate_valid(model, dataset, args)
        #     print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
        #           % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

        #     if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[
        #         1] > best_test_hr:
        #         best_val_ndcg = max(t_valid[0], best_val_ndcg)
        #         best_val_hr = max(t_valid[1], best_val_hr)
        #         best_test_ndcg = max(t_test[0], best_test_ndcg)
        #         best_test_hr = max(t_test[1], best_test_hr)
        #         folder = args.dataset + '_' + args.train_dir
        #         fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        #         fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        #         torch.save(model.state_dict(), os.path.join(folder, fname))

        #     f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
        #     f.flush()
        #     t0 = time.time()
        #     model.train()
        if epoch % 5 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Eval: ', end='')
            
            # 调用评估函数获取指标字典
            #t_test_metrics = evaluate(model, dataset, args)
            t_valid_metrics = evaluate_valid(model, dataset, val_seqs, val_labels, args)
            
            # 提取K=10的指标用于打印和比较
            valid_ndcg1 = t_valid_metrics["NDCG@1"]
            valid_recall1 = t_valid_metrics["Recall@1"]
            valid_ndcg5 = t_valid_metrics["NDCG@5"]
            valid_recall5 = t_valid_metrics["Recall@5"]
            valid_ndcg10 = t_valid_metrics["NDCG@10"]
            valid_recall10 = t_valid_metrics["Recall@10"]

            # test_ndcg1 = t_test_metrics["NDCG@1"]
            # test_recall1 = t_test_metrics["Recall@1"]
            # test_ndcg5 = t_test_metrics["NDCG@5"]
            # test_recall5 = t_test_metrics["Recall@5"]
            # test_ndcg10 = t_test_metrics["NDCG@10"]
            # test_recall10 = t_test_metrics["Recall@10"]
            
            # print(f'epoch:{epoch}, time: {T:.2f}(s), '
            #     f'valid (NDCG@10: {valid_ndcg10:.4f}, Recall@10: {valid_recall10:.4f}), '
            #     f'test (NDCG@10: {test_ndcg10:.4f}, Recall@10: {test_recall10:.4f})')
            print(f'epoch:{epoch},'
                  f'valid (N@1: {valid_ndcg1:.4f}, N@5:{valid_ndcg5:.4f}, N@10:{valid_ndcg10:.4f}, '
                  f'Recall@1: {valid_recall1:.4f}, Recall@5:{valid_recall5:.4f}, Recall@10:{valid_recall10:.4f})')
            # print(f'epoch:{epoch},'
            #       f'test (N@1: {test_ndcg1:.4f}, N@5:{test_ndcg5:.4f}, N@10:{test_ndcg10:.4f}, '
            #       f'Recall@1: {test_recall1:.4f}, Recall@5:{test_recall5:.4f}, Recall@10:{test_recall10:.4f})')
            if (valid_ndcg10 > best_val_ndcg or 
                valid_recall10 > best_val_hr ):
                
                # 更新最佳指标
                best_val_ndcg = max(valid_ndcg10, best_val_ndcg)
                best_val_hr = max(valid_recall10, best_val_hr)
                
                # 保存模型
                folder = args.dataset + '_' + args.train_dir
                fname = f'SASRec.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.' \
                        f'hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
                torch.save(model.state_dict(), os.path.join(folder, fname))
            
            # 记录完整指标到文件
            f.write(f'{epoch} Valid Metrics: {t_valid_metrics}\n')
            # if (valid_ndcg10 > best_val_ndcg or 
            #     valid_recall10 > best_val_hr or 
            #     test_ndcg10 > best_test_ndcg or 
            #     test_recall10 > best_test_hr):
                
            #     # 更新最佳指标
            #     best_val_ndcg = max(valid_ndcg10, best_val_ndcg)
            #     best_val_hr = max(valid_recall10, best_val_hr)
            #     best_test_ndcg = max(test_ndcg10, best_test_ndcg)
            #     best_test_hr = max(test_recall10, best_test_hr)
                
            #     # 保存模型
            #     folder = args.dataset + '_' + args.train_dir
            #     fname = f'SASRec.epoch={epoch}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.' \
            #             f'hidden={args.hidden_units}.maxlen={args.maxlen}.pth'
            #     torch.save(model.state_dict(), os.path.join(folder, fname))
            
            # # 记录完整指标到文件
            # f.write(f'{epoch} Valid Metrics: {t_valid_metrics} Test Metrics: {t_test_metrics}\n')
            f.flush()
            
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    t_test = evaluate(model, dataset, test_seqs, test_labels, args) #evaluate(model, dataset, args)
    #t_val = evaluate_valid(model, dataset, args)
    print('test (NDCG@10: %.4f, ReCall@10: %.4f)' % (t_test['NDCG@10'], t_test['Recall@10']))
    f.write(f'{epoch} Valid Metrics: {t_test}\n')
    f.close()
    sampler.close()
    print("Done")
