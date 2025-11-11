import os
import argparse
import copy
import math
import random
import torch
import tqdm
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from scipy.sparse import csr_matrix
from collections import defaultdict

# copy from https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/models.py
import random

import torch
from torch.utils.data import Dataset

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(dim)

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

# def get_user_seqs(data_file):
#     #lines = open(data_file).readlines()
#     lines = open('data/processed.txt').readlines()
#     user_seq = []
#     item_set = set()
#     for line in lines:
#         user, items = line.strip().split(' ', 1)
#         items = items.split(' ')
#         items = [int(item) for item in items]
#         user_seq.append(items)
#         item_set = item_set | set(items)
#     max_item = max(item_set)

#     num_users = len(lines)
#     num_items = max_item + 2

#     valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
#     test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
#     return user_seq, max_item, valid_rating_matrix, test_rating_matrix

def get_user_seqs(data_file):
    #lines = open(data_file).readlines()
    lines = open(data_file, 'r')
    user_seq = []
    #item_set = set()
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    for line in lines:
        #user, items = line.strip().split(' ', 1)
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        #items = items.split(' ')
        #items = [int(item) for item in items]
        #user_seq.append(items)
        #item_set = item_set | set(items)
    #max_item = max(item_set)
    user_seq = [i for i in User.values()]
    # num_users = len(lines)
    # num_items = max_item + 2
    max_item = itemnum
    num_users = usernum
    num_items = itemnum + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

# class PretrainDataset(Dataset):

#     def __init__(self, args, user_seq, long_sequence):
#         self.args = args
#         self.user_seq = user_seq
#         self.long_sequence = long_sequence
#         self.max_len = args.max_seq_length
#         self.part_sequence = []
#         self.split_sequence()

#     def split_sequence(self):
#         for seq in self.user_seq:
#             input_ids = seq[-(self.max_len+2):-2] # keeping same as train set
#             for i in range(len(input_ids)):
#                 self.part_sequence.append(input_ids[:i+1])

#     def __len__(self):
#         return len(self.part_sequence)

#     def __getitem__(self, index):

#         sequence = self.part_sequence[index] # pos_items
#         # sample neg item for every masked item
#         masked_item_sequence = []
#         neg_items = []
#         # Masked Item Prediction
#         item_set = set(sequence)
#         for item in sequence[:-1]:
#             prob = random.random()
#             if prob < self.args.mask_p:
#                 masked_item_sequence.append(self.args.mask_id)
#                 neg_items.append(neg_sample(item_set, self.args.item_size))
#             else:
#                 masked_item_sequence.append(item)
#                 neg_items.append(item)

#         # add mask at the last position
#         masked_item_sequence.append(self.args.mask_id)
#         neg_items.append(neg_sample(item_set, self.args.item_size))

#         # Segment Prediction
#         if len(sequence) < 2:
#             masked_segment_sequence = sequence
#             pos_segment = sequence
#             neg_segment = sequence
#         else:
#             sample_length = random.randint(1, len(sequence) // 2)
#             start_id = random.randint(0, len(sequence) - sample_length)
#             neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
#             pos_segment = sequence[start_id: start_id + sample_length]
#             neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
#             masked_segment_sequence = sequence[:start_id] + [self.args.mask_id] * sample_length + sequence[
#                                                                                       start_id + sample_length:]
#             pos_segment = [self.args.mask_id] * start_id + pos_segment + [self.args.mask_id] * (
#                         len(sequence) - (start_id + sample_length))
#             neg_segment = [self.args.mask_id] * start_id + neg_segment + [self.args.mask_id] * (
#                         len(sequence) - (start_id + sample_length))

#         assert len(masked_segment_sequence) == len(sequence)
#         assert len(pos_segment) == len(sequence)
#         assert len(neg_segment) == len(sequence)

#         # padding sequence
#         pad_len = self.max_len - len(sequence)
#         masked_item_sequence = [0] * pad_len + masked_item_sequence
#         pos_items = [0] * pad_len + sequence
#         neg_items = [0] * pad_len + neg_items
#         masked_segment_sequence = [0]*pad_len + masked_segment_sequence
#         pos_segment = [0]*pad_len + pos_segment
#         neg_segment = [0]*pad_len + neg_segment

#         masked_item_sequence = masked_item_sequence[-self.max_len:]
#         pos_items = pos_items[-self.max_len:]
#         neg_items = neg_items[-self.max_len:]

#         masked_segment_sequence = masked_segment_sequence[-self.max_len:]
#         pos_segment = pos_segment[-self.max_len:]
#         neg_segment = neg_segment[-self.max_len:]

#         # Associated Attribute Prediction
#         # Masked Attribute Prediction
#         attributes = []
#         for item in pos_items:
#             attribute = [0] * self.args.attribute_size
#             try:
#                 now_attribute = self.args.item2attribute[str(item)]
#                 for a in now_attribute:
#                     attribute[a] = 1
#             except:
#                 pass
#             attributes.append(attribute)


#         assert len(attributes) == self.max_len
#         assert len(masked_item_sequence) == self.max_len
#         assert len(pos_items) == self.max_len
#         assert len(neg_items) == self.max_len
#         assert len(masked_segment_sequence) == self.max_len
#         assert len(pos_segment) == self.max_len
#         assert len(neg_segment) == self.max_len


#         cur_tensors = (torch.tensor(attributes, dtype=torch.long),
#                        torch.tensor(masked_item_sequence, dtype=torch.long),
#                        torch.tensor(pos_items, dtype=torch.long),
#                        torch.tensor(neg_items, dtype=torch.long),
#                        torch.tensor(masked_segment_sequence, dtype=torch.long),
#                        torch.tensor(pos_segment, dtype=torch.long),
#                        torch.tensor(neg_segment, dtype=torch.long),)
#         return cur_tensors

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        target_neg = []
        # print("\n", type(items),"\n")
        # print("\n", items ,"\n")
        # assert 0
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """
    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        # 修改属性
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class S3RecModel(nn.Module):
    def __init__(self, args):
        super(S3RecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        #self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.map_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    # AAP
    def associated_attribute_prediction(self, sequence_output, attribute_embedding):
        '''
        :param sequence_output: [B L H]
        :param attribute_embedding: [arribute_num H]
        :return: scores [B*L tag_num]
        '''
        sequence_output = self.aap_norm(sequence_output) # [B L H]
        sequence_output = sequence_output.view([-1, self.args.hidden_size, 1]) # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1)) # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        '''
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        '''
        sequence_output = self.mip_norm(sequence_output.view([-1,self.args.hidden_size])) # [B*L H]
        target_item = target_item.view([-1,self.args.hidden_size]) # [B*L H]
        score = torch.mul(sequence_output, target_item) # [B*L H]
        return torch.sigmoid(torch.sum(score, -1)) # [B*L]

    # MAP
    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view([-1, self.args.hidden_size, 1])  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1)) # [B*L tag_num]

    # SP sample neg segment
    def segment_prediction(self, context, segment):
        '''
        :param context: [B H]
        :param segment: [B H]
        :return:
        '''
        context = self.sp_norm(context)
        score = torch.mul(context, segment) # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1)) # [B]

    #
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def pretrain(self, attributes, masked_item_sequence, pos_items,  neg_items,
                  masked_segment_sequence, pos_segment, neg_segment):

        # Encode masked sequence
        sequence_emb = self.add_position_embedding(masked_item_sequence)
        sequence_mask = (masked_item_sequence == 0).float() * -1e8
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)

        encoded_layers = self.item_encoder(sequence_emb,
                                          sequence_mask,
                                          output_all_encoded_layers=True)
        # [B L H]
        sequence_output = encoded_layers[-1]

        attribute_embeddings = self.attribute_embeddings.weight
        # AAP
        aap_score = self.associated_attribute_prediction(sequence_output, attribute_embeddings)
        aap_loss = self.criterion(aap_score, attributes.view(-1, self.args.attribute_size).float())
        # only compute loss at non-masked position
        aap_mask = (masked_item_sequence != self.args.mask_id).float() * \
                         (masked_item_sequence != 0).float()
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))

        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)
        mip_distance = torch.sigmoid(pos_score - neg_score)
        mip_loss = self.criterion(mip_distance, torch.ones_like(mip_distance, dtype=torch.float32))
        mip_mask = (masked_item_sequence == self.args.mask_id).float()
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        # MAP
        map_score = self.masked_attribute_prediction(sequence_output, attribute_embeddings)
        map_loss = self.criterion(map_score, attributes.view(-1, self.args.attribute_size).float())
        map_mask = (masked_item_sequence == self.args.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_sequence)
        segment_mask = (masked_segment_sequence == 0).float() * -1e8
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(segment_context,
                                               segment_mask,
                                               output_all_encoded_layers=True)

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]# [B H]
        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.item_encoder(pos_segment_emb,
                                                   pos_segment_mask,
                                                   output_all_encoded_layers=True)
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.item_encoder(neg_segment_emb,
                                                       neg_segment_mask,
                                                       output_all_encoded_layers=True)
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :] # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        sp_loss = torch.sum(self.criterion(sp_distance,
                                           torch.ones_like(sp_distance, dtype=torch.float32)))

        return aap_loss, mip_loss, map_loss, sp_loss

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)
    


    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class PretrainTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = f'AAP-{self.args.aap_weight}-' \
               f'MIP-{self.args.mip_weight}-' \
               f'MAP-{self.args.map_weight}-' \
               f'SP-{self.args.sp_weight}'

        pretrain_data_iter = tqdm.tqdm(enumerate(pretrain_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(pretrain_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            attributes, masked_item_sequence, pos_items, neg_items, \
            masked_segment_sequence, pos_segment, neg_segment = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(attributes,
                                            masked_item_sequence, pos_items, neg_items,
                                            masked_segment_sequence, pos_segment, neg_segment)

            joint_loss = self.args.aap_weight * aap_loss + \
                         self.args.mip_weight * mip_loss + \
                         self.args.map_weight * map_loss + \
                         self.args.sp_weight * sp_loss

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        post_fix = {
            "epoch": epoch,
            "aap_loss_avg": '{:.4f}'.format(aap_loss_avg /num),
            "mip_loss_avg": '{:.4f}'.format(mip_loss_avg /num),
            "map_loss_avg": '{:.4f}'.format(map_loss_avg / num),
            "sp_loss_avg": '{:.4f}'.format(sp_loss_avg / num),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.finetune(input_ids)
                loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.finetune(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # 推荐的结果

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


# def get_user_seqs_long(data_file):
#     lines = open(data_file).readlines()
#     user_seq = []
#     long_sequence = []
#     item_set = set()
#     for line in lines:
#         user, items = line.strip().split(' ', 1)
#         items = items.split(' ')
#         items = [int(item) for item in items]
#         long_sequence.extend(items) # 后面的都是采的负例
#         user_seq.append(items)
#         item_set = item_set | set(items)
#     max_item = max(item_set)

#     return user_seq, max_item, long_sequence

def get_user_seqs_long(data_file):
    lines = open(data_file, 'r')
    user_seq = []
    #item_set = set()
    long_sequence = []
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    for line in lines:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    for i in User.values():
        user_seq.append(i)
        long_sequence.extend(i)
    
    max_item = itemnum
    # lines = open(data_file).readlines()
    # user_seq = []
    # long_sequence = []
    # item_set = set()
    # for line in lines:
    #     user, items = line.strip().split(' ', 1)
    #     items = items.split(' ')
    #     items = [int(item) for item in items]
    #     long_sequence.extend(items) # 后面的都是采的负例
    #     user_seq.append(items)
    #     item_set = item_set | set(items)
    # max_item = max(item_set)

    return user_seq, max_item, long_sequence


# def get_item2attribute_json(data_file):
#     print(data_file)
#     #item2attribute = json.loads(open(data_file, "r", encoding="utf-8"))#json.loads(open(data_file)) #.readline())
#     with open(data_file, "r", encoding="utf-8") as file:
#         item2attribute = json.load(file)
#     attribute_set = set()
#     for item, attributes in item2attribute.items():
#         attribute_set = attribute_set | set(attributes)
#     attribute_size = max(attribute_set) # 331
#     return item2attribute, attribute_size


# parser = argparse.ArgumentParser()

# parser.add_argument('--data_dir', default='./data/', type=str)
# parser.add_argument('--output_dir', default='output/', type=str)
# parser.add_argument('--data_name', default='Beauty', type=str)

# # model args
# parser.add_argument("--model_name", default='Pretrain', type=str)

# parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
# parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
# parser.add_argument('--num_attention_heads', default=2, type=int)
# parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
# parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
# parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
# parser.add_argument("--initializer_range", type=float, default=0.02)
# parser.add_argument('--max_seq_length', default=50, type=int)

# # train args
# parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
# parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
# parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
# parser.add_argument("--no_cuda", action="store_true")
# parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
# parser.add_argument("--seed", default=42, type=int)

# # pre train args
# parser.add_argument("--pre_epochs", type=int, default=300, help="number of pre_train epochs")
# parser.add_argument("--pre_batch_size", type=int, default=100)

# parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
# parser.add_argument("--aap_weight", type=float, default=0.2, help="aap loss weight")
# parser.add_argument("--mip_weight", type=float, default=1.0, help="mip loss weight")
# parser.add_argument("--map_weight", type=float, default=1.0, help="map loss weight")
# parser.add_argument("--sp_weight", type=float, default=0.5, help="sp loss weight")

# parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
# parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
# parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
# parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")


# args = parser.parse_args()

# set_seed(args.seed)
# check_path(args.output_dir)

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

# args.data_file = args.data_dir + args.data_name + '.txt'
# item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'
# # concat all user_seq get a long sequence, from which sample neg segment for SP
# user_seq, max_item, long_sequence = get_user_seqs_long(args.data_file)
# item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

# args.item_size = max_item + 2
# args.mask_id = max_item + 1
# args.attribute_size = attribute_size + 1
# # save model args
# args_str = f'{args.model_name}-{args.data_name}'
# args.log_file = os.path.join(args.output_dir, args_str + '.txt')
# print(args)
# with open(args.log_file, 'a') as f:
#     f.write(str(args) + '\n')

# args.item2attribute = item2attribute

# model = S3RecModel(args=args)
# trainer = PretrainTrainer(model, None, None, None, args)

# for epoch in range(args.pre_epochs):

#     pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
#     pretrain_sampler = RandomSampler(pretrain_dataset)
#     pretrain_dataloader = DataLoader(pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)

#     trainer.pretrain(epoch, pretrain_dataloader)

#     if (epoch+1) % 10 == 0:
#         ckp = f'{args.data_name}-epochs-{epoch+1}.pt'
#         checkpoint_path = os.path.join(args.output_dir, ckp)
#         trainer.save(checkpoint_path)



parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--output_dir', default='output/', type=str)
parser.add_argument('--data_name', default='ui', type=str)
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

# model args
parser.add_argument("--model_name", default='Finetune_full', type=str)
parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
parser.add_argument('--num_attention_heads', default=2, type=int)
parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
parser.add_argument("--initializer_range", type=float, default=0.02)
parser.add_argument('--max_seq_length', default=50, type=int)

# train args
parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
parser.add_argument("--seed", default=42, type=int)

parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

args = parser.parse_args()

set_seed(args.seed)
check_path(args.output_dir)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

args.data_file = args.data_dir + args.data_name + '.txt'
item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
    get_user_seqs(args.data_file)

# item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

args.item_size = max_item + 2
args.mask_id = max_item + 1
# args.attribute_size = attribute_size + 1

# save model args
args_str = f'{args.model_name}-{args.data_name}-{args.ckp}'
args.log_file = os.path.join(args.output_dir, args_str + '.txt')
print(str(args))
with open(args.log_file, 'a') as f:
    f.write(str(args) + '\n')

# args.item2attribute = item2attribute
# set item score in train set to `0` in validation
args.train_matrix = valid_rating_matrix

# save model
checkpoint = args_str + '.pt'
args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

train_dataset = SASRecDataset(args, user_seq, data_type='train')
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

test_dataset = SASRecDataset(args, user_seq, data_type='test')
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)


model = S3RecModel(args=args)

trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                            test_dataloader, args)


if args.do_eval:
    trainer.load(args.checkpoint_path)
    print(f'Load model from {args.checkpoint_path} for test!')
    scores, result_info = trainer.test(0, full_sort=True)

else:
    pretrained_path = os.path.join(args.output_dir, f'{args.data_name}-epochs-{args.ckp}.pt')
    try:
        trainer.load(pretrained_path)
        print(f'Load Checkpoint From {pretrained_path}!')

    except FileNotFoundError:
        print(f'{pretrained_path} Not Found! The Model is same as SASRec')

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)
        # evaluate on NDCG@20
        scores, _ = trainer.valid(epoch, full_sort=True)
        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainer.args.train_matrix = test_rating_matrix
    print('---------------Change to test_rating_matrix!-------------------')
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0, full_sort=True)

print(args_str)
print(result_info)
with open(args.log_file, 'a') as f:
    f.write(args_str + '\n')
    f.write(result_info + '\n')