import os
import math
import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import argparse
import torch.utils.data as data_utils
from tqdm import trange

from abc import *
from tqdm import tqdm
from torch import nn as nn
from collections import defaultdict

import pprint as pp
from datetime import date
from pathlib import Path
from utils import compute_metrics


def load_dataset(data_file):
    lines = open(data_file, 'r')
    #user_seq = []
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
    train, val, test = {}, {}, {}
    for user, items in User.items():
        train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
    return {
        'train': train,
        'val': val,
        'test': test,
        'usernum': usernum,
        'itemnum': itemnum
    }

def setup_train(args):
    set_up_gpu(args)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)
    return export_root


def create_experiment_export_folder(args):
    experiment_dir, experiment_description = args.experiment_dir, args.experiment_description
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)


def set_up_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    args.num_gpu = len(args.device_idx.split(","))


# copy from https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch/blob/master/models/bert.py

def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert'):
        args.mode = 'train'

        args.dataset_code = 'ml-20m'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = 128
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 100
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert'
        args.model_init_seed = 0

        args.bert_dropout = 0.1
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 4
    
    elif args.template.startswith('train_dae'):
        args.mode = 'train'

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'dae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.00
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'

        args.model_code = 'dae'
        args.model_init_seed = 0
        args.dae_num_hidden = 2
        args.dae_hidden_dim = 600
        args.dae_latent_dim = 200
        args.dae_dropout = 0.5

    elif args.template.startswith('train_vae_search_beta'):
        args.mode = 'train'

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'vae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.01
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@10'
        args.total_anneal_steps = 3000 if args.dataset_code == 'ml-1m' else 20000
        args.find_best_beta = True

        args.model_code = 'vae'
        args.model_init_seed = 0
        args.vae_num_hidden = 2
        args.vae_hidden_dim = 600
        args.vae_latent_dim = 200
        args.vae_dropout = 0.5
    
    elif args.template.startswith('train_vae_give_beta'):
        args.mode = 'train'

        args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'vae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.01
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 200
        args.metric_ks = [1, 5, 10, 20, 50, 100]
        args.best_metric = 'NDCG@100'
        args.find_best_beta = False
        args.anneal_cap = 0.342
        args.total_anneal_steps = 3000 if args.dataset_code == 'ml-1m' else 20000

        args.model_code = 'vae'
        args.model_init_seed = 0
        args.vae_num_hidden = 2
        args.vae_hidden_dim = 600
        args.vae_latent_dim = 200
        args.vae_dropout = 0.5

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)  # + self.segment(segment_label)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)




class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = \
           (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics


class BERTTrainer:
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        #super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.args = args
        self.log_period_as_iter = args.log_period_as_iter
        self.device = args.device
        self.model = model.to(self.device)
        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = self._create_optimizer()
        if args.enable_lr_schedule:
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.export_root = export_root
        self.metric_value = 0.0
        self.ce = nn.CrossEntropyLoss(ignore_index=0, size_average=True)
        if not os.path.exists(os.path.join(self.export_root, 'models')):
            os.makedirs(os.path.join(self.export_root, 'models'))
        if not os.path.exists(os.path.join(self.export_root, 'logs')):
            os.makedirs(os.path.join(self.export_root, 'logs'))


    @classmethod
    def code(cls):
        return 'bert'

    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        for epoch in range(self.num_epochs):
            # if epoch == 1:
            #     print(123)
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
            
        #self.writer.close()

    def train_one_epoch(self, epoch, accum_iter):
        self.model.train()
        if self.args.enable_lr_schedule:
            self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch_size = batch[0].size(0)
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

            accum_iter += batch_size

            if self._needs_to_log(accum_iter):
                tqdm_dataloader.set_description('Logging to Tensorboard')
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch+1,
                    'accum_iter': accum_iter,
                }
                log_data.update(average_meter_set.averages())
                # print(log_data.keys())
                # assert 0
                # self.log_extra_train_info(log_data)
                # self.logger_service.log_train(log_data)

        return accum_iter

    def validate(self, epoch, accum_iter):
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch+1,
                'accum_iter': accum_iter,
            }
            log_data.update(average_meter_set.averages())
            

            if epoch % 10 ==0:
                average_metrics = average_meter_set.averages()
                filename = os.path.join(self.export_root, 'logs', 'val_metrics.json')
                if not os.path.exists(filename):
                    with open(filename, 'w', encoding='utf-8') as file:
                        json.dump([], file)

                with open(filename, 'r+', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                    except json.JSONDecodeError:  # 处理空文件或无效JSON
                        data = []
                    
                    data.append(average_metrics)
                    file.seek(0)  # 回到文件开头
                    json.dump(data, file, indent=4, ensure_ascii=False)
                    file.truncate() 
            # self.log_extra_val_info(log_data)
                    
            if log_data[self.best_metric] >= self.metric_value:
                self.metric_value = log_data[self.best_metric]
                torch.save(log_data['state_dict'], os.path.join(self.export_root, 'models', 'best_acc_model.pth'))

            # self.logger_service.log_val(log_data)
            # torch.save(state_dict, os.path.join(path, filename))

    def test(self):
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_metrics(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                                      ['Recall@%d' % k for k in self.metric_ks[:3]]
                description = 'Val: ' + ', '.join(s + ' {:.3f}' for s in description_metrics)
                description = description.replace('NDCG', 'N').replace('Recall', 'R')
                description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError

    def _create_state_dict(self):
        return {
            'model_state_dict': self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

    def _needs_to_log(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.train_batch_size and accum_iter != 0


    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, answers = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        labels = answers # B
        # print(type(scores))
        # print(type(labels))
        
        # #assert 0
        # print(scores.shape, labels.shape)
        
        #print(labels)
        #assert 0
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        metrics = compute_metrics(scores, labels, self.metric_ks)
        # print("metrics1:", metrics)

        # def convert_labels_to_onehot(labels, V):
        #     """
        #     Args:
        #         labels: Tensor of shape (B, 1), values in [1, V-1]
        #         V: Vocabulary size (one-hot output size)
        #     Returns:
        #         onehot: Tensor of shape (B, V) with first position 0
        #     """
        #     B = labels.shape[0]
        #     # Convert labels to 0-based (0 ~ V-2)
        #     labels_0based = labels - 1  # shape (B, 1)
            
        #     # Generate one-hot for 0 ~ V-2 (shape B, V-1)
        #     onehot = F.one_hot(labels_0based.squeeze(1), num_classes=V-1)
            
        #     # Pad a zero column at the beginning (for label 0)
        #     zeros = torch.zeros((B, 1), dtype=onehot.dtype, device=labels.device)
        #     onehot_with_zero = torch.cat([zeros, onehot], dim=1)  # shape (B, V)
            
        #     return onehot_with_zero
        # labels = convert_labels_to_onehot(labels, scores.shape[1])
        # print(scores.shape, labels.shape)
        # metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        # print("metrics2:", metrics)
        #assert 0
        return metrics
    
    

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass

class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x): 
        x = self.bert(x)
        return self.out(x)


class BertDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.seed = seed
        self.rng = random.Random(seed)
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']

        self.user_count = dataset['usernum']
        self.item_count = dataset['itemnum']
        args.num_items = dataset['itemnum'] #len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1
        self.sample_size = args.test_negative_sample_size


        #self.test_negative_samples = self.generate_negative_samples()


    @classmethod
    def code(cls):
        return 'bert'
    
    def generate_negative_samples(self):
    #assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(int(self.seed))
        negative_samples = {}
        # print('\n',self.train,"\n")
        # assert 0
        print('Sampling negative items')
        for user in trange(1, self.user_count+1):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count) + 1
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) + 1
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        if mode == 'test':
            dicttrain = {}
            for i in self.train:
                dicttrain[i] = self.train[i] + self.val[i]
            dataset = BertEvalDataset(dicttrain, answers, self.max_len, self.CLOZE_MASK_TOKEN, None)    
        else:
            dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, None)
        return dataset


# dataset = {'train': train,
#                    'val': val,
#                    'test': test,
#                    'umap': umap,
#                    'smap': smap}

class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]

        seq = seq[-self.max_len:]
        mask_len = self.max_len - len(seq)
        seq = [0] * mask_len + seq

        return torch.LongTensor(seq), torch.LongTensor(answer)



parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--mode', type=str, default='train', choices=['train'])
parser.add_argument('--template', type=str, default='train_bert')

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='ml-20m')
parser.add_argument('--min_rating', type=int, default=4, help='Only keep ratings greater than equal to this value')
parser.add_argument('--min_uc', type=int, default=5, help='Only keep users with more than min_uc ratings')
parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--eval_set_size', type=int, default=500, 
                    help='Size of val and test set. 500 for ML-1m and 10000 for ML-20m recommended')

################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='bert')
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)

################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=100)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=None)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='bert')
# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')
# Finding optimal beta for VAE #
parser.add_argument('--find_best_beta', type=bool, default=False, 
                    help='If set True, the trainer will anneal beta all the way up to 1.0 and find the best beta')
parser.add_argument('--total_anneal_steps', type=int, default=2000, help='The step number when beta reaches 1.0')
parser.add_argument('--anneal_cap', type=float, default=0.2, help='Upper limit of increasing beta. Set this as the best beta found')

################
# Model
################
parser.add_argument('--model_code', type=str, default='bert')
parser.add_argument('--model_init_seed', type=int, default=6)
# BERT #
parser.add_argument('--bert_max_len', type=int, default=None, help='Length of sequence for bert')
parser.add_argument('--bert_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--bert_hidden_units', type=int, default=None, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=None, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=None, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=None, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=None, help='Probability for masking items in the training sequence')
# DAE #
parser.add_argument('--dae_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--dae_num_hidden', type=int, default=0, help='Number of hidden layers in DAE')
parser.add_argument('--dae_hidden_dim', type=int, default=600, help='Dimension of hidden layer in DAE')
parser.add_argument('--dae_latent_dim', type=int, default=200, help="Dimension of latent vector in DAE")
parser.add_argument('--dae_dropout', type=float, default=0.5, help='Probability of input dropout in DAE')
# VAE #
parser.add_argument('--vae_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--vae_num_hidden', type=int, default=0, help='Number of hidden layers in VAE')
parser.add_argument('--vae_hidden_dim', type=int, default=600, help='Dimension of hidden layer in VAE')
parser.add_argument('--vae_latent_dim', type=int, default=200, help="Dimension of latent vector in VAE (K in paper)")
parser.add_argument('--vae_dropout', type=float, default=0.5, help='Probability of input dropout in VAE')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='BERT4Rec')
parser.add_argument('--experiment_description', type=str, default='train')
parser.add_argument('--dataset', default='kuairand_pure', type=str)


################
args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.experiment_dir += '/'+args.dataset

set_template(args)

export_root = setup_train(args)

data_path = 'data/'+args.dataset+'_ui.txt'
dataset = load_dataset(data_path)

print("Total number of users:", dataset['usernum'], "\nTotal number of items:", dataset['itemnum'])
dataloader = BertDataloader(args, dataset)
train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()
model = BERTModel(args)

trainer= BERTTrainer(args, model, train_loader, val_loader, test_loader, export_root)

trainer.train()
trainer.test()


