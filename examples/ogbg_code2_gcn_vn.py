import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch.optim import Adam
from torchvision import transforms
from torchvision.ops import MLP
from tqdm.auto import tqdm

import torch_geometric.nn as gnn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

# Setup

DATASET_NAME = 'ogbg-code2'
DATASET_DIR = 'ogbg_code2'
NUM_VOCAB = 5000
MAX_SEQ_LEN = 5
MAX_DEPTH = 20
BATCH_SIZE = 128
NUM_WORKERS = 0
EMB_DIM = 300
NUM_LAYERS = 5
NUM_EPOCHS = 25
OPT_LR = .001
OPT_CLS = Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = PygGraphPropPredDataset(DATASET_NAME, root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(DATASET_NAME)

node_types_mapping = pd.read_csv(
    osp.join(root, DATASET_DIR, 'mapping', 'typeidx2type.csv.gz'))
num_node_types = len(node_types_mapping['type'])

node_attr_mapping = pd.read_csv(
    osp.join(root, DATASET_DIR, 'mapping', 'attridx2attr.csv.gz'))
num_node_attr = len(node_attr_mapping['attr'])

# Utilities


def get_vocab_mapping(seq_list, num_vocab):
    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind='stable')[:num_vocab]

    vocab2idx = {
        vocab_list[vocab_idx]: idx
        for idx, vocab_idx in enumerate(topvocab)
    }
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    vocab2idx['__UNK__'] = num_vocab
    idx2vocab.append('__UNK__')

    vocab2idx['__EOS__'] = num_vocab + 1
    idx2vocab.append('__EOS__')

    for idx, vocab in enumerate(idx2vocab):
        assert idx == vocab2idx[vocab]

    assert vocab2idx['__EOS__'] == len(idx2vocab) - 1

    return vocab2idx, idx2vocab


def augment_edge(data):
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    edge_index_ast_inverse = torch.stack(
        [edge_index_ast[1], edge_index_ast[0]], 0)
    edge_attr_ast_inverse = torch.cat([
        torch.zeros(edge_index_ast_inverse.size(1), 1),
        torch.ones(edge_index_ast_inverse.size(1), 1)
    ], dim=1)

    attributed_node_idx_in_dfs_order = torch.where(
        data.node_is_attributed.view(-1, ) == 1)[0]

    edge_index_nextoken = torch.stack([
        attributed_node_idx_in_dfs_order[:-1],
        attributed_node_idx_in_dfs_order[1:]
    ], dim=0)
    edge_attr_nextoken = torch.cat([
        torch.ones(edge_index_nextoken.size(1), 1),
        torch.zeros(edge_index_nextoken.size(1), 1)
    ], dim=1)

    edge_index_nextoken_inverse = torch.stack(
        [edge_index_nextoken[1], edge_index_nextoken[0]], dim=0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat([
        edge_index_ast, edge_index_ast_inverse, edge_index_nextoken,
        edge_index_nextoken_inverse
    ], dim=1)
    data.edge_attr = torch.cat([
        edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken,
        edge_attr_nextoken_inverse
    ], dim=0)

    return data


def encode_y_to_arr(data, vocab2idx, max_seq_len):
    data.y_arr = encode_seq_to_arr(data.y, vocab2idx, max_seq_len)
    return data


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(
        0, max_seq_len - len(seq))
    return torch.tensor([[
        vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__']
        for w in augmented_seq
    ]], dtype=torch.long)


def decode_arr_to_seq(arr, idx2vocab):
    eos_idx_list = torch.nonzero(arr == len(idx2vocab) - 1, as_tuple=False)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[:torch.min(eos_idx_list)]
    else:
        clippted_arr = arr
    return [idx2vocab[clippted] for clippted in clippted_arr.cpu()]


# GNN


class ASTNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_node_types, num_node_attr, max_depth):
        super().__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_node_types, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_node_attr, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(
            x[:, 1]) + self.depth_encoder(depth)


class GCNConv(gnn.MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr='add')

        self.lin = torch.nn.Linear(emb_dim, emb_dim)
        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))
        self.edge_encoder = torch.nn.Linear(2, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding,
                             norm=norm)
        out += F.relu(x + self.bias) / deg.view(-1, 1)
        return out

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class VirtualNodeEncoder(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, num_node_types, num_node_attr,
                 max_depth):
        super().__init__()
        self.num_layers = num_layers

        self.ast_node_encoder = ASTNodeEncoder(emb_dim, num_node_types,
                                               num_node_attr, max_depth)

        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        self.virtualnode_embedding.weight.data.zero_()

        self.convs = torch.nn.ModuleList(
            [GCNConv(emb_dim) for _ in range(num_layers)])

        self.batch_norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(emb_dim) for _ in range(num_layers)])

        self.mlp_vn_list = torch.nn.ModuleList([
            MLP(emb_dim, [2 * emb_dim, emb_dim],
                norm_layer=torch.nn.BatchNorm1d) for _ in range(num_layers - 1)
        ]) + [None]

    def forward(self, data_batch):
        vn_embed = data_batch.edge_index.new_zeros(
            data_batch.batch[-1].item() + 1)
        vn_embed = self.virtualnode_embedding(vn_embed)

        h = self.ast_node_encoder(data_batch.x,
                                  data_batch.node_depth.view(-1, ))
        for layer, (conv, batch_norm, mlp_vn) in enumerate(
                zip(self.convs, self.batch_norms, self.mlp_vn_list)):
            h = h + vn_embed[data_batch.batch]
            h = conv(h, data_batch.edge_index, data_batch.edge_attr)
            h = batch_norm(h)

            if layer == self.num_layers - 1:
                break

            h = F.relu(h)

            vn_embed = vn_embed + gnn.global_add_pool(h, data_batch.batch)
            vn_embed = mlp_vn(vn_embed)

        return h


class GCNVirtual(torch.nn.Module):
    '''GCN with virtual nodes'''
    def __init__(self, num_vocab, num_node_types, num_node_attr, max_seq_len,
                 max_depth, num_layers, emb_dim):
        super().__init__()

        assert num_layers >= 2

        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len

        self.node_encoder = VirtualNodeEncoder(num_layers, emb_dim,
                                               num_node_types, num_node_attr,
                                               max_depth)

        self.graph_pred_linear_list = torch.nn.ModuleList([
            torch.nn.Linear(emb_dim, self.num_vocab)
            for _ in range(self.max_seq_len)
        ])

    def forward(self, data_batch):
        h_node = self.node_encoder(data_batch)
        h_graph = gnn.global_mean_pool(h_node, data_batch.batch)
        preds = [lin(h_graph) for lin in self.graph_pred_linear_list]
        return preds


# Training and testing

vocab2idx, idx2vocab = get_vocab_mapping(
    [dataset.data.y[i] for i in split_idx['train']], NUM_VOCAB)
dataset.transform = transforms.Compose(
    [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, MAX_SEQ_LEN)])

train_loader = DataLoader(dataset[split_idx['train']], batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset[split_idx['test']], batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)

model = GCNVirtual(len(vocab2idx), num_node_types, num_node_attr, MAX_SEQ_LEN,
                   MAX_DEPTH, NUM_LAYERS, EMB_DIM)
model.to(device)

opt = OPT_CLS(model.parameters(), lr=OPT_LR)


def train(epoch):
    model.train()

    losses = []
    for data_batch in tqdm(train_loader, desc=f'Epoch {epoch:02d}'):
        data_batch = data_batch.to(device)
        if data_batch.x.shape[0] == 1 or data_batch.batch[-1] == 0:
            continue

        pred_list = model(data_batch)

        loss_seq = [
            F.cross_entropy(pred.float(), y)
            for pred, y in zip(pred_list, data_batch.y_arr.T)
        ]
        loss = torch.stack(loss_seq).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses += [loss.item()]

    return np.mean(losses)


@torch.no_grad()
def test(loader, label):
    model.eval()

    seq_ref_list = []
    seq_pred_list = []
    for data_batch in tqdm(loader, desc=label):
        data_batch = data_batch.to(device)
        if data_batch.x.shape[0] == 1:
            continue
        preds = model(data_batch)
        preds = torch.cat(
            [torch.argmax(pred, 1).view(-1, 1) for pred in preds], 1)
        seq_pred_list += [decode_arr_to_seq(arr, idx2vocab) for arr in preds]
        seq_ref_list += [data_batch.y[i] for i in range(len(data_batch.y))]

    return evaluator.eval({
        "seq_ref": seq_ref_list,
        "seq_pred": seq_pred_list
    })['F1']


for epoch in range(1, NUM_EPOCHS + 1):
    print('Training...')
    train(epoch)

    print('Testing...')
    train_f1 = test(train_loader, 'train')
    val_f1 = test(valid_loader, 'valid')
    test_f1 = test(test_loader, 'test')

    print(f'Train: {train_f1:.4f}, Val: {val_f1:.4f}, Test: {test_f1:.4f}')
