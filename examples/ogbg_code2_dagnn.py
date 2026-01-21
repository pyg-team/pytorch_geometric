import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

import torch_geometric.nn as gnn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Batch

DATASET_NAME = 'ogbg-code2'
DATASET_DIR = 'ogbg_code2'
NUM_VOCAB = 5000
MAX_SEQ_LEN = 5
MAX_DEPTH = 20
BATCH_SIZE = 200
NUM_WORKERS = 10  # speed up making edge masks (see `make_edge_masks`)
EMB_DIM = 300
NUM_LAYERS = 2
NUM_EPOCHS = 50
OPT_LR = .001
OPT_CLS = Adam
MAX_GRAD_NORM = .25
P_DROPOUT = .5  # helps with overfitting
TORCH_SEED = 42

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

torch.manual_seed(TORCH_SEED)

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

    data.edge_index = torch.cat([
        edge_index_ast,
        edge_index_nextoken,
    ], dim=1)
    data.edge_attr = torch.cat([
        edge_attr_ast,
        edge_attr_nextoken,
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


def make_edge_masks(data):
    '''
    Returns:
        a tensor with shape `(D-1, E)`, where `D` is is the depth
        of the dag (aka the number of topological generations),
        and `E` is the number of edges. The `d`th mask indicates which
        edges connect generation `d` to `d+1`.
    '''
    max_depth = data.node_depth.max().item() + 1

    topo_gens = {d: [] for d in range(max_depth)}
    for i, d in enumerate(data.node_depth.squeeze().numpy()):
        topo_gens[d] += [i]

    successors = {u: [] for u in range(data.num_nodes)}
    for u, v in data.edge_index.numpy().T:
        successors[u] += [v]

    edge_mask_list = []
    for d in range(max_depth - 1):
        node_level = topo_gens[d]
        succ = set.union(*[set(successors[n]) for n in node_level])
        node_mask = pyg_utils.index_to_mask(
            torch.tensor(node_level + list(succ)), data.num_nodes)
        edge_mask = \
            node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        edge_mask_list += [edge_mask]

    data.edge_masks = torch.stack(edge_mask_list)
    return data


def collate_edge_masks(edge_masks_list, total_num_edges):
    '''collates list of edge mask tensors from many dags. Since the dags vary
    in depth, edge mask tensors are padded to the maximum depth.
    '''
    max_depth = max(edge_masks.shape[0] for edge_masks in edge_masks_list)

    # output tensor that will contain all the edge masks
    edge_masks_collated = torch.zeros((max_depth, total_num_edges), dtype=bool)

    i = 0
    for edge_masks in edge_masks_list:
        # copy these masks into the output tensor
        depth, num_edges = edge_masks.shape
        if depth > 0:
            edge_masks_collated[:depth, i:(i + num_edges)] = edge_masks
        i += num_edges

    return edge_masks_collated


def collate_fn(data_list):
    data_batch = Batch.from_data_list(data_list, exclude_keys=['edge_masks'])
    edge_masks_list = [data.edge_masks for data in data_list]
    data_batch.edge_masks = collate_edge_masks(edge_masks_list,
                                               data_batch.num_edges)
    return data_batch


def make_dataloader(dataset, shuffle=False):
    return DataLoader(dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=True)


# GNN


class ASTNodeEncoder(nn.Module):
    def __init__(self, emb_dim, num_node_types, num_node_attr, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.type_encoder = nn.Embedding(num_node_types, emb_dim)
        self.attribute_encoder = nn.Embedding(num_node_attr, emb_dim)
        self.depth_encoder = nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x, depth):
        depth = torch.where(depth > self.max_depth, self.max_depth, depth)
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(
            x[:, 1]) + self.depth_encoder(depth)


class AttnConv(gnn.MessagePassing):
    def __init__(self, emb_dim, reverse_flow, num_edge_attr=2):
        flow = 'target_to_source' if reverse_flow else 'source_to_target'
        super().__init__(aggr='add', flow=flow)
        self.edge_lin = nn.Linear(num_edge_attr, emb_dim)
        self.attn_lin = nn.Linear(2 * emb_dim, 1)

    def forward(self, h, h_prev, edge_index, edge_attr):
        edge_emb = self.edge_lin(edge_attr)
        return self.propagate(edge_index, h=h, h_prev=h_prev,
                              edge_emb=edge_emb)

    def message(self, h_j, h_prev_i, edge_emb, index, size_i):
        q, k, v = h_prev_i, h_j + edge_emb, h_j
        alpha_j = self.attn_lin(torch.cat([q, k], -1))
        alpha_j = pyg_utils.softmax(alpha_j, index=index, num_nodes=size_i)
        return alpha_j * v


class DagEncoder(nn.Module):
    def __init__(self, emb_dim, num_node_types, num_node_attr, num_layers,
                 max_depth, p_dropout, reverse_flow=False):
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (0, 1) if not reverse_flow else (1, 0)
        self.ast_node_encoder = ASTNodeEncoder(emb_dim, num_node_types,
                                               num_node_attr, max_depth)
        self.attn_convs = nn.ModuleList(
            [AttnConv(emb_dim, reverse_flow) for _ in range(num_layers)])
        self.gru_cells = nn.ModuleList(
            [nn.GRUCell(emb_dim, emb_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=p_dropout, inplace=True)

    def forward(self, data_batch):
        root_node_mask = ~pyg_utils.index_to_mask(
            data_batch.edge_index[self.i],  # nodes with in-flow
            data_batch.num_nodes)
        leaf_node_mask = ~pyg_utils.index_to_mask(
            data_batch.edge_index[self.j],  # nodes with out-flow
            data_batch.num_nodes)

        # initial node embeddings
        h_prev = self.ast_node_encoder(data_batch.x,
                                       data_batch.node_depth.squeeze())

        # list of leaf node embeddings at each layer; used in readout
        h_leaf_list = []

        # multiple message passing layers
        for attn_conv, gru_cell in zip(self.attn_convs, self.gru_cells):
            edge_masks_it = iter(data_batch.edge_masks) \
                if not self.reverse_flow else \
                iter(reversed(data_batch.edge_masks))

            h_prev = self.dropout(h_prev)

            # stores embeddings for all nodes at this layer
            h = torch.zeros_like(h_prev)

            # separately embed the root nodes, which have no incoming messages
            h[root_node_mask] = gru_cell(h_prev[root_node_mask])

            # pass messages to one topological generation of nodes at a time
            for edge_mask in edge_masks_it:
                # only include edges that connect the previous topo. gen.
                # to the current one
                edge_index_masked = data_batch.edge_index[:, edge_mask]
                edge_attr_masked = data_batch.edge_attr[edge_mask]

                msg = attn_conv(h, h_prev, edge_index_masked, edge_attr_masked)

                # embed only the current topological generation of nodes
                node_mask = pyg_utils.index_to_mask(edge_index_masked[self.i],
                                                    data_batch.num_nodes)
                h[node_mask] = gru_cell(h_prev[node_mask], msg[node_mask])

            h_leaf_list += [h[leaf_node_mask]]
            h_prev = h

        # readout for graph-level embeddings, only using leaf nodes
        h_graph = gnn.global_max_pool(torch.cat(h_leaf_list, 1),
                                      data_batch.batch[leaf_node_mask],
                                      size=data_batch.num_graphs)

        return h_graph


class DAGNN(nn.Module):
    def __init__(self, num_vocab, num_node_types, num_node_attr, max_seq_len,
                 max_depth, num_layers, emb_dim, p_dropout):
        super().__init__()

        self.num_vocab = num_vocab
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim

        # encodes dags using source-to-target message passing
        self.dag_encoder_s2t = DagEncoder(emb_dim, num_node_types,
                                          num_node_attr, num_layers, max_depth,
                                          p_dropout)

        # encodes dags using target-to-source message passing
        self.dag_encoder_t2s = DagEncoder(emb_dim, num_node_types,
                                          num_node_attr, num_layers, max_depth,
                                          p_dropout, reverse_flow=True)

        self.dropout = nn.Dropout(p=p_dropout, inplace=True)

        self.lins_graph_pred = torch.nn.ModuleList([
            torch.nn.Linear(2 * emb_dim * num_layers, self.num_vocab)
            for _ in range(self.max_seq_len)
        ])

    def forward(self, data_batch):
        h_graph_s2t = self.dag_encoder_s2t(data_batch)
        h_graph_t2s = self.dag_encoder_t2s(data_batch)
        h_graph = torch.cat([h_graph_s2t, h_graph_t2s], 1)
        h_graph = self.dropout(h_graph)
        preds = [lin(h_graph) for lin in self.lins_graph_pred]
        return preds


# Training and testing

vocab2idx, idx2vocab = get_vocab_mapping(
    [dataset.data.y[i] for i in split_idx['train']], NUM_VOCAB)
dataset.transform = transforms.Compose([
    augment_edge, make_edge_masks,
    lambda data: encode_y_to_arr(data, vocab2idx, MAX_SEQ_LEN)
])
train_loader = make_dataloader(dataset[split_idx['train']], shuffle=True)
valid_loader = make_dataloader(dataset[split_idx['valid']])
test_loader = make_dataloader(dataset[split_idx['test']])

model = DAGNN(len(vocab2idx), num_node_types, num_node_attr, MAX_SEQ_LEN,
              MAX_DEPTH, NUM_LAYERS, EMB_DIM, P_DROPOUT)
model.to(device)

opt = OPT_CLS(model.parameters(), lr=OPT_LR)


def train(epoch):
    model.train()

    losses = []
    for data_batch in tqdm(train_loader, desc=f'Epoch {epoch:02d}'):
        if data_batch.x.shape[0] == 1 or data_batch.batch[-1] == 0:
            continue

        data_batch.to(device)
        pred_list = model(data_batch)

        loss = torch.stack([
            F.cross_entropy(pred.float(), y)
            for pred, y in zip(pred_list, data_batch.y_arr.T)
        ]).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        opt.step()

        losses += [loss.item()]

    return np.mean(losses)


@torch.no_grad()
def test(loader, label):
    model.eval()

    seq_ref_list = []
    seq_pred_list = []
    for data_batch in tqdm(loader, desc=label):
        if data_batch.x.shape[0] == 1:
            continue

        data_batch.to(device)
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
    avg_loss = train(epoch)
    print('average loss:', avg_loss)

    print('Testing...')
    train_f1 = test(train_loader, label='train')
    val_f1 = test(valid_loader, label='valid')
    test_f1 = test(test_loader, label='test')
    print(f'F1 scores: '
          f'train={train_f1:.4f}, '
          f'val={val_f1:.4f}, '
          f'test={test_f1:.4f}')
