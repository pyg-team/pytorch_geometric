import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import from_scipy_sparse_matrix

import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'ota'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
print(path)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]

with open(osp.join(path,'processed_data.p'), 'rb') as fp:
    all_inputs = pickle.load(fp)

if 'circuit_graph' in locals():
    del circuit_graph

for circuit_name, circuit_data in all_inputs.items():
    df = circuit_data["data_matrix"]
    print(circuit_name)
    node_features = df.values
    node_features = np.delete(node_features, 0, 1)
    node_features = np.array(node_features, dtype=np.float32)
    node_features = node_features[:, 0:16]
    x = torch.Tensor(node_features)
    y = torch.Tensor(circuit_data["target"])
    adj = circuit_data["adjacency_matrix"]
    # print(adj.todense())
    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    print(edge_index,edge_weight)
    exit()
    if 'circuit_graph' in locals() and 'X' in locals():
        X = np.concatenate((X, node_features), axis=0)
        label = circuit_data["target"].reshape((-1, 1))
        y = np.concatenate((y, label), axis=0)
        igraph = circuit_data["adjacency_matrix"]
        circuit_graph = block_diag((circuit_graph, igraph)).tocsr()
    else:
        X = node_features
        y = circuit_data["target"].reshape((-1, 1))
        circuit_graph = circuit_data["adjacency_matrix"]


if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
