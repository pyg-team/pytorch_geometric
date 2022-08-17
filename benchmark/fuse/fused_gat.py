import argparse
import os.path as osp
from collections import defaultdict

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
# the following code requires the installation of dgNN from https://github.com/dgSPARSE/dgNN
from torch_geometric.nn.conv.fused_gat_conv import FusedGATConv
from torch_geometric.utils import to_dgnn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pubmed')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--heads', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()


class dgNN_GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = FusedGATConv(in_channels, hidden_channels, heads,
                                  dropout=0.6, add_self_loops=False)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = FusedGATConv(hidden_channels * heads, out_channels,
                                  heads=1, concat=False, dropout=0.6,
                                  add_self_loops=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6,
                             add_self_loops=False)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6, add_self_loops=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# test with Planetoid dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data',
                'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())

# define data and model
pyg_data = dataset[0].to(device)
dgNN_data = dataset[0].to(device)
dgNN_data.edge_index = to_dgnn(
    dgNN_data.edge_index)  # add one line of data conversion to dgNN

pyg_model = GAT(dataset.num_features, args.hidden_channels,
                dataset.num_classes, args.heads).to(device)
dgNN_model = dgNN_GAT(dataset.num_features, args.hidden_channels,
                      dataset.num_classes, args.heads).to(device)

benchmark_info = defaultdict(dict)
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

# train and test model
for data, model, name in zip([pyg_data, dgNN_data], [pyg_model, dgNN_model],
                             ['pyg', 'dgNN']):
    print('{} train on dataset: {}'.format(name, dataset.name))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005,
                                 weight_decay=5e-4)
    best_val_acc = final_test_acc = 0
    # record training time
    start_time.record()
    for epoch in range(1, args.epochs + 1):
        loss = train(model, optimizer, data)
        # train_acc, val_acc, tmp_test_acc = test(model, data)
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     test_acc = tmp_test_acc
        # log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    end_time.record()
    torch.cuda.synchronize()

    benchmark_info[name]['train_time'] = start_time.elapsed_time(end_time)

    # benchmark_info[name]['train_acc'] = train_acc
    # benchmark_info[name]['val_acc'] = val_acc
    # benchmark_info[name]['test_acc'] = test_acc

print('Benchmark info:')
print(benchmark_info)
