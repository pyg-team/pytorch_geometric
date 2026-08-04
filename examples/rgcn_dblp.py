import copy
import os.path as osp

import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.datasets import DBLP
from torch_geometric.nn import RGCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DBLP')
dataset = DBLP(path)
graph = dataset[0]

num_classes = torch.max(graph['author'].y).item() + 1
graph['conference'].x = torch.randn((graph['conference'].num_nodes, 50))
graph = graph.to(device)
train_mask, val_mask, test_mask = graph['author'].train_mask, graph[
    'author'].val_mask, graph['author'].test_mask
y = graph['author'].y

node_types, edge_types = graph.metadata()
num_nodes = graph['author'].x.shape[0]
num_relations = len(edge_types)
init_sizes = [graph[x].x.shape[1] for x in node_types]
homogeneous_graph = graph.to_homogeneous()
in_feats, hidden_feats = 128, 64


class RGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels,
                              num_relations=num_relations, num_bases=30)
        self.conv2 = RGCNConv(hidden_channels, out_channels,
                              num_relations=num_relations, num_bases=30)
        self.lins = torch.nn.ModuleList()
        for i in range(len(node_types)):
            lin = nn.Linear(init_sizes[i], in_channels)
            self.lins.append(lin)

    def trans_dimensions(self, g):
        data = copy.deepcopy(g)
        for node_type, lin in zip(node_types, self.lins):
            data[node_type].x = lin(data[node_type].x)

        return data

    def forward(self, data):
        data = self.trans_dimensions(data)
        homogeneous_data = data.to_homogeneous()
        edge_index, edge_type = homogeneous_data.edge_index, homogeneous_data.edge_type
        x = self.conv1(homogeneous_data.x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        x = x[:num_nodes]
        x = F.softmax(x, dim=1)
        return x


def train():
    model = RGCN(in_feats, hidden_feats, num_classes).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    min_epochs = 5
    best_val_acc = 0
    final_best_acc = 0
    model.train()
    for epoch in range(100):
        f = model(graph)
        loss = loss_function(f[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # validation
        val_acc, val_loss = test(model, val_mask)
        test_acc, test_loss = test(model, test_mask)
        if epoch + 1 > min_epochs and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_best_acc = test_acc
        print('Epoch{:3d} train_loss {:.5f} val_acc {:.3f} test_acc {:.3f}'.
              format(epoch, loss.item(), val_acc, test_acc))

    return final_best_acc


@torch.no_grad()
def test(model, mask):
    model.eval()
    out = model(graph)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    loss = loss_function(out[mask], y[mask])
    _, pred = out.max(dim=1)
    correct = int(pred[mask].eq(y[mask]).sum().item())
    acc = correct / int(test_mask.sum())

    return acc, loss.item()


def main():
    final_best_acc = train()
    print('RGCN Accuracy:', final_best_acc)


if __name__ == '__main__':
    main()
