import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import train_test_split_edges

torch.manual_seed(12345)

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

# Train/validation/test
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, pos_edge_index, neg_edge_index):

        x = F.relu(self.conv1(data.x, pos_edge_index))
        x = self.conv2(x, pos_edge_index)

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()
    optimizer.zero_grad()

    x, pos_edge_index = data.x, data.train_pos_edge_index

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                       num_nodes=x.size(0))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
        num_neg_samples=pos_edge_index.size(1))

    link_logits = model(pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]
        link_probs = torch.sigmoid(model(pos_edge_index, neg_edge_index))
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        link_probs = link_probs.detach().cpu().numpy()
        link_labels = link_labels.detach().cpu().numpy()
        perfs.append(roc_auc_score(link_labels, link_probs))
    return perfs


best_val_perf = test_perf = 0
for epoch in range(1, 501):
    train_loss = train()
    val_perf, tmp_test_perf = test()
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_perf, test_perf))
