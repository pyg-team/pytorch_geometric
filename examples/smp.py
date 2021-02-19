import torch
import os.path as osp

import torch.nn.functional as F
from torch_geometric.nn import SMPConv
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader


def to_one_hot(data):
    data.x = F.one_hot(data.x.squeeze(-1), num_classes=21).float()
    data.edge_attr = F.one_hot(data.edge_attr - 1, num_classes=3).float()
    return data


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
train_dataset = ZINC(path, subset=True, split='train', transform=to_one_hot)
val_dataset = ZINC(path, subset=True, split='val', transform=to_one_hot)
test_dataset = ZINC(path, subset=True, split='test', transform=to_one_hot)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class SMP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 edge_dim, towers, pre_layers, post_layers):
        super(SMP, self).__init__()

        self.lin = Linear(in_channels + 1, hidden_channels)

        self.conv = SMPConv(hidden_channels, hidden_channels, edge_dim,
                            towers=1)
        print(self.conv)

    def forward(self, x, edge_index, edge_attr, batch):
        x, mask = self.conv.local_context(x, batch)
        x = self.conv(x, edge_index, edge_attr).relu_()
        raise NotImplementedError


model = SMP(train_dataset.num_features, hidden_channels=32, out_channels=1,
            num_layers=12, towers=8, edge_dim=3, pre_layers=2, post_layers=2)

for data in train_loader:
    model(data.x, data.edge_index, data.edge_attr, data.batch)
    break
