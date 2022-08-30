import argparse

import torch
import torch.nn.functional as F
from points.datasets import get_dataset
from points.train_eval import run
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

from torch_geometric.nn import NNConv, fps, global_mean_pool, radius_graph
from torch_geometric.profile import rename_profile_file

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--bf16', action='store_true')
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        nn = Seq(Lin(3, 25), ReLU(), Lin(25, 1 * 64))
        self.conv1 = NNConv(1, 64, nn, aggr='mean')

        nn = Seq(Lin(3, 25), ReLU(), Lin(25, 64 * 64))
        self.conv2 = NNConv(64, 64, nn, aggr='mean')

        nn = Seq(Lin(3, 25), ReLU(), Lin(25, 64 * 128))
        self.conv3 = NNConv(64, 128, nn, aggr='mean')

        self.lin1 = torch.nn.Linear(128, 256)
        self.lin2 = torch.nn.Linear(256, 256)
        self.lin3 = torch.nn.Linear(256, num_classes)

    def forward(self, pos, batch):
        x = pos.new_ones((pos.size(0), 1))

        radius = 0.2
        edge_index = radius_graph(pos, r=radius, batch=batch)
        pseudo = pos[edge_index[1]] - pos[edge_index[0]]
        x = F.relu(self.conv1(x, edge_index, pseudo))

        idx = fps(pos, batch, ratio=0.5)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 0.4
        edge_index = radius_graph(pos, r=radius, batch=batch)
        pseudo = pos[edge_index[1]] - pos[edge_index[0]]
        x = F.relu(self.conv2(x, edge_index, pseudo))

        idx = fps(pos, batch, ratio=0.25)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        pseudo = pos[edge_index[1]] - pos[edge_index[0]]
        x = F.relu(self.conv3(x, edge_index, pseudo))

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


train_dataset, test_dataset = get_dataset(num_points=1024)
model = Net(train_dataset.num_classes)
run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
    args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay,
    args.inference, args.profile, args.bf16)

if args.profile:
    rename_profile_file('points', NNConv.__name__)
