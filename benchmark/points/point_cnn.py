import argparse

import torch
import torch.nn.functional as F
from points.datasets import get_dataset
from points.train_eval import run
from torch.nn import Linear as Lin

from torch_geometric.nn import XConv, fps, global_mean_pool
from torch_geometric.profile import rename_profile_file

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(48, 96, dim=3, kernel_size=12, hidden_channels=64,
                           dilation=2)
        self.conv3 = XConv(96, 192, dim=3, kernel_size=16, hidden_channels=128,
                           dilation=2)
        self.conv4 = XConv(192, 384, dim=3, kernel_size=16,
                           hidden_channels=256, dilation=2)

        self.lin1 = Lin(384, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, num_classes)

    def forward(self, pos, batch):
        x = F.relu(self.conv1(None, pos, batch))

        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv2(x, pos, batch))

        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))

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
    args.inference, args.profile)

if args.profile:
    rename_profile_file('points', XConv.__name__)
