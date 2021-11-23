import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import DynamicEdgeConv, global_max_pool

from points.datasets import get_dataset
from points.train_eval import run

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--weight_decay', type=float, default=0)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        nn = Seq(Lin(6, 64), ReLU(), Lin(64, 64), ReLU(), Lin(64, 64), ReLU())
        self.conv1 = DynamicEdgeConv(nn, k=20, aggr='max')

        nn = Seq(Lin(128, 128), ReLU(), Lin(128, 128), ReLU(), Lin(128, 256),
                 ReLU())
        self.conv2 = DynamicEdgeConv(nn, k=20, aggr='max')

        self.lin0 = Lin(256, 512)

        self.lin1 = Lin(512, 256)
        self.lin2 = Lin(256, 256)
        self.lin3 = Lin(256, num_classes)

    def forward(self, pos, batch):
        x = self.conv1(pos, batch)
        x = self.conv2(x, batch)

        x = F.relu(self.lin0(x))

        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)


train_dataset, test_dataset = get_dataset(num_points=1024)
model = Net(train_dataset.num_classes)
run(train_dataset, test_dataset, model, args.epochs, args.batch_size, args.lr,
    args.lr_decay_factor, args.lr_decay_step_size, args.weight_decay)
