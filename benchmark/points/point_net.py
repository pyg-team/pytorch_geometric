import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import PointConv, radius_graph, fps, global_max_pool

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

        nn = Seq(Lin(3, 64), ReLU(), Lin(64, 64))
        self.conv1 = PointConv(local_nn=nn)

        nn = Seq(Lin(67, 128), ReLU(), Lin(128, 128))
        self.conv2 = PointConv(local_nn=nn)

        nn = Seq(Lin(131, 256), ReLU(), Lin(256, 256))
        self.conv3 = PointConv(local_nn=nn)

        self.lin1 = Lin(256, 256)
        self.lin2 = Lin(256, 256)
        self.lin3 = Lin(256, num_classes)

    def forward(self, pos, batch):
        radius = 0.2
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv1(None, pos, edge_index))

        idx = fps(pos, batch, ratio=0.5)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 0.4
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv2(x, pos, edge_index))

        idx = fps(pos, batch, ratio=0.25)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        radius = 1
        edge_index = radius_graph(pos, r=radius, batch=batch)
        x = F.relu(self.conv3(x, pos, edge_index))

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
