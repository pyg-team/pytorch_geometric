import os
import sys
import random

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ShapeNet2  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.transform import NormalizeScale, CartesianAdj  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import sparse_voxel_max_pool  # noqa
from torch_geometric.visualization.model import show_model  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ShapeNet2')

category = 'Pistol'

transform = CartesianAdj()
train_transform = Compose([NormalizeScale(), transform])
test_transform = Compose([NormalizeScale(), transform])
train_dataset = ShapeNet2(path, category, 'train', train_transform)
val_dataset = ShapeNet2(path, category, 'val', test_transform)
test_dataset = ShapeNet2(path, category, 'test', test_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

min_label = train_dataset.set.target.min()
max_label = train_dataset.set.target.max()
num_classes = max_label - min_label + 1


def upscale(input, cluster):
    cluster = Variable(cluster)
    cluster = cluster.view(-1, 1).expand(cluster.size(0), input.size(1))
    return torch.gather(input, 0, cluster)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.conv2 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)

        self.conv4 = SplineConv(64, 32, dim=3, kernel_size=5)
        self.conv5 = SplineConv(64, 32, dim=3, kernel_size=5)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(64, num_classes)

        self.skip1 = nn.Linear(64, 32)
        self.skip2 = nn.Linear(64, 32)
        self.skip3 = nn.Linear(64, 32)

    def pool_args(self, mean, x):
        if not self.training:
            return 1 / mean, 0
        size = 1 / random.uniform(mean - x, mean + x)
        start = random.uniform(-1 / (mean - x), 0)
        return size, start

    def forward(self, data1):
        data1.input = F.elu(self.conv1(data1.adj, data1.input))
        size, start = self.pool_args(32, 24)
        data2, c1 = sparse_voxel_max_pool(data1, size, start, transform)

        data2.input = F.elu(self.conv2(data2.adj, data2.input))
        size, start = self.pool_args(8, 3)
        data3, c2 = sparse_voxel_max_pool(data2, size, start, transform)

        data3.input = F.elu(self.conv2(data3.adj, data3.input))
        size, start = self.pool_args(4, 2)
        data4, c3 = sparse_voxel_max_pool(data3, size, start, transform)

        data4.input = F.dropout(data4.input, training=self.training)
        data4.input = F.elu(self.fc1(data4.input))

        x = [upscale(data4.input, c3), self.skip3(data3.input)]
        data3.input = F.elu(self.conv4(data3.adj, torch.cat(x, dim=1)))

        x = [upscale(data3.input, c2), self.skip2(data2.input)]
        data2.input = F.elu(self.conv5(data2.adj, torch.cat(x, dim=1)))

        x = [upscale(data2.input, c1), self.skip1(data1.input)]
        data1.input = F.elu(self.conv6(data1.adj, torch.cat(x, dim=1)))

        data1.input = F.dropout(data1.input, training=self.training)
        data1.input = self.fc2(data1.input)

        return F.log_softmax(data1.input, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    if epoch == 31:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        F.nll_loss(model(data), data.target - min_label).backward()
        optimizer.step()


def test(epoch, loader, dataset, str):
    model.eval()
    correct = 0
    iou = 0
    num_runs = 0

    for data in loader:
        data = data.cuda().to_variable(['input'])
        target = data.target - min_label
        pred = model(data).data.max(1)[1]
        correct += pred.eq(target).sum()

        iou_single = 0
        num_runs += 1
        batch_size = data.batch[-1] + 1
        for i in range(num_classes):
            p = pred == i
            t = target == i
            intersection = ((p + t) > 1).sum()
            union = p.sum() + t.sum() - intersection

            if union == 0:
                iou_single += 1
            else:
                iou_single += intersection / union

        iou += batch_size * (iou_single / num_classes)

    acc = correct / dataset.set.num_nodes
    iou = iou / len(dataset)

    print('Epoch:', epoch, str, 'Accuracy:', acc, 'IoU:', iou)


for epoch in range(1, 61):
    train()
    test(epoch, train_loader, train_dataset, 'Train')
    test(epoch, val_loader, val_dataset, 'Val')
    test(epoch, test_loader, test_dataset, 'Test')
