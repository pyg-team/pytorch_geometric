import os
import sys

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ModelNet10  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import (RandomRotate, RandomScale,
                                        RandomTranslate, RandomShear)  # noqa
from torch_geometric.transform import CartesianAdj  # noqa
from torch_geometric.nn.modules import SplineConv  # noqa
from torch_geometric.nn.functional import voxel_avg_pool, voxel_max_pool  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ModelNet10')

train_transform = Compose([
        #RandomRotate([0, 0, 2]),
        #RandomScale(1.1),
        #RandomShear(0.5),
        #RandomTranslate(0.005),
    CartesianAdj()
])
test_transform = CartesianAdj()
train_dataset = ModelNet10(path, True, transform=train_transform)
test_dataset = ModelNet10(path, False, transform=test_transform)
train_loader = DataLoader2(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=1)
num_classes=10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv12 = SplineConv(32, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv5 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv7 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv8 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, data):
        max, min = torch.max(data.pos, 0)[0], torch.min(data.pos,0)[0]
        pool1_cells = ((max-min)/64).cpu().tolist()
        pool2_cells = ((max-min)/32).cpu().tolist()
        pool3_cells = ((max-min)/16).cpu().tolist()
        pool4_cells = ((max-min)/8).cpu().tolist()
        #pool1_cells = torch.mean(pool1_cells)
        #pool2_cells = torch.mean(pool2_cells)
        #pool3_cells = torch.mean(pool3_cells)
        #pool4_cells = torch.mean(pool4_cells)
        #print('Num nodes', data.num_nodes)
        data.input = F.elu(self.conv1(data.adj, data.input))
        data.input = F.elu(self.conv12(data.adj, data.input))
        data, _ = voxel_max_pool(data,pool1_cells, test_transform)
        #print('FirstPool:', data.num_nodes)
        data.input = F.elu(self.conv2(data.adj, data.input))
        data.input = F.elu(self.conv3(data.adj, data.input))
        data, _ = voxel_max_pool(data, pool2_cells, test_transform)
        #print('SecondPool:', data.num_nodes)
        data.input = F.elu(self.conv4(data.adj, data.input))
        data.input = F.elu(self.conv5(data.adj, data.input))
        data, _ = voxel_max_pool(data, pool3_cells, test_transform)
        #print('ThirdPool:', data.num_nodes)
        data.input = F.elu(self.conv6(data.adj, data.input))
        data.input = F.elu(self.conv7(data.adj, data.input))
        data, _ = voxel_max_pool(data, pool4_cells, test_transform)
        #print('FourthPool:', data.num_nodes)
        #print(clusters.size())

        #data.batch = Variable(data.batch.view(-1, 1).expand(data.input.size()))
        #x = scatter_mean(data.batch, data.input)
        x = data.input.mean(0).view(1,-1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    if epoch == 21:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    example = 0
    loss_accum = 0
    loss_count = 0
    for i,data in enumerate(train_loader):
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.target)
        loss_accum += loss.data[0]
        loss_count += 1
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Example',example, 'of',len(train_dataset),'. Loss:', loss_accum/loss_count)
            loss_accum = 0
            loss_count = 0
        example +=1


def test(epoch,loader, ds):
    model.eval()
    correct = 0

    for data in loader:
        data = data.cuda().to_variable(['input'])
        output = model(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(data.target).cpu().sum()
    print(correct)
    print('Epoch:', epoch, 'Test Accuracy:', correct / len(ds))


for epoch in range(1, 31):
    train(epoch)
    test(epoch,train_loader, train_dataset)
    test(epoch,test_loader, test_dataset)

