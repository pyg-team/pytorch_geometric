import os
import sys

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ShapeNet2  # noqa
from torch_geometric.utils import DataLoader2  # noqa
from torch_geometric.transform import CartesianAdj, CartesianLocalAdj, \
    NormalizeScale, RandomRotate, RandomTranslate # noqa
from torch_geometric.nn.modules import SplineConv, Lin  # noqa
from torch_geometric.nn.functional import voxel_max_pool  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'ShapeNet2')

category = 'Pistol'

cell_sizes = {'Airplane': (0.05, 0.08, 0.12),
              'Bag': (0.05, 0.08, 0.12),
              'Cap': (0.05, 0.08, 0.12),
              'Car': (0.05, 0.08, 0.12),
              'Chair': (0.03, 0.05, 0.07),
              'Earphone': (0.03, 0.05, 0.7),
              'Guitar': (0.05, 0.08, 0.12),
              'Knife': (0.03, 0.05, 0.07),
              'Lamp': (0.05, 0.08, 0.12),
              'Laptop': (0.05, 0.08, 0.12),
              'Motorbike': (0.05, 0.08, 0.12),
              'Mug': (0.05, 0.08, 0.12),
              'Pistol': (0.05, 0.08, 0.12),
              'Rocket': (0.015, 0.03, 0.06),
              'Skateboard': (0.05, 0.08, 0.12),
              'Table': (0.05, 0.08, 0.12)}

trans1 = Compose([
    #RandomRotate(3.14),
    #    RandomScale(1.1),
    #    RandomShear(0.5),
    #NormalizeScale(),
    #RandomTranslate(0.005),
    NormalizeScale(),
    CartesianLocalAdj()
])

trans2 = CartesianLocalAdj()
train_dataset = ShapeNet2(path, category, 'train', trans1)
val_dataset = ShapeNet2(path, category, 'val', trans1)
test_dataset = ShapeNet2(path, category, 'test', trans1)
train_loader = DataLoader2(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader2(val_dataset, batch_size=32)
test_loader = DataLoader2(test_dataset, batch_size=32)
test_loader_visualize = DataLoader2(test_dataset, batch_size=1)
train_loader_visualize = DataLoader2(train_dataset, batch_size=1)

labelmin = train_dataset.set.target.min()
labelmax = train_dataset.set.target.max()
unique_labels = np.unique(train_dataset.set.target.numpy(), return_counts=True)

num_classes = labelmax - labelmin + 1

print(labelmin, labelmax, num_classes)
print(unique_labels)

print('Length of train set:', len(train_dataset))
print('Length of val set:  ', len(val_dataset))
print('Length of test set: ', len(test_dataset))


def upscale(input, cluster):
    cluster = Variable(cluster)
    cluster = cluster.view(-1, 1).expand(cluster.size(0), input.size(1))
    return torch.gather(input, 0, cluster)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 64, dim=3, kernel_size=5)
        self.conv12 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv2 = SplineConv(65, 64, dim=3, kernel_size=5)
        self.conv22 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(65, 64, dim=3, kernel_size=5)
        self.conv32 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(65, 64, dim=3, kernel_size=5)
        self.conv42 = SplineConv(64, 64, dim=3, kernel_size=5)
        # self.conv42 = SplineConv(32, 32, dim=3, kernel_size=5)
        self.fc1 = Lin(64, 32)
        self.conv5 = SplineConv(32, 32, dim=3, kernel_size=5)
        #self.conv52 = SplineConv(32, 32, dim=3, kernel_size=5)
        self.conv6 = SplineConv(32, 32, dim=3, kernel_size=5)
        #self.conv62 = SplineConv(32, 32, dim=3, kernel_size=5)
        self.conv7 = SplineConv(32, 32, dim=3, kernel_size=5)
        #self.conv72 = SplineConv(32, 32, dim=3, kernel_size=5)
        self.fc2 = Lin(32, num_classes)
        self.skip1 = Lin(64, 32)
        self.skip2 = Lin(64, 32)
        self.skip3 = Lin(64, 32)


    def forward(self, data1):
        one = Variable(data1.input.data.new(data1.input.size(0), 1).fill_(1))
        data1.input = F.elu(self.conv1(data1.adj, data1.input))
        #data1.input = self.bn1(data1.input)
        data1.input = F.elu(self.conv12(data1.adj, data1.input))
        data2, cluster1 = voxel_max_pool(data1, (1/64), origin=0,
                                         transform=trans2)

        data2.input = torch.cat([data2.input, one[:data2.input.size(0)]], dim=1)
        data2.input = F.elu(self.conv2(data2.adj, data2.input))
        #data2.input = self.bn2(data2.input)
        data2.input = F.elu(self.conv22(data2.adj, data2.input))
        data3, cluster2 = voxel_max_pool(data2, (1/32), origin=0,
                                         transform=trans2)

        data3.input = torch.cat([data3.input, one[:data3.input.size(0)]], dim=1)
        data3.input = F.elu(self.conv3(data3.adj, data3.input))
        #data3.input = self.bn3(data3.input)
        data3.input = F.elu(self.conv32(data3.adj, data3.input))
        data4, cluster3 = voxel_max_pool(data3, (1/16), origin=0,
                                         transform=trans2)

        data4.input = torch.cat([data4.input, one[:data4.input.size(0)]], dim=1)
        data4.input = F.elu(self.conv4(data4.adj, data4.input))
        #data4.input = self.bn4(data4.input)
        data4.input = F.elu(self.conv42(data4.adj, data4.input))
        data4.input = F.elu(self.conv42(data4.adj, data4.input))
        data4.input = F.dropout(data4.input, training=self.training)
        data4.input = F.elu(self.fc1(data4.input))

        data3.input = upscale(data4.input, cluster3) + self.skip3(data3.input)
        #data3.input = torch.cat(
        #    [upscale(data4.input, cluster3), self.skip3(data3.input)], dim=1)
        data3.input = F.elu(self.conv5(data3.adj, data3.input))
        #data3.input = F.elu(self.conv52(data3.adj, data3.input))

        data2.input = upscale(data3.input, cluster2) + self.skip2(data2.input)
        #data2.input = torch.cat(
        #    [upscale(data3.input, cluster2), self.skip2(data2.input)], dim=1)
        data2.input = F.elu(self.conv6(data2.adj, data2.input))
        #data2.input = F.elu(self.conv62(data2.adj, data2.input))

        data1.input = upscale(data2.input, cluster1) + self.skip1(data1.input)
        #data1.input = torch.cat(
        #    [upscale(data2.input, cluster1), self.skip1(data1.input)], dim=1)
        #data2.input = F.elu(self.conv6(data2.adj, data2.input))
        #data2.input = F.elu(self.conv62(data2.adj, data2.input))
        data1.input = F.elu(self.conv7(data1.adj, data1.input))
        #data1.input = F.elu(self.conv72(data1.adj, data1.input))

        data1.input = self.fc2(data1.input)

        return F.log_softmax(data1.input, dim=1)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 61:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 121:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001


    for data in train_loader:
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        output = model(data)
        data.target.data -= labelmin
        loss = F.nll_loss(output, data.target)
        loss.backward()
        optimizer.step()


def test(epoch, loader, print_str):
    model.eval()
    correct = 0
    iou = 0
    num_examples = 0
    num_nodes = 0
    for data in loader:
        data = data.cuda().to_variable()
        output = model(data)
        data.target.data -= labelmin
        pred = output.data.max(1)[1]
        correct += pred.eq(data.target.data).cpu().sum()
        iou_single = 0
        num_labels = 0
        for i in range(num_classes):
            p = pred == i
            t = data.target.data == i

            intersection = ((p + t) > 1).sum()
            union = p.sum() + t.sum() - intersection
            if union == 0:
                iou_single += 1
            else:
                iou_single += intersection / union
            num_labels += 1

        iou += iou_single / num_labels
        num_examples += 1
        num_nodes += data.num_nodes
    acc = correct / num_nodes
    iou = iou / num_examples

    print(print_str, 'Epoch:', epoch, 'Accuracy:', acc, 'IoU:', iou)
    return iou


def show_pointclouds():
    model.eval()
    for data in test_loader_visualize:
        data = data.cuda().to_variable()
        output = model(data)
        pred = output.data.max(1)[1]
        print(data.pos)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax_gt = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2], c=pred,
                   alpha=1, )
        ax.axis('equal')
        ax_gt.scatter(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2],
                      c=data.target.data, alpha=1, )
        ax_gt.axis('equal')
        plt.show()

    for data in train_loader_visualize:
        data = data.cuda().to_variable()
        output = model(data)
        pred = output.data.max(1)[1]
        print(data.pos)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax_gt = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2], c=pred,
                   alpha=1, )
        ax.axis('equal')
        ax_gt.scatter(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2],
                      c=data.target.data, alpha=1, )
        ax_gt.axis('equal')
        plt.show()



for epoch in range(1, 101):
    train(epoch)
    best_val_iou = 0
    best_model_test_iou = 0
    print('----------------------------')
    test(epoch, train_loader, 'Train --')
    val_iou = test(epoch, val_loader, 'Val   --')
    test_iou = test(epoch, test_loader, 'Test  --')
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_model_test_iou = test_iou
    print('Current best model - test IoU:', best_model_test_iou)



show_pointclouds()
