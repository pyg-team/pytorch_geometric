from __future__ import division, print_function

import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import ShapeNet  # noqa
from torch_geometric.transform import CartesianAdj  # noqa
from torch_geometric.utils import DataLoader  # noqa
from torch_geometric.nn.modules import SplineGCN, Lin  # noqa

path = '~/ShapeNet'
train_dataset = ShapeNet(
    path, train=True, category='cap', transform=CartesianAdj())
test_dataset = ShapeNet(
    path, train=False, category='cap', transform=CartesianAdj())

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
num_classes = 2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineGCN(1, 128, dim=3, kernel_size=5)
        self.conv2 = SplineGCN(128, 128, dim=3, kernel_size=5)
        self.conv3 = SplineGCN(128, 128, dim=3, kernel_size=5)
        self.conv4 = SplineGCN(128, 64, dim=3, kernel_size=5)
        self.conv5 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv6 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv7 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv8 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv9 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv10 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv11 = SplineGCN(64, 64, dim=3, kernel_size=5)
        self.conv12 = SplineGCN(64, 64, dim=3, kernel_size=5)

        self.lin1 = Lin(64, 128)
        self.lin2 = Lin(128, num_classes)

    def forward(self, adj, x):
        x = F.elu(self.conv1(adj, x))
        x = F.elu(self.conv2(adj, x))
        x = F.elu(self.conv3(adj, x))
        x = F.elu(self.conv4(adj, x))
        x = F.elu(self.conv5(adj, x))
        x = F.elu(self.conv6(adj, x))
        x = F.elu(self.conv7(adj, x))
        x = F.elu(self.conv8(adj, x))
        x = F.elu(self.conv9(adj, x))
        x = F.elu(self.conv10(adj, x))
        x = F.elu(self.conv11(adj, x))
        x = F.elu(self.conv12(adj, x))
        x = F.elu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x)


model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

        if epoch == 101:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001

    for batch, ((input, (adj, _), _), target) in enumerate(train_loader):
        # weight = target.numpy()
        # weight = np.bincount(weight)
        # weight = torch.LongTensor(weight).float()
        # weight = 1 / weight
        if torch.cuda.is_available():
            input, adj, target = input.cuda(), adj.cuda(), target.cuda()
            # weight = weight.cuda()

        input, target = Variable(input), Variable(target)

        optimizer.zero_grad()
        output = model(adj, input)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Epoch:', epoch, 'Batch:', batch, 'Loss:', loss.data[0])


def test(epoch):
    model.eval()

    iou = 0
    correct = 0

    for (input, (adj, _), _), target in test_loader:
        if torch.cuda.is_available():
            input, adj, target = input.cuda(), adj.cuda(), target.cuda()

        output = model(adj, Variable(input))
        pred = output.data.max(1)[1]
        correct += pred.eq(target).cpu().sum() / input.size(0)

        for i in range(num_classes):
            iou_single = 0
            p = pred == i
            t = target == i

            intersection = ((p + t) > 1).sum()
            union = p.sum() + t.sum() - intersection
            if union == 0:
                iou_single = 1
            else:
                iou_single += intersection / union
            # w = t.sum() / t.size(0)
            w = 1 / num_classes
            iou_single *= w

            iou += iou_single

    acc = correct / len(test_dataset)
    iou = iou / len(test_dataset)
    print('Epoch:', epoch, 'Accuracy:', acc, 'IoU:', iou)


for epoch in range(1, 401):
    train(epoch)
    test(epoch)
