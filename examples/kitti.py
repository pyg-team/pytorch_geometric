import os
import sys
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import KITTI  # noqa
from torch_geometric.utils import DataLoader2  # noqa

from torch_geometric.transform import (RandomRotate, RandomScale,
                                        RandomTranslate, RandomShear,
                                       NormalizeScale)  # noqa
from torch_geometric.transform import CartesianAdj, CartesianLocalAdj, \
    SphericalAdj, Spherical2dAdj  # noqa
from torch_geometric.nn.modules import SplineConv, Lin  # noqa
from torch_geometric.nn.functional import voxel_max_pool  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'KITTI')

batch_size = 1
num_classes = 4

transform = Compose([
    #RandomRotate(3.14),
    #RandomScale(1.1),
    #RandomShear(0.5),
    #RandomTranslate(0.005),
    NormalizeScale(),
    CartesianAdj()
])
train_dataset = KITTI(path, True, transform=transform)
test_dataset = KITTI(path, False, transform=transform)
train_loader = DataLoader2(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader2(test_dataset, batch_size=batch_size)



trans = CartesianAdj()


def upscale(input, cluster):
    cluster = Variable(cluster)
    cluster = cluster.view(-1, 1).expand(cluster.size(0), input.size(1))
    return torch.gather(input, 0, cluster)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5)
        self.conv2 = SplineConv(33, 64, dim=3, kernel_size=5)
        self.conv22 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv3 = SplineConv(65, 64, dim=3, kernel_size=5)
        self.conv32 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv4 = SplineConv(65, 64, dim=3, kernel_size=5)
        self.conv42 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.conv5 = SplineConv(65, 96, dim=3, kernel_size=5)
        self.conv52 = SplineConv(96, 96, dim=3, kernel_size=5)
        self.conv6 = SplineConv(97, 128, dim=3, kernel_size=5)
        self.conv62 = SplineConv(128, 128, dim=3, kernel_size=5)
        self.convRPN1 = SplineConv(65, 64, dim=3, kernel_size=5)
        self.convRPN2 = SplineConv(64, 64, dim=3, kernel_size=5)
        self.convRPN3 = SplineConv(64, 64, dim=3, kernel_size=5)

        self.fc1 = Lin(128,64)
        self.fc2 = Lin(96,64)

        self.skip1 = Lin(32,64)
        self.skip2 = Lin(64,96)
        self.skip2 = Lin(96,128)

    def forward(self, data):
        ones = Variable(data.input.data.new(data.input.size(0), 1).fill_(1))
        data.input = F.elu(self.conv1(data.adj, data.input))
        data1, cluster1 = voxel_max_pool(data, (1 / 64), origin=0, transform=trans)
        x = torch.cat([data1.input, ones[:data1.input.size(0)]], dim=1)
        x = F.elu(self.conv2(data1.adj, x))
        x = self.conv22(data1.adj, x)
        data1.input = F.elu(x + self.skip1(data1.input))
        data2, cluster2 = voxel_max_pool(data1, (1 / 32), origin=0, transform=trans)
        x = torch.cat((data2.input, ones[:data2.input.size(0), :]), dim=1)
        x = F.elu(self.conv3(data2.adj, x))
        x = self.conv32(data2.adj, x)
        data2.input = F.elu(x + data2.input)
        data3, cluster3 = voxel_max_pool(data2, (1 / 16), origin=0, transform=trans)
        x = torch.cat([data3.input, ones[:data3.input.size(0)]], dim=1)
        x = F.elu(self.conv4(data3.adj, x))
        x = self.conv42(data3.adj, x)
        data3.input = F.elu(x + data3.input)
        data4, cluster4 = voxel_max_pool(data3, (1 / 8), origin=0, transform=trans)
        x = torch.cat([data4.input, ones[:data4.input.size(0)]], dim=1)
        x = F.elu(self.conv5(data4.adj, x))
        x = self.conv52(data4.adj, x)
        data4.input = F.elu(x + self.skip2(data4.input))

        data5, cluster5 = voxel_max_pool(data4, (1 / 4), origin=0,
                                         transform=trans)
        x = torch.cat([data5.input, ones[:data5.input.size(0)]], dim=1)
        x = F.elu(self.conv6(data5.adj, x))
        x = self.conv62(data5.adj, x)
        data5.input = F.elu(x + self.skip3(data5.input))

        data5.input = F.dropout(data5.input, training=self.training)
        data5.input = self.fc1(data5.input)

        data5.input = torch.cat(
            [upscale(upscale(data5.input, cluster4),cluster3),
             upscale(data4.input, cluster3),
             self.skip3(data3.input)], dim=1)

        data5.input = F.elu(self.convRPN1(data3.adj, data5.input))
        data5.input = F.elu(self.convRPN2(data3.adj, data5.input))
        data5.input = self.convRPN3(data3.adj, data5.input)

        return F.log_softmax(data5.input[:,:2], dim=1), data5.input[:,2:], data5.pos




model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    #if epoch == 21:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.0001
    example = 0
    for i,data in enumerate(train_loader):
        data = data.cuda().to_variable()
        optimizer.zero_grad()
        class_out, bb_out, positions = model(data)

        # Compute Groundtruth
        bb_gt = torch.cuda.FloatTensor(positions.size(0),7).zero_()
        mask = torch.cuda.IntTensor(positions.size(0),7).zero_()
        class_gt = torch.cuda.FloatTensor(positions.size(0),num_classes).zero_()
        class_gt[:,0] = 1.0

        for i in range(data.target.size(0)):
            # type, alpha, dimensions (h, w, d), location (x, y, z), rotation_y
            target = data.target[i,:]
            pos_target = target[i,5:8]
            anchor_pos = ((positions - pos_target)**2).sum(dim=1).min()[1]
            if class_gt[anchor_pos, 0] == 0.0:
                print('Error: Double Anchor')
            class_gt[anchor_pos, 0] = 0.0
            class_gt[anchor_pos, target[0]] = 1.0
            mask[anchor_pos] = 1
            bb_gt[anchor_pos,:] = target[2:9]
            bb_gt[anchor_pos, 5:8] /= data.scaled_by
            bb_gt[anchor_pos,5:8] -= positions[anchor_pos]

        loss_class = F.nll_loss(class_out, data.target)
        loss_bb = F.mse_loss(bb_out, data.bb,reduce=False)
        loss_bb.data = loss_bb.data * mask.unsqueeze(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb
        loss.backward()
        optimizer.step()
        print('Example',example, 'of',len(train_dataset), '. Loss:',
              loss.data[0], 'ce_loss:', loss_class, 'mse_loss:',loss_bb)
        example += batch_size


def test(epoch, loader, ds, string):
    model.eval()
    correct = 0

    for data in loader:
        data = data.cuda().to_variable(['input'])
        class_out, bb_out, positions = model(data)
        pred = class_out.data.max(1)[1]
        indices = torch.nonzero(pred)

        #print(pred.size(),data.target.size())
        correct += pred.eq(data.target).cpu().sum()
    print(correct)
    print('Epoch:', epoch, string, 'Accuracy:', correct / len(ds))


for epoch in range(1, 31):
    train(epoch)
    test(epoch,train_loader, train_dataset, 'Train')
    test(epoch,test_loader, test_dataset, 'Test')
