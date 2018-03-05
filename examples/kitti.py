import os
import sys
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose


import time
sys.path.insert(0, '.')
sys.path.insert(0, '..')


from kitti_utils import camera_to_lidar_box, show_pointclouds, \
    compute_anchor_gt, mean_average_precision  # noqa
from torch_geometric.datasets import KITTI  # noqa
from torch_geometric.utils import DataLoader2  # noqa

from torch_geometric.transform import (RandomRotate, RandomScale,
                                        RandomTranslate, RandomShear,
                                       NormalizeScale)  # noqa
from torch_geometric.transform import CartesianAdj, CartesianLocalAdj, \
    SphericalAdj, Spherical2dAdj  # noqa
from torch_geometric.nn.modules import SplineConv, Lin  # noqa
from torch_geometric.nn.functional import sparse_voxel_max_pool  # noqa

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '..', 'data', 'KITTI')

batch_size = 1
num_classes = 2
num_anchors = 2

transform = Compose([
    #RandomRotate(3.14),
    #RandomScale(1.1),
    #RandomShear(0.5),
    #RandomTranslate(0.005),
    NormalizeScale(),
    CartesianAdj(0.05)
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
        self.convRPN1 = SplineConv(160, 64, dim=3, kernel_size=5)
        self.convRPN2 = SplineConv(128, 64, dim=3, kernel_size=5)
        self.convRPN3 = SplineConv(128, 64, dim=3, kernel_size=5)
        self.convRPN4 = SplineConv(64, (7+num_classes)*num_anchors,
                                   dim=3, kernel_size=5)

        self.fc1 = Lin(64,64)
        self.fc2 = Lin(64,64)
        self.fc3 = Lin(64,64)
        self.fc0 = Lin(32,32)

        self.skip1 = Lin(32, 64)
        self.skip2 = Lin(64, 96)
        self.skip3 = Lin(64, 64)


    def forward(self, data):
        ones = Variable(data.input.data.new(data.input.size(0), 1).fill_(1))
        data.input = F.elu(self.conv1(data.adj, data.input))
        data1, cluster1 = sparse_voxel_max_pool(data, (1 / 64), 0, trans)
        x = torch.cat([data1.input, ones[:data1.input.size(0)]], dim=1)
        x = F.elu(self.conv2(data1.adj, x))
        x = self.conv22(data1.adj, x)
        data1.input = F.elu(x + self.skip1(data1.input))
        data2, cluster2 = sparse_voxel_max_pool(data1, (1 / 32), 0, trans)
        x = torch.cat((data2.input, ones[:data2.input.size(0), :]), dim=1)
        x = F.elu(self.conv3(data2.adj, x))
        x = self.conv32(data2.adj, x)
        data2.input = F.elu(x + data2.input)
        data3, cluster3 = sparse_voxel_max_pool(data2, (1 / 16), 0, trans)
        x = torch.cat([data3.input, ones[:data3.input.size(0)]], dim=1)
        x = F.elu(self.conv4(data3.adj, x))
        x = self.conv42(data3.adj, x)
        data3.input = F.elu(x + data3.input)
        data4, cluster4 = sparse_voxel_max_pool(data3, (1 / 8), 0, trans)
        x = torch.cat([data4.input, ones[:data4.input.size(0)]], dim=1)
        x = F.elu(self.conv5(data4.adj, x))
        x = self.conv52(data4.adj, x)
        data4.input = F.elu(x + self.skip2(data4.input))

        data3.input = torch.cat(
            [upscale(data4.input, cluster4), self.fc3(data3.input)], dim=1)
        data3.input = F.elu(self.convRPN1(data3.adj, data3.input))
        # data3.input = F.elu(self.conv52(data3.adj, data3.input))

        data2.input = torch.cat(
            [upscale(data3.input, cluster3), self.fc2(data2.input)], dim=1)
        data2.input = F.elu(self.convRPN2(data2.adj, data2.input))
        data2.input = self.convRPN4(data2.adj, data2.input)

        #data1.input = torch.cat(
        #    [upscale(data2.input, cluster2), self.fc1(data1.input)], dim=1)
        #data1.input = F.elu(self.convRPN3(data1.adj, data1.input))
        #data1.input = self.convRPN4(data1.adj, data1.input)
        class_out = data2.input[:,:num_classes*num_anchors].contiguous().view(-1,2)
        bb_out = data2.input[:,num_classes*num_anchors:].contiguous().view(-1,7)

        return F.log_softmax(class_out, dim=1), bb_out, data2.pos, data2




model = Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

class_id = 0


def train(epoch):
    model.train()

    #if epoch == 21:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.0005

    if epoch == 11:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    if epoch == 21:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00005

    #if epoch == 61:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 0.00001
    example = 0
    loss_class_sum = 0
    loss_bb_sum = 0
    loss_sum = 0
    for i,data in enumerate(train_loader):
        data = data.cuda().to_variable()
        data.offset = data.offset[0].cuda()
        optimizer.zero_grad()
        start = time.time()
        class_out, bb_out, positions, data1 = model(data)
        #torch.cuda.synchronize()
        #print('Model Forward:',time.time()-start)
        mask = data.target.data[:, 0] == class_id
        start = time.time()
        if mask.sum()>0:
            mask = mask.view(-1, 1).expand(mask.size(0), 9)
            boxes = data.target.data[mask].view(-1, 9)[:, 2:9]
            boxes = camera_to_lidar_box(boxes)
            boxes[:,3:6] = (boxes[:,3:6] + data.offset) * data.scale[0]
            boxes[:,0:3] *= data.scale[0]
            # Compute Groundtruth
        else:
            boxes = None
        class_gt, bb_gt, mask_gt = compute_anchor_gt(positions, boxes,
                                                  data.offset, data.scale[0])
        #torch.cuda.synchronize()
        #print('Compute Anchor GT:',time.time()-start)
        #start = time.time()
        #show_pointclouds(data1, boxes)
        #print(class_out.max(dim=1)[1].sum().data[0], class_gt.sum())
        # Anchor loss
        weight = torch.FloatTensor([1.0,3.0]).cuda()
        loss_class = F.nll_loss(class_out, Variable(class_gt), weight=weight)
        #mAP = mean_average_precision(boxes, class_out.data, bb_out.data)
        #print(mAP)
        # BB loss
        #bb_out.data = bb_out.data * mask.unsqueeze(1)
        loss_bb = F.smooth_l1_loss(bb_out,  Variable(bb_gt), reduce=False)

        loss_bb = loss_bb * Variable(mask_gt.unsqueeze(1))
        loss_bb[:,0:6] /= data.scale[0]
        if mask_gt.sum() > 0:
            loss_bb = loss_bb.sum()/mask.sum()
        else:
            loss_bb = loss_bb.sum()


        loss = loss_class + loss_bb
        #torch.cuda.synchronize()
        #print('Loss:',time.time()-start)
        start = time.time()
        loss.backward()
        optimizer.step()
        #torch.cuda.synchronize()
        #print('BW:',time.time()-start)

        loss_class_sum += loss_class.data[0]
        loss_bb_sum += loss_bb.data[0]
        loss_sum += loss.data[0]

        example += batch_size
    print('Epoch', epoch, 'Example', example, 'of', len(train_dataset), '. Loss:',
          loss_sum/example, 'ce_loss:', loss_class_sum/example, 'mse_loss:',
          loss_bb_sum/example)

def test(epoch, loader, ds, string):
    model.eval()

    mAP_sum = 0
    for i,data in enumerate(loader):
        data = data.cuda().to_variable()
        data.offset = data.offset[0].cuda()
        optimizer.zero_grad()
        class_out, bb_out, positions, data1 = model(data)
        mask = data.target.data[:, 0] == class_id
        num_targets = mask.sum()
        if mask.sum()>0:
            mask = mask.view(-1, 1).expand(mask.size(0), 9)
            boxes = data.target.data[mask].view(-1, 9)[:, 2:9]
            boxes = camera_to_lidar_box(boxes)
            boxes[:,3:6] = (boxes[:,3:6] + data.offset) * data.scale[0]
            boxes[:,0:3] *= data.scale[0]
            # Compute Groundtruth
        else:
            boxes = None
        #print(class_out.max(dim=1)[1].sum().data[0], num_targets)

        # Translate to global position
        positions = torch.cat([positions, positions],dim=1).view(-1,3)
        bb_out.data[:,3:6] += positions
        # Translate middle to bottom of bb to match gt
        #bb_out.data[:, 5] -= bb_out.data[:, 0]/2
        mAP_sum += mean_average_precision(boxes, class_out.data, bb_out.data,
                                          data.offset, data.scale[0])


        if epoch == 70:
            class_out_mod = class_out.data
            if class_out_mod.max(1)[1].sum()>0:
                pos_anchor_idx = class_out_mod.max(1)[1].nonzero().squeeze()
                pos_anchor_pos = positions[pos_anchor_idx]
                pos_anchor_bbs = bb_out.data[pos_anchor_idx]
                a_idx = pos_anchor_idx % 2
                show_pointclouds(data1, boxes, pos_anchor_bbs, pos_anchor_pos,
                                 a_idx, data.offset, data.scale[0])
            else:
                show_pointclouds(data1, boxes)
    print('Epoch', epoch, string, 'mAP', mAP_sum / len(ds))

for epoch in range(1, 71):
    #test(epoch, train_loader, train_dataset, 'Train')
    train(epoch)
    if epoch%10 == 0:
        test(epoch, train_loader, train_dataset, 'Train')
        test(epoch, test_loader, test_dataset, 'Test')
