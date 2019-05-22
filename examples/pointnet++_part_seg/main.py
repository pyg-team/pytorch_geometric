import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import mean_IoU

from pointnet2_part_seg import Net

category = 'Airplane'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ShapeNet')
train_transform = T.Compose([
    T.NormalizeScale(),
    T.RandomTranslate(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2)
])
test_transform = T.NormalizeScale()
train_dataset = ShapeNet(path, category, train=True, transform=train_transform)
test_dataset = ShapeNet(path, category, train=False, transform=test_transform)
train_loader = DataLoader(
    train_dataset, batch_size=12, shuffle=True, num_workers=6)
test_loader = DataLoader(
    test_dataset, batch_size=12, shuffle=False, num_workers=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.max(dim=1)[1].eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
                i + 1, len(train_loader), total_loss / 10,
                correct_nodes / total_nodes))
            total_loss = correct_nodes = total_nodes = 0


def test(loader):
    model.eval()

    correct_nodes = total_nodes = 0
    ious = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.max(dim=1)[1]
        correct_nodes += pred.eq(data.y).sum().item()
        ious += [mean_IoU(pred, data.y, test_dataset.num_classes, data.batch)]
        total_nodes += data.num_nodes
    return correct_nodes / total_nodes, torch.cat(ious, dim=0).mean().item()


for epoch in range(1, 26):
    train()
    acc, iou = test(test_loader)
    print('Epoch: {:02d}, Acc: {:.4f}, IoU: {:.4f}'.format(epoch, acc, iou))
