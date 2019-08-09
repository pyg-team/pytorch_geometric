import os.path as osp
import os
os.chdir("/home/cy/PycharmProjects/PointCloud")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DynamicEdgeConv, DataParallel, global_max_pool
from torch_geometric.utils import intersection_and_union as i_and_u

category = None # None # 'Airplane'
path = 'ShapeNet'
transform = T.Compose([
    T.FixedPoints(2048),
#     T.RandomTranslate(0.01),
#     T.RandomRotate(15, axis=0),
#     T.RandomRotate(15, axis=1),
#     T.RandomRotate(15, axis=2)
])
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(
    path,
    category,
    train=True,
    transform=transform,
    pre_transform=pre_transform)
test_dataset = ShapeNet(
    path,
    category,
    train=False,
    transform=transform,
    pre_transform=pre_transform)
train_loader = DataListLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=16, drop_last=True)
test_loader = DataListLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=16)

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), nn.GroupNorm(max(1, channels[i] // 16), channels[i]))
        for i in range(1, len(channels))
    ])

class Net(torch.nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super(Net, self).__init__()

        self.tconv1 = DynamicEdgeConv(MLP([2 * 3, 64, 128]), k, aggr)
        self.tmlp1 = MLP([128, 1024])
        self.tmlp2 = MLP([1024, 512, 256, 9])
        
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64, 64]), k, aggr)
        self.lin1 = MLP([64, 1024])

        self.mlp = Seq(
            MLP([1152, 256, 256]), Dropout(0.5), Lin(256, 128), Dropout(0.5),
            Lin(128, out_channels))

    def forward(self, data):
        pos, batch = data.pos, data.batch
        
        t1 = self.tconv1(pos, batch)
        t2 = self.tmlp1(t1)
        t3 = global_max_pool(t2, batch)
        t4 = self.tmlp2(t3)
        pos = torch.matmul(pos.view(-1, 2048, 3), t4.view(-1, 3, 3)).view(-1, 3)
        
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.lin1(x3)
        x5 = global_max_pool(x4, batch)
        x6 = x5.repeat([1, 2048]).view(-1, 1024)
        x7 = torch.cat([x2, x3, x6], dim=1)
        out = self.mlp(x7)
        return F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes, k=30)
model = DataParallel(model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data_list in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(data_list)
        y = torch.cat([data.y for data in data_list]).to(device)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.max(dim=1)[1].eq(y).sum().item()
        total_nodes += sum(data.num_nodes for data in data_list)

        if (i + 1) % 10 == 0:
            print('[{}/{}] Loss: {:.4f}, Train Accuracy: {:.4f}'.format(
                i + 1, len(train_loader), total_loss / 10,
                correct_nodes / total_nodes))
            total_loss = correct_nodes = total_nodes = 0


def test(loader):
    model.eval()

    correct_nodes = total_nodes = 0
    intersections, unions, categories = [], [], []
    for data_list in loader:
        with torch.no_grad():
            out = model(data_list)
        pred = out.max(dim=1)[1]
        y = torch.cat([data.y for data in data_list]).to(device)
        correct_nodes += pred.eq(y).sum().item()
        total_nodes += sum(data.num_nodes for data in data_list)
        b = torch.cat([torch.ones_like(data.y) * i for i, data in enumerate(data_list)]).to(device)
        i, u = i_and_u(pred, y, test_dataset.num_classes, b)
        intersections.append(i.to(torch.device('cpu')))
        unions.append(u.to(torch.device('cpu')))
        c = torch.cat([data.category for data in data_list])
        categories.append(c.to(torch.device('cpu')))

    category = torch.cat(categories, dim=0)
    intersection = torch.cat(intersections, dim=0)
    union = torch.cat(unions, dim=0)

    ious = [[] for _ in range(len(loader.dataset.categories))]
    for j in range(len(loader.dataset)):
        i = intersection[j, loader.dataset.y_mask[category[j]]]
        u = union[j, loader.dataset.y_mask[category[j]]]
        iou = i.to(torch.float) / u.to(torch.float)
        iou[torch.isnan(iou)] = 1
        ious[category[j]].append(iou.mean().item())

    miiou = sum(sum(ious, [])) / len(loader.dataset)
    for cat in range(len(loader.dataset.categories)):
        ious[cat] = torch.tensor(ious[cat]).mean().item()

    return correct_nodes / total_nodes, miiou, torch.tensor(ious).mean().item()


best = 0
for epoch in range(1, 101):
    train()
    acc, miiou, mciou = test(test_loader)
    best = max(best, miiou)
    if epoch % 10 == 0:
        torch.save(model.module.state_dict(), str(epoch) + 'dgcnn.pth')
    print('Epoch: {:02d}, Best: {:.4f}, Acc: {:.4f}, MIoU: {:.4f}, MCIoU: {:.4f}'.format(epoch, best, acc, miiou, mciou))
