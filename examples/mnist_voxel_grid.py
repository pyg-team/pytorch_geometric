import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
transform = T.Cartesian(cat=False)
train_dataset = MNISTSuperpixels(path, True, transform=transform)
test_dataset = MNISTSuperpixels(path, False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
d = train_dataset


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SplineConv(d.num_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.conv3 = SplineConv(64, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(4 * 64, 128)
        self.fc2 = torch.nn.Linear(128, d.num_classes)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        cluster = voxel_grid(data.pos, data.batch, size=5, start=0, end=28)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        cluster = voxel_grid(data.pos, data.batch, size=7, start=0, end=28)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        cluster = voxel_grid(data.pos, data.batch, size=14, start=0, end=27.99)
        x, _ = max_pool_x(cluster, data.x, data.batch, size=4)

        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        F.nll_loss(model(data), data.y).backward()
        optimizer.step()


def test():
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(test_dataset)


for epoch in range(1, 21):
    train(epoch)
    test_acc = test()
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
