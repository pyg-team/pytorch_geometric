import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv
from torch_geometric.utils import intersection_and_union as i_and_u

category = 'Airplane'  # Pass in `None` to train on all categories.
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ShapeNet')
transform = T.Compose([
    T.RandomTranslate(0.01),
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2)
])
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
                         pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test',
                        pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True,
                          num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False,
                         num_workers=6)


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)

        self.mlp = MLP([3 * 64, 1024, 256, 128, out_channels], dropout=0.5,
                       batch_norm=False)

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x0 = torch.cat([x, pos], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return F.log_softmax(out, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes, k=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)


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
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0


@torch.no_grad()
def test(loader):
    model.eval()

    y_mask = loader.dataset.y_mask
    ious = [[] for _ in range(len(loader.dataset.categories))]

    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)

        i, u = i_and_u(pred, data.y, loader.dataset.num_classes, data.batch)
        iou = i.cpu().to(torch.float) / u.cpu().to(torch.float)
        iou[torch.isnan(iou)] = 1

        # Find and filter the relevant classes for each category.
        for iou, category in zip(iou.unbind(), data.category.unbind()):
            ious[category.item()].append(iou[y_mask[category]])

    # Compute mean IoU.
    ious = [torch.stack(iou).mean(0).mean(0) for iou in ious]
    return torch.tensor(ious).mean().item()


for epoch in range(1, 31):
    train()
    iou = test(test_loader)
    print(f'Epoch: {epoch:02d}, Test IoU: {iou:.4f}')
