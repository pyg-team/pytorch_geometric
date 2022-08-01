import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn.pool import knn

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius


class LocalSpatialEncoding(torch.nn.Module):
    def __init__(self, d_in):
        # self.d_in = d_in
        ...

    def forward(self, x, pos, batch):
        ...


class LFAggregationModule(torch.nn.Module):
    def __init__(self, decimation, num_neighbors, nn):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.conv = PointConv(nn, add_self_loops=False)
        # self.locSE = LocalSpatialEncoding(6)

    def forward(self, x, pos, batch):
        # Random Sampling instead of FPS.
        idx = torch.arange(start=0, end=batch.size(0), step=self.decimation)
        row, col = knn(
            pos, pos[idx], self.num_neighbors, batch_x=batch, batch_y=batch[idx]
        )
        # (N, 3+d) -> K, 2d
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        decimation = 4
        k = 16
        # Input channels account for both `pos` and node features.
        self.lfa1_module = LFAggregationModule(decimation, k, MLP([3, 32]))
        self.lfa2_module = LFAggregationModule(decimation, k, MLP([32 + 3, 128]))
        self.lfa3_module = LFAggregationModule(decimation, k, MLP([128 + 3, 256]))
        self.lfa4_module = LFAggregationModule(decimation, k, MLP([256 + 3, 512]))

        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.lfa1_module(*sa0_out)
        sa2_out = self.lfa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)


def train(epoch):
    model.train()

    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == "__main__":
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data/ModelNet10")
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, "10", True, transform, pre_transform)
    test_dataset = ModelNet(path, "10", False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(test_loader)
        print(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")
