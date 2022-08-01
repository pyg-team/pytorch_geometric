from typing import Optional
import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn.pool import knn
import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_max_pool


class GlobalPooling(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class LocSE(torch.nn.Module):
    """This only creates relative encodings, taht will later be processed by Attentive Pooling."""

    def __init__(self, d):
        # TODO: check if need for batch, activation, etc.
        super().__init__()
        self.mlp = MLP([d, d])

    def forward(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        # TODO: add all distahnce, squared distance, etc, to msg
        dist = pos_j - pos_i
        squared_dist = dist * dist
        relative_infos = torch.cat([dist, squared_dist], dim=1)  # K, d
        local_spatial_encoding = self.mlp(relative_infos)  # K, d
        x = torch.cat([x_j, local_spatial_encoding], dim=1)  # K, 2d
        return x


class AttentivePooling(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        # TODO: add batch norm, activation, etc. if needed
        self.nn = MLP(
            [d, d],
            # bias=False, norm=False
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # x : K, 2d, already contains relative positions and features of all points in neighborhood.
        scores = self.nn(x)
        attention_scores = self.softmax(scores)
        x = torch.sum(x * attention_scores, dim=-1, keepdim=True)
        return x


class LFAggregationModule(MessagePassing):
    def __init__(self, decimation, num_neighbors, d_in, d_out):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = MLP(
            [d_in, d_out // 2],
            # act="leaky_relu",
            # act_kwargs={"negative_slope": 0.2},
            # norm=None,
        )
        # output mlp
        self.mlp2 = MLP(
            [d_out, d_out // 2],
            # act="leaky_relu",
            # act_kwargs={"negative_slope": 0.2},
            # norm=None,
        )
        # mlp on input whose result sis added to the output of mlp2
        self.shortcut = MLP(
            [d_in, d_out // 2],
            # act="leaky_relu",
            # act_kwargs={"negative_slope": 0.2},
        )

        self.lse1 = LocSE(d_out // 2)
        self.lse2 = LocSE(d_out // 2)

        self.pool1 = AttentivePooling(d_out)
        self.pool2 = AttentivePooling(d_out)

        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x, pos, batch):
        # (N, 3+d) -> K, 2d

        # Random Sampling instead of FPS.
        idx = torch.arange(start=0, end=batch.size(0), step=self.decimation)
        row, col = knn(
            pos, pos[idx], self.num_neighbors, batch_x=batch, batch_y=batch[idx]
        )
        edge_index = torch.stack([col, row], dim=0)

        # forwar pass, including message passing for
        shortcut_of_x = self.shortcut(x)
        x = self.mlp1(x)
        x = self.propagate(edge_index, x=x, pos=pos)
        x = self.lrelu(x + shortcut_of_x)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

    def message(self, x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:

        ####### LocSE #######

        ####### Attentive Pooling
        # Encode local structure, keeping the same shape.
        x = self.lse1(x_j, pos_i, pos_j)
        x = self.pool1(x)

        x = self.lse2(x_j, pos_i, pos_j)
        x = self.pool2(x)

        return self.mlp2(x)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        decimation = 4
        k = 16
        # Input channels account for both `pos` and node features.
        # TODO: probably turn x = pos+x in each LFA ?
        self.lfa1_module = LFAggregationModule(decimation, k, 3, 32)
        self.lfa2_module = LFAggregationModule(decimation, k, 32 + 3, 128)
        self.lfa3_module = LFAggregationModule(decimation, k, 128 + 3, 256)
        self.lfa4_module = LFAggregationModule(decimation, k, 256 + 3, 512)
        self.pool = GlobalPooling(MLP([512, 1024]))
        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5)

    def forward(self, data):
        data.x = data.pos  # to avoid empty x Tensor.
        in_0 = (data.x, data.pos, data.batch)
        lfa1_out = self.lfa1_module(*in_0)
        lfa2_out = self.lfa2_module(*lfa1_out)
        lfa3_out = self.lfa3_module(*lfa2_out)
        lfa4_out = self.lfa4_module(*lfa3_out)
        encoding = self.pool(*lfa4_out)

        x, _, _ = encoding

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
