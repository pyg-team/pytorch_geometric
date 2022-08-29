import os.path as osp

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, LeakyReLU
from tqdm import tqdm
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_max_pool, BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn


# Default activation and BatchNorm used by RandLaNet authors
lrelu02 = LeakyReLU(negative_slope=0.2)


def bn099(in_channels):
    return BatchNorm(in_channels, momentum=0.99, eps=1e-6)


class GlobalPooling(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(x)
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""

    def __init__(self, d_out):
        super().__init__(aggr="add")
        self.mlp_encoder = MLP([10, d_out // 2], act=lrelu02, norm=bn099(d_out // 2))
        self.mlp_attention = Linear(in_features=d_out, out_features=d_out, bias=False)
        self.mlp_post_attention = MLP([d_out, d_out], act=lrelu02, norm=bn099(d_out))

    def forward(self, edge_indx, x, pos):
        out = self.propagate(edge_indx, x=x, pos=pos)  # N // 4 * d_out
        out = self.mlp_post_attention(out)  # N // 4 * d_out
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
        """
        Local Spatial Encoding (locSE) and attentive pooling of features.

        Args:
            x_j (Tensor): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)

        returns:
            (Tensor): locSE weighted by feature attention scores.

        """
        dist = pos_j - pos_i
        euclidian_dist = torch.sqrt((dist * dist).sum(1, keepdim=True))
        relative_infos = torch.cat(
            [pos_i, pos_j, dist, euclidian_dist], dim=1
        )  # N//4 * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N//4 * K, d
        local_features = torch.cat([x_j, local_spatial_encoding], dim=1)  # N//4 * K, 2d

        # attention will weight the different features of x
        attention_scores = torch.softmax(
            self.mlp_attention(local_features), dim=-1
        )  # N//4 * K, d_out
        return attention_scores * local_features  # N//4 * K, d_out


def subsample_by_decimation(batch, decimation):
    """Subsamples by a decimation factor.

    Each sample needs to be decimated separately to prevent emptying point clouds by accident.

    """
    ends = (
        (torch.argwhere(torch.diff(batch) != 0) + 1)
        .cpu()
        .numpy()
        .squeeze()
        .astype(int)
        .tolist()
    )
    starts = [0] + ends
    ends = ends + [batch.size(0)]
    idx = torch.cat(
        [
            (start + torch.randperm(end - start))[::decimation]
            for start, end in zip(starts, ends)
        ],
        dim=0,
    )
    return idx


class DilatedResidualBlock(MessagePassing):
    def __init__(
        self,
        decimation,
        num_neighbors,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = MLP([d_in, d_out // 8], act=lrelu02, norm=False)
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = MLP([d_in, d_out], act=None, norm=bn099(d_out))
        # MLP on output
        self.mlp2 = MLP([d_out // 2, d_out], act=None, norm=bn099(d_out))

        self.lfa1 = LocalFeatureAggregation(d_out // 4)
        self.lfa2 = LocalFeatureAggregation(d_out // 2)

        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x, pos, batch):
        # Random Sampling by decimation
        idx = subsample_by_decimation(batch, self.decimation)
        row, col = knn(
            pos, pos[idx], self.num_neighbors, batch_x=batch, batch_y=batch[idx]
        )
        edge_index = torch.stack([col, row], dim=0)
        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out // 8
        x = self.lfa1(edge_index, x, pos)  # N, d_out // 2
        x = self.lfa2(edge_index, x, pos)  # N, d_out // 2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out
        return x[idx], pos[idx], batch[idx]  # N // decimation, d_out


class Net(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        decimation: int = 4,
        num_neighboors: int = 16,
        return_logits: bool = False,
    ):
        super().__init__()
        self.return_logits = return_logits
        self.fc0 = Sequential(
            Linear(in_features=num_features, out_features=8), bn099(8)
        )
        self.lfa1_module = DilatedResidualBlock(decimation, num_neighboors, 8, 32)
        self.lfa2_module = DilatedResidualBlock(decimation, num_neighboors, 32, 128)
        self.lfa3_module = DilatedResidualBlock(decimation, num_neighboors, 128, 256)
        self.lfa4_module = DilatedResidualBlock(decimation, num_neighboors, 256, 512)
        self.pool = GlobalPooling(MLP([512, 512]))
        self.mlp = MLP([512, 64, num_classes], dropout=0.5)

    def forward(self, data):
        in_0 = (self.fc0(data.x), data.pos, data.batch)
        lfa1_out = self.lfa1_module(*in_0)
        lfa2_out = self.lfa2_module(*lfa1_out)
        lfa3_out = self.lfa3_module(*lfa2_out)
        lfa4_out = self.lfa4_module(*lfa3_out)
        encoding = self.pool(*lfa4_out)
        x, _, _ = encoding
        logits = self.mlp(x)
        if self.return_logits:
            return logits
        return logits.log_softmax(dim=-1)


class SetPosAsXIfNoX(BaseTransform):
    """Avoid empty x Tensor by using positions as features."""

    def __call__(self, data):
        if not data.x:
            data.x = data.pos
        return data


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
    # FixedPoints acts as a shuffler of Sampled points.
    pre_transform, transform = T.NormalizeScale(), T.Compose(
        [T.SamplePoints(1024), T.FixedPoints(1024, replace=False), SetPosAsXIfNoX()]
    )
    train_dataset = ModelNet(path, "10", True, transform, pre_transform)
    test_dataset = ModelNet(path, "10", False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(3, train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(test_loader)
        print(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")
        scheduler.step()
