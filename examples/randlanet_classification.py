import os.path as osp
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.datasets import ModelNet
from torch_geometric.typing import OptTensor
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn
from torch_geometric.utils import softmax

# Default activation, BatchNorm, and resulting MLP used by RandLA-Net authors
lrelu02_kwargs = {"negative_slope": 0.2}


bn099_kwargs = {"momentum": 0.99, "eps": 1e-6}


class SharedMLP(MLP):
    """SharedMLP with new defauts BN and Activation following tensorflow implementation."""

    def __init__(self, *args, **kwargs):
        # BN + Act always active even at last layer.
        kwargs["plain_last"] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs["act"] = kwargs.get("act", "LeakyReLU")
        kwargs["act_kwargs"] = kwargs.get("act_kwargs", lrelu02_kwargs)
        # BatchNorm with 0.99 momentum and 1e-6 eps by defaut.
        kwargs["norm_kwargs"] = kwargs.get("norm_kwargs", bn099_kwargs)
        super().__init__(*args, **kwargs)


class GlobalPooling(torch.nn.Module):
    """Global Pooling to adapt RandLA-Net to a classification task."""

    def forward(self, x, pos, batch):
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""

    def __init__(self, d_out, num_neighbors):
        super().__init__(aggr="add")
        self.mlp_encoder = SharedMLP([10, d_out // 2])
        self.mlp_attention = SharedMLP([d_out, d_out], bias=False, act=None, norm=None)
        self.mlp_post_attention = SharedMLP([d_out, d_out])
        self.num_neighbors = num_neighbors

    def forward(self, edge_index, x, pos):
        out = self.propagate(edge_index, x=x, pos=pos)  # N, d_out
        out = self.mlp_post_attention(out)  # N, d_out
        return out

    def message(
        self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, index: Tensor
    ) -> Tensor:
        """
        Local Spatial Encoding (locSE) and attentive pooling of features.

        Args:
            x_j (Tensor): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)

        returns:
            (Tensor): locSE weighted by feature attention scores.

        """
        # Encode local neighboorhod structural information
        pos_diff = pos_j - pos_i
        distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        relative_infos = torch.cat(
            [pos_i, pos_j, pos_diff, distance], dim=1
        )  # N * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N * K, d
        local_features = torch.cat([x_j, local_spatial_encoding], dim=1)  # N * K, 2d

        # Attention will weight the different features of x along the neighborhood dimension.
        att_features = self.mlp_attention(local_features)  # N * K, d_out
        att_scores = softmax(att_features, index=index)  # N * K, d_out

        return att_scores * local_features  # N * K, d_out


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
        self.mlp1 = SharedMLP([d_in, d_out // 8])
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = SharedMLP([d_in, d_out], act=None)
        # MLP on output
        self.mlp2 = SharedMLP([d_out // 2, d_out], act=None)

        self.lfa1 = LocalFeatureAggregation(d_out // 4, num_neighbors)
        self.lfa2 = LocalFeatureAggregation(d_out // 2, num_neighbors)

        self.lrelu = torch.nn.LeakyReLU(**lrelu02_kwargs)

    def forward(self, x, pos, batch, return_unsampled_as_well: bool = False):
        idx = subsample_by_decimation(batch, self.decimation)
        row, col = knn(pos, pos, self.num_neighbors, batch_x=batch, batch_y=batch)
        edge_index = torch.stack([col, row], dim=0)

        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out//8
        x = self.lfa1(edge_index, x, pos)  # N, d_out//2
        x = self.lfa2(edge_index, x, pos)  # N, d_out//2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out

        sampled = (x[idx], pos[idx], batch[idx])
        if return_unsampled_as_well:
            # Needed for skip connection in final upsampling
            return sampled, (x, pos, batch)
        return sampled  # N//decim, d_out


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
        self.fc0 = Sequential(Linear(in_features=num_features, out_features=8))
        self.lfa1_module = DilatedResidualBlock(decimation, num_neighboors, 8, 32)
        self.lfa2_module = DilatedResidualBlock(decimation, num_neighboors, 32, 128)
        # self.lfa3_module = DilatedResidualBlock(decimation, num_neighboors, 128, 256)
        # self.lfa4_module = DilatedResidualBlock(decimation, num_neighboors, 256, 512)
        self.mlp1 = SharedMLP([128, 128])
        self.pool = GlobalPooling()
        self.mlp_end = Sequential(
            SharedMLP([128, 32], dropout=[0.5]), Linear(32, num_classes)
        )

    def forward(self, data):
        in_0 = (self.fc0(data.x), data.pos, data.batch)
        lfa1_out = self.lfa1_module(*in_0)
        lfa2_out = self.lfa2_module(*lfa1_out)
        # lfa3_out = self.lfa3_module(*lfa2_out)
        # lfa4_out = self.lfa4_module(*lfa3_out)
        x = self.mlp1(lfa2_out[0])
        x, _, _ = self.pool(x, lfa2_out[1], lfa2_out[2])
        logits = self.mlp_end(x)
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
