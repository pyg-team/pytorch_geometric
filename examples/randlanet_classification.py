import os.path as osp

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, LeakyReLU
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_max_pool, BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn


# Default activation, BatchNorm, and resulting MLP used by RandLaNet authors
lrelu02_kwargs = {"negative_slope": 0.2}


bn099_kwargs = {"momentum": 0.99, "eps": 1e-6}


def default_MLP(*args, **kwargs):
    """MLP with custom activation, bn, and dropout that are always active even an last layer."""
    kwargs["plain_last"] = kwargs.get("plain_last", False)
    kwargs["act"] = kwargs.get("act", "LeakyReLU")
    kwargs["act_kwargs"] = kwargs.get("act_kwargs", lrelu02_kwargs)
    kwargs["norm_kwargs"] = kwargs.get("norm_kwargs", bn099_kwargs)
    return MLP(*args, **kwargs)


class GlobalPooling(torch.nn.Module):
    def forward(self, x, pos, batch):
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""

    def __init__(self, d_out, num_neighbors):
        super().__init__(
            aggr="add"
            # , flow="source_to_target"  # the default: source are knn points, target are neighboorhood centers
        )
        self.mlp_encoder = default_MLP([10, d_out // 2])
        self.mlp_attention = default_MLP(
            [d_out, d_out], bias=False, act=None, norm=None
        )
        self.mlp_post_attention = default_MLP([d_out, d_out])
        self.num_neighbors = num_neighbors

    def forward(self, edge_index, x, pos):
        out = self.propagate(edge_index, x=x, pos=pos)  # N // 4 * d_out
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
        # Encode local neighboorhod structural information
        pos_diff = pos_j - pos_i
        distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        relative_infos = torch.cat(
            [pos_i, pos_j, pos_diff, distance], dim=1
        )  # N//decim * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N//decim * K, d
        local_features = torch.cat(
            [x_j, local_spatial_encoding], dim=1
        )  # N//decim * K, 2d

        # Attention will weight the different features of x
        att_features = self.mlp_attention(local_features)  # N//decim * K, d_out
        d_out = att_features.size(1)
        att_scores = torch.softmax(
            att_features.view(-1, self.num_neighbors, d_out), dim=1
        ).view(
            -1, d_out
        )  # N//decim * K, d_out with transitory N//decim, K, d_out

        return att_scores * local_features  # N//decim * K, d_out


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
        self.mlp1 = default_MLP([d_in, d_out // 8])
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = default_MLP([d_in, d_out], act=None)
        # MLP on output
        self.mlp2 = default_MLP([d_out // 2, d_out], act=None)

        self.lfa1 = LocalFeatureAggregation(d_out // 4, num_neighbors)
        self.lfa2 = LocalFeatureAggregation(d_out // 2, num_neighbors)

        self.lrelu = torch.nn.LeakyReLU(**lrelu02_kwargs)

    def forward(self, x, pos, batch):
        # Random Sampling by decimation
        idx = subsample_by_decimation(batch, self.decimation)
        row, col = knn(
            pos, pos[idx], self.num_neighbors, batch_x=batch, batch_y=batch[idx]
        )
        edge_index = torch.stack([col, row], dim=0)
        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out//8
        x = self.lfa1(edge_index, x, pos)  # N, d_out//2
        x = self.lfa2(edge_index, x, pos)  # N, d_out//2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out
        return x[idx], pos[idx], batch[idx]  # N//decim, d_out


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
        self.mlp1 = default_MLP([128, 128])
        self.pool = GlobalPooling()
        self.mlp_end = Sequential(
            default_MLP([128, 32], dropout=[0.5]), Linear(32, num_classes)
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
