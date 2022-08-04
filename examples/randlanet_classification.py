import os.path as osp
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LeakyReLU, Sequential
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn


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
    """This only creates relative encodings, taht will later be processed by Attentive Pooling."""
    def __init__(self, d_out):
        # TODO: check if need for batch, activation, etc.
        super().__init__(aggr="add")
        self.mlp_encoder = MLP([10, d_out // 2])
        self.mlp_attention = MLP([d_out, d_out])

    def forward(self, edge_indx, x, pos):
        out = self.propagate(edge_indx, x=x, pos=pos)  # N // 4 * d_out
        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor) -> Tensor:
        """_summary_

        Args:
            x_j (Optional[Tensor]): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)

        out1 (Tensor): Concatenation of x_j + MLP(relative_infos)
        out2 (Tensor): out1 multiplied by derived attention scores on the features.

        """
        dist = pos_j - pos_i
        euclidian_dist = torch.sqrt(dist * dist).sum(1, keepdim=True)
        relative_infos = torch.cat([pos_i, pos_j, dist, euclidian_dist],
                                   dim=1)  # N//4 * K, d
        local_spatial_encoding = self.mlp_encoder(
            relative_infos)  # N//4 * K, d
        out1 = torch.cat([x_j, local_spatial_encoding], dim=1)  # N//4 * K, 2d

        # attention will weight the differetn features of x
        attention_scores = torch.softmax(self.mlp_attention(out1),
                                         dim=-1)  # N//4 * K, d_out
        out2 = attention_scores * out1  # N//4 * K, d_out
        return out2


class DilatedResidualBlock(MessagePassing):
    def __init__(self, decimation, num_neighbors, d_in, d_out):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = Sequential(
            MLP(
                [d_in, d_out // 4],
                norm=False,
            ),
            LeakyReLU(negative_slope=0.2),
        )
        # MLP on input whose result is summed with the output of mlp2
        self.shortcut = Sequential(
            MLP([d_in, d_out], ),
            LeakyReLU(negative_slope=0.2),
        )
        # MLP on output
        self.mlp2 = Sequential(
            MLP(
                [d_out, d_out],
                norm=False,
            ),
            LeakyReLU(negative_slope=0.2),
        )

        self.lfa1 = LocalFeatureAggregation(d_out // 2)
        self.lfa2 = LocalFeatureAggregation(d_out)

        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x, pos, batch):
        # Random Sampling by decimation
        idx = torch.arange(start=0, end=batch.size(0), step=self.decimation)
        row, col = knn(pos, pos[idx], self.num_neighbors, batch_x=batch,
                       batch_y=batch[idx])
        edge_index = torch.stack([col, row], dim=0)
        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out // 4
        x = self.lfa1(edge_index, x, pos)  # N, d_out
        x = self.lfa2(edge_index, x, pos)  # N, d_out
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out
        x, pos, batch = x[idx], pos[idx], batch[idx]  # N // decimation, d_out
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self, decimation: int = 4, num_neighboors: int = 16):
        super().__init__()
        self.lfa1_module = DilatedResidualBlock(decimation, num_neighboors, 3, 32)
        self.lfa2_module = DilatedResidualBlock(decimation, num_neighboors, 32, 128)
        self.lfa3_module = DilatedResidualBlock(decimation, num_neighboors, 128, 256)
        self.lfa4_module = DilatedResidualBlock(decimation, num_neighboors, 256, 512)
        self.pool = GlobalPooling(MLP([512, 1024]))
        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5)

    def forward(self, data):
        # Avoid empty x Tensor by using positions as features.
        if not data.x:
            data.x = data.pos
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
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..",
                    "data/ModelNet10")
    # FixedPoints acts as a shuffler of Sampled points.
    pre_transform, transform = T.NormalizeScale(), T.Compose(
        [T.SamplePoints(1024),
         T.FixedPoints(1024, replace=False)])
    train_dataset = ModelNet(path, "10", True, transform, pre_transform)
    test_dataset = ModelNet(path, "10", False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                gamma=0.5)

    for epoch in range(1, 201):
        train(epoch)
        test_acc = test(test_loader)
        print(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")
        scheduler.step()
