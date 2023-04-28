"""
An adaptation of RandLA-Net to the classification task, which was not
addressed in the paper:
RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
Reference: https://arxiv.org/abs/1911.11236
"""
import os.path as osp

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import MaxAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn.pool.decimation import decimation_indices
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import softmax

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

# Default activation and batch norm parameters used by RandLA-Net:
lrelu02_kwargs = {'negative_slope': 0.2}
bn099_kwargs = {'momentum': 0.01, 'eps': 1e-6}


class SharedMLP(MLP):
    """SharedMLP following RandLA-Net paper."""
    def __init__(self, *args, **kwargs):
        # BN + Act always active even at last layer.
        kwargs['plain_last'] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs['act'] = kwargs.get('act', 'LeakyReLU')
        kwargs['act_kwargs'] = kwargs.get('act_kwargs', lrelu02_kwargs)
        # BatchNorm with 1 - 0.99 = 0.01 momentum
        # and 1e-6 eps by defaut (tensorflow momentum != pytorch momentum)
        kwargs['norm_kwargs'] = kwargs.get('norm_kwargs', bn099_kwargs)
        super().__init__(*args, **kwargs)


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""
    def __init__(self, channels):
        super().__init__(aggr='add')
        self.mlp_encoder = SharedMLP([10, channels // 2])
        self.mlp_attention = SharedMLP([channels, channels], bias=False,
                                       act=None, norm=None)
        self.mlp_post_attention = SharedMLP([channels, channels])

    def forward(self, edge_index, x, pos):
        out = self.propagate(edge_index, x=x, pos=pos)  # N, d_out
        out = self.mlp_post_attention(out)  # N, d_out
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                index: Tensor) -> Tensor:
        """Local Spatial Encoding (locSE) and attentive pooling of features.

        Args:
            x_j (Tensor): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)
            index (Tensor): index of centroid positions
                (e.g. [0,...,0,1,...,1,...,N,...,N])

        returns:
            (Tensor): locSE weighted by feature attention scores.

        """
        # Encode local neighboorhod structural information
        pos_diff = pos_j - pos_i
        distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        relative_infos = torch.cat([pos_i, pos_j, pos_diff, distance],
                                   dim=1)  # N * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N * K, d
        local_features = torch.cat([x_j, local_spatial_encoding],
                                   dim=1)  # N * K, 2d

        # Attention will weight the different features of x
        # along the neighborhood dimension.
        att_features = self.mlp_attention(local_features)  # N * K, d_out
        att_scores = softmax(att_features, index=index)  # N * K, d_out

        return att_scores * local_features  # N * K, d_out


class DilatedResidualBlock(torch.nn.Module):
    def __init__(
        self,
        num_neighbors,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = SharedMLP([d_in, d_out // 8])
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = SharedMLP([d_in, d_out], act=None)
        # MLP on output
        self.mlp2 = SharedMLP([d_out // 2, d_out], act=None)

        self.lfa1 = LocalFeatureAggregation(d_out // 4)
        self.lfa2 = LocalFeatureAggregation(d_out // 2)

        self.lrelu = torch.nn.LeakyReLU(**lrelu02_kwargs)

    def forward(self, x, pos, batch):
        edge_index = knn_graph(pos, self.num_neighbors, batch=batch, loop=True)

        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out//8
        x = self.lfa1(edge_index, x, pos)  # N, d_out//2
        x = self.lfa2(edge_index, x, pos)  # N, d_out//2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out

        return x, pos, batch


def decimate(tensors, ptr: Tensor, decimation_factor: int):
    """Decimates each element of the given tuple of tensors."""
    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, ptr_decim


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
        self.decimation = decimation
        # An option to return logits instead of log probabilities:
        self.return_logits = return_logits
        self.fc0 = Linear(in_features=num_features, out_features=8)
        # 2 DilatedResidualBlock converges better than 4 on ModelNet.
        self.block1 = DilatedResidualBlock(num_neighboors, 8, 32)
        self.block2 = DilatedResidualBlock(num_neighboors, 32, 128)
        self.mlp1 = SharedMLP([128, 128])
        self.max_agg = MaxAggregation()
        self.mlp_classif = SharedMLP([128, 32], dropout=[0.5])
        self.fc_classif = Linear(32, num_classes)

    def forward(self, x, pos, batch, ptr):
        x = x if x is not None else pos
        b1 = self.block1(self.fc0(x), pos, batch)
        b1_decimated, ptr1 = decimate(b1, ptr, self.decimation)

        b2 = self.block2(*b1_decimated)
        b2_decimated, _ = decimate(b2, ptr1, self.decimation)

        x = self.mlp1(b2_decimated[0])
        x = self.max_agg(x, b2_decimated[2])

        x = self.mlp_classif(x)
        logits = self.fc_classif(x)

        return logits if self.return_logits else logits.log_softmax(dim=-1)


def train(epoch):
    model.train()

    total_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch, data.ptr)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += data.num_graphs * float(loss)
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.pos, data.batch, data.ptr)
        correct += int((out.argmax(dim=-1) == data.y).sum())
    return correct / len(loader.dataset)


if __name__ == '__main__':
    path = osp.dirname(osp.realpath(__file__))
    path = osp.join(path, '..', 'data/ModelNet10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, 32, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, 32, shuffle=False, num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(3, train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                gamma=0.5)

    for epoch in range(1, 201):
        loss = train(epoch)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
        scheduler.step()
