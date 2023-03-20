import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, Sequential

from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import reset
from torch_geometric.nn.models.dimenet import triplets
from torch_geometric.nn.models.schnet import ShiftedSoftplus
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


class GaussianFilter(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (float(offset[1]) - float(offset[0]))**2
        self.register_buffer('offset', offset)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist.pow(2))


class NodeBlock(torch.nn.Module):
    def __init__(self, hidden_node_channels: int, hidden_edge_channels: int):
        super().__init__()
        self.lin_c1 = Linear(hidden_node_channels + hidden_edge_channels,
                             2 * hidden_node_channels)

        # BN was added based on previous studies.
        # ref: https://github.com/txie-93/cgcnn/blob/master/cgcnn/model.py
        self.bn_c1 = BatchNorm1d(2 * hidden_node_channels)
        self.bn = BatchNorm1d(hidden_node_channels)

    def reset_parameters(self):
        self.lin_c1.reset_parameters()
        self.bn_c1.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, node_emb: Tensor, edge_emb: Tensor, i: Tensor) -> Tensor:
        c1 = torch.cat([node_emb[i], edge_emb], dim=1)
        c1 = self.bn_c1(self.lin_c1(c1))
        c1_filter, c1_core = c1.chunk(2, dim=1)
        c1_filter = c1_filter.sigmoid()
        c1_core = c1_core.tanh()
        c1_emb = scatter(c1_filter * c1_core, i, dim=0,
                         dim_size=node_emb.size(0), reduce='sum')
        c1_emb = self.bn(c1_emb)

        return (node_emb + c1_emb).tanh()


class EdgeBlock(torch.nn.Module):
    def __init__(self, hidden_node_channels: int, hidden_edge_channels: int):
        super().__init__()
        self.lin_c2 = Linear(hidden_node_channels, 2 * hidden_edge_channels)
        self.lin_c3 = Linear(
            3 * hidden_node_channels + 2 * hidden_edge_channels,
            2 * hidden_edge_channels,
        )

        # BN was added based on previous studies.
        # ref: https://github.com/txie-93/cgcnn/blob/master/cgcnn/model.py
        self.bn_c2 = BatchNorm1d(2 * hidden_edge_channels)
        self.bn_c3 = BatchNorm1d(2 * hidden_edge_channels)
        self.bn_c2_2 = BatchNorm1d(hidden_edge_channels)
        self.bn_c3_2 = BatchNorm1d(hidden_edge_channels)

    def reset_parameters(self):
        self.lin_c2.reset_parameters()
        self.lin_c3.reset_parameters()
        self.bn_c2.reset_parameters()
        self.bn_c3.reset_parameters()
        self.bn_c2_2.reset_parameters()
        self.bn_c3_2.reset_parameters()

    def forward(
        self,
        node_emb: Tensor,
        edge_emb: Tensor,
        i: Tensor,
        j: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        idx_k: Tensor,
        idx_ji: Tensor,
        idx_kj: Tensor,
    ) -> Tensor:
        c2 = node_emb[i] * node_emb[j]
        c2 = self.bn_c2(self.lin_c2(c2))
        c2_filter, c2_core = c2.chunk(2, dim=1)
        c2_filter = c2_filter.sigmoid()
        c2_core = c2_core.tanh()
        c2_emb = self.bn_c2_2(c2_filter * c2_core)

        c3 = torch.cat([
            node_emb[idx_i],
            node_emb[idx_j],
            node_emb[idx_k],
            edge_emb[idx_ji],
            edge_emb[idx_kj],
        ], dim=1)
        c3 = self.bn_c3(self.lin_c3(c3))
        c3_filter, c3_core = c3.chunk(2, dim=1)
        c3_filter = c3_filter.sigmoid()
        c3_core = c3_core.tanh()
        c3_emb = scatter(c3_filter * c3_core, idx_ji, dim=0,
                         dim_size=edge_emb.size(0), reduce='sum')
        c3_emb = self.bn_c3_2(c3_emb)

        return (edge_emb + c2_emb + c3_emb).tanh()


class GNNFF(torch.nn.Module):
    r"""The Graph Neural Network Force Field (GNNFF) from the
    `"Accurate and scalable graph neural network force field and molecular
    dynamics with direct force architecture"
    <https://www.nature.com/articles/s41524-021-00543-3>`_ paper.
    :class:`GNNFF` directly predicts atomic forces from automatically
    extracted features of the local atomic environment that are
    translationally-invariant, but rotationally-covariant to the coordinate of
    the atoms.

    Args:
        hidden_node_channels (int): Hidden node embedding size.
        hidden_edge_channels (int): Hidden edge embedding size.
        num_layers (int): Number of message passing blocks.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
    """
    def __init__(
        self,
        hidden_node_channels: int,
        hidden_edge_channels: int,
        num_layers: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        self.node_emb = Sequential(
            Embedding(95, hidden_node_channels),
            ShiftedSoftplus(),
            Linear(hidden_node_channels, hidden_node_channels),
            ShiftedSoftplus(),
            Linear(hidden_node_channels, hidden_node_channels),
        )
        self.edge_emb = GaussianFilter(0.0, 5.0, hidden_edge_channels)

        self.node_blocks = ModuleList([
            NodeBlock(hidden_node_channels, hidden_edge_channels)
            for _ in range(num_layers)
        ])
        self.edge_blocks = ModuleList([
            EdgeBlock(hidden_node_channels, hidden_edge_channels)
            for _ in range(num_layers)
        ])

        self.force_predictor = Sequential(
            Linear(hidden_edge_channels, hidden_edge_channels),
            ShiftedSoftplus(),
            Linear(hidden_edge_channels, hidden_edge_channels),
            ShiftedSoftplus(),
            Linear(hidden_edge_channels, 1),
        )

    def reset_parameters(self):
        reset(self.node_emb)
        self.edge_emb.reset_parameters()
        for node_block in self.node_blocks:
            node_block.reset_parameters()
        for edge_block in self.edge_blocks:
            edge_block.reset_parameters()
        reset(self.force_predictor)

    def forward(self, z: Tensor, pos: Tensor,
                batch: OptTensor = None) -> Tensor:
        """"""
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances and unit vector:
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        unit_vec = (pos[i] - pos[j]) / dist.view(-1, 1)

        # Embedding blocks:
        node_emb = self.node_emb(z)
        edge_emb = self.edge_emb(dist)

        # Message passing blocks:
        for node_block, edge_block in zip(self.node_blocks, self.edge_blocks):
            node_emb = node_block(node_emb, edge_emb, i)
            edge_emb = edge_block(node_emb, edge_emb, i, j, idx_i, idx_j,
                                  idx_k, idx_ji, idx_kj)

        # Force prediction block:
        force = self.force_predictor(edge_emb) * unit_vec

        return scatter(force, i, dim=0, reduce='sum')
