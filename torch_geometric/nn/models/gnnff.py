from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Sequential
from torch_scatter import scatter
from torch_sparse import SparseTensor

from torch_geometric.nn import radius_graph
from torch_geometric.typing import OptTensor


class AtomType2NodeEmb(torch.nn.Module):
    def __init__(self, num_atom_types: int, hidden_node_channels: int):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.lin_emb = Sequential(
            Linear(num_atom_types, hidden_node_channels),
            ShiftedSoftplus(),
            Linear(hidden_node_channels, hidden_node_channels),
            ShiftedSoftplus(),
            Linear(hidden_node_channels, hidden_node_channels),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.lin_emb(
            F.one_hot(z - 1, self.num_atom_types).to(torch.float))


class GaussianFilter(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class NodeUpdate(torch.nn.Module):
    def __init__(self, hidden_node_channels: int, hidden_edge_channels: int):
        super().__init__()
        self.lin_c1 = Linear(hidden_node_channels + hidden_edge_channels,
                             2 * hidden_node_channels)
        # BN was added based on previous studies.
        # ref: https://github.com/txie-93/cgcnn/blob/master/cgcnn/model.py
        self.bn_c1 = torch.nn.BatchNorm1d(2 * hidden_node_channels)
        self.bn = torch.nn.BatchNorm1d(hidden_node_channels)

    def forward(self, node_emb: Tensor, edge_emb: Tensor, i: Tensor) -> Tensor:
        c1 = torch.cat([node_emb[i], edge_emb], dim=1)
        c1 = self.bn_c1(self.lin_c1(c1))
        c1_filter, c1_core = c1.chunk(2, dim=1)
        c1_filter = torch.sigmoid(c1_filter)
        c1_core = torch.tanh(c1_core)
        c1_emb = self.bn(
            scatter(c1_filter * c1_core, i, dim=0, dim_size=node_emb.size(0)))

        return torch.tanh(node_emb + c1_emb)


class EdgeUpdate(torch.nn.Module):
    def __init__(self, hidden_node_channels: int, hidden_edge_channels: int):
        super().__init__()
        self.lin_c2 = Linear(hidden_node_channels, 2 * hidden_edge_channels)
        self.lin_c3 = Linear(
            3 * hidden_node_channels + 2 * hidden_edge_channels,
            2 * hidden_edge_channels)
        # BN was added based on previous studies.
        # ref: https://github.com/txie-93/cgcnn/blob/master/cgcnn/model.py
        self.bn_c2 = torch.nn.BatchNorm1d(2 * hidden_edge_channels)
        self.bn_c3 = torch.nn.BatchNorm1d(2 * hidden_edge_channels)
        self.bn_c2_2 = torch.nn.BatchNorm1d(hidden_edge_channels)
        self.bn_c3_2 = torch.nn.BatchNorm1d(hidden_edge_channels)

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
        c2_filter = torch.sigmoid(c2_filter)
        c2_core = torch.tanh(c2_core)
        c2_emb = self.bn_c2_2(c2_filter * c2_core)

        c3 = torch.cat(
            [
                node_emb[idx_i],
                node_emb[idx_j],
                node_emb[idx_k],
                edge_emb[idx_ji],
                edge_emb[idx_kj],
            ],
            dim=1,
        )
        c3 = self.bn_c3(self.lin_c3(c3))
        c3_filter, c3_core = c3.chunk(2, dim=1)
        c3_filter = torch.sigmoid(c3_filter)
        c3_core = torch.tanh(c3_core)
        c3_emb = self.bn_c3_2(
            scatter(c3_filter * c3_core, idx_ji, dim=0,
                    dim_size=edge_emb.size(0)))

        return torch.tanh(edge_emb + c2_emb + c3_emb)


class MessagePassing(torch.nn.Module):
    def __init__(self, hidden_node_channels: int, hidden_edge_channels: int):
        super().__init__()
        self.node_update = NodeUpdate(hidden_node_channels,
                                      hidden_edge_channels)
        self.edge_update = EdgeUpdate(hidden_node_channels,
                                      hidden_edge_channels)

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
        node_emb = self.node_update(node_emb, edge_emb, i)
        edge_emb = self.edge_update(node_emb, edge_emb, i, j, idx_i, idx_j,
                                    idx_k, idx_ji, idx_kj)

        return node_emb, edge_emb


class ForcePredictor(torch.nn.Module):
    def __init__(self, hidden_edge_channels: int):
        super().__init__()
        self.lin = Sequential(
            Linear(hidden_edge_channels, hidden_edge_channels),
            ShiftedSoftplus(),
            Linear(hidden_edge_channels, hidden_edge_channels),
            ShiftedSoftplus(),
            Linear(hidden_edge_channels, 1),
        )

    def forward(self, edge_emb: Tensor, unit_vec: Tensor):
        return self.lin(edge_emb) * unit_vec


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class GNNFF(torch.nn.Module):
    r"""Accurate and scalable graph neural network force field (GNNFF) from the
    `"Accurate and scalable graph neural network force field and molecular
    dynamics with direct force architecture"
    <https://www.nature.com/articles/s41524-021-00543-3>`_ paper.
    GNNFF directly predict atomic forces from automatically extracted features
    of the local atomic environment that are translationally-invariant,
    but rotationally-covariant to the coordinate of the atoms.

    Args:
        hidden_node_channels (int): Hidden node embedding size.
        hidden_edge_channels (int): Hidden edge embedding size.
        num_message_passings (int): Number of message passing blocks.
        cutoff (float): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int): Maximum number of neighbors
            for each atom. (default: :obj:`32`)
        num_atom_types (int): Number of atom types. (default: :obj:`100`)
    """
    def __init__(
        self,
        hidden_node_channels: int,
        hidden_edge_channels: int,
        num_message_passings: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        num_atom_types: int = 100,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.node_emb = AtomType2NodeEmb(num_atom_types, hidden_node_channels)
        self.edge_emb = GaussianFilter(0.0, 5.0, hidden_edge_channels)

        self.message_passings = ModuleList([
            MessagePassing(hidden_node_channels, hidden_edge_channels)
            for _ in range(num_message_passings)
        ])

        self.force_predictor = ForcePredictor(hidden_edge_channels)

    def reset_parameters(self):
        pass

    def triplets(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, z: Tensor, pos: Tensor,
                batch: OptTensor = None) -> Tensor:
        """"""
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances and unit vecs.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        unit_vec = (pos[i] - pos[j]) / dist.view(-1, 1)

        # Embedding block.
        node_emb = self.node_emb(z)
        edge_emb = self.edge_emb(dist)

        # Message passing blocks.
        for message_passing in self.message_passings:
            node_emb, edge_emb = message_passing(node_emb, edge_emb, i, j,
                                                 idx_i, idx_j, idx_k, idx_ji,
                                                 idx_kj)

        # Force prediction block.
        force = self.force_predictor(edge_emb, unit_vec)

        return scatter(force, i, dim=0)
