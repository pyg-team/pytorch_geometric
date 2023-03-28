from typing import List, Optional

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphConv
from torch_geometric.testing import withPackage
from torch_geometric.utils import trim_to_layer


def test_trim_to_layer_basic():
    x = torch.arange(4)
    edge_index = torch.tensor([[1, 2, 3], [0, 1, 2]])
    edge_weight = torch.arange(3)

    num_sampled_nodes_per_hop = [1, 1, 1, 1]
    num_sampled_edges_per_hop = [1, 1, 1]

    x, edge_index, edge_weight = trim_to_layer(
        layer=0,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
    )
    assert torch.equal(x, torch.arange(4))
    assert edge_index.tolist() == [[1, 2, 3], [0, 1, 2]]
    assert torch.equal(edge_weight, torch.arange(3))

    x, edge_index, edge_weight = trim_to_layer(
        layer=1,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
    )
    assert torch.equal(x, torch.arange(3))
    assert edge_index.tolist() == [[1, 2], [0, 1]]
    assert torch.equal(edge_weight, torch.arange(2))

    x, edge_index, edge_weight = trim_to_layer(
        layer=2,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
    )
    assert torch.equal(x, torch.arange(2))
    assert edge_index.tolist() == [[1], [0]]
    assert torch.equal(edge_weight, torch.arange(1))


def test_trim_to_layer_hetero():
    x = {'v': torch.arange(4)}
    edge_index = {('v', 'to', 'v'): torch.tensor([[1, 2, 3], [0, 1, 2]])}
    edge_weight = {('v', 'to', 'v'): torch.arange(3)}

    num_sampled_nodes_per_hop = {'v': [1, 1, 1, 1]}
    num_sampled_edges_per_hop = {('v', 'to', 'v'): [1, 1, 1]}

    x, edge_index, edge_weight = trim_to_layer(
        layer=1,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
    )
    assert torch.equal(x['v'], torch.arange(3))
    assert edge_index['v', 'to', 'v'].tolist() == [[1, 2], [0, 1]]
    assert torch.equal(edge_weight['v', 'to', 'v'], torch.arange(2))


class GNN(torch.nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()

        self.convs = torch.nn.ModuleList(
            GraphConv(16, 16) for _ in range(num_layers))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_sampled_nodes: Optional[List[int]] = None,
        num_sampled_edges: Optional[List[int]] = None,
    ) -> Tensor:
        for i, conv in enumerate(self.convs):
            if num_sampled_nodes is not None:
                x, edge_index, edge_weight = trim_to_layer(
                    i, num_sampled_nodes, num_sampled_edges, x, edge_index,
                    edge_weight)
            x = conv(x, edge_index, edge_weight)
        return x


@withPackage('pyg_lib')
def test_trim_to_layer_with_neighbor_loader():
    x = torch.randn(14, 16)
    edge_index = torch.tensor([
        [2, 3, 4, 5, 7, 7, 10, 11, 12, 13],
        [0, 1, 2, 3, 2, 3, 7, 7, 7, 7],
    ])
    edge_weight = torch.rand(edge_index.size(1))
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    loader = NeighborLoader(
        data,
        num_neighbors=[1, 2, 4],
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))

    model = GNN(num_layers=3)
    out1 = model(batch.x, batch.edge_index, batch.edge_weight)[:2]
    assert out1.size() == (2, 16)

    out2 = model(batch.x, batch.edge_index, batch.edge_weight,
                 batch.num_sampled_nodes, batch.num_sampled_edges)[:2]
    assert out2.size() == (2, 16)

    assert torch.allclose(out1, out2)
