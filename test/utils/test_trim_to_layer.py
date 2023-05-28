from typing import List, Optional

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphConv
from torch_geometric.testing import withPackage
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import trim_to_layer
from torch_geometric.utils.trim_to_layer import trim_sparse_tensor


@withPackage('torch_sparse')
def test_trim_sparse_tensor():
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]])
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=[5, 5])

    adj = trim_sparse_tensor(adj, num_nodes=3, num_seed_nodes=1)

    row, col, _ = adj.coo()
    assert row.tolist() == [0, 0]
    assert col.tolist() == [1, 2]


def test_trim_to_layer_basic():
    x0 = torch.arange(4)
    edge_index0 = torch.tensor([[1, 2, 3], [0, 1, 2]])
    edge_weight0 = torch.arange(3)

    num_sampled_nodes_per_hop = [1, 1, 1]
    num_sampled_edges_per_hop = [1, 1, 1]

    x1, edge_index1, edge_weight1 = trim_to_layer(
        layer=0,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        x=x0,
        edge_index=edge_index0,
        edge_attr=edge_weight0,
    )
    assert torch.equal(x1, torch.arange(4))
    assert edge_index1.tolist() == [[1, 2, 3], [0, 1, 2]]
    assert torch.equal(edge_weight1, torch.arange(3))

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj0 = SparseTensor.from_edge_index(edge_index0, edge_weight0, (4, 4))
        x1, adj_t1, _ = trim_to_layer(
            layer=0,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            x=x0,
            edge_index=adj0.t(),
            edge_attr=edge_weight0,
        )
        adj1 = adj_t1.t()
        assert adj1.sizes() == [4, 4]

        row, col, value = adj1.coo()
        assert torch.equal(x1, torch.arange(4))
        assert row.tolist() == [1, 2, 3]
        assert col.tolist() == [0, 1, 2]
        assert torch.equal(value, torch.arange(3))

    x2, edge_index2, edge_weight2 = trim_to_layer(
        layer=1,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        x=x1,
        edge_index=edge_index1,
        edge_attr=edge_weight1,
    )
    assert torch.equal(x2, torch.arange(3))
    assert edge_index2.tolist() == [[1, 2], [0, 1]]
    assert torch.equal(edge_weight2, torch.arange(2))

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj1 = SparseTensor.from_edge_index(edge_index1, edge_weight1, (4, 4))
        x2, adj_t2, _ = trim_to_layer(
            layer=1,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            x=x1,
            edge_index=adj1.t(),
        )
        adj2 = adj_t2.t()
        assert adj2.sizes() == [3, 3]

        row, col, value = adj2.coo()
        assert torch.equal(x2, torch.arange(3))
        assert row.tolist() == [1, 2]
        assert col.tolist() == [0, 1]
        assert torch.equal(value, torch.arange(2))

    x3, edge_index3, edge_weight3 = trim_to_layer(
        layer=2,
        num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        x=x2,
        edge_index=edge_index2,
        edge_attr=edge_weight2,
    )
    assert torch.equal(x3, torch.arange(2))
    assert edge_index3.tolist() == [[1], [0]]
    assert torch.equal(edge_weight3, torch.arange(1))

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index2, edge_weight2, (3, 3))
        x3, adj_t3, _ = trim_to_layer(
            layer=2,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
            x=x2,
            edge_index=adj2.t(),
        )
        adj3 = adj_t3.t()
        assert adj3.sizes() == [2, 2]

        row, col, value = adj3.coo()
        assert torch.equal(x3, torch.arange(2))
        assert row.tolist() == [1]
        assert col.tolist() == [0]
        assert torch.equal(value, torch.arange(1))


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
