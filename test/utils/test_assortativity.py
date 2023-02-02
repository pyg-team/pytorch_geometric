import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import assortativity


def test_assortativity():
    # completely assortative graph
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    out = assortativity(edge_index)
    assert pytest.approx(out, abs=1e-5) == 1.0

    # SpraseTensor version
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)
    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=6)
    data = ToSparseTensor()(data)
    edge_index_as_sparse = SparseTensor(row=data.adj_t.storage.row(),
                                        col=data.adj_t.storage.col(),
                                        value=data.adj_t.storage.value())
    out = assortativity(edge_index_as_sparse)
    assert pytest.approx(out, abs=1e-5) == 1.0

    # completely disassortative graph
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 5, 5, 5, 5],
                               [5, 5, 5, 5, 5, 0, 1, 2, 3, 4]])
    out = assortativity(edge_index)
    assert pytest.approx(out, abs=1e-5) == -1.0

    # SpraseTensor version
    edge_weight = torch.randn(edge_index.size(1))
    edge_attr = torch.randn(edge_index.size(1), 8)
    data = Data(edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, num_nodes=6)
    data = ToSparseTensor()(data)
    edge_index_as_sparse = SparseTensor(row=data.adj_t.storage.row(),
                                        col=data.adj_t.storage.col(),
                                        value=data.adj_t.storage.value())
    out = assortativity(edge_index_as_sparse)
    assert pytest.approx(out, abs=1e-5) == -1.0
