import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.testing.graph_store import MyGraphStore


def test_graph_store():
    graph_store = MyGraphStore()
    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])

    def assert_equal_tensor_tuple(expected, actual):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            assert torch.equal(expected[i], actual[i])

    # We put all three tensor types: COO, CSR, and CSC, and we get them back
    # to confirm that `GraphStore` works as intended.
    coo = adj.coo()[:-1]
    csr = adj.csr()[:-1]
    csc = adj.csc()[-2::-1]  # (row, colptr)

    # Put:
    graph_store['edge', EdgeLayout.COO] = coo
    graph_store['edge', 'csr'] = csr
    graph_store['edge', 'csc'] = csc

    # Get:
    assert_equal_tensor_tuple(coo, graph_store['edge', 'coo'])
    assert_equal_tensor_tuple(csr, graph_store['edge', 'csr'])
    assert_equal_tensor_tuple(csc, graph_store['edge', 'csc'])

    # Get attrs:
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 3

    with pytest.raises(KeyError):
        _ = graph_store['edge_2', 'coo']
