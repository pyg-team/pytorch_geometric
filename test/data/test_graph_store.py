import torch
from torch_sparse import SparseTensor

from torch_geometric.data.graph_store import (
    EdgeAttr,
    EdgeLayout,
    EdgeTensorType,
    GraphStore,
)


class MyGraphStore(GraphStore):
    def __init__(self):
        super().__init__()
        self.store = {}

    @staticmethod
    def key(attr: EdgeAttr) -> str:
        return (attr.edge_type or '<default>') + '_' + str(attr.layout)

    @staticmethod
    def from_key(key: str) -> EdgeAttr:
        edge_type, layout = key.split('_')
        if edge_type == '<default>':
            edge_type = None
        return EdgeAttr(layout=EdgeLayout(layout.split('.')[1]),
                        edge_type=edge_type)

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        self.store[MyGraphStore.key(edge_attr)] = edge_index

    def _get_edge_index(self, edge_attr: EdgeAttr) -> EdgeTensorType:
        return self.store.get(MyGraphStore.key(edge_attr), None)

    def get_all_edge_types(self):
        return [MyGraphStore.from_key(key) for key in self.store]


def test_graph_store():
    graph_store = MyGraphStore()
    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])

    def assert_edge_tensor_type_equal(expected: EdgeTensorType,
                                      actual: EdgeTensorType):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            assert torch.equal(expected[i], actual[i])

    # We put all three tensor types: COO, CSR, and CSC, and we get them back
    # to confirm that `GraphStore` works as intended.
    coo = adj.coo()[:-1]
    csr = adj.csr()[:-1]
    csc = adj.t().csr()[:-1]

    # Put:
    graph_store['edge', 'coo'] = coo
    graph_store['edge', 'csr'] = csr
    graph_store['edge', 'csc'] = csc

    # Get:
    assert_edge_tensor_type_equal(coo, graph_store['edge', 'coo'])
    assert_edge_tensor_type_equal(csr, graph_store['edge', 'csr'])
    assert_edge_tensor_type_equal(csc, graph_store['edge', 'csc'])
