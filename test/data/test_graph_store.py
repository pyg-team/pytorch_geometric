import torch

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
        return attr.edge_type or '<default>'

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        self.store[MyGraphStore.key(edge_attr)] = edge_index

    def _get_edge_index(self, edge_attr: EdgeAttr) -> EdgeTensorType:
        return self.store.get(MyGraphStore.key(edge_attr), None)


def test_graph_store():
    m = MyGraphStore()
    edge_index = torch.LongTensor([[0, 1], [1, 2]])

    # Put:
    m.put_edge_index(edge_index, layout=EdgeLayout.COO, edge_type='a')
    m['b', EdgeLayout.CSR] = edge_index

    # Get:
    assert torch.equal(m.get_edge_index(edge_type='a', layout=EdgeLayout.COO),
                       edge_index)

    assert torch.equal(m['b', EdgeLayout.CSR], edge_index)
