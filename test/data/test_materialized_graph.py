import torch

from torch_geometric.data.materialized_graph import (
    EdgeAttr,
    EdgeLayout,
    EdgeTensorType,
    MaterializedGraph,
)


class MyMaterializedGraph(MaterializedGraph):
    def __init__(self):
        super().__init__()
        self.store = {}

    @staticmethod
    def key(attr: EdgeAttr) -> str:
        return attr.edge_type or '<default>'

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        self.store[MyMaterializedGraph.key(edge_attr)] = edge_index

    def _get_edge_index(self, edge_attr: EdgeAttr) -> EdgeTensorType:
        return self.store.get(MyMaterializedGraph.key(edge_attr), None)

    def get_all_edge_types(self):
        if len(self.store) == 0 and self.store[0] == '<default>':
            return None
        return list(self.store.keys())


def test_materialized_graph():
    m = MyMaterializedGraph()
    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    m.put_edge_index(edge_index, layout=EdgeLayout.COO, edge_type='a')
    assert torch.equal(m.get_edge_index(edge_type='a'), edge_index)
