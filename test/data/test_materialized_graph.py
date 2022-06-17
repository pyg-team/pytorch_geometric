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
        return f'{attr.edge_type or "<default>"}.{attr.layout}'

    @staticmethod
    def from_key(key: str) -> EdgeAttr:
        edge_type, layout = key.split('.')
        if edge_type == '<default>':
            edge_type = None
        return EdgeAttr(layout=EdgeLayout(layout), edge_type=edge_type)

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        self.store[MyMaterializedGraph.key(edge_attr)] = edge_index

    def _get_edge_index(self, edge_attr: EdgeAttr) -> EdgeTensorType:
        return self.store.get(MyMaterializedGraph.key(edge_attr), None)

    def get_all_edge_types(self):
        return [MyMaterializedGraph.from_key(key) for key in self.store]


def test_materialized_graph():
    m = MyMaterializedGraph()
    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    m.put_edge_index(edge_index, layout=EdgeLayout.COO, edge_type='a')
    assert torch.equal(m.get_edge_index(edge_type='a', layout=EdgeLayout.COO),
                       edge_index)
