from typing import Dict, Optional, Tuple

from torch import Tensor

from torch_geometric.data import EdgeAttr, GraphStore
from torch_geometric.typing import EdgeTensorType


class MyGraphStore(GraphStore):
    def __init__(self):
        super().__init__()
        self.store: Dict[EdgeAttr, Tuple[Tensor, Tensor]] = {}

    @staticmethod
    def key(attr: EdgeAttr) -> str:
        return (attr.edge_type, attr.layout.value, attr.is_sorted, attr.size)

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        self.store[MyGraphStore.key(edge_attr)] = edge_index

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        return self.store.get(MyGraphStore.key(edge_attr), None)

    def get_all_edge_attrs(self):
        return [EdgeAttr(*key) for key in self.store]
