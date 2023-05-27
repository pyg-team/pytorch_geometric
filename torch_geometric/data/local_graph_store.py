from typing import Dict, List, Optional, Tuple

from torch import Tensor

from torch_geometric.data import EdgeAttr, GraphStore
from torch_geometric.typing import EdgeTensorType


class LocalGraphStore(GraphStore):
    r""" This class extends from GraphStore and use the store dict to save 
    the graph topology and also use edge_ids dict to save the global eids
    """
    def __init__(self):
        super().__init__()
        self.store: Dict[Tuple, Tuple[Tensor, Tensor]] = {}
        r""" save the graph topo - edge_index """
        
        self.edge_ids: Dict[Tuple, Tuple[Tensor, Tensor]] = {}
        r""" save the edge ids """

    @staticmethod
    def key(attr: EdgeAttr) -> Tuple:
        return (attr.edge_type, attr.layout.value, attr.is_sorted, attr.size)

    def set_edge_ids(self, edge_ids: Tensor,
                       edge_attr: EdgeAttr):
        self.edge_ids[self.key(edge_attr)] = edge_ids

    def get_edge_ids(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        return self.edge_ids.get(self.key(edge_attr), None)

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        self.store[self.key(edge_attr)] = edge_index

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        return self.store.get(self.key(edge_attr), None)

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        return self.store.pop(self.key(edge_attr), None) is not None

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        return [EdgeAttr(*key) for key in self.store.keys()]
