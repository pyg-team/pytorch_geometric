from typing import Dict, List, Optional, Tuple

from torch import Tensor

from torch_geometric.data import EdgeAttr, GraphStore
from torch_geometric.typing import EdgeTensorType


class LocalGraphStore(GraphStore):
    r"""This class implements the :class:`torch_geometric.data.GraphStore`
    interface to act as a local graph store for distributed training.
    """
    def __init__(self):
        super().__init__()
        self._edge_index: Dict[Tuple, EdgeTensorType] = {}
        self._edge_id: Dict[Tuple, Tensor] = {}

    @staticmethod
    def key(attr: EdgeAttr) -> Tuple:
        return (attr.edge_type, attr.layout.value, attr.is_sorted, attr.size)

    def put_edge_id(self, edge_id: Tensor, *args, **kwargs) -> bool:
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        self._edge_id[self.key(edge_attr)] = edge_id
        return True

    def get_edge_id(self, *args, **kwargs) -> Optional[EdgeTensorType]:
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._edge_id[self.key(edge_attr)]

    def remove_edge_id(self, *args, **kwargs) -> bool:
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._edge_id.pop(self.key(edge_attr), None) is not None

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        self._edge_index[self.key(edge_attr)] = edge_index
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        return self._edge_index.get(self.key(edge_attr), None)

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        return self._edge_index.pop(self.key(edge_attr), None) is not None

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        return [EdgeAttr(*key) for key in self._edge_index.keys()]
