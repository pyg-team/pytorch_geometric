from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from torch_geometric.typing import EdgeTensorType
from torch_geometric.utils.mixin import CastMixin


class EdgeLayout(Enum):
    COO = 'coo'
    CSC = 'csc'
    CSR = 'csr'


@dataclass
class EdgeAttr(CastMixin):
    r"""Defines the attributes of an :obj:`GraphStore` edge."""

    # The type of the edge
    edge_type: Optional[Any]

    # The layout of the edge representation
    layout: EdgeLayout

    # Whether the edge index is sorted, by destination node. Useful for
    # avoiding sorting costs when performing neighbor sampling, and only
    # meaningful for COO (CSC and CSR are sorted by definition)
    is_sorted: bool = False

    # TODO support num_nodes here, default None, so users can specify this
    # instead of relying on default inferral

    def __init__(self, edge_type: Optional[Any], layout: EdgeLayout,
                 is_sorted: bool = False):
        self.edge_type = edge_type
        self.layout = EdgeLayout(layout)
        self.is_sorted = is_sorted


class GraphStore:
    def __init__(self, edge_attr_cls: Any = EdgeAttr):
        r"""Initializes the graph store. Implementor classes can customize the
        ordering and required nature of their :class:`EdgeAttr` edge attributes
        by subclassing :class:`EdgeAttr` and passing the subclass as
        :obj:`edge_attr_cls`."""
        super().__init__()
        self.__dict__['_edge_attr_cls'] = edge_attr_cls

    # Core ####################################################################

    @abstractmethod
    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        pass

    def put_edge_index(self, edge_index: EdgeTensorType, *args,
                       **kwargs) -> bool:
        r"""Synchronously adds an edge_index tensor to the graph store.

        Args:
            tensor(EdgeTensorType): an edge_index in a format specified in
            attr.
            **attr(EdgeAttr): the edge attributes.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        edge_attr.layout = EdgeLayout(edge_attr.layout)

        # Override is_sorted for CSC and CSR:
        edge_attr.is_sorted = edge_attr.is_sorted or (edge_attr.layout in [
            EdgeLayout.CSC, EdgeLayout.CSR
        ])
        return self._put_edge_index(edge_index, edge_attr)

    @abstractmethod
    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        pass

    def get_edge_index(self, *args, **kwargs) -> EdgeTensorType:
        r"""Synchronously gets an edge_index tensor from the materialized
        graph.

        Args:
            **attr(EdgeAttr): the edge attributes.

        Returns:
            EdgeTensorType: an edge_index tensor corresonding to the provided
            attributes, or None if there is no such tensor.

        Raises:
            KeyError: if the edge index corresponding to attr was not found.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        edge_attr.layout = EdgeLayout(edge_attr.layout)
        edge_index = self._get_edge_index(edge_attr)
        if edge_index is None:
            raise KeyError(f"An edge corresponding to '{edge_attr}' was not "
                           f"found")
        return edge_index

    # TODO implement coo(), csc(), csr() methods on GraphStore, which perform
    # conversions of edge indices between formats. These conversions can also
    # automatically be performed in `get_edge_index`

    # Additional methods ######################################################

    @abstractmethod
    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        pass

    # Python built-ins ########################################################

    def __setitem__(self, key: EdgeAttr, value: EdgeTensorType):
        key = self._edge_attr_cls.cast(key)
        self.put_edge_index(value, key)

    def __getitem__(self, key: EdgeAttr) -> Optional[EdgeTensorType]:
        key = self._edge_attr_cls.cast(key)
        return self.get_edge_index(key)
