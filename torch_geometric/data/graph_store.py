from abc import abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from torch import LongTensor

from torch_geometric.typing import EdgeTensorType, EdgeType
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

    # Whether the edge index is sorted
    is_sorted: bool = False


class GraphStore(MutableMapping):
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
        return self._put_edge_index(edge_index, edge_attr)

    @abstractmethod
    def _get_edge_index(self, edge_attr: EdgeAttr) -> EdgeTensorType:
        return None

    def get_edge_index(self, *args, **kwargs) -> Optional[EdgeTensorType]:
        r"""Synchronously gets an edge_index tensor from the materialized
        graph.

        Args:
            **attr(EdgeAttr): the edge attributes.

        Returns:
            EdgeTensorType: an edge_index tensor corresonding to the provided
            attributes, or None if there is no such tensor.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        edge_attr.layout = EdgeLayout(edge_attr.layout)
        return self._get_edge_index(edge_attr)

    # TODO implement coo(), csc(), csr() methods on GraphStore, which perform
    # conversions of edge indices between formats. These conversions can also
    # automatically be performed in `get_edge_index`

    @abstractmethod
    def get_all_edge_types(self) -> List[EdgeAttr]:
        r"""Obtain all edge attributes of edges that are stored in the
        materialized graph."""
        raise NotImplementedError

    def sample(
        self,
        input_nodes: Union[LongTensor, Dict[EdgeType, LongTensor]],
        *args,
        **kwargs,
    ) -> None:
        r"""Samples the graph store to obtain a sampled subgraph, utilizing
        sampling routines provided by `torch_sparse`.

        Args:
            input_nodes: the input nodes to sample from, either provided as a
                :class:`torch.LongTensor` or a dictionary mapping node type to
                a list of input nodes.

        Notes:
            Most sampling methods operate efficiently on sorted edge indices.
            This class makes no guarantees on the formats of the edge indices
            that are passed as input. Implementations should account for this
            (e.g. by converting all edge indices to CSC) before sampling.
        """
        # NOTE for the future: this is expected to call a C++ implementation
        # in pyg-lib.
        return None

    # Python built-ins ########################################################

    def __setitem__(self, key: EdgeAttr, value: EdgeTensorType):
        key = self._edge_attr_cls.cast(key)
        self.put_edge_index(value, key)

    def __getitem__(self, key: EdgeAttr) -> EdgeTensorType:
        key = self._edge_attr_cls.cast(key)
        return self.get_edge_index(key)

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
