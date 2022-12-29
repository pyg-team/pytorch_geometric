r"""
This class defines the abstraction for a backend-agnostic graph store. The
goal of the graph store is to abstract away all graph edge index memory
management so that varying implementations can allow for independent scale-out.

This particular graph store abstraction makes a few key assumptions:
* The edge indices we care about storing are represented either in COO, CSC,
  or CSR format. They can be uniquely identified by an edge type (in PyG,
  this is a tuple of the source node, relation type, and destination node).
* Edge indices are static once they are stored in tthe grah. That is, we do not
  support dynamic modification of edge indices once they have been inserted
  into the graph store.

It is the job of a graph store implementor class to handle these assumptions
properly. For example, a simple in-memory graph store implementation may
concatenate all metadata values with an edge index and use this as a unique
index in a KV store. More complicated implementations may choose to partition
the graph in interesting manners based on the provided metadata.
"""
import copy
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.typing import EdgeTensorType, EdgeType, OptTensor
from torch_geometric.utils.mixin import CastMixin

# The output of converting between two types in the GraphStore is a Tuple of
# dictionaries: row, col, and perm. The dictionaries are keyed by the edge
# type of the input edge attribute.
#   * The row dictionary contains the row tensor for COO, the row pointer for
#     CSR, or the row tensor for CSC
#   * The col dictionary contains the col tensor for COO, the col tensor for
#     CSR, or the col pointer for CSC
#   * The perm dictionary contains the permutation of edges that was applied
#     in converting between formats, if applicable.
ConversionOutputType = Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor],
                             Dict[EdgeType, OptTensor]]

ptr2ind = torch.ops.torch_sparse.ptr2ind
ind2ptr = torch.ops.torch_sparse.ind2ptr


class EdgeLayout(Enum):
    COO = 'coo'
    CSC = 'csc'
    CSR = 'csr'


@dataclass
class EdgeAttr(CastMixin):
    r"""Defines the attributes of a :obj:`GraphStore` edge.
    It holds all the parameters necessary to uniquely identify an edge from
    the :class:`GraphStore`.

    Note that the order of the attributes is important; this is the order in
    which attributes must be provided for indexing calls. :class:`GraphStore`
    implementations can define a different ordering by overriding
    :meth:`EdgeAttr.__init__`.
    """

    # The type of the edge:
    edge_type: EdgeType

    # The layout of the edge representation:
    layout: EdgeLayout

    # Whether the edge index is sorted by destination node. Useful for
    # avoiding sorting costs when performing neighbor sampling, and only
    # meaningful for COO (CSC is sorted and CSR is not sorted by definition):
    is_sorted: bool = False

    # The number of source and destination nodes in this edge type:
    size: Optional[Tuple[int, int]] = None

    # NOTE we define __init__ to force-cast layout
    def __init__(
        self,
        edge_type: EdgeType,
        layout: EdgeLayout,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
    ):
        layout = EdgeLayout(layout)

        if layout == EdgeLayout.CSR and is_sorted:
            raise ValueError("Cannot create a 'CSR' edge attribute with "
                             "option 'is_sorted=True'")

        if layout == EdgeLayout.CSC:
            is_sorted = True

        self.edge_type = edge_type
        self.layout = layout
        self.is_sorted = is_sorted
        self.size = size


class GraphStore:
    r"""An abstract base class to access edges from a remote graph store.

    Args:
        edge_attr_cls (EdgeAttr, optional): A user-defined
            :class:`EdgeAttr` class to customize the required attributes and
            their ordering to uniquely identify edges. (default: :obj:`None`)
    """
    def __init__(self, edge_attr_cls: Optional[Any] = None):
        super().__init__()
        self.__dict__['_edge_attr_cls'] = edge_attr_cls or EdgeAttr

    # Core (CRUD) #############################################################

    @abstractmethod
    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        r"""To be implemented by :class:`GraphStore` subclasses."""
        pass

    def put_edge_index(self, edge_index: EdgeTensorType, *args,
                       **kwargs) -> bool:
        r"""Synchronously adds an :obj:`edge_index` tuple to the
        :class:`GraphStore`.
        Returns whether insertion was successful.

        Args:
            tensor (Tuple[torch.Tensor, torch.Tensor]): The :obj:`edge_index`
                tuple in a format specified in :class:`EdgeAttr`.
            **kwargs (EdgeAttr): Any relevant edge attributes that
                correspond to the :obj:`edge_index` tuple. See the
                :class:`EdgeAttr` documentation for required and optional
                attributes.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._put_edge_index(edge_index, edge_attr)

    @abstractmethod
    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        r"""To be implemented by :class:`GraphStore` subclasses."""
        pass

    def get_edge_index(self, *args, **kwargs) -> EdgeTensorType:
        r"""Synchronously obtains an :obj:`edge_index` tuple from the
        :class:`GraphStore`.

        Args:
            **kwargs (EdgeAttr): Any relevant edge attributes that
                correspond to the :obj:`edge_index` tuple. See the
                :class:`EdgeAttr` documentation for required and optional
                attributes.

        Raises:
            KeyError: If the :obj:`edge_index` corresponding to the input
                :class:`EdgeAttr` was not found.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        edge_index = self._get_edge_index(edge_attr)
        if edge_index is None:
            raise KeyError(f"'edge_index' for '{edge_attr}' not found")
        return edge_index

    @abstractmethod
    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        r"""To be implemented by :class:`GraphStore` subclasses."""
        pass

    def remove_edge_index(self, *args, **kwargs) -> bool:
        r"""Synchronously deletes an :obj:`edge_index` tuple from the
        :class:`GraphStore`.
        Returns whether deletion was successful.

        Args:
            **kwargs (EdgeAttr): Any relevant edge attributes that
                correspond to the :obj:`edge_index` tuple. See the
                :class:`EdgeAttr` documentation for required and optional
                attributes.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._remove_edge_index(edge_attr)

    @abstractmethod
    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        r"""Obtains all edge attributes stored in the :class:`GraphStore`."""
        pass

    # Layout Conversion #######################################################

    def coo(
        self,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        r"""Obtains the edge indices in the :class:`GraphStore` in COO
        format.

        Args:
            edge_types (List[Any], optional): The edge types of edge indices
                to obtain. If set to :obj:`None`, will return the edge indices
                of all existing edge types. (default: :obj:`None`)
            store (bool, optional): Whether to store converted edge indices in
                the :class:`GraphStore`. (default: :obj:`False`)
        """
        return self._edges_to_layout(EdgeLayout.COO, edge_types, store)

    def csr(
        self,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        r"""Obtains the edge indices in the :class:`GraphStore` in CSR
        format.

        Args:
            edge_types (List[Any], optional): The edge types of edge indices
                to obtain. If set to :obj:`None`, will return the edge indices
                of all existing edge types. (default: :obj:`None`)
            store (bool, optional): Whether to store converted edge indices in
                the :class:`GraphStore`. (default: :obj:`False`)
        """
        return self._edges_to_layout(EdgeLayout.CSR, edge_types, store)

    def csc(
        self,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        r"""Obtains the edge indices in the :class:`GraphStore` in CSC
        format.

        Args:
            edge_types (List[Any], optional): The edge types of edge indices
                to obtain. If set to :obj:`None`, will return the edge indices
                of all existing edge types. (default: :obj:`None`)
            store (bool, optional): Whether to store converted edge indices in
                the :class:`GraphStore`. (default: :obj:`False`)
        """
        return self._edges_to_layout(EdgeLayout.CSC, edge_types, store)

    # Python built-ins ########################################################

    def __setitem__(self, key: EdgeAttr, value: EdgeTensorType):
        self.put_edge_index(value, key)

    def __getitem__(self, key: EdgeAttr) -> Optional[EdgeTensorType]:
        return self.get_edge_index(key)

    def __delitem__(self, key: EdgeAttr):
        return self.remove_edge_index(key)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    # Helper methods ##########################################################

    def _edge_to_layout(
        self,
        attr: EdgeAttr,
        layout: EdgeLayout,
        store: bool = False,
    ) -> Tuple[Tensor, Tensor, OptTensor]:

        (row, col), perm = self.get_edge_index(attr), None

        if layout == EdgeLayout.COO:  # COO output requested:
            if attr.layout == EdgeLayout.CSR:  # CSR->COO
                row = ptr2ind(row, col.numel())
            elif attr.layout == EdgeLayout.CSC:  # CSC->COO
                col = ptr2ind(col, row.numel())

        elif layout == EdgeLayout.CSR:  # CSR output requested:
            if attr.layout == EdgeLayout.CSC:  # CSC->COO
                col = ptr2ind(col, row.numel())

            if attr.layout != EdgeLayout.CSR:  # COO->CSR
                row, perm = row.sort()  # Cannot be sorted by destination.
                col = col[perm]
                num_rows = attr.size[0] if attr.size else int(row.max()) + 1
                row = ind2ptr(row, num_rows)

        else:  # CSC output requested:
            if attr.layout == EdgeLayout.CSR:  # CSR->COO
                row = ptr2ind(row, col.numel())

            if attr.layout != EdgeLayout.CSC:  # COO->CSC
                if not attr.is_sorted:  # Not sorted by destination.
                    col, perm = col.sort()
                    row = row[perm]
                num_cols = attr.size[1] if attr.size else int(col.max()) + 1
                col = ind2ptr(col, num_cols)

        if attr.layout != layout and store:
            attr = copy.copy(attr)
            attr.layout = layout
            if perm is not None:
                attr.is_sorted = False
            self.put_edge_index((row, col), attr)

        return row, col, perm

    def _edges_to_layout(
        self,
        layout: EdgeLayout,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:

        # Obtain all edge attributes, grouped by type:
        edge_type_attrs: Dict[EdgeType, List[EdgeAttr]] = defaultdict(list)
        for attr in self.get_all_edge_attrs():
            edge_type_attrs[attr.edge_type].append(attr)

        # Check that requested edge types exist and filter:
        if edge_types is not None:
            for edge_type in edge_types:
                if edge_type not in edge_type_attrs:
                    raise ValueError(f"The 'edge_index' of type '{edge_type}' "
                                     f"was not found in the graph store.")

            edge_type_attrs = {
                key: attr
                for key, attr in edge_type_attrs.items() if key in edge_types
            }

        # Convert layout from its most favorable original layout:
        row_dict, col_dict, perm_dict = {}, {}, {}
        for edge_type, attrs in edge_type_attrs.items():
            layouts = [attr.layout for attr in attrs]

            if layout in layouts:  # No conversion needed.
                attr = attrs[layouts.index(layout)]
            elif EdgeLayout.COO in layouts:  # Prefer COO for conversion.
                attr = attrs[layouts.index(EdgeLayout.COO)]
            elif EdgeLayout.CSC in layouts:
                attr = attrs[layouts.index(EdgeLayout.CSC)]
            elif EdgeLayout.CSR in layouts:
                attr = attrs[layouts.index(EdgeLayout.CSR)]

            row_dict[edge_type], col_dict[edge_type], perm_dict[edge_type] = (
                self._edge_to_layout(attr, layout, store))

        return row_dict, col_dict, perm_dict
