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

Major TODOs for future implementation:
* `sample` behind the graph store interface
"""
import copy
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.typing import (
    Adj,
    EdgeTensorType,
    EdgeType,
    OptTensor,
    SparseTensor,
)
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
        edge_type: Any,
        layout: EdgeLayout,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
    ):
        layout = EdgeLayout(layout)

        if layout == EdgeLayout.CSR and is_sorted:
            raise ValueError("Cannot create a CSR edge attribute with option "
                             "'is_sorted=True'")

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
        pass

    def remove_edge_index(self, *args, **kwargs) -> bool:
        r"""Synchronously deletes an :obj:`edge_index` tensor from the graph
        store.

        Args:
            attr (EdgeAttr): The edge attributes.
        """
        edge_attr = self._edge_attr_cls.cast(*args, **kwargs)
        return self._remove_edge_index(edge_attr)

    @abstractmethod
    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        r"""Returns all edge attributes stored in the graph store."""
        pass

    # Layout Conversion #######################################################

    def coo(
        self,
        edge_types: Optional[List[Any]] = None,
        replace: bool = False,
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
        return self._all_edges_to_layout(EdgeLayout.COO, edge_types, store)

    def csr(
        self,
        edge_types: Optional[List[Any]] = None,
        replace: bool = False,
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
        return self._all_edges_to_layout(EdgeLayout.CSR, edge_types, store)

    def csc(
        self,
        edge_types: Optional[List[Any]] = None,
        replace: bool = False,
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
        return self._all_edges_to_layout(EdgeLayout.CSC, edge_types, store)

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
        replace: bool = False,
    ) -> Tuple[Tensor, Tensor, OptTensor]:
        r"""Converts an :obj:`edge_index` in the graph store to the desired
        output layout, by fetching the :obj:`edge_index` and performing
        in-memory conversion. Implementations that support conversion within
        the graph store can override this method."""
        (row, col), perm = self.get_edge_index(attr), None

        if layout == EdgeLayout.COO:  # COO output:
            if attr.layout == EdgeLayout.CSR:
                row = ptr2ind(row, col.numel())
            elif attr.layout == EdgeLayout.CSC:
                col = ptr2ind(col, row.numel())

        elif layout == EdgeLayout.CSR:  # CSR output:
            if attr.layout == EdgeLayout.CSC:
                col = ptr2ind(col, row.numel())

            if attr.layout != EdgeLayout.CSR:
                perm = row.argsort()
                row, col, = row[perm], col[perm]
                num_rows = attr.size[0] if attr.size else int(row.max()) + 1
                row = ind2ptr(row, num_rows)

        else:  # CSC output:
            if attr.layout == EdgeLayout.CSR:
                row = ptr2ind(row, col.numel())

            if attr.layout == EdgeLayout.CSR or (attr.layout == EdgeLayout.COO
                                                 and not attr.is_sorted):
                perm = col.argsort()
                row, col, = row[perm], col[perm]

            if attr.layout != EdgeLayout.CSC:
                num_cols = attr.size[0] if attr.size else int(col.max()) + 1
                col = ind2ptr(col, num_cols)

        if replace:
            if perm is not None:
                warnings.warn(
                    f"The 'edge_index' of type '{attr.edge_type}' and layout "
                    f"'{attr.layout.value}' could not be converted to layout "
                    f"'{layout.value}' without permuting its data. As a "
                    f"result, the converted 'edge_index' is not being "
                    f"replaced in the graph store.")
            elif attr.layout != layout:
                self.remove_edge_index(attr)
                attr = copy.copy(attr)
                attr.layout = layout
                self.put_edge_index((row, col), attr)

        return row, col, perm

    def _edges_to_layout(
        self,
        layout: EdgeLayout,
        edge_types: Optional[List[Any]] = None,
        replace: bool = False,
    ) -> ConversionOutputType:
        r"""Converts all edge indices in the graph store to the desired output
        layout."""
        # Obtain all edge attributes, grouped by type:
        all_edge_attrs = self.get_all_edge_attrs()
        edge_type_to_attrs: Dict[EdgeType, List[EdgeAttr]] = defaultdict(list)
        for attr in all_edge_attrs:
            edge_type_to_attrs[attr.edge_type].append(attr)

        # Edge types to convert:
        for edge_type in (edge_types or []):
            if edge_type not in edge_type_to_attrs:
                raise ValueError(f"The 'edge_index' of type '{edge_type}' was "
                                 f"not found in the graph store.")
        edge_types = edge_types or [attr.edge_type for attr in all_edge_attrs]

        # Convert layout from its most favorable original layout:
        row_dict, col_dict, perm_dict = {}, {}, {}
        for edge_type in edge_types:
            attrs = edge_type_to_attrs[edge_type]
            layouts = [attr.layout for attr in attrs]

            if layout in layouts:  # No conversion needed.
                attr = attrs[layouts.index(layout)]
            elif EdgeLayout.COO in layouts:  # Prefer COO for conversion.
                attr = attrs[layouts.index(EdgeLayout.COO)]
            elif EdgeLayout.CSC in layouts:
                attr = attrs[layouts.index(EdgeLayout.CSC)]
            elif EdgeLayout.CSR in layouts:
                attr = attrs[layouts.index(EdgeLayout.CSR)]

            row, col, perm = self._edge_to_layout(attr, layout, replace)

            row_dict[edge_type] = row
            col_dict[edge_type] = col
            perm_dict[edge_type] = perm

        return row_dict, col_dict, perm_dict


# Data and HeteroData utilities ###############################################

EDGE_LAYOUT_TO_ATTR_NAME = {
    EdgeLayout.COO: 'edge_index',
    EdgeLayout.CSR: 'adj',
    EdgeLayout.CSC: 'adj_t',
}


def edge_tensor_type_to_adj_type(
    attr: EdgeAttr,
    tensor_tuple: EdgeTensorType,
) -> Adj:
    r"""Converts an EdgeTensorType tensor tuple to a PyG Adj tensor."""
    src, dst = tensor_tuple

    if attr.layout == EdgeLayout.COO:  # COO: (row, col)
        assert src.dim() == 1 and dst.dim() == 1 and src.numel() == dst.numel()

        if src.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=src.device)

        if (src[0].storage().data_ptr() == dst[1].storage().data_ptr()
                and src.storage_offset() < dst.storage_offset()):
            # Do not copy if the tensor tuple is constructed from the same
            # storage (instead, return a view):
            out = torch.empty(0, dtype=src.dtype)
            out.set_(src.storage(), storage_offset=src.storage_offset(),
                     size=(src.size()[0] + dst.size()[0], ))
            return out.view(2, -1)

        return torch.stack([src, dst], dim=0)

    elif attr.layout == EdgeLayout.CSR:  # CSR: (rowptr, col)
        return SparseTensor(rowptr=src, col=dst, is_sorted=True,
                            sparse_sizes=attr.size)

    elif attr.layout == EdgeLayout.CSC:  # CSC: (row, colptr)
        # CSC is a transposed adjacency matrix, so rowptr is the compressed
        # column and col is the uncompressed row.
        sparse_sizes = None if attr.size is None else (attr.size[1],
                                                       attr.size[0])
        return SparseTensor(rowptr=dst, col=src, is_sorted=True,
                            sparse_sizes=sparse_sizes)
    raise ValueError(f"Bad edge layout (got '{attr.layout}')")


def adj_type_to_edge_tensor_type(layout: EdgeLayout,
                                 edge_index: Adj) -> EdgeTensorType:
    r"""Converts a PyG Adj tensor to an EdgeTensorType equivalent."""
    if isinstance(edge_index, Tensor):
        return (edge_index[0], edge_index[1])  # (row, col)
    if layout == EdgeLayout.COO:
        return edge_index.coo()[:-1]  # (row, col)
    elif layout == EdgeLayout.CSR:
        return edge_index.csr()[:-1]  # (rowptr, col)
    else:
        return edge_index.csr()[-2::-1]  # (row, colptr)
