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
from torch_sparse import SparseTensor

from torch_geometric.typing import Adj, EdgeTensorType, EdgeType, OptTensor
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
ConversionOutputType = Tuple[Dict[str, Tensor], Dict[str, Tensor],
                             Dict[str, OptTensor]]


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

    # The type of the edge
    edge_type: Optional[EdgeType]

    # The layout of the edge representation
    layout: Optional[EdgeLayout] = None

    # Whether the edge index is sorted, by destination node. Useful for
    # avoiding sorting costs when performing neighbor sampling, and only
    # meaningful for COO (CSC and CSR are sorted by definition)
    is_sorted: bool = False

    # The number of source and destination nodes in this edge type
    size: Optional[Tuple[int, int]] = None

    # NOTE we define __init__ to force-cast layout
    def __init__(
        self,
        edge_type: Optional[Any],
        layout: Optional[EdgeLayout] = None,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
    ):
        self.edge_type = edge_type
        self.layout = EdgeLayout(layout) if layout else None
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
        assert edge_attr.layout is not None
        edge_attr.layout = EdgeLayout(edge_attr.layout)

        # Override is_sorted for CSC and CSR:
        edge_attr.is_sorted = edge_attr.is_sorted or (edge_attr.layout in [
            EdgeLayout.CSC, EdgeLayout.CSR
        ])
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
        assert edge_attr.layout is not None
        edge_attr.layout = EdgeLayout(edge_attr.layout)
        # Override is_sorted for CSC and CSR:
        # TODO treat is_sorted specially in this function, where is_sorted=True
        # returns an edge index sorted by column.
        edge_attr.is_sorted = edge_attr.is_sorted or (edge_attr.layout in [
            EdgeLayout.CSC, EdgeLayout.CSR
        ])
        edge_index = self._get_edge_index(edge_attr)
        if edge_index is None:
            raise KeyError(f"An edge corresponding to '{edge_attr}' was not "
                           f"found")
        return edge_index

    @abstractmethod
    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        r"""Obtains all edge attributes stored in the :class:`GraphStore`."""
        pass

    # Layout Conversion #######################################################

    def _edge_to_layout(
        self,
        attr: EdgeAttr,
        layout: EdgeLayout,
    ) -> Tuple[Tensor, Tensor, OptTensor]:
        r"""Converts an :obj:`edge_index` tuple in the :class:`GraphStore` to
        the desired output layout by fetching the :obj:`edge_index` and
        performing in-memory conversion."""
        from_tuple = self.get_edge_index(attr)

        if layout == EdgeLayout.COO:
            if attr.layout == EdgeLayout.CSR:
                col = from_tuple[1]
                row = torch.ops.torch_sparse.ptr2ind(from_tuple[0],
                                                     col.numel())
            else:
                row = from_tuple[0]
                col = torch.ops.torch_sparse.ptr2ind(from_tuple[1],
                                                     row.numel())
            perm = None

        elif layout == EdgeLayout.CSR:
            # We convert to CSR by converting to CSC on the transpose
            if attr.layout == EdgeLayout.COO:
                adj = edge_tensor_type_to_adj_type(
                    attr, (from_tuple[1], from_tuple[0]))
            else:
                adj = edge_tensor_type_to_adj_type(attr, from_tuple).t()

            # NOTE we set is_sorted=False here as is_sorted refers to
            # the edge_index being sorted by the destination node
            # (column), but here we deal with the transpose
            attr_copy = copy.copy(attr)
            attr_copy.is_sorted = False
            attr_copy.size = None if attr.size is None else (attr.size[1],
                                                             attr.size[0])

            # Actually rowptr, col, perm
            row, col, perm = to_csc(adj, attr_copy, device='cpu')

        else:
            adj = edge_tensor_type_to_adj_type(attr, from_tuple)

            # Actually colptr, row, perm
            col, row, perm = to_csc(adj, attr, device='cpu')

        return row, col, perm

    # TODO support `replace` to replace the existing edge index.
    def _all_edges_to_layout(
        self,
        layout: EdgeLayout,
        edge_types: Optional[List[Any]] = None,
        store: bool = False,
    ) -> ConversionOutputType:
        r"""Converts all edge attributes in the graph store to the desired
        layout, by fetching all edge indices and performing conversion on
        the caller instance. Implementations that support conversion within
        the graph store can override this method."""
        # Obtain all edge attributes, grouped by type:
        all_edge_attrs = self.get_all_edge_attrs()
        edge_type_to_attrs: Dict[Any, List[EdgeAttr]] = defaultdict(list)
        for attr in all_edge_attrs:
            edge_type_to_attrs[attr.edge_type].append(attr)

        # Edge types to convert:
        edge_types = edge_types or [attr.edge_type for attr in all_edge_attrs]
        for edge_type in edge_types:
            if edge_type not in edge_type_to_attrs:
                raise ValueError(
                    f"The edge index {edge_type} was not found in the graph "
                    f"store.")

        # Convert layouts for each attribute from its most favorable original
        # layout to the desired layout. Store permutations of edges if
        # necessary as part of the conversion:
        row_dict, col_dict, perm_dict = {}, {}, {}
        for edge_type in edge_types:
            edge_type_attrs = edge_type_to_attrs[edge_type]
            edge_type_layouts = [attr.layout for attr in edge_type_attrs]

            # Ignore if requested layout is already present:
            if layout in edge_type_layouts:
                from_attr = edge_type_attrs[edge_type_layouts.index(layout)]
                row, col = self.get_edge_index(from_attr)
                perm = None

            # Convert otherwise:
            else:
                # Pick the most favorable layout to convert from. We prefer
                # COO to CSC/CSR:
                from_attr = None
                if EdgeLayout.COO in edge_type_layouts:
                    from_attr = edge_type_attrs[edge_type_layouts.index(
                        EdgeLayout.COO)]
                elif EdgeLayout.CSC in edge_type_layouts:
                    from_attr = edge_type_attrs[edge_type_layouts.index(
                        EdgeLayout.CSC)]
                else:
                    from_attr = edge_type_attrs[edge_type_layouts.index(
                        EdgeLayout.CSR)]

                row, col, perm = self._edge_to_layout(from_attr, layout)

            row_dict[from_attr.edge_type] = row
            col_dict[from_attr.edge_type] = col
            perm_dict[from_attr.edge_type] = perm

            if store and layout not in edge_type_layouts:
                # We do not store converted edge indices if this conversion
                # results in a permutation of nodes in the original edge index.
                # This is to exercise an abundance of caution in the case that
                # there are edge attributes.
                if perm is not None:
                    warnings.warn(f"Edge index {from_attr.edge_type} with "
                                  f"layout {from_attr.layout} was not sorted "
                                  f"by destination node, so conversion to "
                                  f"{layout} resulted in a permutation of "
                                  f"the order of edges. As a result, the "
                                  f"converted edge is not being re-stored in "
                                  f"the graph store. Please sort the edge "
                                  f"index and set 'is_sorted=True' to avoid "
                                  f"this warning.")
                else:
                    is_sorted = (layout != EdgeLayout.COO)
                    self.put_edge_index((row, col),
                                        EdgeAttr(from_attr.edge_type, layout,
                                                 is_sorted, from_attr.size))

        return row_dict, col_dict, perm_dict

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
        return self._all_edges_to_layout(EdgeLayout.COO, edge_types, store)

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
        return self._all_edges_to_layout(EdgeLayout.CSR, edge_types, store)

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
        return self._all_edges_to_layout(EdgeLayout.CSC, edge_types, store)

    # Python built-ins ########################################################

    def __setitem__(self, key: EdgeAttr, value: EdgeTensorType):
        key = self._edge_attr_cls.cast(key)
        self.put_edge_index(value, key)

    def __getitem__(self, key: EdgeAttr) -> Optional[EdgeTensorType]:
        key = self._edge_attr_cls.cast(key)
        return self.get_edge_index(key)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


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


###############################################################################


def to_csc(
    adj: Adj,
    edge_attr: EdgeAttr,
    device: Optional[torch.device] = None,
    share_memory: bool = False,
) -> Tuple[Tensor, Tensor, OptTensor]:
    # Convert the graph data into a suitable format for sampling (CSC format).
    # Returns the `colptr` and `row` indices of the graph, as well as an
    # `perm` vector that denotes the permutation of edges.
    # Since no permutation of edges is applied when using `SparseTensor`,
    # `perm` can be of type `None`.
    perm: Optional[Tensor] = None
    layout = edge_attr.layout
    is_sorted = edge_attr.is_sorted
    size = edge_attr.size

    if layout == EdgeLayout.CSR:
        colptr, row, _ = adj.csc()
    elif layout == EdgeLayout.CSC:
        colptr, row, _ = adj.csr()
    else:
        if size is None:
            raise ValueError(
                f"Edge {edge_attr.edge_type} cannot be converted "
                f"to a different type without specifying 'size' for "
                f"the source and destination node types (got {size}). "
                f"Please specify these parameters for successful execution.")
        (row, col) = adj
        if not is_sorted:
            perm = (col * size[0]).add_(row).argsort()
            row = row[perm]
        colptr = torch.ops.torch_sparse.ind2ptr(col[perm], size[1])

    colptr = colptr.to(device)
    row = row.to(device)
    perm = perm.to(device) if perm is not None else None

    if not colptr.is_cuda and share_memory:
        colptr.share_memory_()
        row.share_memory_()
        if perm is not None:
            perm.share_memory_()

    return colptr, row, perm
