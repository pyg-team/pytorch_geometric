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

from torch_geometric.typing import Adj, EdgeTensorType, OptTensor
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
    r"""Defines the attributes of an :obj:`GraphStore` edge."""

    # The type of the edge
    edge_type: Optional[Any]

    # The layout of the edge representation
    layout: EdgeLayout

    # Whether the edge index is sorted, by destination node. Useful for
    # avoiding sorting costs when performing neighbor sampling, and only
    # meaningful for COO (CSC and CSR are sorted by definition)
    is_sorted: bool = False

    # The number of nodes in this edge type. If set to None, will attempt to
    # infer with the simple heuristic int(self.edge_index.max()) + 1
    size: Optional[Tuple[int, int]] = None

    # NOTE we define __init__ to force-cast layout
    def __init__(
        self,
        edge_type: Optional[Any],
        layout: EdgeLayout,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
    ):
        self.edge_type = edge_type
        self.layout = EdgeLayout(layout)
        self.is_sorted = is_sorted
        self.size = size


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
        # Override is_sorted for CSC and CSR:
        edge_attr.is_sorted = edge_attr.is_sorted or (edge_attr.layout in [
            EdgeLayout.CSC, EdgeLayout.CSR
        ])
        edge_index = self._get_edge_index(edge_attr)
        if edge_index is None:
            raise KeyError(f"An edge corresponding to '{edge_attr}' was not "
                           f"found")
        return edge_index

    # Layout Conversion #######################################################

    # TODO support `replace` to replace the existing edge index.
    def _to_layout(self, layout: EdgeLayout,
                   store: bool = False) -> ConversionOutputType:
        # Obtain all edge attributes, grouped by type:
        edge_attrs = self.get_all_edge_attrs()
        edge_type_to_attrs: Dict[Any, List[EdgeAttr]] = defaultdict(list)
        for attr in edge_attrs:
            edge_type_to_attrs[attr.edge_type].append(attr)

        # Convert layouts for each attribute from its most favorable original
        # layout to the desired layout. Store permutations of edges if
        # necessary as part of the conversion:
        row_dict, col_dict, perm_dict = {}, {}, {}
        for edge_attrs in edge_type_to_attrs.values():
            edge_layouts = [edge_attr.layout for edge_attr in edge_attrs]

            # Ignore if requested layout is already present:
            if layout in edge_layouts:
                from_attr = edge_attrs[edge_layouts.index(layout)]
                row, col = self.get_edge_index(from_attr)
                perm = None

            # Convert otherwise:
            else:
                # Pick the most favorable layout to convert from. We prefer
                # COO to CSC/CSR:
                from_attr = None
                if EdgeLayout.COO in edge_layouts:
                    from_attr = edge_attrs[edge_layouts.index(EdgeLayout.COO)]
                elif EdgeLayout.CSC in edge_layouts:
                    from_attr = edge_attrs[edge_layouts.index(EdgeLayout.CSC)]
                else:
                    from_attr = edge_attrs[edge_layouts.index(EdgeLayout.CSR)]

                from_tuple = self.get_edge_index(from_attr)

                # Convert to the new layout:
                if layout == EdgeLayout.COO:
                    if from_attr.layout == EdgeLayout.CSR:
                        col = from_tuple[1]
                        row = torch.ops.torch_sparse.ptr2ind(
                            from_tuple[0], col.numel())
                    else:
                        row = from_tuple[0]
                        col = torch.ops.torch_sparse.ptr2ind(
                            from_tuple[1], row.numel())
                    perm = None

                elif layout == EdgeLayout.CSR:
                    # We convert to CSR by converting to CSC on the transpose
                    if from_attr.layout == EdgeLayout.COO:
                        adj = edge_tensor_type_to_adj_type(
                            from_attr, (from_tuple[1], from_tuple[0]))
                    else:
                        adj = edge_tensor_type_to_adj_type(
                            from_attr, from_tuple).t()

                    # NOTE we set is_sorted=False here as is_sorted refers to
                    # the edge_index being sorted by the destination node
                    # (column), but here we deal with the transpose
                    from_attr_copy = copy.copy(from_attr)
                    from_attr_copy.is_sorted = False
                    from_attr_copy.size = None if from_attr.size is None else (
                        from_attr.size[1], from_attr.size[0])

                    # Actually rowptr, col, perm
                    row, col, perm = to_csc(adj, from_attr_copy, device='cpu')

                else:
                    adj = edge_tensor_type_to_adj_type(from_attr, from_tuple)

                    # Actually colptr, row, perm
                    col, row, perm = to_csc(adj, from_attr, device='cpu')

            row_dict[from_attr.edge_type] = row
            col_dict[from_attr.edge_type] = col
            perm_dict[from_attr.edge_type] = perm

            if store and layout not in edge_layouts:
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
                    self._put_edge_index((row, col),
                                         EdgeAttr(from_attr.edge_type, layout,
                                                  is_sorted, from_attr.size))

        return row_dict, col_dict, perm_dict

    def coo(self, store: bool = False) -> ConversionOutputType:
        r"""Converts the edge indices in the graph store to COO format,
        optionally storing the converted edge indices in the graph store."""
        return self._to_layout(EdgeLayout.COO, store)

    def csr(self, store: bool = False) -> ConversionOutputType:
        r"""Converts the edge indices in the graph store to CSR format,
        optionally storing the converted edge indices in the graph store."""
        return self._to_layout(EdgeLayout.CSR, store)

    def csc(self, store: bool = False) -> ConversionOutputType:
        r"""Converts the edge indices in the graph store to CSC format,
        optionally storing the converted edge indices in the graph store."""
        return self._to_layout(EdgeLayout.CSC, store)

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

    if attr.layout == EdgeLayout.COO:
        # COO: (row, col)
        if (src[0].storage().data_ptr() == dst[1].storage().data_ptr()
                and src.storage_offset() < dst.storage_offset()):
            # Do not copy if the tensor tuple is constructed from the same
            # storage (instead, return a view):
            out = torch.empty(0, dtype=src.dtype)
            out.set_(src.storage(), storage_offset=src.storage_offset(),
                     size=(src.size()[0] + dst.size()[0], ))
            return out.view(2, -1)
        return torch.stack(tensor_tuple)
    elif attr.layout == EdgeLayout.CSR:
        # CSR: (rowptr, col)
        return SparseTensor(rowptr=src, col=dst, is_sorted=True,
                            sparse_sizes=attr.size)
    elif attr.layout == EdgeLayout.CSC:
        # CSC: (row, colptr) is a transposed adjacency matrix, so rowptr
        # is the compressed column and col is the uncompressed row.
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
                f"Please specify these parameters for successful execution. ")
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
