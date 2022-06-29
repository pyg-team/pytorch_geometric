import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.typing import Adj, EdgeTensorType, OptTensor
from torch_geometric.utils.mixin import CastMixin

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
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None

    # NOTE we define __init__ to force-cast layout
    def __init__(
        self,
        edge_type: Optional[Any],
        layout: EdgeLayout,
        is_sorted: bool = False,
        num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        self.edge_type = edge_type
        self.layout = EdgeLayout(layout)
        self.is_sorted = is_sorted
        self.num_nodes = num_nodes

    @property
    def num_nodes_tuple(self):
        if self.num_nodes is None or isinstance(self.num_nodes, tuple):
            return self.num_nodes
        return (self.num_nodes, self.num_nodes)


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

    def _to_layout(self, to_layout: EdgeLayout,
                   store: bool = False) -> ConversionOutputType:
        class _DataArgument(object):
            pass

        # Obtain all edge attributes, grouped by type:
        edge_attrs = self.get_all_edge_attrs()
        edge_type_to_attrs: Dict[Any, List[EdgeAttr]] = defaultdict(list)
        for attr in edge_attrs:
            edge_type_to_attrs[attr.edge_type].append(attr)

        # Convert layouts for each attribute from its most favorable original
        # layout to the desired layout. Store permutations of edges if
        # necessary as part of the conversion:
        row_dict, col_dict, perm_dict = {}, {}, {}
        for _, edge_attrs in edge_type_to_attrs.items():
            edge_layouts = [edge_attr.layout for edge_attr in edge_attrs]

            # Ignore if requested layout is already present:
            if to_layout in edge_layouts:
                from_attr = edge_attrs[edge_layouts.index(to_layout)]
                row, col = self.get_edge_index(from_attr)
                perm = None

            # Convert otherwise
            else:
                # Pick the most favorable layout to convert from. We always
                # prefer CSR or CSC to COO:
                from_attr = None
                if EdgeLayout.CSC in edge_layouts:
                    from_attr = edge_attrs[edge_layouts.index(EdgeLayout.CSC)]
                elif EdgeLayout.CSR in edge_layouts:
                    from_attr = edge_attrs[edge_layouts.index(EdgeLayout.CSR)]
                else:
                    from_attr = edge_attrs[edge_layouts.index(EdgeLayout.COO)]

                from_tuple = self.get_edge_index(from_attr)

                # Convert to the new layout:
                if to_layout == EdgeLayout.COO:
                    if from_attr.layout == EdgeLayout.CSR:
                        adj = SparseTensor(rowptr=from_tuple[0],
                                           col=from_tuple[1])
                    else:
                        adj = SparseTensor(rowptr=from_tuple[1],
                                           col=from_tuple[0]).t()
                    row, col = adj.coo()
                    perm = None

                elif to_layout == EdgeLayout.CSR:
                    data_argument = _DataArgument()

                    # We convert to CSR by converting to CSC on the transpose
                    if from_attr.layout == EdgeLayout.COO:
                        adj = edge_tensor_type_to_adj_type(
                            from_attr, (from_tuple[1], from_tuple[0]))
                    else:
                        adj = edge_tensor_type_to_adj_type(
                            from_attr, from_tuple)
                        adj = adj.t()

                    attr_name = EDGE_LAYOUT_TO_ATTR_NAME[from_attr.layout]
                    setattr(data_argument, attr_name, adj)

                    # Actually rowptr, col, perm
                    row, col, perm = to_csc(
                        data_argument, device='cpu',
                        is_sorted=from_attr.is_sorted,
                        num_nodes=from_attr.num_nodes_tuple)

                else:
                    data_argument = _DataArgument()
                    adj = edge_tensor_type_to_adj_type(from_attr, from_tuple)
                    attr_name = EDGE_LAYOUT_TO_ATTR_NAME[from_attr.layout]
                    setattr(data_argument, attr_name, adj)

                    # Actually colptr, row, perm
                    col, row, perm = to_csc(
                        data_argument, device='cpu',
                        is_sorted=from_attr.is_sorted,
                        num_nodes=from_attr.num_nodes_tuple)

            row_dict[from_attr.edge_type] = row
            col_dict[from_attr.edge_type] = col
            perm_dict[from_attr.edge_type] = perm

            if store and to_layout not in edge_layouts:
                # We do not store converted edge indices if this conversion
                # results in a permutation of nodes in the original edge index.
                # This is to exercise an abundance of caution in the case that
                # there are edge attributes.
                if perm is not None:
                    warnings.warn(f"Edge index {from_attr.edge_type} with "
                                  f"layout {from_attr.layout} was not sorted "
                                  f"by destination node, so conversion to "
                                  f"{to_layout} resulted in a permutation of "
                                  f"the order of edges. As a result, the "
                                  f"converted edge is not being re-stored in "
                                  f"the graph store. Please sort the edge "
                                  f"index and set 'is_sorted=True' to avoid "
                                  f"this warning.")
                else:
                    self._put_edge_index(
                        (row, col),
                        EdgeAttr(from_attr.edge_type, to_layout, True,
                                 from_attr.num_nodes))

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
        if (src[0].storage().data_ptr() == dst[1].storage().data_ptr()):
            # Do not copy if the tensor tuple is constructed from the same
            # storage (instead, return a view):
            out = torch.empty(0, dtype=src.dtype)
            out.set_(src.storage(), storage_offset=0,
                     size=(src.size()[0] + dst.size()[0], ))
            return out.view(2, -1)
        return torch.stack(tensor_tuple)
    elif attr.layout == EdgeLayout.CSR:
        # CSR: (rowptr, col)
        return SparseTensor(rowptr=src, col=dst, is_sorted=True)
    elif attr.layout == EdgeLayout.CSC:
        # CSC: (row, colptr) this is a transposed adjacency matrix, so rowptr
        # is the compressed column and col is the uncompressed row.
        return SparseTensor(rowptr=dst, col=src, is_sorted=True)
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
    data: Any,
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
    num_nodes: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor, OptTensor]:
    # Convert the graph data into a suitable format for sampling (CSC format).
    # Returns the `colptr` and `row` indices of the graph, as well as an
    # `perm` vector that denotes the permutation of edges.
    # Since no permutation of edges is applied when using `SparseTensor`,
    # `perm` can be of type `None`.
    perm: Optional[Tensor] = None

    if hasattr(data, 'adj'):
        colptr, row, _ = data.adj.csc()

    elif hasattr(data, 'adj_t'):
        colptr, row, _ = data.adj_t.csr()

    elif hasattr(data, 'edge_index'):
        (row, col) = data.edge_index
        size = (num_nodes[0] or data.size(0), num_nodes[1] or data.size(1))
        if not is_sorted:
            perm = (col * size[0]).add_(row).argsort()
            row = row[perm]
        colptr = torch.ops.torch_sparse.ind2ptr(col[perm], size[1])
    else:
        raise AttributeError("Data object does not contain attributes "
                             "'adj', 'adj_t' or 'edge_index'")

    colptr = colptr.to(device)
    row = row.to(device)
    perm = perm.to(device) if perm is not None else None

    if not colptr.is_cuda and share_memory:
        colptr.share_memory_()
        row.share_memory_()
        if perm is not None:
            perm.share_memory_()

    return colptr, row, perm
