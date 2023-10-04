import inspect
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

WITH_PT20 = int(torch.__version__.split('.')[0]) >= 2
WITH_PT21 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 1
WITH_PT111 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 11
WITH_PT112 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 12
WITH_PT113 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 13

if not hasattr(torch, 'sparse_csc'):
    torch.sparse_csc = -1

try:
    import pyg_lib  # noqa
    WITH_PYG_LIB = True
    WITH_GMM = WITH_PT20 and hasattr(pyg_lib.ops, 'grouped_matmul')
    WITH_SEGMM = hasattr(pyg_lib.ops, 'segment_matmul')
    if WITH_SEGMM and 'pytest' in sys.modules and torch.cuda.is_available():
        # NOTE `segment_matmul` is currently bugged on older NVIDIA cards which
        # let our GPU tests on CI crash. Try if this error is present on the
        # current GPU and disable `WITH_SEGMM`/`WITH_GMM` if necessary.
        # TODO Drop this code block once `segment_matmul` is fixed.
        try:
            x = torch.randn(3, 4, device='cuda')
            ptr = torch.tensor([0, 2, 3], device='cuda')
            weight = torch.randn(2, 4, 4, device='cuda')
            out = pyg_lib.ops.segment_matmul(x, ptr, weight)
        except RuntimeError:
            WITH_GMM = False
            WITH_SEGMM = False
    WITH_SAMPLED_OP = hasattr(pyg_lib.ops, 'sampled_add')
    WITH_INDEX_SORT = hasattr(pyg_lib.ops, 'index_sort')
    WITH_METIS = hasattr(pyg_lib, 'partition')
    WITH_WEIGHTED_NEIGHBOR_SAMPLE = ('edge_weight' in inspect.signature(
        pyg_lib.sampler.neighbor_sample).parameters)
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
                      f"Disabling its usage. Stacktrace: {e}")
    pyg_lib = object
    WITH_PYG_LIB = False
    WITH_GMM = False
    WITH_SEGMM = False
    WITH_SAMPLED_OP = False
    WITH_INDEX_SORT = False
    WITH_METIS = False
    WITH_WEIGHTED_NEIGHBOR_SAMPLE = False

try:
    import torch_scatter  # noqa
    WITH_TORCH_SCATTER = True
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(f"An issue occurred while importing 'torch-scatter'. "
                      f"Disabling its usage. Stacktrace: {e}")
    torch_scatter = object
    WITH_TORCH_SCATTER = False

try:
    import torch_cluster  # noqa
    WITH_TORCH_CLUSTER = True
    WITH_TORCH_CLUSTER_BATCH_SIZE = 'batch_size' in torch_cluster.knn.__doc__
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(f"An issue occurred while importing 'torch-cluster'. "
                      f"Disabling its usage. Stacktrace: {e}")
    WITH_TORCH_CLUSTER = False
    WITH_TORCH_CLUSTER_BATCH_SIZE = False

    class TorchCluster:
        def __getattr__(self, key: str):
            raise ImportError(f"'{key}' requires 'torch-cluster'")

    torch_cluster = TorchCluster()

try:
    import torch_spline_conv  # noqa
    WITH_TORCH_SPLINE_CONV = True
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(
            f"An issue occurred while importing 'torch-spline-conv'. "
            f"Disabling its usage. Stacktrace: {e}")
    WITH_TORCH_SPLINE_CONV = False

try:
    import torch_sparse  # noqa
    from torch_sparse import SparseStorage, SparseTensor
    WITH_TORCH_SPARSE = True
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
                      f"Disabling its usage. Stacktrace: {e}")
    WITH_TORCH_SPARSE = False

    class SparseStorage:
        def __init__(
            self,
            row: Optional[Tensor] = None,
            rowptr: Optional[Tensor] = None,
            col: Optional[Tensor] = None,
            value: Optional[Tensor] = None,
            sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
            rowcount: Optional[Tensor] = None,
            colptr: Optional[Tensor] = None,
            colcount: Optional[Tensor] = None,
            csr2csc: Optional[Tensor] = None,
            csc2csr: Optional[Tensor] = None,
            is_sorted: bool = False,
            trust_data: bool = False,
        ):
            raise ImportError("'SparseStorage' requires 'torch-sparse'")

    class SparseTensor:
        def __init__(
            self,
            row: Optional[Tensor] = None,
            rowptr: Optional[Tensor] = None,
            col: Optional[Tensor] = None,
            value: Optional[Tensor] = None,
            sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
            is_sorted: bool = False,
            trust_data: bool = False,
        ):
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        @classmethod
        def from_edge_index(
            self,
            edge_index: Tensor,
            edge_attr: Optional[Tensor] = None,
            sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
            is_sorted: bool = False,
            trust_data: bool = False,
        ) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        @classmethod
        def from_dense(self, mat: Tensor,
                       has_value: bool = True) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def size(self, dim: int) -> int:
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def nnz(self) -> int:
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def is_cuda(self) -> bool:
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def has_value(self) -> bool:
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def set_value(self, value: Optional[Tensor],
                      layout: Optional[str] = None) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def fill_value(self, fill_value: float,
                       dtype: Optional[torch.dtype] = None) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def coo(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def csr(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def requires_grad(self) -> bool:
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        def to_torch_sparse_csr_tensor(
            self,
            dtype: Optional[torch.dtype] = None,
        ) -> Tensor:
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

    class torch_sparse:
        @staticmethod
        def matmul(src: SparseTensor, other: Tensor,
                   reduce: str = "sum") -> Tensor:
            raise ImportError("'matmul' requires 'torch-sparse'")

        @staticmethod
        def sum(src: SparseTensor, dim: Optional[int] = None) -> Tensor:
            raise ImportError("'sum' requires 'torch-sparse'")

        @staticmethod
        def mul(src: SparseTensor, other: Tensor) -> SparseTensor:
            raise ImportError("'mul' requires 'torch-sparse'")

        @staticmethod
        def set_diag(src: SparseTensor, values: Optional[Tensor] = None,
                     k: int = 0) -> SparseTensor:
            raise ImportError("'set_diag' requires 'torch-sparse'")

        @staticmethod
        def fill_diag(src: SparseTensor, fill_value: float,
                      k: int = 0) -> SparseTensor:
            raise ImportError("'fill_diag' requires 'torch-sparse'")

        @staticmethod
        def masked_select_nnz(src: SparseTensor, mask: Tensor,
                              layout: Optional[str] = None) -> SparseTensor:
            raise ImportError("'masked_select_nnz' requires 'torch-sparse'")


try:
    import torch_frame  # noqa
    WITH_TORCH_FRAME = True
    from torch_frame import TensorFrame
except Exception:
    WITH_TORCH_FRAME = False

    class TensorFrame:
        pass


try:
    import intel_extension_for_pytorch  # noqa
    WITH_IPEX = True
except Exception:
    WITH_IPEX = False


class MockTorchCSCTensor:
    def __init__(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.size = size

    def t(self) -> Tensor:  # Only support accessing its transpose:
        from torch_geometric.utils import to_torch_csr_tensor
        size = self.size
        return to_torch_csr_tensor(
            self.edge_index.flip([0]),
            self.edge_attr,
            size[::-1] if isinstance(size, (tuple, list)) else size,
        )


# Types for accessing data ####################################################

# Node-types are denoted by a single string, e.g.: `data['paper']`:
NodeType = str

# Edge-types are denotes by a triplet of strings, e.g.:
# `data[('author', 'writes', 'paper')]
EdgeType = Tuple[str, str, str]

NodeOrEdgeType = Union[NodeType, EdgeType]

DEFAULT_REL = 'to'
EDGE_TYPE_STR_SPLIT = '__'


class EdgeTypeStr(str):
    r"""A helper class to construct serializable edge types by merging an edge
    type tuple into a single string."""
    def __new__(cls, *args):
        if isinstance(args[0], (list, tuple)):
            # Unwrap `EdgeType((src, rel, dst))` and `EdgeTypeStr((src, dst))`:
            args = tuple(args[0])

        if len(args) == 1 and isinstance(args[0], str):
            args = args[0]  # An edge type string was passed.

        elif len(args) == 2 and all(isinstance(arg, str) for arg in args):
            # A `(src, dst)` edge type was passed - add `DEFAULT_REL`:
            args = (args[0], DEFAULT_REL, args[1])
            args = EDGE_TYPE_STR_SPLIT.join(args)

        elif len(args) == 3 and all(isinstance(arg, str) for arg in args):
            # A `(src, rel, dst)` edge type was passed:
            args = EDGE_TYPE_STR_SPLIT.join(args)

        else:
            raise ValueError(f"Encountered invalid edge type '{args}'")

        return str.__new__(cls, args)

    def to_tuple(self) -> EdgeType:
        r"""Returns the original edge type."""
        out = tuple(self.split(EDGE_TYPE_STR_SPLIT))
        if len(out) != 3:
            raise ValueError(f"Cannot convert the edge type '{self}' to a "
                             f"tuple since it holds invalid characters")
        return out


# There exist some short-cuts to query edge-types (given that the full triplet
# can be uniquely reconstructed, e.g.:
# * via str: `data['writes']`
# * via Tuple[str, str]: `data[('author', 'paper')]`
QueryType = Union[NodeType, EdgeType, str, Tuple[str, str]]

Metadata = Tuple[List[NodeType], List[EdgeType]]

# A representation of a feature tensor
FeatureTensorType = Union[Tensor, np.ndarray]

# A representation of an edge index, following the possible formats:
#   * COO: (row, col)
#   * CSC: (row, colptr)
#   * CSR: (rowptr, col)
EdgeTensorType = Tuple[Tensor, Tensor]

# Types for message passing ###################################################

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]

MaybeHeteroNodeTensor = Union[Tensor, Dict[NodeType, Tensor]]
MaybeHeteroEdgeTensor = Union[Tensor, Dict[EdgeType, Tensor]]

# Types for sampling ##########################################################

InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]
