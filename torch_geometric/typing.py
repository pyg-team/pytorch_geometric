import inspect
import os
import sys
import typing
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor

try:
    from typing import TypeAlias  # type: ignore
except ImportError:
    from typing_extensions import TypeAlias

WITH_PT20 = int(torch.__version__.split('.')[0]) >= 2
WITH_PT21 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 1
WITH_PT22 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 2
WITH_PT23 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 3
WITH_PT24 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 4
WITH_PT25 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 5
WITH_PT26 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 6
WITH_PT111 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 11
WITH_PT112 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 12
WITH_PT113 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 13

WITH_WINDOWS = os.name == 'nt'
NO_MKL = 'USE_MKL=OFF' in torch.__config__.show() or WITH_WINDOWS

MAX_INT64 = torch.iinfo(torch.int64).max

if WITH_PT20:
    INDEX_DTYPES: Set[torch.dtype] = {
        torch.int32,
        torch.int64,
    }
elif not typing.TYPE_CHECKING:  # pragma: no cover
    INDEX_DTYPES: Set[torch.dtype] = {
        torch.int64,
    }

if not hasattr(torch, 'sparse_csc'):
    torch.sparse_csc = torch.sparse_coo

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
    WITH_SOFTMAX = hasattr(pyg_lib.ops, 'softmax_csr')
    WITH_INDEX_SORT = hasattr(pyg_lib.ops, 'index_sort')
    WITH_METIS = hasattr(pyg_lib, 'partition')
    WITH_EDGE_TIME_NEIGHBOR_SAMPLE = ('edge_time' in inspect.signature(
        pyg_lib.sampler.neighbor_sample).parameters)
    WITH_WEIGHTED_NEIGHBOR_SAMPLE = ('edge_weight' in inspect.signature(
        pyg_lib.sampler.neighbor_sample).parameters)
    try:
        torch.classes.pyg.CPUHashMap
        WITH_CPU_HASH_MAP = True
    except Exception:
        WITH_CPU_HASH_MAP = False
    try:
        torch.classes.pyg.CUDAHashMap
        WITH_CUDA_HASH_MAP = True
    except Exception:
        WITH_CUDA_HASH_MAP = False
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
                      f"Disabling its usage. Stacktrace: {e}")
    pyg_lib = object
    WITH_PYG_LIB = False
    WITH_GMM = False
    WITH_SEGMM = False
    WITH_SAMPLED_OP = False
    WITH_SOFTMAX = False
    WITH_INDEX_SORT = False
    WITH_METIS = False
    WITH_EDGE_TIME_NEIGHBOR_SAMPLE = False
    WITH_WEIGHTED_NEIGHBOR_SAMPLE = False
    WITH_CPU_HASH_MAP = False
    WITH_CUDA_HASH_MAP = False

if WITH_CPU_HASH_MAP:
    CPUHashMap: TypeAlias = torch.classes.pyg.CPUHashMap
else:

    class CPUHashMap:  # type: ignore
        def __init__(self, key: Tensor) -> None:
            raise ImportError("'CPUHashMap' requires 'pyg-lib'")

        def get(self, query: Tensor) -> Tensor:
            raise ImportError("'CPUHashMap' requires 'pyg-lib'")


if WITH_CUDA_HASH_MAP:
    CUDAHashMap: TypeAlias = torch.classes.pyg.CUDAHashMap
else:

    class CUDAHashMap:  # type: ignore
        def __init__(self, key: Tensor) -> None:
            raise ImportError("'CUDAHashMap' requires 'pyg-lib'")

        def get(self, query: Tensor) -> Tensor:
            raise ImportError("'CUDAHashMap' requires 'pyg-lib'")


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
        def __getattr__(self, key: str) -> Any:
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

    class SparseStorage:  # type: ignore
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

        def value(self) -> Optional[Tensor]:
            raise ImportError("'SparseStorage' requires 'torch-sparse'")

        def rowcount(self) -> Tensor:
            raise ImportError("'SparseStorage' requires 'torch-sparse'")

    class SparseTensor:  # type: ignore
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

        @property
        def storage(self) -> SparseStorage:
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

    class torch_sparse:  # type: ignore
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
    torch_frame = object
    WITH_TORCH_FRAME = False

    class TensorFrame:  # type: ignore
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
    type tuple into a single string.
    """
    edge_type: tuple[str, str, str]

    def __new__(cls, *args: Any) -> 'EdgeTypeStr':
        if isinstance(args[0], (list, tuple)):
            # Unwrap `EdgeType((src, rel, dst))` and `EdgeTypeStr((src, dst))`:
            args = tuple(args[0])

        if len(args) == 1 and isinstance(args[0], str):
            arg = args[0]  # An edge type string was passed.
            edge_type = tuple(arg.split(EDGE_TYPE_STR_SPLIT))
            if len(edge_type) != 3:
                raise ValueError(f"Cannot convert the edge type '{arg}' to a "
                                 f"tuple since it holds invalid characters")

        elif len(args) == 2 and all(isinstance(arg, str) for arg in args):
            # A `(src, dst)` edge type was passed - add `DEFAULT_REL`:
            edge_type = (args[0], DEFAULT_REL, args[1])
            arg = EDGE_TYPE_STR_SPLIT.join(edge_type)

        elif len(args) == 3 and all(isinstance(arg, str) for arg in args):
            # A `(src, rel, dst)` edge type was passed:
            edge_type = tuple(args)
            arg = EDGE_TYPE_STR_SPLIT.join(args)

        else:
            raise ValueError(f"Encountered invalid edge type '{args}'")

        out = str.__new__(cls, arg)
        out.edge_type = edge_type  # type: ignore
        return out

    def to_tuple(self) -> EdgeType:
        r"""Returns the original edge type."""
        if len(self.edge_type) != 3:
            raise ValueError(f"Cannot convert the edge type '{self}' to a "
                             f"tuple since it holds invalid characters")
        return self.edge_type

    def __reduce__(self) -> tuple[Any, Any]:
        return (self.__class__, (self.edge_type, ))


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
MaybeHeteroAdjTensor = Union[Tensor, Dict[EdgeType, Adj]]
MaybeHeteroEdgeTensor = Union[Tensor, Dict[EdgeType, Tensor]]

# Types for sampling ##########################################################

InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]

# Serialization ###############################################################

if WITH_PT24:
    torch.serialization.add_safe_globals([
        SparseTensor,
        SparseStorage,
        TensorFrame,
        MockTorchCSCTensor,
        EdgeTypeStr,
    ])
