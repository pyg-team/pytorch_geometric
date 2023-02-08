import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from torch import Tensor

try:
    import pyg_lib  # noqa
    WITH_PYG_LIB = True
    WITH_INDEX_SORT = WITH_PYG_LIB and hasattr(pyg_lib.ops, 'index_sort')
except (ImportError, OSError) as e:
    if isinstance(e, OSError):
        warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
                      f"Disabling its usage. Stacktrace: {e}")
    pyg_lib = object
    WITH_PYG_LIB = False
    WITH_INDEX_SORT = False

try:
    import torch_scatter  # noqa
    WITH_TORCH_SCATTER = True
except (ImportError, OSError) as e:
    if isinstance(e, OSError):
        warnings.warn(f"An issue occurred while importing 'torch-scatter'. "
                      f"Disabling its usage. Stacktrace: {e}")
    torch_scatter = object
    WITH_TORCH_SCATTER = False

try:
    import torch_sparse  # noqa
    from torch_sparse import SparseTensor
    WITH_TORCH_SPARSE = True
except (ImportError, OSError) as e:
    if isinstance(e, OSError):
        warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
                      f"Disabling its usage. Stacktrace: {e}")
    torch_sparse = object
    WITH_TORCH_SPARSE = False

    class SparseTensor:
        def __init__(self, *args, **kwargs):
            raise ImportError("'SparseTensor' requires 'torch-sparse'")

        @classmethod
        def from_edge_index(cls, *args, **kwargs) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'torch-sparse'")


# Types for accessing data ####################################################

# Node-types are denoted by a single string, e.g.: `data['paper']`:
NodeType = str

# Edge-types are denotes by a triplet of strings, e.g.:
# `data[('author', 'writes', 'paper')]
EdgeType = Tuple[str, str, str]

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

# Types for sampling ##########################################################

InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]
