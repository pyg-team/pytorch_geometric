from typing import Tuple, Optional, Union, Callable

from torch import Tensor
from torch_sparse import SparseTensor

# Data Types
NodeType = str
EdgeType = Tuple[str, str, str]
QueryType = Union[str, NodeType, EdgeType, Tuple[str, str]]
AttrType = Union[str, Tuple[str, Callable]]

# Message Passing Types
Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]
