from typing import Tuple, Optional, Union

from torch import Tensor
from torch_sparse import SparseTensor

# Types for accesing data:

# Node-types are denoted by a single string, e.g.: `data['paper']`
NodeType = str

# Edge-types are denotes by a tripled of strings, e.g.:
# `data[('author', 'writes', 'paper')]
EdgeType = Tuple[str, str, str]

# There exist some short-cuts to query edge-types (given that the full triplet
# can be uniquely reconstructed, e.g.:
# `data['writes']` or `data[('author', 'paper')]`
QueryType = Union[str, NodeType, EdgeType, Tuple[str, str]]

# Types for message passing:
Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]
