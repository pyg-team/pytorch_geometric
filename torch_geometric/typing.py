from typing import Tuple, Optional, Union, List

from torch import Tensor
from torch_sparse import SparseTensor

NodeType = str
EdgeType = Tuple[str, str, str]
Metadata = Tuple[List[NodeType], List[EdgeType]]

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]
