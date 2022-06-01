from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_sparse import SparseTensor

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
FeatureTensorType = Union[torch.Tensor, np.ndarray]

# Types for message passing ###################################################
class ProxyTensor(object):
    def __init__():
        return super()

AnyTensor = Union[Tensor, ProxyTensor]

Adj = Union[AnyTensor, SparseTensor]
OptTensor = Optional[AnyTensor]
PairTensor = Tuple[AnyTensor, AnyTensor]
OptPairTensor = Tuple[AnyTensor, Optional[AnyTensor]]
PairOptTensor = Tuple[Optional[AnyTensor], Optional[AnyTensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[AnyTensor]

# Types for sampling ##########################################################

InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]
NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]
