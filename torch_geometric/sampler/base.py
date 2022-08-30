from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, Tuple, Union, List
import torch
from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore

# An input to a sampler is either a list or tensor of node indices:
SamplerInput = Union[List[int], torch.Tensor]


# A sampler output contains the following information.
#   * input_size: the number of input nodes to begin sampling from.
#   * node: a tensor of output nodes resulting from sampling. In the
#       heterogeneous case, this is a dictionary mapping node types to the
#       associated output tensors.
#   * row: a tensor of edge indices that correspond to the row values of the
#       edges in the sampled subgraph. Note that these indices must be
#       re-indexed from 0..n-1 corresponding to the nodes in the 'node' tensor.
#       In the heterogeneous case, this is a dictionary mapping edge types to
#       the associated output tensors.
#   * row: a tensor of edge indices that correspond to the column values of the
#       edges in the sampled subgraph. Note that these indices must be
#       re-indexed from 0..n-1 corresponding to the nodes in the 'node' tensor.
#       In the heterogeneous case, this is a dictionary mapping edge types to
#       the associated output tensors.
#   * edge: a tensor of the indices in the original graph; e.g. to be used to
#       obtain edge attributes.
class SamplerOutput(NamedTuple):
    input_size: int
    node: Union[torch.Tensor, Dict[NodeType, torch.Tensor]]
    row: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    col: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    edge: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]


class BaseSampler(ABC):
    r"""A base class that initializes a graph sampler and provides a `sample`
    routine that performs sampling on an input list or tensor of node indices.
    """
    @abstractmethod
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        **kwargs,
    ):
        r"""Initializes a sampler with common data it will need to perform a
        `sample` call. Note that this data will be replicated across processes,
        so it is best to limit the amount of information stored here."""
        pass

    @abstractmethod
    def sample(self, index: SamplerInput, **kwargs) -> SamplerOutput:
        pass

    def __call__(self, index: SamplerInput, **kwargs) -> SamplerOutput:
        if not isinstance(index, torch.LongTensor):
            index = torch.LongTensor(index)
        return self.sample(index, **kwargs)
