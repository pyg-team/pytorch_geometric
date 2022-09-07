from abc import ABC, abstractmethod
from typing import Dict, NamedTuple, Union

import torch

from torch_geometric.typing import EdgeType, NodeType, OptTensor

# An input to a node-based sampler is a tensor of node indices:
NodeSamplerInput = torch.Tensor

# An input to an edge-based sampler consists of four tensors:
#   * The row of the edge index in COO format
#   * The column of the edge index in COO format
#   * The labels of the edges
#   * (Optionally) the time attribute corresponding to the edge label
EdgeSamplerInput = Union[torch.Tensor, torch.Tensor, torch.Tensor, OptTensor]


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
    node: Union[torch.Tensor, Dict[NodeType, torch.Tensor]]
    row: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    col: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    edge: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    input_size: int


class BaseSampler(ABC):
    r"""A base class that initializes a graph sampler and provides a `sample`
    routine that performs sampling on an input list or tensor of node indices.
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        r"""Initializes a sampler with common data it will need to perform a
        `sample` call. Note that this data will be replicated across processes,
        so it is best to limit the amount of information stored here."""
        pass

    def sample_from_nodes(
        self,
        index: NodeSamplerInput,
        *args,
        **kwargs,
    ) -> SamplerOutput:
        raise NotImplementedError

    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        *args,
        **kwargs,
    ) -> SamplerOutput:
        raise NotImplementedError
