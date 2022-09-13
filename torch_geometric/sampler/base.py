from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from torch_geometric.typing import EdgeType, NodeType, OptTensor

# An input to a node-based sampler is a tensor of node indices:
NodeSamplerInput = torch.Tensor

# An input to an edge-based sampler consists of four tensors:
#   * The row of the edge index in COO format
#   * The column of the edge index in COO format
#   * The labels of the edges
#   * (Optionally) the time attribute corresponding to the edge label
EdgeSamplerInput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, OptTensor]


# A sampler output contains the following information.
#   * node: a tensor of `n` output nodes resulting from sampling. In the
#       heterogeneous case, this is a dictionary mapping node types to the
#       associated output tensors, each with potentially varying length.
#   * row: a tensor of edge indices that correspond to the COO row values of
#       the edges in the sampled subgraph. Note that these indices must be
#       re-indexed from 0..n-1 corresponding to the nodes in the 'node' tensor.
#       In the heterogeneous case, this is a dictionary mapping edge types to
#       the associated COO row tensors.
#   * col: a tensor of edge indices that correspond to the COO column values of
#       the edges in the sampled subgraph. Note that these indices must be
#       re-indexed from 0..n-1 corresponding to the nodes in the 'node' tensor.
#       In the heterogeneous case, this is a dictionary mapping edge types to
#       the associated COO column tensors.
#   * edge: a tensor of the indices of the sampled edges in the original graph.
#       This tensor is used to obtain edge attributes from the original graph;
#       if no edge attributes are present, it may be omitted.
#   * batch: a tensor identifying the seed node for each sampled node.
#   * metadata: any additional metadata required by a loader using the sampler
#       output.
# There exist both homogeneous and heterogeneous versions.
@dataclass
class SamplerOutput:
    node: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    edge: torch.Tensor
    batch: Optional[torch.Tensor] = None
    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Optional[Any] = None


@dataclass
class HeteroSamplerOutput:
    node: Dict[NodeType, torch.Tensor]
    row: Dict[EdgeType, torch.Tensor]
    col: Dict[EdgeType, torch.Tensor]
    edge: Dict[EdgeType, torch.Tensor]
    batch: Optional[Dict[NodeType, torch.Tensor]] = None
    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Optional[Any] = None


class BaseSampler(ABC):
    r"""A base class that initializes a graph sampler and provides a `sample`
    routine that performs sampling on an input list or tensor of node indices.

    .. warning ::
        Any data stored in the sampler will be _replicated_ across data loading
        workers that use the sampler. That is, each data loading worker has its
        own instance of a sampler. As such, it is recommended to limit the
        amount of information stored in the sampler, and to initialize all this
        information at `__init__`.
    """
    @abstractmethod
    def sample_from_nodes(
        self,
        index: NodeSamplerInput,
        **kwargs,
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        r"""Performs sampling from the nodes specified in 'index', returning
        a sampled subgraph in the specified output format."""
        raise NotImplementedError

    @abstractmethod
    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        **kwargs,
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        r"""Performs sampling from the edges specified in 'index', returning
        a sampled subgraph in the specified output format."""
        raise NotImplementedError

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        r"""If the sampler performs any modification of edge ordering in the
        original graph, this function is expected to return the permutation
        tensor that defines the permutation from the edges in the original
        graph and the edges used in the sampler. If no such permutation was
        applied, a default None tensor is returned. For heterogeneous graphs,
        the expected return type is a permutation tensor for each edge type."""
        return None
