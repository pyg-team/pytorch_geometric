from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Union

import torch

from torch_geometric.typing import EdgeType, NodeType

# An input to a sampler is a tensor of node indices:
SamplerInput = torch.Tensor


# A sampler output contains the following information:
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
#   * metadata: any additional metadata required by a loader using the sampler
#       output.
# There exist both homogeneous and heterogeneous versions.
class SamplerOutput(NamedTuple):
    node: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    edge: torch.Tensor

    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Any

    # TODO(manan): include a 'batch' attribute that assigns each node to an
    # example; this is necessary for integration with pyg-lib


class HeteroSamplerOutput(NamedTuple):
    node: Dict[NodeType, torch.Tensor]
    row: Dict[EdgeType, torch.Tensor]
    col: Dict[EdgeType, torch.Tensor]
    edge: Dict[EdgeType, torch.Tensor]

    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Any

    # TODO(manan): include a 'batch' attribute that assigns each node to an
    # example; this is necessary for integration with pyg-lib


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
    def sample(
        self,
        index: SamplerInput,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        pass

    def __call__(
        self,
        index: SamplerInput,
        **kwargs,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        if isinstance(index, (list, tuple)):
            index = torch.tensor(index)
        return self.sample(index, **kwargs)
