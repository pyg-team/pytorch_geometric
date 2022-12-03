import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from torch import Tensor

from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils.mixin import CastMixin

# An input to a node-based sampler consists of two tensors:
#  * The example indices
#  * The node indices
#  * The timestamps of the given node indices (optional)
NodeSamplerInput = Tuple[Tensor, Tensor, OptTensor]

# An input to an edge-based sampler consists of four tensors:
#   * The example indices
#   * The row of the edge index in COO format
#   * The column of the edge index in COO format
#   * The labels of the edges (optional)
#   * The time attribute corresponding to the edge label (optional)
EdgeSamplerInput = Tuple[Tensor, Tensor, Tensor, OptTensor, OptTensor]


@dataclass
class SamplerOutput:
    r"""The sampling output of a :class:`BaseSampler` on homogeneous graphs.

    Args:
        node (torch.Tensor): The sampled nodes in the original graph.
        row (torch.Tensor): The source node indices of the sampled subgraph.
            Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
            corresponding to the nodes in the :obj:`node` tensor.
        col (torch.Tensor): The destination node indices of the sampled
            subgraph.
            Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
            corresponding to the nodes in the :obj:`node` tensor.
        edge (torch.Tensor, optional): The sampled edges in the original graph.
            This tensor is used to obtain edge features from the original
            graph. If no edge attributes are present, it may be omitted.
        batch (torch.Tensor, optional): The vector to identify the seed node
            for each sampled node. Can be present in case of disjoint subgraph
            sampling per seed node. (default: :obj:`None`)
        metadata: (Any, optional): Additional metadata information.
            (default: :obj:`None`)
    """
    node: Tensor
    row: Tensor
    col: Tensor
    edge: OptTensor
    batch: OptTensor = None
    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Optional[Any] = None


@dataclass
class HeteroSamplerOutput:
    r"""The sampling output of a :class:`BaseSampler` on heterogeneous graphs.

    Args:
        node (Dict[str, torch.Tensor]): The sampled nodes in the original graph
            for each node type.
        row (Dict[Tuple[str, str, str], torch.Tensor]): The source node indices
            of the sampled subgraph for each edge type.
            Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
            corresponding to the nodes in the :obj:`node` tensor of the source
            node type.
        col (Dict[Tuple[str, str, str], torch.Tensor]): The destination node
            indices of the sampled subgraph for each edge type.
            Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
            corresponding to the nodes in the :obj:`node` tensor of the
            destination node type.
        edge (Dict[Tuple[str, str, str], torch.Tensor], optional): The sampled
            edges in the original graph for each edge type.
            This tensor is used to obtain edge features from the original
            graph. If no edge attributes are present, it may be omitted.
        batch (Dict[str, torch.Tensor], optional): The vector to identify the
            seed node for each sampled node for each node type. Can be present
            in case of disjoint subgraph sampling per seed node.
            (default: :obj:`None`)
        metadata: (Any, optional): Additional metadata information.
            (default: :obj:`None`)
    """
    node: Dict[NodeType, Tensor]
    row: Dict[EdgeType, Tensor]
    col: Dict[EdgeType, Tensor]
    edge: Optional[Dict[EdgeType, Tensor]]
    batch: Optional[Dict[NodeType, Tensor]] = None
    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Optional[Any] = None


class NegativeSamplingStrategy(Enum):
    # 'binary': Randomly sample negative edges in the graph.
    binary = 'binary'
    # 'triplet': Randomly sample negative destination nodes for each positive
    # source node.
    triplet = 'triplet'


@dataclass
class NegativeSamplingConfig(CastMixin):
    strategy: NegativeSamplingStrategy
    amount: Union[int, float] = 1

    def __init__(
        self,
        strategy: Union[NegativeSamplingStrategy, str],
        amount: Union[int, float] = 1,
    ):
        self.strategy = NegativeSamplingStrategy(strategy)
        self.amount = amount

        if self.amount <= 0:
            raise ValueError(f"The attribute 'amount' needs to be positive "
                             f"for '{self.__class__.__name__}' "
                             f"(got {self.amount})")

        if self.is_triplet():
            if self.amount != math.ceil(self.amount):
                raise ValueError(f"The attribute 'amount' needs to be an "
                                 f"integer for '{self.__class__.__name__}' "
                                 f"with 'triplet' negative sampling "
                                 f"(got {self.amount}).")
            self.amount = math.ceil(self.amount)

    def is_binary(self) -> bool:
        return self.strategy == NegativeSamplingStrategy.binary

    def is_triplet(self) -> bool:
        return self.strategy == NegativeSamplingStrategy.triplet


class BaseSampler(ABC):
    r"""An abstract base class that initializes a graph sampler and provides
    :meth:`sample_from_nodes` and :meth:`sample_from_edges` routines.

    .. note ::

        Any data stored in the sampler will be *replicated* across data loading
        workers that use the sampler since each data loading worker holds its
        own instance of a sampler.
        As such, it is recommended to limit the amount of information stored in
        the sampler.
    """
    @abstractmethod
    def sample_from_nodes(
        self,
        index: NodeSamplerInput,
        **kwargs,
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        r"""Performs sampling from the nodes specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        The :obj:`index` is a tuple holding the following information:

        1. The example indices of the seed nodes
        2. The node indices to start sampling from
        3. The timestamps of the given seed nodes (optional)
        """
        pass

    @abstractmethod
    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        **kwargs,
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        r"""Performs sampling from the edges specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        The :obj:`index` is a tuple holding the following information:

        1. The example indices of the seed links
        2. The source node indices to start sampling from
        3. The destination node indices to start sampling from
        4. The labels of the seed links (optional)
        5. The timestamps of the given seed nodes (optional)
        """
        pass

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        r"""If the sampler performs any modification of edge ordering in the
        original graph, this function is expected to return the permutation
        tensor that defines the permutation from the edges in the original
        graph and the edges used in the sampler. If no such permutation was
        applied, :obj:`None` is returned. For heterogeneous graphs, the
        expected return type is a permutation tensor for each edge type."""
        return None
