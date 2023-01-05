import math
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils.mixin import CastMixin


@dataclass
class NodeSamplerInput(CastMixin):
    r"""The sampling input of
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_nodes`.

    Args:
        input_id (torch.Tensor, optional): The indices of the data loader input
            of the current mini-batch.
        node (torch.Tensor): The indices of seed nodes to start sampling from.
        time (torch.Tensor, optional): The timestamp for the seed nodes.
            (default: :obj:`None`)
        input_type (str, optional): The input node type (in case of sampling in
            a heterogeneous graph). (default: :obj:`None`)
    """
    input_id: OptTensor
    node: Tensor
    time: OptTensor = None
    input_type: Optional[NodeType] = None

    def __getitem__(self, index: Union[Tensor, Any]) -> 'NodeSamplerInput':
        if not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)

        return NodeSamplerInput(
            self.input_id[index] if self.input_id is not None else index,
            self.node[index],
            self.time[index] if self.time is not None else None,
            self.input_type,
        )


@dataclass
class EdgeSamplerInput(CastMixin):
    r"""The sampling input of
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.

    Args:
        input_id (torch.Tensor, optional): The indices of the data loader input
            of the current mini-batch.
        row (torch.Tensor): The source node indices of seed links to start
            sampling from.
        col (torch.Tensor): The destination node indices of seed links to start
            sampling from.
        label (torch.Tensor, optional): The label for the seed links.
            (default: :obj:`None`)
        time (torch.Tensor, optional): The timestamp for the seed links.
            (default: :obj:`None`)
        input_type (Tuple[str, str, str], optional): The input edge type (in
            case of sampling in a heterogeneous graph). (default: :obj:`None`)
    """
    input_id: OptTensor
    row: Tensor
    col: Tensor
    label: OptTensor = None
    time: OptTensor = None
    input_type: Optional[EdgeType] = None

    def __getitem__(self, index: Union[Tensor, Any]) -> 'EdgeSamplerInput':
        if not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)

        return EdgeSamplerInput(
            self.input_id[index] if self.input_id is not None else index,
            self.row[index],
            self.col[index],
            self.label[index] if self.label is not None else None,
            self.time[index] if self.time is not None else None,
            self.input_type,
        )


@dataclass
class SamplerOutput(CastMixin):
    r"""The sampling output of a :class:`~torch_geometric.sampler.BaseSampler`
    on homogeneous graphs.

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
class HeteroSamplerOutput(CastMixin):
    r"""The sampling output of a :class:`~torch_geometric.sampler.BaseSampler`
    on heterogeneous graphs.

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


class NegativeSamplingMode(Enum):
    # 'binary': Randomly sample negative edges in the graph.
    binary = 'binary'
    # 'triplet': Randomly sample negative destination nodes for each positive
    # source node.
    triplet = 'triplet'


@dataclass
class NegativeSampling(CastMixin):
    r"""The negative sampling configuration of a
    :class:`~torch_geometric.sampler.BaseSampler` when calling
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.

    Args:
        mode (str): The negative sampling mode
            (:obj:`"binary"` or :obj:`"triplet"`).
            If set to :obj:`"binary"`, will randomly sample negative links
            from the graph.
            If set to :obj:`"triplet"`, will randomly sample negative
            destination nodes for each positive source node.
        amount (int or float, optional): The ratio of sampled negative edges to
            the number of positive edges. (default: :obj:`1`)
        weight (torch.Tensor, optional): A node-level vector determining the
            sampling of nodes. Does not necessariyl need to sum up to one.
            If not given, negative nodes will be sampled uniformly.
            (default: :obj:`None`)
    """
    mode: NegativeSamplingMode
    amount: Union[int, float] = 1
    weight: Optional[Tensor] = None

    def __init__(
        self,
        mode: Union[NegativeSamplingMode, str],
        amount: Union[int, float] = 1,
        weight: Optional[Tensor] = None,
    ):
        self.mode = NegativeSamplingMode(mode)
        self.amount = amount
        self.weight = weight

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
        return self.mode == NegativeSamplingMode.binary

    def is_triplet(self) -> bool:
        return self.mode == NegativeSamplingMode.triplet

    def sample(self, num_samples: int,
               num_nodes: Optional[int] = None) -> Tensor:
        r"""Generates :obj:`num_samples` negative samples."""
        if self.weight is None:
            if num_nodes is None:
                raise ValueError(
                    f"Cannot sample negatives in '{self.__class__.__name__}' "
                    f"without passing the 'num_nodes' argument")
            return torch.randint(num_nodes, (num_samples, ))

        if num_nodes is not None and self.weight.numel() != num_nodes:
            raise ValueError(
                f"The 'weight' attribute in '{self.__class__.__name__}' "
                f"needs to match the number of nodes {num_nodes} "
                f"(got {self.weight.numel()})")
        return torch.multinomial(self.weight, num_samples, replacement=True)


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

        Args:
            index (NodeSamplerInput): The node sampler input object.
        """
        raise NotImplementedError

    def sample_from_edges(
        self,
        index: EdgeSamplerInput,
        neg_sampling: Optional[NegativeSampling] = None,
    ) -> Union[HeteroSamplerOutput, SamplerOutput]:
        r"""Performs sampling from the edges specified in :obj:`index`,
        returning a sampled subgraph in the specified output format.

        The :obj:`index` is a tuple holding the following information:

        1. The example indices of the seed links
        2. The source node indices to start sampling from
        3. The destination node indices to start sampling from
        4. The labels of the seed links (optional)
        5. The timestamps of the given seed nodes (optional)

        Args:
            index (EdgeSamplerInput): The edge sampler input object.
            neg_sampling (NegativeSampling, optional): The negative sampling
                configuration. (default: :obj:`None`)
        """
        raise NotImplementedError

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        r"""If the sampler performs any modification of edge ordering in the
        original graph, this function is expected to return the permutation
        tensor that defines the permutation from the edges in the original
        graph and the edges used in the sampler. If no such permutation was
        applied, :obj:`None` is returned. For heterogeneous graphs, the
        expected return type is a permutation tensor for each edge type."""
        return None
