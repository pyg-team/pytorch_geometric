import copy
import math
import warnings
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.sampler.utils import to_bidirectional
from torch_geometric.typing import EdgeType, EdgeTypeStr, NodeType, OptTensor
from torch_geometric.utils.mixin import CastMixin


class DataType(Enum):
    r"""The data type a sampler is operating on."""
    homogeneous = 'homogeneous'
    heterogeneous = 'heterogeneous'
    remote = 'remote'

    @classmethod
    def from_data(cls, data: Any):
        if isinstance(data, Data):
            return cls.homogeneous
        elif isinstance(data, HeteroData):
            return cls.heterogeneous
        elif (isinstance(data, (list, tuple)) and len(data) == 2
              and isinstance(data[0], FeatureStore)
              and isinstance(data[1], GraphStore)):
            return cls.remote

        raise ValueError(f"Expected a 'Data', 'HeteroData', or a tuple of "
                         f"'FeatureStore' and 'GraphStore' "
                         f"(got '{type(data)}')")


class SubgraphType(Enum):
    r"""The type of the returned subgraph."""
    directional = 'directional'
    bidirectional = 'bidirectional'
    induced = 'induced'


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
        num_sampled_nodes (List[int], optional): The number of sampled nodes
            per hop. (default: :obj:`None`)
        num_sampled_edges (List[int], optional): The number of sampled edges
            per hop. (default: :obj:`None`)
        metadata: (Any, optional): Additional metadata information.
            (default: :obj:`None`)
    """
    node: Tensor
    row: Tensor
    col: Tensor
    edge: OptTensor
    batch: OptTensor = None
    num_sampled_nodes: Optional[List[int]] = None
    num_sampled_edges: Optional[List[int]] = None
    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Optional[Any] = None

    def to_bidirectional(self) -> 'SamplerOutput':
        r"""Converts the sampled subgraph into a bidirectional variant, in
        which all sampled edges are guaranteed to be bidirectional."""
        out = copy.copy(self)

        out.row, out.col, out.edge = to_bidirectional(
            row=self.row,
            col=self.col,
            rev_row=self.row,
            rev_col=self.col,
            edge_id=self.edge,
            rev_edge_id=self.edge,
        )
        out.num_sampled_nodes = out.num_sampled_edges = None

        return out


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
        num_sampled_nodes (Dict[str, List[int]], optional): The number of
            sampled nodes for each node type and each layer.
            (default: :obj:`None`)
        num_sampled_edges (Dict[EdgeType, List[int]], optional): The number of
            sampled edges for each edge type and each layer.
            (default: :obj:`None`)
        metadata: (Any, optional): Additional metadata information.
            (default: :obj:`None`)
    """
    node: Dict[NodeType, Tensor]
    row: Dict[EdgeType, Tensor]
    col: Dict[EdgeType, Tensor]
    edge: Dict[EdgeType, OptTensor]
    batch: Optional[Dict[NodeType, Tensor]] = None
    num_sampled_nodes: Optional[Dict[NodeType, List[int]]] = None
    num_sampled_edges: Optional[Dict[EdgeType, List[int]]] = None
    # TODO(manan): refine this further; it does not currently define a proper
    # API for the expected output of a sampler.
    metadata: Optional[Any] = None

    def to_bidirectional(self) -> 'SamplerOutput':
        r"""Converts the sampled subgraph into a bidirectional variant, in
        which all sampled edges are guaranteed to be bidirectional."""
        out = copy.copy(self)
        out.row = copy.copy(self.row)
        out.col = copy.copy(self.col)
        out.edge = copy.copy(self.edge)

        src_dst_dict = defaultdict(list)
        edge_types = self.row.keys()
        edge_types = [k for k in edge_types if not k[1].startswith('rev_')]
        for edge_type in edge_types:
            src, rel, dst = edge_type
            rev_edge_type = (dst, f'rev_{rel}', src)

            if src == dst and rev_edge_type not in self.row:
                out.row[edge_type], out.col[edge_type], _ = to_bidirectional(
                    row=self.row[edge_type],
                    col=self.col[edge_type],
                    rev_row=self.row[edge_type],
                    rev_col=self.col[edge_type],
                )
                if out.edge is not None:
                    out.edge[edge_type] = None

            elif rev_edge_type in self.row:
                out.row[edge_type], out.col[edge_type], _ = to_bidirectional(
                    row=self.row[edge_type],
                    col=self.col[edge_type],
                    rev_row=self.row[rev_edge_type],
                    rev_col=self.col[rev_edge_type],
                )
                out.row[rev_edge_type] = out.col[edge_type]
                out.col[rev_edge_type] = out.row[edge_type]
                if out.edge is not None:
                    out.edge[edge_type] = None
                    out.edge[rev_edge_type] = None

            else:  # Find the reverse edge type (if it is unique):
                if len(src_dst_dict) == 0:  # Create mapping lazily.
                    for key in self.row.keys():
                        v1, _, v2 = key
                        src_dst_dict[(v1, v2)].append(key)

                if len(src_dst_dict[(dst, src)]) == 1:
                    rev_edge_type = src_dst_dict[(dst, src)][0]
                    row, col, _ = to_bidirectional(
                        row=self.row[edge_type],
                        col=self.col[edge_type],
                        rev_row=self.row[rev_edge_type],
                        rev_col=self.col[rev_edge_type],
                    )
                    out.row[edge_type] = row
                    out.col[edge_type] = col
                    if out.edge is not None:
                        out.edge[edge_type] = None

                else:
                    warnings.warn(f"Cannot convert to bidirectional graph "
                                  f"since the edge type {edge_type} does not "
                                  f"seem to have a reverse edge type")

        out.num_sampled_nodes = out.num_sampled_edges = None

        return out


@dataclass(frozen=True)
class NumNeighbors:
    r"""The number of neighbors to sample in a homogeneous or heterogeneous
    graph. In heterogeneous graphs, may also take in a dictionary denoting
    the amount of neighbors to sample for individual edge types.

    Args:
        values (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for individual edge types.
        default (List[int], optional): The default number of neighbors for edge
            types not specified in :obj:`values`. (default: :obj:`None`)
    """
    values: Union[List[int], Dict[EdgeTypeStr, List[int]]]
    default: Optional[List[int]] = None

    def __init__(
        self,
        values: Union[List[int], Dict[EdgeType, List[int]]],
        default: Optional[List[int]] = None,
    ):
        if isinstance(values, (tuple, list)) and default is not None:
            raise ValueError(f"'default' must be set to 'None' in case a "
                             f"single list is given as the number of "
                             f"neighbors (got '{type(default)})'")

        if isinstance(values, dict):
            values = {EdgeTypeStr(key): value for key, value in values.items()}

        # Write to `__dict__` since dataclass is annotated with `frozen=True`:
        self.__dict__['values'] = values
        self.__dict__['default'] = default

    def _get_values(
        self,
        edge_types: Optional[List[EdgeType]] = None,
        mapped: bool = False,
    ) -> Union[List[int], Dict[Union[EdgeType, EdgeTypeStr], List[int]]]:

        if edge_types is not None:
            if isinstance(self.values, (tuple, list)):
                default = self.values
            elif isinstance(self.values, dict):
                default = self.default
            else:
                assert False

            out = {}
            for edge_type in edge_types:
                edge_type_str = EdgeTypeStr(edge_type)
                if edge_type_str in self.values:
                    out[edge_type_str if mapped else edge_type] = (
                        self.values[edge_type_str])
                else:
                    if default is None:
                        raise ValueError(f"Missing number of neighbors for "
                                         f"edge type '{edge_type}'")
                    out[edge_type_str if mapped else edge_type] = default

        elif isinstance(self.values, dict) and not mapped:
            out = {key.to_tuple(): value for key, value in self.values.items()}

        else:
            out = copy.copy(self.values)

        if isinstance(out, dict):
            num_hops = set(len(v) for v in out.values())
            if len(num_hops) > 1:
                raise ValueError(f"Number of hops must be the same across all "
                                 f"edge types (got {len(num_hops)} different "
                                 f"number of hops)")

        return out

    def get_values(
        self,
        edge_types: Optional[List[EdgeType]] = None,
    ) -> Union[List[int], Dict[EdgeType, List[int]]]:
        r"""Returns the number of neighbors.

        Args:
            edge_types (List[Tuple[str, str, str]], optional): The edge types
                to generate the number of neighbors for. (default: :obj:`None`)
        """
        if '_values' in self.__dict__:
            return self.__dict__['_values']

        values = self._get_values(edge_types, mapped=False)

        self.__dict__['_values'] = values
        return values

    def get_mapped_values(
        self,
        edge_types: Optional[List[EdgeType]] = None,
    ) -> Union[List[int], Dict[str, List[int]]]:
        r"""Returns the number of neighbors.
        For heterogeneous graphs, a dictionary is returned in which edge type
        tuples are converted to strings.

        Args:
            edge_types (List[Tuple[str, str, str]], optional): The edge types
                to generate the number of neighbors for. (default: :obj:`None`)
        """
        if '_mapped_values' in self.__dict__:
            return self.__dict__['_mapped_values']

        values = self._get_values(edge_types, mapped=True)

        self.__dict__['_mapped_values'] = values
        return values

    @property
    def num_hops(self) -> int:
        r"""Returns the number of hops."""
        if '_num_hops' in self.__dict__:
            return self.__dict__['_num_hops']

        if isinstance(self.values, (tuple, list)):
            num_hops = max(len(self.values), len(self.default or []))
        else:  # isinstance(self.values, dict):
            num_hops = max([0] + [len(v) for v in self.values.values()])
            num_hops = max(num_hops, len(self.default or []))

        self.__dict__['_num_hops'] = num_hops
        return num_hops

    def __len__(self) -> int:
        r"""Returns the number of hops."""
        return self.num_hops


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
