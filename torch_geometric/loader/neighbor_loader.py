from typing import Callable, Dict, List, Optional, Tuple, Union

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.node_loader import NodeLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor


class NeighborLoader(NodeLoader):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`num_neighbors` denotes how many neighbors are
    sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import NeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = NeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=data.train_mask,
        )

        sampled_data = next(iter(loader))
        print(sampled_data.batch_size)
        >>> 128

    By default, the data loader will only include the edges that were
    originally sampled (:obj:`directed = True`).
    This option should only be used in case the number of hops is equivalent to
    the number of GNN layers.
    In case the number of GNN layers is greater than the number of hops,
    consider setting :obj:`directed = False`, which will include all edges
    between all sampled nodes (but is slightly slower as a result).

    Furthermore, :class:`~torch_geometric.loader.NeighborLoader` works for both
    **homogeneous** graphs stored via :class:`~torch_geometric.data.Data` as
    well as **heterogeneous** graphs stored via
    :class:`~torch_geometric.data.HeteroData`.
    When operating in heterogeneous graphs, up to :obj:`num_neighbors`
    neighbors will be sampled for each :obj:`edge_type`.
    However, more fine-grained control over
    the amount of sampled neighbors of individual edge types is possible:

    .. code-block:: python

        from torch_geometric.datasets import OGB_MAG
        from torch_geometric.loader import NeighborLoader

        hetero_data = OGB_MAG(path)[0]

        loader = NeighborLoader(
            hetero_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper', hetero_data['paper'].train_mask),
        )

        sampled_hetero_data = next(iter(loader))
        print(sampled_hetero_data['paper'].batch_size)
        >>> 128

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.NeighborLoader`, see
        `examples/hetero/to_hetero_mag.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`_.

    The :class:`~torch_geometric.loader.NeighborLoader` will return subgraphs
    where global node indices are mapped to local indices corresponding to this
    specific subgraph. However, often times it is desired to map the nodes of
    the current subgraph back to the global node indices. The
    :class:`~torch_geometric.loader.NeighborLoader` will include this mapping
    as part of the :obj:`data` object:

    .. code-block:: python

        loader = NeighborLoader(data, ...)
        sampled_data = next(iter(loader))
        print(sampled_data.n_id)  # Global node index of each node in batch.

    In particular, the data loader will add the following attributes to the
    returned mini-batch:

    * :obj:`batch_size` The number of seed nodes (first nodes in the batch)
    * :obj:`n_id` The global node index for every sampled node
    * :obj:`e_id` The global edge index for every sampled edge
    * :obj:`input_id`: The global index of the :obj:`input_nodes`
    * :obj:`num_sampled_nodes`: The number of sampled nodes in each hop
    * :obj:`num_sampled_edges`: The number of sampled edges in each hop

    Args:
        data (Any): A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            If an entry is set to :obj:`-1`, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        input_time (torch.Tensor, optional): Optional values to override the
            timestamp for the input nodes given in :obj:`input_nodes`. If not
            set, will use the timestamps in :obj:`time_attr` as default (if
            present). The :obj:`time_attr` needs to be set for this to work.
            (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        subgraph_type (SubgraphType or str, optional): The type of the returned
            subgraph.
            If set to :obj:`"directional"`, the returned subgraph only holds
            the sampled (directed) edges which are necessary to compute
            representations for the sampled seed nodes.
            If set to :obj:`"bidirectional"`, sampled edges are converted to
            bidirectional edges.
            If set to :obj:`"induced"`, the returned subgraph contains the
            induced subgraph of all sampled nodes.
            (default: :obj:`"directional"`)
        disjoint (bool, optional): If set to :obj: `True`, each seed node will
            create its own disjoint subgraph.
            If set to :obj:`True`, mini-batch outputs will have a :obj:`batch`
            vector holding the mapping of nodes to their respective subgraph.
            Will get automatically set to :obj:`True` in case of temporal
            sampling. (default: :obj:`False`)
        temporal_strategy (str, optional): The sampling strategy when using
            temporal sampling (:obj:`"uniform"`, :obj:`"last"`).
            If set to :obj:`"uniform"`, will sample uniformly across neighbors
            that fulfill temporal constraints.
            If set to :obj:`"last"`, will sample the last `num_neighbors` that
            fulfill temporal constraints.
            (default: :obj:`"uniform"`)
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for either the nodes or edges in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier or equal timestamp than the center node.
            (default: :obj:`None`)
        weight_attr (str, optional): The name of the attribute that denotes
            edge weights in the graph.
            If set, weighted/biased sampling will be used such that neighbors
            are more likely to get sampled the higher their edge weights are.
            Edge weights do not need to sum to one, but must be non-negative,
            finite and have a non-zero sum within local neighborhoods.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        transform_sampler_output (callable, optional): A function/transform
            that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
            returns a transformed version. (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column.
            If :obj:`time_attr` is set, additionally requires that rows are
            sorted according to time within individual neighborhoods.
            This avoids internal re-sorting of the data and can improve
            runtime and memory efficiency. (default: :obj:`False`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returned data in each worker's subprocess.
            If set to :obj:`False`, will filter the returned data in the main
            process.
            If set to :obj:`None`, will automatically infer the decision based
            on whether data partially lives on the GPU
            (:obj:`filter_per_worker=True`) or entirely on the CPU
            (:obj:`filter_per_worker=False`).
            There exists different trade-offs for setting this option.
            Specifically, setting this option to :obj:`True` for in-memory
            datasets will move all features to shared memory, which may result
            in too many open file handles. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional[NeighborSampler] = None,
        directed: bool = True,  # Deprecated.
        **kwargs,
    ):
        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")

        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                weight_attr=weight_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                directed=directed,
            )

        super().__init__(
            data=data,
            node_sampler=neighbor_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
