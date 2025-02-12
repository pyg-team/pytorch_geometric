from typing import Callable, Dict, List, Optional, Tuple, Union

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.link_loader import LinkLoader
from torch_geometric.sampler import NegativeSampling, NeighborSampler
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputEdges, OptTensor


class LinkNeighborLoader(LinkLoader):
    r"""A link-based data loader derived as an extension of the node-based
    :class:`torch_geometric.loader.NeighborLoader`.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, this loader first selects a sample of edges from the
    set of input edges :obj:`edge_label_index` (which may or not be edges in
    the original graph) and then constructs a subgraph from all the nodes
    present in this list by sampling :obj:`num_neighbors` neighbors in each
    iteration.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import LinkNeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = LinkNeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            edge_label_index=data.edge_index,
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
                 train_mask=[1368], val_mask=[1368], test_mask=[1368],
                 edge_label_index=[2, 128])

    It is additionally possible to provide edge labels for sampled edges, which
    are then added to the batch:

    .. code-block:: python

        loader = LinkNeighborLoader(
            data,
            num_neighbors=[30] * 2,
            batch_size=128,
            edge_label_index=data.edge_index,
            edge_label=torch.ones(data.edge_index.size(1))
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
                 train_mask=[1368], val_mask=[1368], test_mask=[1368],
                 edge_label_index=[2, 128], edge_label=[128])

    The rest of the functionality mirrors that of
    :class:`~torch_geometric.loader.NeighborLoader`, including support for
    heterogeneous graphs.
    In particular, the data loader will add the following attributes to the
    returned mini-batch:

    * :obj:`n_id` The global node index for every sampled node
    * :obj:`e_id` The global edge index for every sampled edge
    * :obj:`input_id`: The global index of the :obj:`edge_label_index`
    * :obj:`num_sampled_nodes`: The number of sampled nodes in each hop
    * :obj:`num_sampled_edges`: The number of sampled edges in each hop

    .. note::
        Negative sampling is currently implemented in an approximate
        way, *i.e.* negative edges may contain false negatives.

    .. warning::
        Note that the sampling scheme is independent from the edge we are
        making a prediction for.
        That is, by default supervision edges in :obj:`edge_label_index`
        **will not** get masked out during sampling.
        In case there exists an overlap between message passing edges in
        :obj:`data.edge_index` and supervision edges in
        :obj:`edge_label_index`, you might end up sampling an edge you are
        making a prediction for.
        You can generally avoid this behavior (if desired) by making
        :obj:`data.edge_index` and :obj:`edge_label_index` two disjoint sets of
        edges, *e.g.*, via the
        :class:`~torch_geometric.transforms.RandomLinkSplit` transformation and
        its :obj:`disjoint_train_ratio` argument.

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
        edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The edge indices for which neighbors are sampled to create
            mini-batches.
            If set to :obj:`None`, all edges will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the edge type and corresponding edge indices.
            (default: :obj:`None`)
        edge_label (Tensor, optional): The labels of edge indices for
            which neighbors are sampled. Must be the same length as
            the :obj:`edge_label_index`. If set to :obj:`None` its set to
            `torch.zeros(...)` internally. (default: :obj:`None`)
        edge_label_time (Tensor, optional): The timestamps for edge indices
            for which neighbors are sampled. Must be the same length as
            :obj:`edge_label_index`. If set, temporal sampling will be
            used such that neighbors are guaranteed to fulfill temporal
            constraints, *i.e.*, neighbors have an earlier timestamp than
            the ouput edge. The :obj:`time_attr` needs to be set for this
            to work. (default: :obj:`None`)
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
        neg_sampling (NegativeSampling, optional): The negative sampling
            configuration.
            For negative sampling mode :obj:`"binary"`, samples can be accessed
            via the attributes :obj:`edge_label_index` and :obj:`edge_label` in
            the respective edge type of the returned mini-batch.
            In case :obj:`edge_label` does not exist, it will be automatically
            created and represents a binary classification task (:obj:`0` =
            negative edge, :obj:`1` = positive edge).
            In case :obj:`edge_label` does exist, it has to be a categorical
            label from :obj:`0` to :obj:`num_classes - 1`.
            After negative sampling, label :obj:`0` represents negative edges,
            and labels :obj:`1` to :obj:`num_classes` represent the labels of
            positive edges.
            Note that returned labels are of type :obj:`torch.float` for binary
            classification (to facilitate the ease-of-use of
            :meth:`F.binary_cross_entropy`) and of type
            :obj:`torch.long` for multi-class classification (to facilitate the
            ease-of-use of :meth:`F.cross_entropy`).
            For negative sampling mode :obj:`"triplet"`, samples can be
            accessed via the attributes :obj:`src_index`, :obj:`dst_pos_index`
            and :obj:`dst_neg_index` in the respective node types of the
            returned mini-batch.
            :obj:`edge_label` needs to be :obj:`None` for :obj:`"triplet"`
            negative sampling mode.
            If set to :obj:`None`, no negative sampling strategy is applied.
            (default: :obj:`None`)
        neg_sampling_ratio (int or float, optional): The ratio of sampled
            negative edges to the number of positive edges.
            Deprecated in favor of the :obj:`neg_sampling` argument.
            (default: :obj:`None`)
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for either the nodes or edges in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier or equal timestamp than the center node.
            Only used if :obj:`edge_label_time` is set. (default: :obj:`None`)
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
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        edge_label_time: OptTensor = None,
        replace: bool = False,
        subgraph_type: Union[SubgraphType, str] = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        neg_sampling: Optional[NegativeSampling] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
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
        if (edge_label_time is not None) != (time_attr is not None):
            raise ValueError(
                f"Received conflicting 'edge_label_time' and 'time_attr' "
                f"arguments: 'edge_label_time' is "
                f"{'set' if edge_label_time is not None else 'not set'} "
                f"while 'time_attr' is "
                f"{'set' if time_attr is not None else 'not set'}. "
                f"Both arguments must be provided for temporal sampling.")

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
            link_sampler=neighbor_sampler,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            edge_label_time=edge_label_time,
            neg_sampling=neg_sampling,
            neg_sampling_ratio=neg_sampling_ratio,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
