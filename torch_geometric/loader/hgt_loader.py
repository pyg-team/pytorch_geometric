from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor

from torch_geometric.data import FeatureStore, GraphStore, HeteroData
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler import HGTSampler
from torch_geometric.typing import NodeType


class HGTLoader(NodeLoader):
    r"""The Heterogeneous Graph Sampler from the `"Heterogeneous Graph
    Transformer" <https://arxiv.org/abs/2003.01332>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    :class:`~torch_geometric.data.HGTLoader` tries to (1) keep a similar
    number of nodes and edges for each type and (2) keep the sampled sub-graph
    dense to minimize the information loss and reduce the sample variance.

    Methodically, :class:`~torch_geometric.data.HGTLoader` keeps track of a
    node budget for each node type, which is then used to determine the
    sampling probability of a node.
    In particular, the probability of sampling a node is determined by the
    number of connections to already sampled nodes and their node degrees.
    With this, :class:`~torch_geometric.data.HGTLoader` will sample a fixed
    amount of neighbors for each node type in each iteration, as given by the
    :obj:`num_samples` argument.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    .. note::

        For an example of using :class:`~torch_geometric.data.HGTLoader`, see
        `examples/hetero/to_hetero_mag.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/to_hetero_mag.py>`_.

    .. code-block:: python

        from torch_geometric.loader import HGTLoader
        from torch_geometric.datasets import OGB_MAG

        hetero_data = OGB_MAG(path)[0]

        loader = HGTLoader(
            hetero_data,
            # Sample 512 nodes per type and per iteration for 4 iterations
            num_samples={key: [512] * 4 for key in hetero_data.node_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper', hetero_data['paper'].train_mask),
        )

        sampled_hetero_data = next(iter(loader))
        print(sampled_data.batch_size)
        >>> 128

    Args:
        data (Any): A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        num_samples (List[int] or Dict[str, List[int]]): The number of nodes to
            sample in each iteration and for each node type.
            If given as a list, will sample the same amount of nodes for each
            node type.
        input_nodes (str or Tuple[str, torch.Tensor]): The indices of nodes for
            which neighbors are sampled to create mini-batches.
            Needs to be passed as a tuple that holds the node type and
            corresponding node indices.
            Node indices need to be either given as a :obj:`torch.LongTensor`
            or :obj:`torch.BoolTensor`.
            If node indices are set to :obj:`None`, all nodes of this specific
            type will be considered.
        transform (callable, optional): A function/transform that takes in
            an a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        transform_sampler_output (callable, optional): A function/transform
            that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
            returns a transformed version. (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column. This avoids internal
            re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returning data in each worker's subprocess rather than in the
            main process.
            Setting this to :obj:`True` for in-memory datasets is generally not
            recommended:
            (1) it may result in too many open file handles,
            (2) it may slown down data loading,
            (3) it requires operating on CPU tensors.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[HeteroData, Tuple[FeatureStore, GraphStore]],
        num_samples: Union[List[int], Dict[NodeType, List[int]]],
        input_nodes: Union[NodeType, Tuple[NodeType, Optional[Tensor]]],
        is_sorted: bool = False,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: bool = False,
        **kwargs,
    ):
        hgt_sampler = HGTSampler(
            data,
            num_samples=num_samples,
            is_sorted=is_sorted,
            share_memory=kwargs.get('num_workers', 0) > 0,
        )

        super().__init__(
            data=data,
            node_sampler=hgt_sampler,
            input_nodes=input_nodes,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
