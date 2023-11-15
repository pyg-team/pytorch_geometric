from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.mixin import AffinityMixin
from torch_geometric.loader.utils import (
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    get_input_nodes,
    infer_filter_per_worker,
)
from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.typing import InputNodes, OptTensor


class NodeLoader(torch.utils.data.DataLoader, AffinityMixin):
    r"""A data loader that performs mini-batch sampling from node information,
    using a generic :class:`~torch_geometric.sampler.BaseSampler`
    implementation that defines a
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_nodes` function and
    is supported on the provided input :obj:`data` object.

    Args:
        data (Any): A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        node_sampler (torch_geometric.sampler.BaseSampler): The sampler
            implementation to be used with this loader.
            Needs to implement
            :meth:`~torch_geometric.sampler.BaseSampler.sample_from_nodes`.
            The sampler implementation must be compatible with the input
            :obj:`data` object.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of seed nodes to start sampling from.
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
        transform (callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        transform_sampler_output (callable, optional): A function/transform
            that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
            returns a transformed version. (default: :obj:`None`)
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
        custom_cls (HeteroData, optional): A custom
            :class:`~torch_geometric.data.HeteroData` class to return for
            mini-batches in case of remote backends. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        node_sampler: BaseSampler,
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional[HeteroData] = None,
        input_id: OptTensor = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        # Remove for PyTorch Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        # Get node type (or `None` for homogeneous graphs):
        input_type, input_nodes, input_id = get_input_nodes(
            data, input_nodes, input_id)

        self.data = data
        self.node_sampler = node_sampler
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls

        self.input_data = NodeSamplerInput(
            input_id=input_id,
            node=input_nodes,
            time=input_time,
            input_type=input_type,
        )

        iterator = range(input_nodes.size(0))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        r"""Samples a subgraph from a batch of input nodes."""
        input_data: NodeSamplerInput = self.input_data[index]

        out = self.node_sampler.sample_from_nodes(input_data)

        if self.filter_per_worker:  # Execute `filter_fn` in the worker process
            out = self.filter_fn(out)

        return out

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object to be used downstream.
        """
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        if isinstance(out, SamplerOutput):
            if isinstance(self.data, Data):
                data = filter_data(  #
                    self.data, out.node, out.row, out.col, out.edge,
                    self.node_sampler.edge_permutation)

            else:  # Tuple[FeatureStore, GraphStore]
                edge_index = torch.stack([out.row, out.col])
                data = Data(edge_index=edge_index)

                # TODO Respect `custom_cls`.
                # TODO Integrate features.

                # Hack to detect whether we are in a distributed setting.
                if (self.node_sampler.__class__.__name__ ==
                        'DistNeighborSampler'):
                    # Metadata entries are populated in
                    # `DistributedNeighborSampler._collate_fn()`
                    data.x = out.metadata[-3]
                    data.y = out.metadata[-2]
                    data.edge_attr = out.metadata[-1]

            if 'n_id' not in data:
                data.n_id = out.node
            if out.edge is not None and 'e_id' not in data:
                edge = out.edge.to(torch.long)
                perm = self.node_sampler.edge_permutation
                data.e_id = perm[edge] if perm is not None else edge

            data.batch = out.batch
            data.num_sampled_nodes = out.num_sampled_nodes
            data.num_sampled_edges = out.num_sampled_edges

            data.input_id = out.metadata[0]
            data.seed_time = out.metadata[1]
            data.batch_size = out.metadata[0].size(0)

        elif isinstance(out, HeteroSamplerOutput):
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(  #
                    self.data, out.node, out.row, out.col, out.edge,
                    self.node_sampler.edge_permutation)

            else:  # Tuple[FeatureStore, GraphStore]
                if (self.node_sampler.__class__.__name__ ==
                        'DistNeighborSampler'):
                    import torch_geometric.distributed as dist
                    data = dist.utils.filter_dist_store(
                        *self.data, out.node, out.row, out.col, out.edge,
                        self.custom_cls, out.metadata)
                else:
                    data = filter_custom_store(  #
                        *self.data, out.node, out.row, out.col, out.edge,
                        self.custom_cls)

            for key, node in out.node.items():
                if 'n_id' not in data[key]:
                    data[key].n_id = node

            for key, edge in (out.edge or {}).items():
                if edge is not None and 'e_id' not in data[key]:
                    edge = edge.to(torch.long)
                    perm = self.node_sampler.edge_permutation
                    if perm is not None and perm.get(key, None) is not None:
                        edge = perm[key][edge]
                    data[key].e_id = edge

            data.set_value_dict('batch', out.batch)
            data.set_value_dict('num_sampled_nodes', out.num_sampled_nodes)
            data.set_value_dict('num_sampled_edges', out.num_sampled_edges)

            input_type = self.input_data.input_type
            data[input_type].input_id = out.metadata[0]
            data[input_type].seed_time = out.metadata[1]
            data[input_type].batch_size = out.metadata[0].size(0)

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()

        # if not self.is_cuda_available and not self.cpu_affinity_enabled:
        # TODO: Add manual page for best CPU practices
        # link = ...
        # Warning('Dataloader CPU affinity opt is not enabled, consider '
        #          'switching it on with enable_cpu_affinity() or see CPU '
        #          f'best practices for PyG [{link}])')

        # Execute `filter_fn` in the main process:
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __enter__(self):
        return self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
