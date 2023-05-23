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
    get_edge_label_index,
    infer_filter_per_worker,
)
from torch_geometric.sampler import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    NegativeSampling,
    SamplerOutput,
)
from torch_geometric.typing import InputEdges, OptTensor


class LinkLoader(torch.utils.data.DataLoader, AffinityMixin):
    r"""A data loader that performs mini-batch sampling from link information,
    using a generic :class:`~torch_geometric.sampler.BaseSampler`
    implementation that defines a
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges` function and
    is supported on the provided input :obj:`data` object.

    .. note::
        Negative sampling is currently implemented in an approximate
        way, *i.e.* negative edges may contain false negatives.

    Args:
        data (Any): A :class:`~torch_geometric.data.Data`,
            :class:`~torch_geometric.data.HeteroData`, or
            (:class:`~torch_geometric.data.FeatureStore`,
            :class:`~torch_geometric.data.GraphStore`) data object.
        link_sampler (torch_geometric.sampler.BaseSampler): The sampler
            implementation to be used with this loader.
            Needs to implement
            :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.
            The sampler implementation must be compatible with the input
            :obj:`data` object.
        edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The edge indices, holding source and destination nodes to start
            sampling from.
            If set to :obj:`None`, all edges will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the edge type and corresponding edge indices.
            (default: :obj:`None`)
        edge_label (Tensor, optional): The labels of edge indices from which to
            start sampling from. Must be the same length as
            the :obj:`edge_label_index`. (default: :obj:`None`)
        edge_label_time (Tensor, optional): The timestamps of edge indices from
            which to start sampling from. Must be the same length as
            :obj:`edge_label_index`. If set, temporal sampling will be
            used such that neighbors are guaranteed to fulfill temporal
            constraints, *i.e.*, neighbors have an earlier timestamp than
            the ouput edge. The :obj:`time_attr` needs to be set for this
            to work. (default: :obj:`None`)
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
            (default: :obj:`None`).
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
        link_sampler: BaseSampler,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        edge_label_time: OptTensor = None,
        neg_sampling: Optional[NegativeSampling] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
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

        if neg_sampling_ratio is not None and neg_sampling_ratio != 0.0:
            # TODO: Deprecation warning.
            neg_sampling = NegativeSampling("binary", neg_sampling_ratio)

        # Get edge type (or `None` for homogeneous graphs):
        input_type, edge_label_index = get_edge_label_index(
            data, edge_label_index)

        self.data = data
        self.link_sampler = link_sampler
        self.neg_sampling = NegativeSampling.cast(neg_sampling)
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls

        if (self.neg_sampling is not None and self.neg_sampling.is_binary()
                and edge_label is not None and edge_label.min() == 0):
            # Increment labels such that `zero` now denotes "negative".
            edge_label = edge_label + 1

        if (self.neg_sampling is not None and self.neg_sampling.is_triplet()
                and edge_label is not None):
            raise ValueError("'edge_label' needs to be undefined for "
                             "'triplet'-based negative sampling. Please use "
                             "`src_index`, `dst_pos_index` and "
                             "`neg_pos_index` of the returned mini-batch "
                             "instead to differentiate between positive and "
                             "negative samples.")

        self.input_data = EdgeSamplerInput(
            input_id=input_id,
            row=edge_label_index[0].clone(),
            col=edge_label_index[1].clone(),
            label=edge_label,
            time=edge_label_time,
            input_type=input_type,
        )

        iterator = range(edge_label_index.size(1))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input edges."""
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        r"""Samples a subgraph from a batch of input edges."""
        input_data: EdgeSamplerInput = self.input_data[index]

        out = self.link_sampler.sample_from_edges(
            input_data, neg_sampling=self.neg_sampling)

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
            data = filter_data(self.data, out.node, out.row, out.col, out.edge,
                               self.link_sampler.edge_permutation)

            if 'n_id' not in data:
                data.n_id = out.node
            if out.edge is not None and 'e_id' not in data:
                data.e_id = out.edge

            data.batch = out.batch
            data.input_id = out.metadata[0]

            if self.neg_sampling is None or self.neg_sampling.is_binary():
                data.edge_label_index = out.metadata[1]
                data.edge_label = out.metadata[2]
                data.edge_label_time = out.metadata[3]
            elif self.neg_sampling.is_triplet():
                data.src_index = out.metadata[1]
                data.dst_pos_index = out.metadata[2]
                data.dst_neg_index = out.metadata[3]
                data.seed_time = out.metadata[4]
                # Sanity removals in case `edge_label_index` and
                # `edge_label_time` are attributes of the base `data` object:
                del data.edge_label_index  # Sanity removals.
                del data.edge_label_time

        elif isinstance(out, HeteroSamplerOutput):
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(self.data, out.node, out.row,
                                          out.col, out.edge,
                                          self.link_sampler.edge_permutation)
            else:  # Tuple[FeatureStore, GraphStore]
                data = filter_custom_store(*self.data, out.node, out.row,
                                           out.col, out.edge, self.custom_cls)

            for key, node in out.node.items():
                if 'n_id' not in data[key]:
                    data[key].n_id = node

            for key, edge in (out.edge or {}).items():
                if 'e_id' not in data[key]:
                    data[key].e_id = edge

            data.set_value_dict('batch', out.batch)

            input_type = self.input_data.input_type
            data[input_type].input_id = out.metadata[0]

            if self.neg_sampling is None or self.neg_sampling.is_binary():
                data[input_type].edge_label_index = out.metadata[1]
                data[input_type].edge_label = out.metadata[2]
                data[input_type].edge_label_time = out.metadata[3]
            elif self.neg_sampling.is_triplet():
                data[input_type[0]].src_index = out.metadata[1]
                data[input_type[-1]].dst_pos_index = out.metadata[2]
                data[input_type[-1]].dst_neg_index = out.metadata[3]
                data[input_type[0]].seed_time = out.metadata[4]
                data[input_type[-1]].seed_time = out.metadata[4]
                # Sanity removals in case `edge_label_index` and
                # `edge_label_time` are attributes of the base `data` object:
                if input_type in data.edge_types:
                    del data[input_type].edge_label_index
                    del data[input_type].edge_label_time

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()

        # Execute `filter_fn` in the main process:
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
