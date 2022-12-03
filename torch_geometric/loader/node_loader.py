import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import psutil
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.base import DataLoaderIterator, WorkerInitWrapper
from torch_geometric.loader.utils import (
    InputData,
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    get_input_nodes,
    get_numa_nodes_cores,
)
from torch_geometric.sampler.base import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.typing import InputNodes, OptTensor


class NodeLoader(torch.utils.data.DataLoader):
    r"""A data loader that performs neighbor sampling from node information,
    using a generic :class:`~torch_geometric.sampler.BaseSampler`
    implementation that defines a :meth:`sample_from_nodes` function and is
    supported on the provided input :obj:`data` object.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        node_sampler (torch_geometric.sampler.BaseSampler): The sampler
            implementation to be used with this loader. Note that the
            sampler implementation must be compatible with the input data
            object.
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
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returning data in each worker's subprocess rather than in the
            main process.
            Setting this to :obj:`True` is generally not recommended:
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
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        node_sampler: BaseSampler,
        input_nodes: InputNodes = None,
        input_time: OptTensor = None,
        transform: Callable = None,
        filter_per_worker: bool = False,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Get node type (or `None` for homogeneous graphs):
        node_type, input_nodes = get_input_nodes(data, input_nodes)

        self.data = data
        self.node_type = node_type
        self.node_sampler = node_sampler
        self.input_data = InputData(input_nodes, input_time)
        self.transform = transform
        self.filter_per_worker = filter_per_worker

        # TODO: Unify DL affinitization in `BaseDataLoader` class
        # CPU Affinitization for loader and compute cores
        self.num_workers = kwargs.get('num_workers', 0)
        self.is_cuda_available = torch.cuda.is_available()
        self.cpu_affinity_enabled = False
        worker_init_fn = WorkerInitWrapper(kwargs.pop('worker_init_fn', None))

        iterator = range(input_nodes.size(0))
        super().__init__(iterator, collate_fn=self.collate_fn,
                         worker_init_fn=worker_init_fn, **kwargs)

    def collate_fn(self, index: NodeSamplerInput) -> Any:
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
        if isinstance(out, SamplerOutput):
            data = filter_data(self.data, out.node, out.row, out.col, out.edge,
                               self.node_sampler.edge_permutation)
            data.batch = out.batch
            data.input_id = out.metadata
            data.batch_size = out.metadata.size(0)

        elif isinstance(out, HeteroSamplerOutput):
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(self.data, out.node, out.row,
                                          out.col, out.edge,
                                          self.node_sampler.edge_permutation)
            else:  # Tuple[FeatureStore, GraphStore]
                data = filter_custom_store(*self.data, out.node, out.row,
                                           out.col, out.edge)

            for key, batch in (out.batch or {}).items():
                data[key].batch = batch
            data[self.node_type].input_id = out.metadata
            data[self.node_type].batch_size = out.metadata.size(0)

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

    @contextmanager
    def enable_cpu_affinity(self, loader_cores: Optional[List[int]] = None):
        r"""A context manager to enable CPU affinity for data loader workers
        (only used when running on CPU devices).

        Affinitization places data loader workers threads on specific CPU
        cores. In effect, it allows for more efficient local memory allocation
        and reduces remote memory calls.
        Every time a process or thread moves from one core to another,
        registers and caches need to be flushed and reloaded. This can become
        very costly if it happens often, and our threads may also no longer be
        close to their data, or be able to share data in a cache.

        .. warning::
            If you want to further affinitize compute threads
            (*i.e.* with OMP), please make sure that you exclude
            :obj:`loader_cores` from the list of cores available for compute.
            This will cause core oversubsription and exacerbate performance.

        .. code-block:: python

            loader = NeigborLoader(data, num_workers=3)
            with loader.enable_cpu_affinity(loader_cores=[1,2,3]):
                for batch in loader:
                    pass

        This will be gradually extended to increase performance on dual socket
        CPUs.

        Args:
            loader_cores ([int], optional): List of CPU cores to which data
                loader workers should affinitize to.
                By default, :obj:`cpu0` is reserved for all auxiliary threads
                and ops.
                The :class:`DataLoader` wil affinitize to cores starting at
                :obj:`cpu1`. (default: :obj:`node0_cores[1:num_workers]`)
        """
        if not self.is_cuda_available:
            if not self.num_workers > 0:
                raise ValueError(
                    f"'enable_cpu_affinity' should be used with at least one "
                    f"worker (got {self.num_workers})")
            if loader_cores and len(loader_cores) != self.num_workers:
                raise ValueError(
                    f"The number of loader cores (got {len(loader_cores)}) "
                    f"in 'enable_cpu_affinity' should match with the number "
                    f"of workers (got {self.num_workers})")

            worker_init_fn_old = self.worker_init_fn
            affinity_old = psutil.Process().cpu_affinity()
            nthreads_old = torch.get_num_threads()
            loader_cores = loader_cores[:] if loader_cores else None

            def init_fn(worker_id):
                try:
                    psutil.Process().cpu_affinity([loader_cores[worker_id]])
                except IndexError:
                    raise ValueError(f"Cannot use CPU affinity for worker ID "
                                     f"{worker_id} on CPU {loader_cores}")

                worker_init_fn_old(worker_id)

            if loader_cores is None:

                numa_info = get_numa_nodes_cores()

                if numa_info and len(numa_info[0]) > self.num_workers:
                    # Take one thread per each node 0 core:
                    node0_cores = [cpus[0] for core_id, cpus in numa_info[0]]
                else:
                    node0_cores = list(range(psutil.cpu_count(logical=False)))

                if len(node0_cores) - 1 < self.num_workers:
                    raise ValueError(
                        f"More workers (got {self.num_workers}) than "
                        f"available cores (got {len(node0_cores) - 1})")

                # Set default loader core IDs:
                loader_cores = node0_cores[1:self.num_workers + 1]

            try:
                # Set CPU affinity for dataloader:
                self.worker_init_fn = init_fn
                self.cpu_affinity_enabled = True
                logging.info(f'{self.num_workers} data loader workers '
                             f'are assigned to CPUs {loader_cores}')
                yield
            finally:
                # Restore omp_num_threads and cpu affinity:
                psutil.Process().cpu_affinity(affinity_old)
                torch.set_num_threads(nthreads_old)
                self.worker_init_fn = worker_init_fn_old
                self.cpu_affinity_enabled = False
        else:
            yield
