from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Tuple, Union

import psutil
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.base import DataLoaderIterator
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
        # CPU Affinitization for loader and compute cores
        self.num_workers = kwargs.get('num_workers', 0)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cpu_affinity_enabled = False
        worker_init_fn = WorkerInitWrapper(kwargs.get('worker_init_fn', None))

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

        # if self.device.type == 'cpu' and not self.cpu_affinity_enabled:
        # TODO: Add manual page for best CPU practices for PyG and switch on warning message
        # link = ...
        # Warning(f'Dataloader CPU affinity opt is not enabled, consider switching it on '
        #             f'(see enable_cpu_affinity() or CPU best practices for PyG [{link}])')
        # Execute `filter_fn` in the main process:
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __enter__(self):
        return self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @contextmanager
    def enable_cpu_affinity(self, loader_cores: List[int] = None):
        """ Helper method for enabling cpu affinity for compute threads and dataloader workers
        Only for CPU devices
        Uses NUMA node 0 by default for multi-node systems

        Args:
            loader_cores : [int] (optional)
                List of cpu cores to which dataloader workers should affinitize to.
                By default cpu0 is reserved for all auxiliary threads & ops.
                DataLoader wil affinitize to cores starting at cpu1.
                default: node0_cores[1:num_workers]
        Usage:
        with dataloader.enable_cpu_affinity(losfrt):
        <training loop>
        """
        if self.device.type == 'cpu':
            if not self.num_workers > 0:
                raise Exception(
                    'ERROR: affinity should be used with at least one DL worker'
                )
            if loader_cores and len(loader_cores) != self.num_workers:
                raise Exception(
                    'ERROR: cpu_affinity incorrect '
                    f'number of loader_cores={loader_cores} for num_workers={self.num_workers}')

            worker_init_fn_old = self.worker_init_fn
            affinity_old = psutil.Process().cpu_affinity()
            nthreads_old = torch.get_num_threads()
            loader_cores = loader_cores[:] if loader_cores else []

            def init_fn(worker_id):
                try:
                    psutil.Process().cpu_affinity([loader_cores[worker_id]])
                except:
                    raise Exception(
                        f'ERROR: cannot use affinity id={worker_id} cpu={loader_cores}')

                worker_init_fn_old(worker_id)

            if not loader_cores:
                numa_info = get_numa_nodes_cores()
                if numa_info and len(numa_info[0]) > self.num_workers:
                    # take one thread per each node 0 core
                    node0_cores = [cpus[0] for core_id, cpus in numa_info[0]]
                else:
                    node0_cores = list(range(psutil.cpu_count(logical=False)))

                if len(node0_cores) - 1 <= self.num_workers:
                    raise Exception('ERROR: more workers than available cores')

                # set default loader core ids
                loader_cores = loader_cores or node0_cores[1:self.num_workers +
                                                           1]

            try:
                # set cpu affinity for dataloader
                self.worker_init_fn = init_fn
                self.cpu_affinity_enabled = True
                print(
                    f"{self.num_workers} DataLoader workers are assigned to CPUs {loader_cores}"
                )
                yield
            finally:
                # restore omp_num_threads and cpu affinity
                psutil.Process().cpu_affinity(affinity_old)
                torch.set_num_threads(nthreads_old)
                self.worker_init_fn = worker_init_fn_old
                self.cpu_affinity_enabled = False
        else:
            yield


class WorkerInitWrapper(object):
    """Wraps the :attr:`worker_init_fn` argument of the DataLoader to set the number of
    OMP threads to 1 for PyTorch DataLoader workers.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, worker_id):
        torch.set_num_threads(1)
        if self.func is not None:
            self.func(worker_id)
