import logging
from contextlib import contextmanager
from typing import Any, Callable, List, Optional

import psutil
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter

from torch_geometric.loader.utils import get_numa_nodes_cores


class DataLoaderIterator:
    r"""A data loader iterator extended by a simple post transformation
    function :meth:`transform_fn`. While the iterator may request items from
    different sub-processes, :meth:`transform_fn` will always be executed in
    the main process.

    This iterator is used in PyG's sampler classes, and is responsible for
    feature fetching and filtering data objects after sampling has taken place
    in a sub-process. This has the following advantages:

    * We do not need to share feature matrices across processes which may
      prevent any errors due to too many open file handles.
    * We can execute any expensive post-processing commands on the main thread
      with full parallelization power (which usually executes faster).
    * It lets us naturally support data already being present on the GPU.
    """
    def __init__(self, iterator: _BaseDataLoaderIter, transform_fn: Callable):
        self.iterator = iterator
        self.transform_fn = transform_fn

    def __iter__(self) -> 'DataLoaderIterator':
        return self

    def _reset(self, loader: Any, first_iter: bool = False):
        self.iterator._reset(loader, first_iter)

    def __len__(self) -> int:
        return len(self.iterator)

    def __next__(self) -> Any:
        return self.transform_fn(next(self.iterator))


class WorkerInitWrapper:
    r"""Wraps the :attr:`worker_init_fn` argument for PyTorch data loader
    workers."""
    def __init__(self, func):
        self.func = func

    def __call__(self, worker_id):
        if self.func is not None:
            self.func(worker_id)


class AffinityMixin:
    def __init__(self, dataloader: torch.utils.data.DataLoader = None):
        self.dataloader = dataloader
        if not torch.cuda.is_available():
            if not self.dataloader.num_workers > 0:
                raise ValueError(
                    f"'enable_cpu_affinity' should be used with at least one "
                    f"worker (got {self.num_workers})")
            print("Dataloader CPU affinitization enabled!")
        else:
            raise ValueError("Can't enable CPU affintity for GPU device.")

    @contextmanager
    def enable_cpu_affinity(self, loader_cores):
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
            with loader.enable_cpu_affinity(loader_cores=[0, 1, 2]):
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
                :obj:`cpu0`. (default: :obj:`node0_cores[:num_workers]`)
        """
        if loader_cores and len(loader_cores) != self.dataloader.num_workers:
            raise ValueError(
                f"The number of loader cores (got {len(loader_cores)}) "
                f"in 'enable_cpu_affinity' should match with the number "
                f"of workers (got {self.dataloader.num_workers})")
        worker_init_fn_old = WorkerInitWrapper(self.dataloader.worker_init_fn)
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

            if numa_info and len(numa_info[0]) > self.dataloader.num_workers:
                # Take one thread per each node 0 core:
                node0_cores = [cpus[0] for core_id, cpus in numa_info[0]]
            else:
                node0_cores = list(range(psutil.cpu_count(logical=False)))

            if len(node0_cores) < self.dataloader.num_workers:
                raise ValueError(
                    f"More workers (got {self.dataloader.num_workers}) than "
                    f"available cores (got {len(node0_cores)})")

            # Set default loader core IDs:
            loader_cores = node0_cores[:self.dataloader.num_workers]

        try:
            # Set CPU affinity for dataloader:
            self.dataloader.worker_init_fn = init_fn
            logging.info(f'{self.dataloader.num_workers} data loader workers '
                         f'are assigned to CPUs {loader_cores}')
            yield
        finally:
            # Restore omp_num_threads and cpu affinity:
            psutil.Process().cpu_affinity(affinity_old)
            torch.set_num_threads(nthreads_old)
            self.dataloader.worker_init_fn = worker_init_fn_old
            self.dataloader.cpu_affinity_enabled = False
