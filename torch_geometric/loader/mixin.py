import glob
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict

import psutil
import torch


def get_numa_nodes_cores() -> Dict[str, Any]:
    """ Returns numa nodes info in format:

    ..code-block::

        {<node_id>: [(<core_id>, [<sibling_thread_id_0>, <sibling_thread_id_1>
        ...]), ...], ...}

        # For example:
        {0: [(0, [0, 4]), (1, [1, 5])], 1: [(2, [2, 6]), (3, [3, 7])]}

    If not available, returns an empty dictionary.
    """
    numa_node_paths = glob.glob('/sys/devices/system/node/node[0-9]*')

    if not numa_node_paths:
        return {}

    nodes = {}
    try:
        for node_path in numa_node_paths:
            numa_node_id = int(os.path.basename(node_path)[4:])

            thread_siblings = {}
            for cpu_dir in glob.glob(os.path.join(node_path, 'cpu[0-9]*')):
                cpu_id = int(os.path.basename(cpu_dir)[3:])
                if cpu_id > 0:
                    with open(os.path.join(cpu_dir,
                                           'online')) as core_online_file:
                        core_online = int(
                            core_online_file.read().splitlines()[0])
                else:
                    core_online = 1  # cpu0 is always online (special case)
                if core_online == 1:
                    with open(os.path.join(cpu_dir, 'topology',
                                           'core_id')) as core_id_file:
                        core_id = int(core_id_file.read().strip())
                        if core_id in thread_siblings:
                            thread_siblings[core_id].append(cpu_id)
                        else:
                            thread_siblings[core_id] = [cpu_id]

            nodes[numa_node_id] = sorted([(k, sorted(v))
                                          for k, v in thread_siblings.items()])

    except (OSError, ValueError, IndexError, IOError):
        Warning('Failed to read NUMA info')
        return {}

    return nodes


class WorkerInitWrapper:
    r"""Wraps the :attr:`worker_init_fn` argument for
    :class:`torch.utils.data.DataLoader` workers."""
    def __init__(self, func):
        self.func = func

    def __call__(self, worker_id):
        if self.func is not None:
            self.func(worker_id)


class AffinityMixin:
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
        # if not torch.cuda.is_available():
        if not self.num_workers > 0:
            raise ValueError(
                f"'enable_cpu_affinity' should be used with at least one "
                f"worker (got {self.num_workers})")
        # else:
        #     raise ValueError("Can't enable CPU affintity for GPU device")

        if loader_cores and len(loader_cores) != self.num_workers:
            raise ValueError(
                f"The number of loader cores (got {len(loader_cores)}) "
                f"in 'enable_cpu_affinity' should match with the number "
                f"of workers (got {self.num_workers})")
        worker_init_fn_old = WorkerInitWrapper(self.worker_init_fn)
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

            if len(node0_cores) < self.num_workers:
                raise ValueError(
                    f"More workers (got {self.num_workers}) than available "
                    f"cores (got {len(node0_cores)})")

            # Set default loader core IDs:
            loader_cores = node0_cores[:self.num_workers]

        try:
            # Set CPU affinity for dataloader:
            self.worker_init_fn = init_fn
            logging.info(f"{self.num_workers} data loader workers are "
                         f"assigned to CPUs {loader_cores}")
            yield
        finally:
            # Restore omp_num_threads and cpu affinity:
            psutil.Process().cpu_affinity(affinity_old)
            torch.set_num_threads(nthreads_old)
            self.worker_init_fn = worker_init_fn_old
            self.cpu_affinity_enabled = False
