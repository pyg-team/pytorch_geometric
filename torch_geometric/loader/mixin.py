import glob
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import psutil
import torch

from torch_geometric.data.hetero_data import HeteroData


def get_numa_nodes_cores() -> Dict[str, Any]:
    """Parses numa nodes information into a dictionary.

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
            numa_node_id = int(osp.basename(node_path)[4:])

            thread_siblings = {}
            for cpu_dir in glob.glob(os.path.join(node_path, 'cpu[0-9]*')):
                cpu_id = int(os.path.basename(cpu_dir)[3:])
                if cpu_id > 0:
                    with open(
                        os.path.join(cpu_dir, 'online')
                    ) as core_online_file:
                        core_online = int(
                            core_online_file.read().splitlines()[0]
                        )
                else:
                    core_online = 1  # cpu0 is always online (special case)
                if core_online == 1:
                    with open(
                        os.path.join(cpu_dir, 'topology', 'core_id')
                    ) as core_id_file:
                        core_id = int(core_id_file.read().strip())
                        if core_id in thread_siblings:
                            thread_siblings[core_id].append(cpu_id)
                        else:
                            thread_siblings[core_id] = [cpu_id]

            nodes[numa_node_id] = sorted(
                [(k, sorted(v)) for k, v in thread_siblings.items()]
            )

    except (OSError, ValueError, IndexError, IOError):
        Warning('Failed to read NUMA info')
        return {}

    return nodes


class WorkerInitWrapper:
    r"""Wraps the :attr:`worker_init_fn` argument for
    :class:`torch.utils.data.DataLoader` workers.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, worker_id):
        if self.func is not None:
            self.func(worker_id)


class MemMixin:
    r"""A context manager to enable logging of dataloder workers
    memory consumption.
    """

    def mem_init_fn(self, worker_id):
        proc = psutil.Process(os.getpid())
        logging.debug(
            f"worker {worker_id} PID-{proc.pid} is using "
            f"{round((proc.memory_info().rss / 1024 ** 2),2)} MB of memory"
        )
        # Chain init_fn's
        self.worker_init_fn_old(worker_id)

    @contextmanager
    def enable_memory_log(self):
        self.worker_init_fn_old = WorkerInitWrapper(self.worker_init_fn)
        try:
            self.worker_init_fn = self.mem_init_fn
            yield
        finally:
            self.worker_init_fn = self.worker_init_fn_old


class MultithreadMixin:
    r"""A context manager to enable multithreading in dataloader workers.
    It changes the default value of threads used in the sampler from 1 to `worker_threads`.
    """

    def mt_init_fn(self, worker_id):
        try:
            torch.set_num_threads(int(self.worker_threads))
        except IndexError:
            raise ValueError(f"Cannot set multithreading for {worker_id}")
        # Chain init_fn's
        self.worker_init_fn_old(worker_id)

    @contextmanager
    def enable_multithreading(
        self,
        worker_threads: Optional[int] = None,
    ):
        r"""Enables multithreading in worker subprocess.

        Args:
            worker_threads ([int], optional): Number of cores for sampling.
            By defualt it uses half of all avaialbe CPU cores
            (default: : `torch.get_num_threads() / 2`)

        .. code-block:: python
        def run():
            loader = NeigborLoader(data, num_workers=3)
            with loader.enable_multithreading(10):
                for batch in loader:
                    pass
        if if __name__ == '__main__':
            torch.set_start_method('spawn')
            run()
        """
        if worker_threads:
            self.worker_threads = worker_threads
        else:
            self.worker_threads = int(
                torch.get_num_threads() / self.num_workers
            )

        self.worker_init_fn_old = WorkerInitWrapper(self.worker_init_fn)

        if not self.num_workers > 0:
            raise ValueError(
                f"'enable_multithread_sampling' should be used with at least one "
                f"worker (got {self.num_workers})"
            )
        if worker_threads > torch.get_num_threads():
            raise ValueError(
                f"'worker_threads' should be smaller than the total available "
                f"number of threads (max is {torch.get_num_threads()} "
                f"got {worker_threads})"
            )
        if torch.multiprocessing.get_context()._name != 'spawn':
            raise ValueError(
                f"'enable_multithread_sampling' can only be used with 'spawn' multiprocessing context "
                f"(got {torch.multiprocessing.get_context()._name})"
            )

        try:
            self.worker_init_fn = self.mt_init_fn
            logging.debug(
                f"Using {self.worker_threads} threads in each sampling worker"
            )
            yield
        finally:
            self.worker_init_fn = self.worker_init_fn_old


class AffinityMixin:
    r"""A context manager to enable CPU affinity for data loader workers
    (only used when running on CPU devices).

    Affinitization places data loader workers threads on specific CPU cores.
    In effect, it allows for more efficient local memory allocation and reduces
    remote memory calls.
    Every time a process or thread moves from one core to another, registers
    and caches need to be flushed and reloaded.
    This can become very costly if it happens often, and our threads may also
    no longer be close to their data, or be able to share data in a cache.

    See `here <https://pytorch-geometric.readthedocs.io/en/latest/advanced/
    cpu_affinity.html>`__ for the accompanying tutorial.

    .. warning::

        To correctly affinitize compute threads (*i.e.* with KMP_AFFINITY),
        please make sure that you exclude :obj:`loader_cores` from the
        list of cores available for the main process.
        This will cause core oversubsription and exacerbate performance.

    .. code-block:: python
        loader = NeigborLoader(data, num_workers=3)
        with loader.enable_cpu_affinity(loader_cores=[0, 1, 2]):
            for batch in loader:
                pass
    """

    def aff_init_fn(self, worker_id):
        try:
            worker_cores = self.loader_cores[worker_id]
            if not isinstance(worker_cores, List):
                worker_cores = [worker_cores]

            if torch.multiprocessing.get_context()._name == 'spawn':
                torch.set_num_threads(len(worker_cores))

            psutil.Process().cpu_affinity(worker_cores)

        except IndexError:
            raise ValueError(
                f"Cannot use CPU affinity for worker ID "
                f"{worker_id} on CPU {self.loader_cores}"
            )

        # Chain init_fn's:
        self.worker_init_fn_old(worker_id)

    @contextmanager
    def enable_cpu_affinity(
        self,
        loader_cores: Optional[Union[List[List[int]], List[int]]] = None,
    ):
        r"""Enables CPU affinity.

        Args:
            loader_cores ([int], optional): List of CPU cores to which data
                loader workers should affinitize to.
                By default, it will affinitize to numa0 cores. If used with 'spawn' multiprocessing context, it will automatically enable multithreading and use multiple cores per eachg sampler.
        """
        if not self.num_workers > 0:
            raise ValueError(
                f"'enable_cpu_affinity' should be used with at least one "
                f"worker (got {self.num_workers})"
            )
        if loader_cores and len(loader_cores) != self.num_workers:
            raise ValueError(
                f"The number of loader cores (got {len(loader_cores)}) "
                f"in 'enable_cpu_affinity' should match with the number "
                f"of workers (got {self.num_workers})"
            )
        if isinstance(self.data, HeteroData):
            raise UserWarning(
                f"Due to conflicting parallelization methods it is not advised "
                f"to use affinitization with hetero datasets. "
                f"Use `enable_multithreading` for better performance. "
            )

        self.worker_init_fn_old = WorkerInitWrapper(self.worker_init_fn)
        self.loader_cores = loader_cores[:] if loader_cores else None
        if loader_cores is None:
            numa_info = get_numa_nodes_cores()

            if numa_info and len(numa_info[0]) > self.num_workers:
                # Take one thread per each node 0 core:
                node0_cores = [cpus[0] for core_id, cpus in numa_info[0]]
                node0_cores.sort()
            else:
                node0_cores = list(range(psutil.cpu_count(logical=False)))

            if len(node0_cores) < self.num_workers:
                raise ValueError(
                    f"More workers (got {self.num_workers}) than available "
                    f"cores (got {len(node0_cores)})"
                )

            # Set default loader core IDs:
            if torch.multiprocessing.get_context()._name == 'spawn':
                work_thread_pool = int(len(node0_cores) / self.num_workers)
                self.loader_cores = [
                    list(
                        range(work_thread_pool * i, work_thread_pool * (i + 1))
                    )
                    for i in range(self.num_workers)
                ]
            else:
                self.loader_cores = node0_cores[: self.num_workers]
        try:
            # Set CPU affinity for dataloader:
            self.worker_init_fn = self.aff_init_fn
            logging.debug(
                f"{self.num_workers} data loader workers are "
                f"assigned to CPUs {loader_cores}"
            )
            yield
        finally:
            self.worker_init_fn = self.worker_init_fn_old
