import glob
import logging
import os
import os.path as osp
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
import torch

from torch_geometric.data import HeteroData


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
            for cpu_dir in glob.glob(osp.join(node_path, 'cpu[0-9]*')):
                cpu_id = int(osp.basename(cpu_dir)[3:])
                if cpu_id > 0:
                    with open(osp.join(cpu_dir, 'online')) as core_online_file:
                        core_online = int(
                            core_online_file.read().splitlines()[0])
                else:
                    core_online = 1  # cpu0 is always online (special case)
                if core_online == 1:
                    with open(osp.join(cpu_dir, 'topology',
                                       'core_id')) as core_id_file:
                        core_id = int(core_id_file.read().strip())
                        if core_id in thread_siblings:
                            thread_siblings[core_id].append(cpu_id)
                        else:
                            thread_siblings[core_id] = [cpu_id]

            nodes[numa_node_id] = sorted([(k, sorted(v))
                                          for k, v in thread_siblings.items()])

    except (OSError, ValueError, IndexError):
        Warning('Failed to read NUMA info')
        return {}

    return nodes


class WorkerInitWrapper:
    r"""Wraps the :attr:`worker_init_fn` argument for
    :class:`torch.utils.data.DataLoader` workers.
    """
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __call__(self, worker_id: int) -> None:
        if self.func is not None:
            self.func(worker_id)


class LogMemoryMixin:
    r"""A context manager to enable logging of memory consumption in
    :class:`~torch.utils.data.DataLoader` workers.
    """
    def _mem_init_fn(self, worker_id: int) -> None:
        proc = psutil.Process(os.getpid())
        memory = proc.memory_info().rss / (1024 * 1024)
        logging.debug(f"Worker {worker_id} @ PID {proc.pid}: {memory:.2f} MB")

        # Chain worker init functions:
        self._old_worker_init_fn(worker_id)

    @contextmanager
    def enable_memory_log(self) -> None:
        self._old_worker_init_fn = WorkerInitWrapper(self.worker_init_fn)
        try:
            self.worker_init_fn = self._mem_init_fn
            yield
        finally:
            self.worker_init_fn = self._old_worker_init_fn


class MultithreadingMixin:
    r"""A context manager to enable multi-threading in
    :class:`~torch.utils.data.DataLoader` workers.
    It changes the default value of threads used in the loader from :obj:`1`
    to :obj:`worker_threads`.
    """
    def _mt_init_fn(self, worker_id: int) -> None:
        try:
            torch.set_num_threads(int(self._worker_threads))
        except IndexError as e:
            raise ValueError(f"Cannot set {self.worker_threads} threads "
                             f"in worker {worker_id}") from e

        # Chain worker init functions:
        self._old_worker_init_fn(worker_id)

    @contextmanager
    def enable_multithreading(
        self,
        worker_threads: Optional[int] = None,
    ) -> None:
        r"""Enables multithreading in worker subprocesses.
        This option requires to change the start method from :obj:`"fork"` to
        :obj:`"spawn"`.

        .. code-block:: python

            def run():
                loader = NeigborLoader(data, num_workers=3)
                with loader.enable_multithreading(10):
                    for batch in loader:
                        pass

            if __name__ == '__main__':
                torch.set_start_method('spawn')
                run()

        Args:
            worker_threads (int, optional): The number of threads to use in
                each worker process.
                By default, it uses half of all available CPU cores.
                (default: :obj:`torch.get_num_threads() // num_workers`)
        """
        if worker_threads is None:
            worker_threads = torch.get_num_threads() // self.num_workers

        self._worker_threads = worker_threads

        if not self.num_workers > 0:
            raise ValueError(f"'enable_multithreading' needs to be performed "
                             f"with at least one worker "
                             f"(got {self.num_workers})")

        if worker_threads > torch.get_num_threads():
            raise ValueError(f"'worker_threads' should be smaller than the "
                             f"total available number of threads "
                             f"{torch.get_num_threads()} "
                             f"(got {worker_threads})")

        context = torch.multiprocessing.get_context()._name
        if context != 'spawn':
            raise ValueError(f"'enable_multithreading' can only be used with "
                             f"the 'spawn' multiprocessing context "
                             f"(got {context})")

        self._old_worker_init_fn = WorkerInitWrapper(self.worker_init_fn)
        try:
            logging.debug(f"Using {worker_threads} threads in each worker")
            self.worker_init_fn = self._mt_init_fn
            yield
        finally:
            self.worker_init_fn = self._old_worker_init_fn


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

        To correctly affinitize compute threads (*i.e.* with
        :obj:`KMP_AFFINITY`), please make sure that you exclude
        :obj:`loader_cores` from the list of cores available for the main
        process.
        This will cause core oversubsription and exacerbate performance.

    .. code-block:: python

        loader = NeigborLoader(data, num_workers=3)
        with loader.enable_cpu_affinity(loader_cores=[0, 1, 2]):
            for batch in loader:
                pass

    """
    def _aff_init_fn(self, worker_id: int) -> None:
        try:
            worker_cores = self.loader_cores[worker_id]
            if not isinstance(worker_cores, List):
                worker_cores = [worker_cores]

            if torch.multiprocessing.get_context()._name == 'spawn':
                torch.set_num_threads(len(worker_cores))

            psutil.Process().cpu_affinity(worker_cores)

        except IndexError as e:
            raise ValueError(f"Cannot use CPU affinity for worker ID "
                             f"{worker_id} on CPU {self.loader_cores}") from e

        # Chain worker init functions:
        self._old_worker_init_fn(worker_id)

    @contextmanager
    def enable_cpu_affinity(
        self,
        loader_cores: Optional[Union[List[List[int]], List[int]]] = None,
    ) -> None:
        r"""Enables CPU affinity.

        Args:
            loader_cores ([int], optional): List of CPU cores to which data
                loader workers should affinitize to.
                By default, it will affinitize to :obj:`numa0` cores.
                If used with :obj:`"spawn"` multiprocessing context, it will
                automatically enable multithreading and use multiple cores
                per each worker.
        """
        if not self.num_workers > 0:
            raise ValueError(
                f"'enable_cpu_affinity' should be used with at least one "
                f"worker (got {self.num_workers})")
        if loader_cores and len(loader_cores) != self.num_workers:
            raise ValueError(
                f"The number of loader cores (got {len(loader_cores)}) "
                f"in 'enable_cpu_affinity' should match with the number "
                f"of workers (got {self.num_workers})")
        if isinstance(self.data, HeteroData):
            warnings.warn(
                "Due to conflicting parallelization methods it is not advised "
                "to use affinitization with 'HeteroData' datasets. "
                "Use `enable_multithreading` for better performance.")

        self.loader_cores = loader_cores[:] if loader_cores else None
        if self.loader_cores is None:
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
                    f"cores (got {len(node0_cores)})")

            # Set default loader core IDs:
            if torch.multiprocessing.get_context()._name == 'spawn':
                work_thread_pool = int(len(node0_cores) / self.num_workers)
                self.loader_cores = [
                    list(
                        range(
                            work_thread_pool * i,
                            work_thread_pool * (i + 1),
                        )) for i in range(self.num_workers)
                ]
            else:
                self.loader_cores = node0_cores[:self.num_workers]

        self._old_worker_init_fn = WorkerInitWrapper(self.worker_init_fn)
        try:
            self.worker_init_fn = self._aff_init_fn
            logging.debug(f"{self.num_workers} data loader workers are "
                          f"assigned to CPUs {self.loader_cores}")
            yield
        finally:
            self.worker_init_fn = self._old_worker_init_fn
