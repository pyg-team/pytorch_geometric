import atexit
import collections
import functools
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Set

from torch._C._distributed_rpc import _is_current_rpc_agent_set
from torch.distributed import rpc

from .dist_context import DistContext, DistRole

_rpc_init_lock = threading.RLock()

_rpc_worker_names: Dict[DistRole, List[str]] = None


def rpc_is_initialized():
    return _is_current_rpc_agent_set()


def _require_initialized(func):
    return rpc.api._require_initialized(func)


@_require_initialized
def get_rpc_worker_names() -> Dict[DistRole, List[str]]:
    return _rpc_worker_names


## All gather objects from all role groups.


@_require_initialized
def global_all_gather(obj, timeout=None):
    r""" Gathers objects from all groups in a list."""
    if timeout is None:
        return rpc.api._all_gather(obj)
    return rpc.api._all_gather(obj, timeout=timeout)


@_require_initialized
def global_barrier(timeout=None):
    r""" Block until all local and remote RPC processes. """
    try:
        global_all_gather(obj=None, timeout=timeout)
    except RuntimeError as ex:
        logging.error("Failed to respond to global_barrier")


## RPC initialization and shutdown


def init_rpc(current_ctx: DistContext, master_addr: str, master_port: int,
             num_rpc_threads: int = 16, rpc_timeout: float = 240):

    with _rpc_init_lock:
        if rpc_is_initialized() is True:
            return
        if rpc_is_initialized() is None:
            raise RuntimeError(
                "'init_rpc': error to re-init rpc after shutdown.")

        ctx = current_ctx
        if ctx is None:
            raise RuntimeError("'init_rpc': dist_context has not been set.")

        options = rpc.TensorPipeRpcBackendOptions(
            _transports=['ibv', 'uv'], _channels=['mpt_uv', 'basic'],
            num_worker_threads=num_rpc_threads, rpc_timeout=rpc_timeout,
            init_method=f'tcp://{master_addr}:{master_port}')

        rpc.init_rpc(name=ctx.worker_name, rank=ctx.global_rank,
                     world_size=ctx.global_world_size,
                     rpc_backend_options=options)

        global _rpc_worker_names
        _rpc_worker_names = {}
        gathered_results = global_all_gather(
            obj=(ctx.role, ctx.world_size, ctx.rank), timeout=rpc_timeout)
        for worker_name, (role, world_size,
                          node_rank) in gathered_results.items():
            worker_list = _rpc_worker_names.get(role, None)
            if worker_list is None:
                worker_list = [None for _ in range(world_size)]
            else:
                if len(worker_list) != world_size:
                    raise RuntimeError(
                        "'init_rpc': world size inconsistent with others.")

            worker_list[node_rank] = worker_name
            _rpc_worker_names[role] = worker_list

        global_barrier(timeout=rpc_timeout)


def shutdown_rpc(graceful=True):
    if rpc_is_initialized() is True:
        rpc.shutdown(graceful=graceful)


atexit.register(shutdown_rpc, False)

## RPC synchronization and routing with data partition mapping.


class RpcRouter():
    r""" A router to get the worker based on partition id."""
    def __init__(self, partition2workers: List[List[str]]):
        for pidx, rpc_worker_list in enumerate(partition2workers):
            if len(rpc_worker_list) == 0:
                raise ValueError("no rpc worker is in worklist'.")
        self.partition2workers = partition2workers
        self.rpc_worker_indexs = [0 for _ in range(len(partition2workers))]

    def get_to_worker(self, data_partition_idx: int) -> str:
        rpc_worker_list = self.partition2workers[data_partition_idx]
        worker_idx = self.rpc_worker_indexs[data_partition_idx]
        router_worker = rpc_worker_list[worker_idx]
        self.rpc_worker_indexs[data_partition_idx] = \
            (worker_idx + 1) % len(rpc_worker_list)
        return router_worker


@_require_initialized
def rpc_partition2workers(current_ctx: DistContext, num_data_partitions: int,
                          current_partition_idx: int):
    r""" all_gather to get the mapping between partition and workers """
    ctx = current_ctx
    partition2workers = [[] for _ in range(num_data_partitions)]
    gathered_results = global_all_gather(
        (ctx.role, num_data_partitions, current_partition_idx))
    for worker_name, (role, nparts, idx) in gathered_results.items():
        partition2workers[idx].append(worker_name)
    return partition2workers


class RpcCallBase(ABC):
    r""" A wrapper base for rpc call in remote processes."""
    def __init__(self):
        pass

    @abstractmethod
    def rpc_sync(self, *args, **kwargs):
        pass

    @abstractmethod
    def rpc_async(self, *args, **kwargs):
        pass


_rpc_call_lock = threading.RLock()
_rpc_call_id: int = 0
_rpc_call_pool: Dict[int, RpcCallBase] = {}


@_require_initialized
def rpc_register(call: RpcCallBase):
    r""" Register a call for rpc requests """
    global _rpc_call_id, _rpc_call_pool
    with _rpc_call_lock:
        call_id = _rpc_call_id
        _rpc_call_id += 1
        if call_id in _rpc_call_pool:
            raise RuntimeError(f"'rpc_register': Registered function twice.")
        _rpc_call_pool[call_id] = call

    return call_id


def _rpc_remote_async_call(call_id, *args, **kwargs):
    r""" Entry for rpc requests """
    return _rpc_call_pool.get(call_id).rpc_async(*args, **kwargs)


def _rpc_remote_sync_all(call_id, *args, **kwargs):
    r""" Entry for rpc requests """
    return _rpc_call_pool.get(call_id).rpc_sync(*args, **kwargs)


@_require_initialized
def rpc_request_async(worker_name, call_id, args=None, kwargs=None):
    r""" Perform a rpc request asynchronously and return a future. """
    return rpc.rpc_async(to=worker_name, func=_rpc_remote_async_call,
                         args=(call_id, *args), kwargs=kwargs)


@_require_initialized
def rpc_request_sync(worker_name, call_id, args=None, kwargs=None):
    r""" Perform a rpc request synchronously and return the results.
    """
    fut = rpc.rpc_async(to=worker_name, func=_rpc_remote_sync_call,
                        args=(call_id, *args), kwargs=kwargs)

    return fut.wait()
