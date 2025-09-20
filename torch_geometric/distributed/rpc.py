import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from torch.distributed import rpc

from torch_geometric.distributed.dist_context import DistContext, DistRole

try:
    from torch._C._distributed_rpc import _is_current_rpc_agent_set
except Exception:

    def _is_current_rpc_agent_set() -> bool:
        return False


_rpc_init_lock = threading.RLock()


def rpc_is_initialized() -> bool:
    return _is_current_rpc_agent_set()


def rpc_require_initialized(func: Callable) -> Callable:
    if hasattr(rpc, 'api'):
        return rpc.api._require_initialized(func)
    return func


@rpc_require_initialized
def global_all_gather(obj, timeout: Optional[int] = None) -> Any:
    r"""Gathers objects from all groups in a list."""
    if timeout is None:
        return rpc.api._all_gather(obj)
    return rpc.api._all_gather(obj, timeout=timeout)


@rpc_require_initialized
def global_barrier(timeout: Optional[int] = None) -> None:
    r"""Block until all local and remote RPC processes."""
    try:
        global_all_gather(obj=None, timeout=timeout)
    except RuntimeError:
        logging.error('Failed to respond to global barrier')


def init_rpc(
    current_ctx: DistContext,
    master_addr: str,
    master_port: int,
    num_rpc_threads: int = 16,
    rpc_timeout: float = 240.0,
    rpc_worker_names: Optional[Dict[DistRole, List[str]]] = None,
):
    with _rpc_init_lock:
        if rpc_is_initialized():
            return

        if current_ctx is None:
            raise RuntimeError("'dist_context' has not been set in 'init_rpc'")

        options = rpc.TensorPipeRpcBackendOptions(
            _transports=['ibv', 'uv'],
            _channels=['mpt_uv', 'basic'],
            num_worker_threads=num_rpc_threads,
            rpc_timeout=rpc_timeout,
            init_method=f'tcp://{master_addr}:{master_port}',
        )

        rpc.init_rpc(
            name=current_ctx.worker_name,
            rank=current_ctx.global_rank,
            world_size=current_ctx.global_world_size,
            rpc_backend_options=options,
        )

        global_barrier(timeout=rpc_timeout)


def shutdown_rpc(id: str = None, graceful: bool = True,
                 timeout: float = 240.0):
    with _rpc_init_lock:
        if rpc_is_initialized():
            logging.info(f"Shutdown RPC in {id}"
                         f"{' gracefully' if graceful else ''}")
            rpc.shutdown(graceful, timeout)
        else:
            logging.info(f'RPC in {id} not initialized.')


class RPCRouter:
    r"""A router to get the worker based on the partition ID."""
    def __init__(self, partition_to_workers: List[List[str]]):
        for rpc_worker_list in partition_to_workers:
            if len(rpc_worker_list) == 0:
                raise ValueError('No RPC worker is in worker list')
        self.partition_to_workers = partition_to_workers
        self.rpc_worker_indices = [0 for _ in range(len(partition_to_workers))]

    def get_to_worker(self, partition_idx: int) -> str:
        rpc_worker_list = self.partition_to_workers[partition_idx]
        worker_idx = self.rpc_worker_indices[partition_idx]
        router_worker = rpc_worker_list[worker_idx]
        self.rpc_worker_indices[partition_idx] = ((worker_idx + 1) %
                                                  len(rpc_worker_list))
        return router_worker


@rpc_require_initialized
def rpc_partition_to_workers(
    current_ctx: DistContext,
    num_partitions: int,
    current_partition_idx: int,
):
    r"""Performs an :obj:`all_gather` to get the mapping between partition and
    workers.
    """
    ctx = current_ctx
    partition_to_workers = [[] for _ in range(num_partitions)]
    gathered_results = global_all_gather(
        (ctx.role, num_partitions, current_partition_idx))
    for worker_name, (_, _, idx) in gathered_results.items():
        partition_to_workers[idx].append(worker_name)
    return partition_to_workers


class RPCCallBase(ABC):
    r"""A wrapper base class for RPC calls in remote processes."""
    @abstractmethod
    def rpc_sync(self, *args, **kwargs):
        pass

    @abstractmethod
    def rpc_async(self, *args, **kwargs):
        pass


_rpc_call_lock = threading.RLock()
_rpc_call_id: int = 0
_rpc_call_pool: Dict[int, RPCCallBase] = {}


@rpc_require_initialized
def rpc_register(call: RPCCallBase) -> int:
    r"""Registers a call for RPC requests."""
    global _rpc_call_id

    with _rpc_call_lock:
        call_id = _rpc_call_id
        _rpc_call_id += 1
        if call_id in _rpc_call_pool:
            raise RuntimeError("Registered function twice in 'rpc_register'")
        _rpc_call_pool[call_id] = call

    return call_id


def _rpc_async_call(call_id: int, *args, **kwargs):
    r"""Entry point for RPC requests."""
    return _rpc_call_pool.get(call_id).rpc_async(*args, **kwargs)


@rpc_require_initialized
def rpc_async(worker_name: str, call_id: int, args=None, kwargs=None):
    r"""Performs an asynchronous RPC request and returns a future."""
    return rpc.rpc_async(
        to=worker_name,
        func=_rpc_async_call,
        args=(call_id, *args),
        kwargs=kwargs,
    )


def _rpc_sync_call(call_id: int, *args, **kwargs):
    r"""Entry point for synchronous RPC requests."""
    return _rpc_call_pool.get(call_id).rpc_sync(*args, **kwargs)


@rpc_require_initialized
def rpc_sync(worker_name: str, call_id: int, args=None, kwargs=None):
    r"""Performs a synchronous RPC request and returns a future."""
    future = rpc.rpc_async(
        to=worker_name,
        func=_rpc_sync_call,
        args=(call_id, *args),
        kwargs=kwargs,
    )
    return future.wait()
