import atexit
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch.multiprocessing as mp

from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.dist_neighbor_sampler import close_sampler
from torch_geometric.distributed.rpc import global_barrier, init_rpc


class DistLoader:
    r"""A base class for creating distributed data loading routines.

    Args:
        current_ctx (DistContext): Distributed context info of the current
            process.
        rpc_worker_names (Dict[DistRole, List[str]]): RPC workers identifiers.
        master_addr (str, optional): RPC address for distributed loader
            communication.
            Refers to the IP address of the master node. (default: :obj:`None`)
        master_port (int or str, optional): The open port for RPC communication
            with the master node. (default: :obj:`None`)
        channel (mp.Queue, optional): A communication channel for messages.
            (default: :obj:`None`)
        num_rpc_threads (int, optional): The number of threads in the
            thread-pool used by
            :class:`~torch.distributed.rpc.TensorPipeAgent` to execute
            requests. (default: :obj:`16`)
        rpc_timeout (int, optional): The default timeout in seconds for RPC
            requests.
            If the RPC has not completed in this timeframe, an exception will
            be raised.
            Callers can override this timeout for
            individual RPCs in :meth:`~torch.distributed.rpc.rpc_sync` and
            :meth:`~torch.distributed.rpc.rpc_async` if necessary.
            (default: :obj:`180`)
    """
    def __init__(
        self,
        current_ctx: DistContext,
        rpc_worker_names: Dict[DistRole, List[str]],
        master_addr: Optional[str] = None,
        master_port: Optional[Union[int, str]] = None,
        channel: Optional[mp.Queue] = None,
        num_rpc_threads: int = 16,
        rpc_timeout: int = 180,
        **kwargs,
    ):
        if master_addr is None and os.environ.get('MASTER_ADDR') is not None:
            master_addr = os.environ['MASTER_ADDR']
        if master_addr is None:
            raise ValueError(f"Missing master address for RPC communication "
                             f"in '{self.__class__.__name__}'. Try to provide "
                             f"it or set it via the 'MASTER_ADDR' environment "
                             f"variable.")

        if master_port is None and os.environ.get('MASTER_PORT') is not None:
            master_port = int(os.environ['MASTER_PORT'])
        if master_port is None:
            raise ValueError(f"Missing master port for RPC communication in "
                             f"'{self.__class__.__name__}'. Try to provide it "
                             f"or set it via the 'MASTER_ADDR' environment "
                             f"variable.")

        assert num_rpc_threads > 0
        assert rpc_timeout > 0

        self.current_ctx = current_ctx
        self.rpc_worker_names = rpc_worker_names
        self.master_addr = master_addr
        self.master_port = master_port
        self.channel = channel or mp.Queue()
        self.pid = mp.current_process().pid
        self.num_rpc_threads = num_rpc_threads
        self.rpc_timeout = rpc_timeout
        self.num_workers = kwargs.get('num_workers', 0)

        logging.info(f"[{self}] MASTER_ADDR={master_addr}, "
                     f"MASTER_PORT={master_port}")

        if self.num_workers == 0:  # Initialize RPC in main process:
            self.worker_init_fn(0)

    def channel_get(self, out: Any) -> Any:
        if self.channel is not None:
            out = self.channel.get()
            logging.debug(f"[{self}] Retrieved message")
        return out

    def worker_init_fn(self, worker_id: int):
        try:
            num_sampler_proc = self.num_workers if self.num_workers > 0 else 1
            self.current_ctx_worker = DistContext(
                world_size=self.current_ctx.world_size * num_sampler_proc,
                rank=self.current_ctx.rank * num_sampler_proc + worker_id,
                global_world_size=self.current_ctx.world_size *
                num_sampler_proc,
                global_rank=self.current_ctx.rank * num_sampler_proc +
                worker_id,
                group_name='mp_sampling_worker',
            )

            init_rpc(
                current_ctx=self.current_ctx_worker,
                rpc_worker_names={},
                master_addr=self.master_addr,
                master_port=self.master_port,
                num_rpc_threads=self.num_rpc_threads,
                rpc_timeout=self.rpc_timeout,
            )
            assert hasattr(self, 'neighbor_sampler')
            self.neighbor_sampler.register_sampler_rpc()
            self.neighbor_sampler.init_event_loop()
            # close RPC & worker group at exit:
            atexit.register(close_sampler, worker_id, self.neighbor_sampler)
            global_barrier(timeout=10)  # Wait for all workers to initialize.

        except RuntimeError:
            raise RuntimeError(f"`{self}.init_fn()` could not initialize the "
                               f"worker loop of the neighbor sampler")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pid={self.pid})'
