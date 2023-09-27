import atexit
import logging
import os
from typing import Dict, List, Optional, Union

import torch.multiprocessing as mp

from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.dist_neighbor_sampler import close_sampler
from torch_geometric.distributed.rpc import global_barrier, init_rpc
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput


class DistLoader:
    def __init__(
        self,
        current_ctx: DistContext,
        rpc_worker_names: Dict[DistRole, List[str]],
        master_addr: str,
        master_port: Union[int, str],
        channel: mp.Queue(),
        num_rpc_threads: Optional[int] = 16,
        rpc_timeout: Optional[int] = 180,
        **kwargs,
    ):
        """
        Args:
            current_ctx (DistContext): Distributed context info of the current process.
            rpc_worker_names (Dict[DistRole, List[str]]): RPC workers identifiers.
            master_addr (str): RPC address for distributed loaders communication,
                IP of the master node.
            master_port (Union[int, str]): Open port for RPC communication with
                the master node.
            channel (mp.Queue): A communication channel for sample messages that
                allows for asynchronous processing of the sampler calls.
                num_rpc_threads (Optional[int], optional): The number of threads in the
                thread-pool used by
                :class:`~torch.distributed.rpc.TensorPipeAgent` to execute
                requests (default: 16).
            rpc_timeout (Optional[int], optional): The default timeout,
                in seconds, for RPC requests (default: 60 seconds). If the RPC has not
                completed in this timeframe, an exception indicating so will
                be raised. Callers can override this timeout for individual
                RPCs in :meth:`~torch.distributed.rpc.rpc_sync` and
                :meth:`~torch.distributed.rpc.rpc_async` if necessary. (default: 180)

        Raises:
            ValueError: If RPC mater port or master address are not specified.
        """
        self.channel = channel
        self.current_ctx = current_ctx
        self.rpc_worker_names = rpc_worker_names
        self.pid = mp.current_process().pid
        self.num_workers = kwargs.get('num_workers', 0)

        if master_addr is not None:
            self.master_addr = str(master_addr)
        elif os.environ.get('MASTER_ADDR') is not None:
            self.master_addr = os.environ['MASTER_ADDR']
        else:
            raise ValueError(
                f"'{self}': missing master address "
                "for rpc communication, try to provide it or set it "
                "with environment variable 'MASTER_ADDR'")
        if master_port is not None:
            self.master_port = int(master_port)
        elif os.environ.get('MASTER_PORT') is not None:
            self.master_port = int(os.environ['MASTER_PORT'])  # + 1
        else:
            raise ValueError(
                f"'{self}': missing master port "
                "for rpc communication, try to provide it or set it "
                "with environment variable 'MASTER_ADDR'")
        logging.info(
            f"{self} MASTER_ADDR: {self.master_addr} MASTER_PORT: {self.master_port}"
        )

        self.num_rpc_threads = num_rpc_threads
        if self.num_rpc_threads is not None:
            assert self.num_rpc_threads > 0
        self.rpc_timeout = rpc_timeout
        if self.rpc_timeout is not None:
            assert self.rpc_timeout > 0

        # init rpc in main process
        if self.num_workers == 0:
            self.worker_init_fn(0)

    def channel_get(self, out) -> Union[SamplerOutput, HeteroSamplerOutput]:
        if self.channel:
            out = self.channel.get()
            logging.debug(
                f'{repr(self)} retrieved Sampler result from PyG MSG channel')
        return out

    def worker_init_fn(self, worker_id):
        try:
            num_sampler_proc = (self.num_workers
                                if self.num_workers > 0 else 1)
            self.current_ctx_worker = DistContext(
                world_size=self.current_ctx.world_size * num_sampler_proc,
                rank=self.current_ctx.rank * num_sampler_proc + worker_id,
                global_world_size=self.current_ctx.world_size *
                num_sampler_proc,
                global_rank=self.current_ctx.rank * num_sampler_proc +
                worker_id, group_name='mp_sampling_worker')

            init_rpc(current_ctx=self.current_ctx_worker, rpc_worker_names={},
                     master_addr=self.master_addr,
                     master_port=self.master_port,
                     num_rpc_threads=self.num_rpc_threads,
                     rpc_timeout=self.rpc_timeout)
            self.neighbor_sampler.register_sampler_rpc()
            self.neighbor_sampler.init_event_loop()
            # close rpc & worker group at exit
            atexit.register(close_sampler, worker_id, self.neighbor_sampler)
            # wait for all workers to init
            global_barrier(timeout=10)

        except RuntimeError:
            raise RuntimeError(
                f"init_fn() defined in {self} didn't initialize the worker_loop of {self.neighbor_sampler}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()-PID{self.pid}"
