import atexit
import torch
import torch.multiprocessing as mp
import os
from typing import Optional, Dict, Union, List
from .rpc import init_rpc, global_barrier
from .dist_neighbor_sampler import DistNeighborSampler, close_sampler
from .dist_context import DistContext, DistRole


class DistLoader():

    def __init__(self,
                 neighbor_sampler: DistNeighborSampler,
                 current_ctx: DistContext,
                 rpc_worker_names: Dict[DistRole, List[str]],
                 master_addr: str,
                 master_port: Union[int, str],
                 # Optional[Union[ChannelBase, mp.Queue()]],
                 channel: mp.Queue(),
                 num_rpc_threads: Optional[int] = 16,
                 rpc_timeout: Optional[int] = 180,
                 device: Optional[torch.device] = torch.device('cpu'),
                 **kwargs,
                 ):

        self.neighbor_sampler = neighbor_sampler
        self.channel = channel

        self.current_ctx = current_ctx
        self.rpc_worker_names = rpc_worker_names
        self.pid = mp.current_process().pid
        self.device = device
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_workers = kwargs.get('num_workers', 0)

        if master_addr is not None:
            self.master_addr = str(master_addr)
        elif os.environ.get('MASTER_ADDR') is not None:
            self.master_addr = os.environ['MASTER_ADDR']
        else:
            raise ValueError(
                f"'{self.__class__.__name__}': missing master address "
                "for rpc communication, try to provide it or set it "
                "with environment variable 'MASTER_ADDR'")
        if master_port is not None:
            self.master_port = int(master_port)
        elif os.environ.get('MASTER_PORT') is not None:
            self.master_port = int(os.environ['MASTER_PORT'])  # + 1
        else:
            raise ValueError(
                f"'{self.__class__.__name__}': missing master port "
                "for rpc communication, try to provide it or set it "
                "with environment variable 'MASTER_ADDR'")
        print(
            f"{repr(self)} MASTER_ADDR: {self.master_addr} MASTER_PORT: {self.master_port}")

        self.num_rpc_threads = num_rpc_threads
        if self.num_rpc_threads is not None:
            assert self.num_rpc_threads > 0
        self.rpc_timeout = rpc_timeout
        if self.rpc_timeout is not None:
            assert self.rpc_timeout > 0

        if self.num_workers == 0:
            self.worker_init_fn(0)

    def worker_init_fn(self, worker_id):
        try:
            num_sampler_proc = (
                self.num_workers if self.num_workers > 0 else 1)
            self.current_ctx_worker = DistContext(
                world_size=self.current_ctx.world_size * num_sampler_proc,
                rank=self.current_ctx.rank * num_sampler_proc + worker_id,
                global_world_size=self.current_ctx.world_size *
                num_sampler_proc, global_rank=self.current_ctx.rank *
                num_sampler_proc + worker_id, group_name='mp_sampling_worker')

            self.sampler_rpc_worker_names = {}
            init_rpc(
                current_ctx=self.current_ctx_worker,
                rpc_worker_names=self.sampler_rpc_worker_names,
                master_addr=self.master_addr,
                master_port=self.master_port,
                num_rpc_threads=self.num_rpc_threads,
                rpc_timeout=self.rpc_timeout
            )
            self.neighbor_sampler.register_sampler_rpc()
            self.neighbor_sampler.init_event_loop()
            # close rpc & worker group at exit
            atexit.register(close_sampler, worker_id, self.neighbor_sampler)
            # wait for all workers to init
            global_barrier()

        except RuntimeError:
            raise RuntimeError(
                f"init_fn() defined in {repr(self)} didn't initialize the worker_loop of {repr(self.neighbor_sampler)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()-PID{self.pid}@{self.device}"
