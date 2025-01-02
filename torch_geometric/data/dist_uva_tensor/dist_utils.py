"""Utilities for launching distributed GNN tasks."""
import os

import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
_LOCAL_ROOT_GLOBAL_RANK = None
nprocs_per_node = 1

def _standalone_pytorch_launcher(rank, world_size):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12367'
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



def init_process_group_per_node():
    """Initialize the distributed process group for each node."""
    if _LOCAL_PROCESS_GROUP is None:
        create_process_group_per_node()
        return
    else:
        assert dist.get_process_group_ranks(
            _LOCAL_PROCESS_GROUP)[0] == _LOCAL_ROOT_GLOBAL_RANK
        return


def create_process_group_per_node():
    """Create local process groups for each distinct node."""
    global _LOCAL_PROCESS_GROUP, _LOCAL_ROOT_GLOBAL_RANK
    global nprocs_per_node
    assert _LOCAL_PROCESS_GROUP is None and _LOCAL_ROOT_GLOBAL_RANK is None
    if not dist.is_initialized():
        # Assume it's a standalone run with single-gpu
        _standalone_pytorch_launcher(rank=0, world_size=1)
        nprocs_per_node = dist.get_world_size()
    else:
        nprocs_per_node = int(os.environ.get('LOCAL_WORLD_SIZE',
                            os.environ.get('SLURM_NTASKS_PER_NODE')))

    world_size = dist.get_world_size()
    rank = dist.get_rank() if dist.is_initialized() else 0
    assert world_size % nprocs_per_node == 0

    num_nodes = world_size // nprocs_per_node
    node_id = rank // nprocs_per_node
    for i in range(num_nodes):
        node_ranks = list(range(i * nprocs_per_node,
                                (i + 1) * nprocs_per_node))
        pg = dist.new_group(node_ranks)
        if i == node_id:
            _LOCAL_PROCESS_GROUP = pg
    assert _LOCAL_PROCESS_GROUP is not None
    _LOCAL_ROOT_GLOBAL_RANK = dist.get_process_group_ranks(
        _LOCAL_PROCESS_GROUP)[0]


def get_local_process_group():
    """Return a torch distributed process group for a subset of all local processes in the same node."""
    assert _LOCAL_PROCESS_GROUP is not None
    return _LOCAL_PROCESS_GROUP


def get_local_root():
    """Return the global rank corresponding to the local root process."""
    assert _LOCAL_ROOT_GLOBAL_RANK is not None
    return _LOCAL_ROOT_GLOBAL_RANK


def get_local_rank():
    """Return the local rank of the current process."""
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(_LOCAL_PROCESS_GROUP)


def get_local_size():
    """Return the number of processes in the local process group."""
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_world_size(_LOCAL_PROCESS_GROUP)
