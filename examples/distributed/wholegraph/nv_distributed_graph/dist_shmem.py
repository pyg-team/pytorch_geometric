"""Utilities for launching distributed GNN tasks."""
import os

import torch.distributed as dist
import torch.multiprocessing as mp

_LOCAL_PROCESS_GROUP = None
_LOCAL_ROOT_GLOBAL_RANK = None
_LOCAL_ROOT_AUTH_KEY = None
nprocs_per_node = 1


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
    assert dist.is_initialized(
    ), "torch.distributed is not initialized. Please call torch.distributed.init_process_group() first."

    nprocs_per_node = int(
        os.environ.get('LOCAL_WORLD_SIZE',
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


def _sync_auth_key(local_group, local_root):
    """Synchronize the authentication key across local process group or node.

    NOTE: In the context of MPI or torchrun, where all processes are launched concurrently and independently,
          synchronized authentication key allows local processes seamlessly exchange data during pickling process.
    """
    global _LOCAL_ROOT_AUTH_KEY
    assert _LOCAL_ROOT_AUTH_KEY is None
    if dist.get_rank() == local_root:
        authkey = [bytes(mp.current_process().authkey)]
        _LOCAL_ROOT_AUTH_KEY = authkey[0]
    else:
        authkey = [None]

    dist.broadcast_object_list(authkey, src=local_root, group=local_group)
    if authkey[0] != bytes(mp.current_process().authkey):
        mp.current_process().authkey = authkey[0]
        _LOCAL_ROOT_AUTH_KEY = authkey[0]
    assert _LOCAL_ROOT_AUTH_KEY == bytes(mp.current_process().authkey)


def _is_authkey_sync():
    global _LOCAL_ROOT_AUTH_KEY
    return _LOCAL_ROOT_AUTH_KEY == bytes(mp.current_process().authkey)


def to_shmem(dataset):
    """Move the dataset into shared memory.

    NOTE: This function performs dataset dumping/loading via a customizble pickler from the multiprocessing module.
          Frameworks (e.g., DGL and PyTorch) have the capability to customize the pickling process for their specific
          objects (e.g., DGLGraph or PyTorch Tensor), which involves moving the objects to shared memory at the local
          root (ForkingPickler.dumps), and then making them accessible to all local processes (ForkingPickler.loads).
    Parameters
    ----------
    dataset : Tuple or List of supported objects
        The objects can be DGLGraph and Pytorch Tensor, or any customized objects with the same mechanism
        of using shared memory during pickling process.

    Returns:
    -------
    dataset : Reconstructed dataset in shared memory
        Returned dataset preserves the same object hierarchy of the input.

    """
    local_root = get_local_root()
    local_group = get_local_process_group()
    if not _is_authkey_sync():  # if authkey not synced, sync the key
        _sync_auth_key(local_group, local_root)
    if dist.get_rank() == local_root:
        # each non-root process should have a dedicated pickle.dumps()
        handles = [None] + [
            bytes(mp.reductions.ForkingPickler.dumps(dataset))
            for _ in range(dist.get_world_size(group=local_group) - 1)
        ]
    else:
        handles = [None] * dist.get_world_size(group=local_group)
    dist.broadcast_object_list(handles, src=local_root, group=local_group)
    handle = handles[dist.get_rank(group=local_group)]
    if dist.get_rank() != local_root:
        # only non-root process performs pickle.loads()
        dataset = mp.reductions.ForkingPickler.loads(handle)
    dist.barrier(
        group=local_group
    )  # necessary to prevent unexpected close of any procs beyond this function
    return dataset
