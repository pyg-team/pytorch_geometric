from typing import List, Tuple, Union

import pylibwholegraph.torch as wgth
import torch
import torch.distributed as dist

from . import dist_shmem

_wm_global = False


def init_wholegraph():
    global _wm_global

    if _wm_global is True:
        return wgth.comm.get_global_communicator("nccl")
    dist_shmem.init_process_group_per_node()
    local_size = dist_shmem.get_local_size()
    local_rank = dist_shmem.get_local_rank()

    wgth.init(dist.get_rank(), dist.get_world_size(), local_rank,
              local_size=local_size, wm_log_level="info")
    print(
        f"[Rank {dist.get_rank()}] WholeGraph Initialization: "
        f"{dist.get_world_size()} GPUs are used with {local_size} GPUs per node."
    )
    global_comm = wgth.comm.get_global_communicator("nccl")
    _wm_global = True
    return global_comm


def finalize_wholegraph():
    global _wm_global
    if _wm_global is False:
        return
    wgth.finalize()
    _wm_global = False


def nvlink_network():
    r"""Check if the current hardware supports cross-node NVLink network.
    """
    if not _wm_global:
        init_wholegraph()

    global_comm = wgth.comm.get_global_communicator("nccl")
    local_size = dist_shmem.get_local_size()
    world_size = dist.get_world_size()

    # Intra-node communication
    if local_size == world_size:
        # use WholeGraph to check if the current hardware supports direct p2p
        return global_comm.support_type_location('continuous', 'cuda')

    # Check for multi-node support
    is_cuda_supported = global_comm.support_type_location('continuous', 'cuda')
    is_cpu_supported = global_comm.support_type_location('continuous', 'cpu')

    if is_cuda_supported and is_cpu_supported:
        return True

    return False


def copy_host_global_tensor_to_local(wm_tensor, host_tensor, wm_comm):
    local_tensor, local_start = wm_tensor.get_local_tensor(host_view=False)
    ## enable these checks when the wholegraph is updated to 24.10
    #local_ref_start = wm_tensor.get_local_entry_start()
    #local_ref_count = wm_tensor.get_local_entry_count()
    #assert local_start == local_ref_start
    #assert local_tensor.shape[0] == local_ref_count
    local_tensor.copy_(host_tensor[local_start:local_start +
                                   local_tensor.shape[0]])
    wm_comm.barrier()


def create_pyg_subgraph(WG_SampleOutput) -> Tuple:
    # PyG_SampleOutput (node, row, col, edge, batch...):
    # node (torch.Tensor): The sampled nodes in the original graph.
    # row (torch.Tensor): The source node indices of the sampled subgraph.
    #                     Indices must be within {0, ..., num_nodes - 1} where num_nodes is the number of nodes in sampled graph.
    # col (torch.Tensor): The destination node indices of the sampled subgraph. Indices must be within {0, ..., num_nodes - 1}
    # edge (torch.Tensor, optional): The sampled edges in the original graph. (for obtain edge features from the original graph)
    # batch (torch.Tensor, optional): The vector to identify the seed node for each sampled node in case of disjoint subgraph
    #                                  sampling per seed node. (None)
    # num_sampled_nodes (List[int], optional): The number of sampled nodes per hop.
    # num_sampled_edges (List[int], optional): The number of sampled edges per hop.
    sampled_nodes_list, edge_indice_list, csr_row_ptr_list, csr_col_ind_list = WG_SampleOutput
    num_sampled_nodes = []
    node = sampled_nodes_list[0]

    for hop in range(len(sampled_nodes_list) - 1):
        sampled_nodes = len(sampled_nodes_list[hop]) - len(
            sampled_nodes_list[hop + 1])
        num_sampled_nodes.append(sampled_nodes)
    num_sampled_nodes.append(len(sampled_nodes_list[-1]))
    num_sampled_nodes.reverse()

    layers = len(edge_indice_list)
    num_sampled_edges = [len(csr_col_ind_list[-1])]
    # Loop in reverse order, starting from the second last layer
    for layer in range(layers - 2, -1, -1):
        num_sampled_edges.append(
            len(csr_col_ind_list[layer] - len(csr_col_ind_list[layer + 1])))

    row = csr_col_ind_list[0]  # rows
    col = edge_indice_list[0][1]  # dst node

    edge = None
    batch = None
    return node, row, col, edge, batch, num_sampled_nodes, num_sampled_edges


def create_wg_dist_tensor(
        shape: list,
        dtype: torch.dtype,
        location: str = "cpu",
        partition_book: Union[List[int],
                              None] = None,  # default is even partition
        backend: str = "nccl",  # default is nccl; support nccl, vmm, nvshmem...
        **kwargs):
    """Create a WholeGraph-managed distributed tensor.

    Parameters
    ----------
    shape : list
        The shape of the tensor. It has to be a two-dimensional and one-dimensional tensor for now.
        The first dimension typically is the number of nodes.
        The second dimension is the feature/embedding dimension.
    dtype : torch.dtype
        The dtype of the tensor. The data type has to be the one in the deep learning framework.
    location : str, optional
        The desired location to store the embedding [ "cpu" | "cuda" ]
    partition_book : list, optional
        The partition book for the embedding tensor. The length of the partition book should be the same as the number of ranks.
    backend : str, optional
        The backend for the distributed tensor [ "nccl" | "vmm" | "nvshmem" ] (nvshmem not turned on in this example)
    """
    global_comm = init_wholegraph()

    if backend == "nccl":
        embedding_wholememory_type = "distributed"
    elif backend == "vmm":
        embedding_wholememory_type = "continuous"
    elif backend == "nvshmem":
        raise NotImplementedError("NVSHMEM backend has not turned on yet.")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    embedding_wholememory_location = location

    if "cache_policy" in kwargs:
        assert len(shape) == 2, "The shape of the embedding tensor must be 2D."
        cache_policy = kwargs['cache_policy']
        kwargs.pop('cache_policy')

        wm_embedding = wgth.create_embedding(
            global_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            dtype,
            shape,
            cache_policy=cache_policy,  # disable cache for now
            #embedding_entry_partition=partition_book,
            **kwargs
            #tensor_entry_partition=None  # important to do load balance
        )
    else:
        assert len(shape) == 2 or len(
            shape) == 1, "The shape of the tensor must be 2D or 1D."
        wm_embedding = wgth.create_wholememory_tensor(
            global_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            shape,
            dtype,
            strides=None,
            #tensor_entry_partition=partition_book  # important to do load balance
        )
    return wm_embedding


def create_wg_dist_tensor_from_files(
        file_list: List[str],
        shape: list,
        dtype: torch.dtype,
        location: str = "cpu",
        partition_book: Union[List[int],
                              None] = None,  # default is even partition
        backend: str = "nccl",  # default is nccl; support nccl, vmm, nvshmem...
        **kwargs):
    """Create a WholeGraph-managed distributed tensor from a list of files.

    Parameters
    ----------
    file_list : list
        The list of files to load the embedding tensor.
    shape : list
        The shape of the tensor. It has to be a two-dimensional and one-dimensional tensor for now.
        The first dimension typically is the number of nodes.
        The second dimension is the feature/embedding dimension.
    dtype : torch.dtype
        The dtype of the tensor. The data type has to be the one in the deep learning framework.
    location : str, optional
        The desired location to store the embedding [ "cpu" | "cuda" ]
    partition_book : list, optional
        The partition book for the embedding tensor. The length of the partition book should be the same as the number of ranks.
    backend : str, optional
        The backend for the distributed tensor [ "nccl" | "vmm" | "nvshmem" ] (nvshmem not turned on in this example)
    """
    global_comm = init_wholegraph()

    if backend == "nccl":
        embedding_wholememory_type = "distributed"
    elif backend == "vmm":
        embedding_wholememory_type = "continuous"
    elif backend == "nvshmem":
        raise NotImplementedError("NVSHMEM backend has not turned on yet.")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    embedding_wholememory_location = location

    if "cache_policy" in kwargs:
        assert len(shape) == 2, "The shape of the embedding tensor must be 2D."
        cache_policy = kwargs['cache_policy']
        kwargs.pop('cache_policy')

        wm_embedding = wgth.create_embedding_from_filelist(
            global_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            file_list,
            dtype,
            shape[1],
            cache_policy=cache_policy,  # disable cache for now
            #embedding_entry_partition=partition_book,
            **kwargs)
    else:
        assert len(shape) == 2 or len(
            shape) == 1, "The shape of the tensor must be 2D or 1D."
        last_dim_size = 0 if len(shape) == 1 else shape[1]
        wm_embedding = wgth.create_wholememory_tensor_from_filelist(
            global_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            file_list,
            dtype,
            last_dim_size,
            #tensor_entry_partition=partition_book  # important to do load balance
        )
    return wm_embedding
