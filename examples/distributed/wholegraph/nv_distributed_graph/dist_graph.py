from typing import Any, List, Union, Literal, Optional
import numpy as np

import torch

from . import dist_shmem
from . import dist_tensor

class DistGraphCSC:
    """ Distributed Graph Store based on DistTensors for Compressed Sparse Column (CSC) format.
    Only support homogeneous graph for now.
    Parameters
    ----------
    node_tensor : torch.Tensor
        The node tensor.
    edge_tensor : torch.Tensor
    """
    def __init__(
        self,
        col_ptrs_src: Optional[Union[torch.Tensor, str, List[str]]] = None,
        row_indx_src : Optional[Union[torch.Tensor, str, List[str]]] = None,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        pinned_shared: Optional[bool] = False,
        partition_book: Optional[Union[List[int], None]] = None,  # location memtype ?? backend?? ; engine; comm =  vmm/nccl ..
        backend: Optional[str] = "nccl", # reserved this for future use
        *args,
        **kwargs,
    ):
    # optionally to save node/edge feature tensors (view)
        self.data = {} # place holder for the hetergenous graph
        self.device = device
        if partition_book is not None:
            raise NotImplementedError("Uneven partition of 1-D disttensor is not turned on yet.")

        if pinned_shared:
            dist_shmem.init_process_group_per_node()
            # load the original dataset in the first process and share it with others
            col_ptrs = None
            row_indx = None
            if dist_shmem.get_local_rank() == 0:
                if isinstance(col_ptrs_src, torch.Tensor) and isinstance(row_indx_src, torch.Tensor):
                    col_ptrs = col_ptrs_src
                    row_indx = row_indx_src
                elif col_ptrs_src.endswith('.pt') and row_indx_src.endswith('.pt'):
                    col_ptrs = torch.load(col_ptrs_src, mmap=True)
                    row_indx = torch.load(row_indx_src, mmap=True)
                elif col_ptrs_src.endswith('.npy') and row_indx_src.endswith('.npy'):
                    col_ptrs = torch.from_numpy(np.load(col_ptrs_src, mmap_mode='c'))
                    row_indx = torch.from_numpy(np.load(row_indx_src, mmap_mode='c'))
                else:
                    raise ValueError("Unsupported file format.")

            self.col_ptrs = dist_shmem.to_shmem(col_ptrs)
            self.row_indx = dist_shmem.to_shmem(row_indx)
        else:
            # 2-gather approach here only
            self.col_ptrs = dist_tensor.DistTensor(col_ptrs_src, device = device, backend = backend)
            self.row_indx = dist_tensor.DistTensor(row_indx_src, device = device, backend = backend)

    @property
    def num_nodes(self):
        return self.col_ptrs.shape[0] - 1

    @property
    def num_edges(self):
        return self.row_indx.shape[0]

    def __getitem__ (self, name: str) -> Any:
        return self.data[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.data[name] = value
        return

    def transform_nodes(self, nodes):
        """Transform all seed nodes from every rank to the local seed nodes

        Args:
            nodes (_type_): _description_
        """
        pass

    def transform_edges(self, edges):
        """Transform all seed edges from every rank to the local seed edges

        Args:
            edges (_type_): _description_
        """
        pass

    def transform_graph() : #back to graph
        pass
