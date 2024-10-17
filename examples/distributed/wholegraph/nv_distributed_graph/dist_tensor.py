import atexit
from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torch.distributed as dist

import pylibwholegraph
from .wholegraph import create_wg_dist_tensor, create_wg_dist_tensor_from_files, finalize_wholegraph, _wm_global
from .wholegraph import copy_host_global_tensor_to_local

class DistTensor:
    _instance_count = 0
    """
    WholeGraph-backed Distributed Tensor Interface for PyTorch.
    Parameters
    ----------
    src: Optional[Union[torch.Tensor, str, List[str]]]
        The source of the tensor. It can be a torch.Tensor on host, a file path, or a list of file paths.
        When the source is omitted, the tensor will be load later.
    shape : Optional[list, tuple]
        The shape of the tensor. It has to be a one- or two-dimensional tensor for now.
        When the shape is omitted, the `src` has to be specified and must be `pt` or `npy` file paths.
    dtype : Optional[torch.dtype]
        The dtype of the tensor. The data type has to be the one in the deep learning framework.
        Whne the dtype is omitted, the `src` has to be specified and must be `pt` or `npy` file paths.
    device : Optional[Literal["cpu", "cuda"]] = "cpu"
        The desired location to store the embedding [ "cpu" | "cuda" ]. Default is "cpu", i.e., pinned-host memory.
    partition_book : Union[List[int], None] = None
        1-D Range partition based on entry (dim-0). partition_book[i] determines the
        entry count of rank i and shoud be a positive integer; the sum of partition_book should equal to shape[0].
        Entries will be equally partitioned if None.
    backend : Optional[Literal["vmm", "nccl", "nvshmem", "chunked"]] = "nccl"
        The backend used for communication. Default is "nccl".
    """
    def __init__(
        self,
        src: Optional[Union[torch.Tensor, str, List[str]]] = None,
        shape: Optional[Union[list, tuple]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Optional[Union[List[int], None]] = None,  # location memtype ?? backend?? ; engine; comm =  vmm/nccl ..
        backend: Optional[str] = "nccl",
        *args,
        **kwargs,
    ):
        if DistTensor._instance_count == 0 and _wm_global is False:
            # Register the cleanup function for safty exit
            atexit.register(finalize_wholegraph)

        self._tensor = None # WholeMemory tensor for now. In future, we may support other types of distributed tensors.
        self._device = device
        if src is None:
            # Create an empty WholeGraph tensor
            assert shape is not None, "Please specify the shape of the tensor."
            assert dtype is not None, "Please specify the dtype of the tensor."
            assert len(shape) == 1 or len(shape) == 2, "The shape of the tensor must be 1D or 2D."
            self._tensor = create_wg_dist_tensor(list(shape), dtype, device, partition_book, backend, *args, **kwargs)
            self._dtype = dtype
        else:
            if isinstance(src, list):
                # A list of file paths for a tensor
                # Only support the binary file format directly loaded via WM API for now
                # TODO (@liuc): support merging multiple pt or npy files to create a tensor
                assert shape is not None and dtype is not None, "For now, read from multiple files are only supported in binary format."
                self._tensor = create_wg_dist_tensor_from_files(src, shape, dtype, device, partition_book, backend, *args, **kwargs)
                #self._tensor.from_filelist(src)
                self._dtype = dtype
            else:
                if isinstance(src, torch.Tensor):
                    self._tensor = create_wg_dist_tensor(list(src.shape), src.dtype, device, partition_book, backend, *args, **kwargs)
                    self._dtype = src.dtype
                    host_tensor = src
                elif isinstance(src, str) and src.endswith('.pt'):
                    host_tensor = torch.load(src, mmap=True)
                    self._tensor = create_wg_dist_tensor(list(host_tensor.shape), host_tensor.dtype, device, partition_book, backend, *args, **kwargs)
                    self._dtype = host_tensor.dtype
                elif isinstance(src, str) and src.endswith('.npy'):
                    host_tensor = torch.from_numpy(np.load(src, mmap_mode='c'))
                    self._dtype = host_tensor.dtype
                    self._tensor = create_wg_dist_tensor(list(host_tensor.shape), host_tensor.dtype, device, partition_book, backend, *args, **kwargs)
                else:
                    raise ValueError("Unsupported source type. Please provide a torch.Tensor, a file path, or a list of file paths.")

                self.load_from_global_tensor(host_tensor)
        DistTensor._instance_count += 1 # increase the instance count to track for resource cleanup

    def load_from_global_tensor(self, tensor):
        # input pytorch host tensor (mmapped or in shared host memory), and copy to wholegraph tensor
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        self._dtype = tensor.dtype
        if isinstance(self._tensor, pylibwholegraph.torch.WholeMemoryEmbedding):
            _tensor = self._tensor.get_embedding_tensor()
        else:
            _tensor = self._tensor
        copy_host_global_tensor_to_local(_tensor, tensor, _tensor.get_comm())

    def load_from_local_tensor(self, tensor):
        # input pytorch host tensor (mmapped or in shared host memory), and copy to wholegraph tensor
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        assert self._tensor.local_shape == tensor.shape, "The shape of the tensor does not match the shape of the local tensor."
        assert self._dtype == tensor.dtype, "The dtype of the tensor does not match the dtype of the local tensor."
        if isinstance(self._tensor, pylibwholegraph.torch.WholeMemoryEmbedding):
            self._tensor.get_embedding_tensor().get_local_tensor().copy_(tensor)
        else:
            self._tensor.get_local_tensor().copy_(tensor)


    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, device: Optional[Literal["cpu", "cuda"]] = "cpu", partition_book: Union[List[int], None] = None, backend: Optional[str] = 'nccl'):
        """
        Create a WholeGraph-backed Distributed Tensor from a PyTorch tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The PyTorch tensor to be copied to the WholeGraph tensor.
        device : str, optional
            The desired location to store the embedding [ "cpu" | "cuda" ]. Default is "cpu".
        backend : str, optional
            The backend used for communication. Default is "nccl".

        Returns
        -------
        DistTensor
            The WholeGraph-backed Distributed Tensor.
        """
        return cls(src=tensor, device=device, partition_book=partition_book, backend=backend)

    @classmethod
    def from_file(cls, file_path: str, device: Optional[Literal["cpu", "cuda"]] = "cpu", partition_book: Union[List[int], None] = None, backend: Optional[str] = 'nccl'):
        """
        Create a WholeGraph-backed Distributed Tensor from a file.

        Parameters
        ----------
        file_path : str
            The file path to the tensor. The file can be in the format of PyTorch tensor or NumPy array.
        device : str, optional
            The desired location to store the embedding [ "cpu" | "cuda" ]. Default is "cpu".
        backend : str, optional
            The backend used for communication. Default is "nccl".

        Returns
        -------
        DistTensor
            The WholeGraph-backed Distributed Tensor.
        """
        return cls(src=file_path, device=device, partition_book=partition_book, backend=backend)


    def __setitem__(self, idx: torch.Tensor, val: torch.Tensor):
        """
        Set the embeddings for the specified node indices.
        This call must be called by all processes.

        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        val : torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        idx = idx.cuda()
        val = val.cuda()

        if val.dtype != self.dtype:
            val = val.to(self.dtype)
        self._tensor.scatter(val, idx)

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings for the specified node indices (remotely).
        This call must be called by all processes.

        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        Returns
        -------
        torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        idx = idx.cuda()
        output_tensor = self._tensor.gather(idx)  # output_tensor is on cuda by default
        return output_tensor

    def get_local_tensor(self, host_view=False):
        """
        Get the local embedding tensor and its element offset at current rank.

        Returns
        -------
        (torch.Tensor, int)
            Tuple of local torch Tensor (converted from DLPack) and its offset.
        """
        local_tensor, offset = self._tensor.get_local_tensor(host_view = host_view)
        return local_tensor

    def get_local_offset(self):
        """
        Get the local embedding tensor and its element offset at current rank.

        Returns
        -------
        (torch.Tensor, int)
            Tuple of local torch Tensor (converted from DLPack) and its offset.
        """
        _, offset = self._tensor.get_local_tensor()
        return offset

    def get_comm(self):
        """
        Get the communicator of the WholeGraph embedding.

        Returns
        -------
        WholeMemoryCommunicator
            The WholeGraph global communicator of the WholeGraph embedding.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        return self._tensor.get_comm()

    @property
    def dim(self):
        return self._tensor.dim()

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        if self._tensor is None:
            return f"<DistTensor: No tensor loaded>"

        # Format the output similar to PyTorch
        tensor_repr = f"DistTensor("
        tensor_repr += f"shape={self._tensor.shape}, dtype={self._dtype}, device='{self._device}')"
        return tensor_repr

    def __del__(self):
        # Decrease instance count when an instance is deleted
        DistTensor._instance_count -= 1
        if DistTensor._instance_count == 0:
            finalize_wholegraph()

class DistEmbedding(DistTensor):
    """
    WholeGraph-backed Distributed Embedding Interface for PyTorch.
    Parameters
    ----------
    src: Optional[Union[torch.Tensor, str, List[str]]]
        The source of the tensor. It can be a torch.Tensor on host, a file path, or a list of file paths.
        When the source is omitted, the tensor will be load later.
    shape : Optional[list, tuple]
        The shape of the tensor. It has to be a one- or two-dimensional tensor for now.
        When the shape is omitted, the `src` has to be specified and must be `pt` or `npy` file paths.
    dtype : Optional[torch.dtype]
        The dtype of the tensor. The data type has to be the one in the deep learning framework.
        Whne the dtype is omitted, the `src` has to be specified and must be `pt` or `npy` file paths.
    device : Optional[Literal["cpu", "cuda"]] = "cpu"
        The desired location to store the embedding [ "cpu" | "cuda" ]. Default is "cpu", i.e., pinned-host memory.
    partition_book : Union[List[int], None] = None
        1-D Range partition based on entry (dim-0). partition_book[i] determines the
        entry count of rank i and shoud be a positive integer; the sum of partition_book should equal to shape[0].
        Entries will be equally partitioned if None.
    backend : Optional[Literal["vmm", "nccl", "nvshmem", "chunked"]] = "nccl"
        The backend used for communication. Default is "nccl".
    cache_policy : Optional[WholeMemoryCachePolicy] = None
        The cache policy for the tensor if it is an embedding. Default is None.
    gather_sms : Optional[int] = -1
        Whether to gather the embeddings on all GPUs. Default is False.
    round_robin_size: int = 0
        continuous embedding size of a rank using round robin shard strategy
    name : Optional[str]
        The name of the tensor.
    """
    def __init__(
        self,
        src: Optional[Union[torch.Tensor, str, List[str]]] = None,
        shape: Optional[Union[list, tuple]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Union[List[int], None] = None,
        backend: Optional[str] = "nccl",
        cache_policy = None, #Optional[pylibwholegraph.WholeMemoryCachePolicy] = None,
        gather_sms: Optional[int] = -1,
        round_robin_size: int = 0,
        name: Optional[str] = None,
    ):
        self._name = name

        super().__init__(src, shape, dtype, device, partition_book, backend, cache_policy=cache_policy, gather_sms=gather_sms, round_robin_size=round_robin_size)
        self._embedding =  self._tensor # returned _tensor is a WmEmbedding object
        self._tensor = self._embedding.get_embedding_tensor()

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        device: Literal["cpu", "cuda"] = "cpu",
        partition_book: Union[List[int], None] = None,
        name: Optional[str] = None,
        cache_policy = None,
        *args,
        **kwargs
    ):
        """
        Create a WholeGraph-backed Distributed Embedding (hooked with PyT's grad tracing) from a PyTorch tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The PyTorch tensor to be copied to the WholeGraph tensor.
        device : str, optional
            The desired location to store the embedding [ "cpu" | "cuda" ]. Default is "cpu".
        name : str, optional
            The name of the tensor.

        Returns
        -------
        DistEmbedding
            The WholeGraph-backed Distributed Tensor.
        """
        return cls(tensor, device, partition_book, name, cache_policy, *args, **kwargs)

    @classmethod
    def from_file(
        cls,
        file_path: str,
        device: Literal["cpu", "cuda"] = "cpu",
        partition_book: Union[List[int], None] = None,
        name: Optional[str] = None,
        cache_policy = None,
        *args,
        **kwargs
    ):
        """
        Create a WholeGraph-backed Distributed Tensor from a file.

        Parameters
        ----------
        file_path : str
            The file path to the tensor. The file can be in the format of PyTorch tensor or NumPy array.
        device : str, optional
            The desired location to store the embedding [ "cpu" | "cuda" ]. Default is "cpu".
        name : str, optional
            The name of the tensor.

        Returns
        -------
        DistTensor
            The WholeGraph-backed Distributed Tensor.
        """
        return cls(file_path, device, partition_book, name, cache_policy, *args, **kwargs)


    def __setitem__(self, idx: torch.Tensor, val: torch.Tensor):
        """
        Set the embeddings for the specified node indices.
        This call must be called by all processes.

        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        val : torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        idx = idx.cuda()
        val = val.cuda()

        if val.dtype != self.dtype:
            val = val.to(self.dtype)
        self._embedding.get_embedding_tensor().scatter(val, idx)

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings for the specified node indices (remotely).
        This call must be called by all processes.

        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        Returns
        -------
        torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        idx = idx.cuda()
        output_tensor = self._embedding.gather(idx)  # output_tensor is on cuda by default
        return output_tensor

    @property
    def name(self):
        return self._name

    def __repr__(self):
        if self._embedding is None:
            return f"<DistEmbedding: No embedding loaded, Name: {self._name}>"

        # Format the output similar to PyTorch
        tensor_repr = f"DistEmbedding("
        if self._name:
            tensor_repr += f"name={self._name}, "
        tensor_repr += f"shape={self.shape}, dtype={self.dtype}, device='{self.device}')"
        return tensor_repr

    def __del__(self):
        super().__del__()