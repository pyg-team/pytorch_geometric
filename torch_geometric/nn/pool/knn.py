import warnings
from typing import NamedTuple, Optional

import torch
from torch import Tensor

from torch_geometric.utils import cumsum, degree, to_dense_batch


class KNNOutput(NamedTuple):
    score: Tensor
    index: Tensor


class KNNIndex:
    r"""A base class to perform fast :math:`k`-nearest neighbor search
    (:math:`k`-NN) via the :obj:`faiss` library.

    Please ensure that :obj:`faiss` is installed by running

    .. code-block:: bash

        pip install faiss-cpu
        # or
        pip install faiss-gpu

    depending on whether to plan to use GPU-processing for :math:`k`-NN search.

    Args:
        index_factory (str, optional): The name of the index factory to use,
            *e.g.*, :obj:`"IndexFlatL2"` or :obj:`"IndexFlatIP"`. See `here
            <https://github.com/facebookresearch/faiss/wiki/
            The-index-factory>`_ for more information.
        emb (torch.Tensor, optional): The data points to add.
            (default: :obj:`None`)
        reserve (int, optional): The number of elements to reserve memory for
            before re-allocating (GPU-only). (default: :obj:`None`)
    """
    def __init__(
        self,
        index_factory: Optional[str] = None,
        emb: Optional[Tensor] = None,
        reserve: Optional[int] = None,
    ):
        warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

        import faiss

        self.index_factory = index_factory
        self.index: Optional[faiss.Index] = None
        self.reserve = reserve

        if emb is not None:
            self.add(emb)

    @property
    def numel(self) -> int:
        r"""The number of data points to search in."""
        if self.index is None:
            return 0
        return self.index.ntotal

    def _create_index(self, channels: int):
        import faiss
        return faiss.index_factory(channels, self.index_factory)

    def add(self, emb: Tensor):
        r"""Adds new data points to the :class:`KNNIndex` to search in.

        Args:
            emb (torch.Tensor): The data points to add.
        """
        import faiss
        import faiss.contrib.torch_utils

        if emb.dim() != 2:
            raise ValueError(f"'emb' needs to be two-dimensional "
                             f"(got {emb.dim()} dimensions)")

        if self.index is None:
            self.index = self._create_index(emb.size(1))

            if emb.device != torch.device('cpu'):
                self.index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(),
                    emb.device.index,
                    self.index,
                )

                if self.reserve is not None:
                    if hasattr(self.index, 'reserveMemory'):
                        self.index.reserveMemory(self.reserve)
                    else:
                        warnings.warn(f"'{self.index.__class__.__name__}' "
                                      f"does not support pre-allocation of "
                                      f"memory")

            self.index.train(emb)

        self.index.add(emb.detach())

    def search(
        self,
        emb: Tensor,
        k: int,
        exclude_links: Optional[Tensor] = None,
    ) -> KNNOutput:
        r"""Search for the :math:`k` nearest neighbors of the given data
        points. Returns the distance/similarity score of the nearest neighbors
        and their indices.

        Args:
            emb (torch.Tensor): The data points to add.
            k (int): The number of nearest neighbors to return.
            exclude_links (torch.Tensor): The links to exclude from searching.
                Needs to be a COO tensor of shape :obj:`[2, num_links]`, where
                :obj:`exclude_links[0]` refers to indices in :obj:`emb`, and
                :obj:`exclude_links[1]` refers to the data points in the
                :class:`KNNIndex`. (default: :obj:`None`)
        """
        if self.index is None:
            raise RuntimeError(f"'{self.__class__.__name__}' is not yet "
                               "initialized. Please call `add(...)` first.")

        if emb.dim() != 2:
            raise ValueError(f"'emb' needs to be two-dimensional "
                             f"(got {emb.dim()} dimensions)")

        query_k = k

        if exclude_links is not None:
            deg = degree(exclude_links[0], num_nodes=emb.size(0)).max()
            query_k = k + int(deg.max() if deg.numel() > 0 else 0)

        query_k = min(query_k, self.numel)

        if k > 2048:  # `faiss` supports up-to `k=2048`:
            warnings.warn(f"Capping 'k' to faiss' upper limit of 2048 "
                          f"(got {k}). This may cause some relevant items to "
                          f"not be retrieved.")
        elif query_k > 2048:
            warnings.warn(f"Capping 'k' to faiss' upper limit of 2048 "
                          f"(got {k} which got extended to {query_k} due to "
                          f"the exclusion of existing links). This may cause "
                          f"some relevant items to not be retrieved.")
            query_k = 2048

        score, index = self.index.search(emb.detach(), query_k)

        if exclude_links is not None:
            # Drop indices to exclude by converting to flat vector:
            flat_exclude = self.numel * exclude_links[0] + exclude_links[1]

            offset = torch.arange(
                start=0,
                end=self.numel * index.size(0),
                step=self.numel,
                device=index.device,
            ).view(-1, 1)
            flat_index = (index + offset).view(-1)

            notin = torch.isin(flat_index, flat_exclude).logical_not_()

            score = score.view(-1)[notin]
            index = index.view(-1)[notin]

            # Only maintain top-k scores:
            count = notin.view(-1, query_k).sum(dim=1)
            cum_count = cumsum(count)

            batch = torch.arange(count.numel(), device=count.device)
            batch = batch.repeat_interleave(count, output_size=cum_count[-1])

            batch_arange = torch.arange(count.sum(), device=count.device)
            batch_arange = batch_arange - cum_count[batch]

            mask = batch_arange < k
            score = score[mask]
            index = index[mask]

            if count.min() < k:  # Fill with dummy scores:
                batch = batch[mask]
                score, _ = to_dense_batch(
                    score,
                    batch,
                    fill_value=float('-inf'),
                    max_num_nodes=k,
                    batch_size=emb.size(0),
                )
                index, _ = to_dense_batch(
                    index,
                    batch,
                    fill_value=-1,
                    max_num_nodes=k,
                    batch_size=emb.size(0),
                )

            score = score.view(-1, k)
            index = index.view(-1, k)

        return KNNOutput(score, index)

    def get_emb(self) -> Tensor:
        r"""Returns the data points stored in the :class:`KNNIndex`."""
        if self.index is None:
            raise RuntimeError(f"'{self.__class__.__name__}' is not yet "
                               "initialized. Please call `add(...)` first.")

        return self.index.reconstruct_n(0, self.numel)


class L2KNNIndex(KNNIndex):
    r"""Performs fast :math:`k`-nearest neighbor search (:math:`k`-NN) based on
    the :math:`L_2` metric via the :obj:`faiss` library.

    Args:
        emb (torch.Tensor, optional): The data points to add.
            (default: :obj:`None`)
    """
    def __init__(self, emb: Optional[Tensor] = None):
        super().__init__(index_factory=None, emb=emb)

    def _create_index(self, channels: int):
        import faiss
        return faiss.IndexFlatL2(channels)


class MIPSKNNIndex(KNNIndex):
    r"""Performs fast :math:`k`-nearest neighbor search (:math:`k`-NN) based on
    the maximum inner product via the :obj:`faiss` library.

    Args:
        emb (torch.Tensor, optional): The data points to add.
            (default: :obj:`None`)
    """
    def __init__(self, emb: Optional[Tensor] = None):
        super().__init__(index_factory=None, emb=emb)

    def _create_index(self, channels: int):
        import faiss
        return faiss.IndexFlatIP(channels)


class ApproxL2KNNIndex(KNNIndex):
    r"""Performs fast approximate :math:`k`-nearest neighbor search
    (:math:`k`-NN) based on the the :math:`L_2` metric via the :obj:`faiss`
    library.
    Hyperparameters needs to be tuned for speed-accuracy trade-off.

    Args:
        num_cells (int): The number of cells.
        num_cells_to_visit (int): The number of cells that are visited to
            perform to search.
        bits_per_vector (int): The number of bits per sub-vector.
        emb (torch.Tensor, optional): The data points to add.
            (default: :obj:`None`)
        reserve (int, optional): The number of elements to reserve memory for
            before re-allocating (GPU only). (default: :obj:`None`)
    """
    def __init__(
        self,
        num_cells: int,
        num_cells_to_visit: int,
        bits_per_vector: int,
        emb: Optional[Tensor] = None,
        reserve: Optional[int] = None,
    ):
        self.num_cells = num_cells
        self.num_cells_to_visit = num_cells_to_visit
        self.bits_per_vector = bits_per_vector
        super().__init__(index_factory=None, emb=emb, reserve=reserve)

    def _create_index(self, channels: int):
        import faiss
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(channels),
            channels,
            self.num_cells,
            self.bits_per_vector,
            8,
            faiss.METRIC_L2,
        )
        index.nprobe = self.num_cells_to_visit
        return index


class ApproxMIPSKNNIndex(KNNIndex):
    r"""Performs fast approximate :math:`k`-nearest neighbor search
    (:math:`k`-NN) based on the maximum inner product via the :obj:`faiss`
    library.
    Hyperparameters needs to be tuned for speed-accuracy trade-off.

    Args:
        num_cells (int): The number of cells.
        num_cells_to_visit (int): The number of cells that are visited to
            perform to search.
        bits_per_vector (int): The number of bits per sub-vector.
        emb (torch.Tensor, optional): The data points to add.
            (default: :obj:`None`)
        reserve (int, optional): The number of elements to reserve memory for
            before re-allocating (GPU only). (default: :obj:`None`)
    """
    def __init__(
        self,
        num_cells: int,
        num_cells_to_visit: int,
        bits_per_vector: int,
        emb: Optional[Tensor] = None,
        reserve: Optional[int] = None,
    ):
        self.num_cells = num_cells
        self.num_cells_to_visit = num_cells_to_visit
        self.bits_per_vector = bits_per_vector
        super().__init__(index_factory=None, emb=emb, reserve=reserve)

    def _create_index(self, channels: int):
        import faiss
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatIP(channels),
            channels,
            self.num_cells,
            self.bits_per_vector,
            8,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.nprobe = self.num_cells_to_visit
        return index
