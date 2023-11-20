import warnings
from typing import NamedTuple, Optional

import torch
from torch import Tensor


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
        index_factory (str): The name of the index factory to use, *e.g.*,
            :obj:`"IndexFlatL2"` or :obj:`"IndexFlatIP"`. See `here
            <https://github.com/facebookresearch/faiss/wiki/
            The-index-factory>`_ for more information.
        emb (torch.Tensor, optional): The data points to add.
            (default: :obj:`None`)
    """
    def __init__(self, index_factory: str, emb: Optional[Tensor] = None):
        warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')

        import faiss

        self.numel = 0
        self.index_factory = index_factory
        self.index: Optional[faiss.Index] = None

        if emb is not None:
            self.add(emb)

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

        self.numel += emb.size(0)
        self.index.add(emb.detach())

    def search(self, emb: Tensor, k: int) -> KNNOutput:
        r"""Search for the :math:`k` nearest neighbors of the given data
        points. Returns the distance/similarity score of the nearest neighbors
        and their indices.

        Args:
            emb (torch.Tensor): The data points to add.
            k (int): The number of nearest neighbors to return.
        """
        if self.index is None:
            raise RuntimeError(f"'{self.__class__.__name__}' is not yet "
                               "initialized. Please call `add(...)` first.")

        if emb.dim() != 2:
            raise ValueError(f"'emb' needs to be two-dimensional "
                             f"(got {emb.dim()} dimensions)")

        return KNNOutput(*self.index.search(emb.detach(), k))

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
