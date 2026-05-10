import math
import statistics
from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame


class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


def _mps_freeze_at(
    dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
    batch_size: int,
) -> Optional[int]:
    """Return the iteration index at which to freeze the MPS graph cache.

    PyG's ``Batch.from_data_list`` concatenates graphs into a single tensor
    whose shape is ``(sum_nodes, sum_edges)``.  With a shuffled DataLoader the
    joint shape space grows without bound, so every batch triggers a new
    MPSGraph compilation that is cached indefinitely.

    This helper computes the effective 2-D shape space from the dataset
    statistics and derives the optimal freeze point:

    .. code-block:: text

        eff_2d = (
            sqrt(BS) * std(node_counts) * sqrt(BS) * std(edge_counts) * 2π
        )
        freeze_at = max(5, int(sqrt(eff_2d)))

    Returns ``None`` when MPS is unavailable, ``freeze_graph_cache`` is not
    present (requires PyTorch ≥ 2.13), or the dataset does not expose graph
    size statistics.

    Reference: pytorch/pytorch#182648
    """
    if not (torch.backends.mps.is_available()
            and hasattr(torch.mps, 'freeze_graph_cache')):
        return None

    try:
        # Sample at most 2000 graphs to keep __init__ fast for large datasets.
        n = len(dataset)  # type: ignore[arg-type]
        if n < 2:
            return None
        step = max(1, n // 2000)
        node_counts = []
        edge_counts = []
        for i in range(0, n, step):
            d = dataset[i]
            if hasattr(d, 'num_nodes') and d.num_nodes is not None:
                node_counts.append(int(d.num_nodes))
            if hasattr(d, 'num_edges') and d.num_edges is not None:
                edge_counts.append(int(d.num_edges))
    except Exception:
        return None

    if len(node_counts) < 2 or len(edge_counts) < 2:
        return None

    std_n = statistics.stdev(node_counts)
    std_e = statistics.stdev(edge_counts)
    if std_n == 0 or std_e == 0:
        return None

    eff_2d = (math.sqrt(batch_size) * std_n * math.sqrt(batch_size) * std_e *
              2 * math.pi)
    return max(5, int(math.sqrt(eff_2d)))


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.

    .. note::

        On Apple Silicon (MPS backend), :class:`DataLoader` automatically
        calls :func:`torch.mps.freeze_graph_cache` after a short warmup phase.
        MPSGraph compiles a new computation graph for every unique
        ``(sum_nodes, sum_edges)`` shape; with shuffled batches the cache
        grows without bound and causes unbounded RSS growth.  Freezing after
        warmup makes the cache read-only: hits are served from compiled graphs,
        rare shapes compile on-the-fly and are discarded.  This requires
        PyTorch ≥ 2.13 (pytorch/pytorch#182648); on earlier versions the
        behaviour is unchanged.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )

        self._mps_freeze_at = _mps_freeze_at(dataset, batch_size)
        self._mps_frozen = False

    def __iter__(self):
        for i, batch in enumerate(super().__iter__()):
            if (not self._mps_frozen and self._mps_freeze_at is not None
                    and i >= self._mps_freeze_at):
                torch.mps.freeze_graph_cache()
                self._mps_frozen = True
            yield batch
