from typing import Iterator, List, Optional

import torch

from torch_geometric.data import Dataset


class DynamicBatchSampler(torch.utils.data.sampler.Sampler[List[int]]):
    r"""Dynamically adds samples to mini-batch up to a maximum size (number of
    nodes). When data samples have a wide range in sizes, specifying a
    mini-batch size in terms of number of samples is not ideal and can cause
    CUDA OOM errors.

    Therefore, the number of steps per epoch is ambiguous, depending on the
    order of the samples. By default the ``__len__()`` will be None, fine for
    most cases but progress bars will be `infinite`. Alternatively,
    ``num_steps`` can be supplied to cap the number of mini-batches produced
    by the sampler.

    **Usage:**

    .. code-block:: python

        from torch_geometric.loader import DataLoader, DynamicBatchSampler

        batch_sampler = DynamicBatchSampler(10000, dataset)
        loader = DataLoader(dataset, batch_sampler=sampler, ...)

    Args:
        max_num (int): size of mini-batch to aim for in number of nodes or
            edges.
        data_source (Dataset): dataset to sample from, need this so that
            potential samples can be sized to create the dynamic batch.
        entity (str, optional): `nodes` or `edges` to measure batch size.
            (default: `nodes`)
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        skip_too_big (bool, optional): set to ``True`` to skip samples which
            can't fit in a batch by itself. (default: ``False``).
        num_steps (int, optional): The number of mini-batches to draw for a
            single epoch. If set to :obj:`None`, will iterate through all the
            underlying data, but __len__ will be :obj:`None` since it will be
            ambiguous. (default: :obj:`None`)
    """
    def __init__(self, max_num: int, data_source: Dataset,
                 entity: str = 'nodes', shuffle: bool = False,
                 skip_too_big: bool = False, num_steps: Optional[int] = None):
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError("`max_num` should be a positive integer value, "
                             "but got max_num={max_num}.")
        if entity not in ['nodes', 'edges']:
            raise ValueError("`entity` choice should be either "
                             f"`nodes` or `edges, not {entity}.")

        self.max_num = max_num
        self.data_source = data_source
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        if num_steps is None:
            num_steps = len(data_source)
        self.num_steps = num_steps

        self.entity = f"num_{entity}"

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        batch_n = 0
        num_steps = 0
        current_place = 0

        if self.shuffle:
            self.idxs = torch.randperm(len(self.data_source), dtype=torch.long)
        else:
            self.idxs = torch.arange(len(self.data_source), dtype=torch.long)

        # Main iteration loop
        while (current_place < len(self.data_source)
               and num_steps < self.num_steps):
            # Fill batch
            for idx in self.idxs[current_place:]:
                # Size of sample
                n = getattr(self.data_source[idx], self.entity)

                if batch_n + n > self.max_num:
                    if batch_n == 0:
                        if self.skip_too_big:
                            continue
                        else:
                            raise RuntimeError(
                                f"Size of a single data example ({n} @ index "
                                "{idx}) is larger than max_num"
                                "({self.max_num})")

                    # Mini-batch filled
                    break

                # Add sample to current batch
                batch.append(idx.item())
                current_place += 1
                batch_n += n

            yield batch
            batch = []
            batch_n = 0
            num_steps += 1

    def __len__(self) -> int:
        return self.num_steps
