import torch
import collections
from torch.utils.data.dataloader import (DataLoader as DefaultDataLoader,
                                         default_collate)

from ..sparse.stack import stack


def collate(batch):
    if torch.is_tensor(batch[0]) and batch[0].is_sparse:
        return stack(batch)

    if torch.is_tensor(batch[0]):
        # Do a cat instead of a stack!
        return torch.cat(batch, dim=0)

    elif isinstance(batch[0], collections.Mapping):
        return {key: collate([d[key] for d in batch]) for key in batch[0]}

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    return default_collate(batch)


class DataLoader(DefaultDataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader, self).__init__(dataset, collate_fn=collate, **kwargs)
