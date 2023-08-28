from typing import List, Union

import torch
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Data, Dataset


def collate_fn(data_list: List[Data]) -> Batch:
    batch = Batch()
    for key in data_list[0].keys():
        batch[key] = default_collate([data[key] for data in data_list])
    return batch


class DenseDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which batches data objects from a
    :class:`torch_geometric.data.dataset` to a
    :class:`torch_geometric.data.Batch` object by stacking all attributes in a
    new dimension.

    .. note::

        To make use of this data loader, all graph attributes in the dataset
        need to have the same shape.
        In particular, this data loader should only be used when working with
        *dense* adjacency matrices.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`drop_last` or
            :obj:`num_workers`.
    """
    def __init__(self, dataset: Union[Dataset, List[Data]],
                 batch_size: int = 1, shuffle: bool = False, **kwargs):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, **kwargs)
