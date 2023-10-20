from typing import List, Union

import torch

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData


def collate_fn(data_list):
    return data_list


class DataListLoader(torch.utils.data.DataLoader):
    r"""A data loader which batches data objects from a
    :class:`torch_geometric.data.dataset` to a :python:`Python` list.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    .. note::

        This data loader should be used for multi-GPU support via
        :class:`torch_geometric.nn.DataParallel`.

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
    def __init__(self, dataset: Union[Dataset, List[BaseData]],
                 batch_size: int = 1, shuffle: bool = False, **kwargs):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, **kwargs)
