import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
from torch._six import container_abcs, string_classes, int_classes


class Collater(object):
    def __init__(self, follow_batch):
        self.follow_batch = follow_batch

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch), **kwargs)


class DataListLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a python list.

    .. note::

        This data loader should be used for multi-gpu support via
        :class:`torch_geometric.nn.DataParallel`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataListLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=lambda data_list: data_list, **kwargs)


class DenseCollater(object):
    def collate(self, data_list):
        batch = Batch()
        for key in data_list[0].keys:
            batch[key] = default_collate([d[key] for d in data_list])
        return batch

    def __call__(self, batch):
        return self.collate(batch)


class DenseDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    .. note::

        To make use of this data loader, all graphs in the dataset needs to
        have the same shape for each its attributes.
        Therefore, this data loader should only be used when working with
        *dense* adjacency matrices.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DenseDataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=DenseCollater(), **kwargs)
