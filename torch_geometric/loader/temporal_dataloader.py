from torch.utils.data import _utils
from torch.utils.data.dataloader import _BaseDataLoaderIter

from torch_geometric.loader import DataLoader


class _SingleProcessTemporalDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessTemporalDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset[index]
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data


class TemporalDataLoader(DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.TemporalData` to a mini-batch.

    Args:
        data (TemporalData): The :obj:`~torch_geometric.data.TemporalData`
            from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(self, data, batch_size, **kwargs):
        super().__init__(data, batch_size=batch_size, shuffle=False, **kwargs)

    def __iter__(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessTemporalDataLoaderIter(self)
        else:
            return super().__iter__()
