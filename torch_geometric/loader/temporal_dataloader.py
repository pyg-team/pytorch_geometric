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
        dataset (TemporalData): The :obj:`temporalData` from which to load 
            the data.
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
    """

    def __iter__(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessTemporalDataLoaderIter(self)
        else:
            return super().__iter__()
