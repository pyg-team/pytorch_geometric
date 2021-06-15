import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import TemporalData


class TemporalDataset(torch.utils.data.Dataset):
    r"""Converts a :class:`torch_geometric.data.TemporalDataset`
    object to a PyTorch Dataset.

    Args:
        temporal_data (TemporalData): TemporalData object that contains all
            data in the dynamic graph dataset
    """

    def __init__(self, temporal_data: TemporalData):
        self.temporal_data = temporal_data

    def __len__(self):
        r"""Gets the number of samples in the dataset."""
        return len(self.temporal_data.src)

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx`."""
        return self.temporal_data[idx]


class TemporalDataCollater(object):
    r"""Collates :class:`torch_geometric.data.TemporalDataset` objects
    for batching."""

    def collate(self, temporal_data_list):
        batch = TemporalData()
        for key in temporal_data_list[0].keys:
            batch[key] = default_collate(
                [d[key] for d in temporal_data_list]).squeeze(1)
        return batch

    def __call__(self, batch):
        return self.collate(batch)


class TemporalDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.TemporalDataset` to a mini-batch.

    Args:
        dataset (TemporalDataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset: TemporalDataset, batch_size=1, shuffle=False,
                 **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super(TemporalDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=TemporalDataCollater(), **kwargs)
