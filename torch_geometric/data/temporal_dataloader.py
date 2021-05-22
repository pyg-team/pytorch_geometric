import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import TemporalData


class TemporalDataset(torch.utils.data.Dataset):
    """
    Converts a TemporalData object to a PyTorch Dataset
    """

    def __init__(self, temporal_data: TemporalData):
        self.temporal_data = temporal_data

    def __len__(self):
        return len(self.temporal_data.src)

    def __getitem__(self, idx):
        return self.temporal_data[idx]


class TemporalDataCollater(object):
    """
    Collates TemporalData objects for batching
    """

    def collate(self, temporal_data_list):
        batch = TemporalData()
        for key in temporal_data_list[0].keys:
            batch[key] = default_collate(
                [d[key] for d in temporal_data_list]).squeeze(1)
        return batch

    def __call__(self, batch):
        return self.collate(batch)


class TemporalDataLoader(torch.utils.data.DataLoader):
    """
    PyTorch DataLoader for TemporalData objects
    """

    def __init__(self, dataset: TemporalDataset, batch_size=1, shuffle=False,
                 **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super(TemporalDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=TemporalDataCollater(), **kwargs)
