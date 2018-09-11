import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda batch: Batch.from_data_list(batch),
            **kwargs)


class DenseDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        def dense_collate(data_list):
            batch = Batch()
            for key in data_list[0].keys:
                batch[key] = default_collate([d[key] for d in data_list])
            return batch

        super(DenseDataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=dense_collate, **kwargs)
