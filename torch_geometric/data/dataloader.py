import torch.utils.data
from torch_geometric.data import Batch


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            collate_fn=lambda batch: Batch.from_data_list(batch),
            **kwargs)
