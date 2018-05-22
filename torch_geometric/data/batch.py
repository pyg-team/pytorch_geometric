import torch
from torch_geometric.data import Data


class Batch(Data):
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        keys = data_list[0].keys
        assert 'batch' not in keys

        batch = Batch()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum = 0
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in keys:
                item = data[key]
                item = item + cumsum if batch.cumsum(key, item) else item
                batch[key].append(item)
            cumsum += num_nodes

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        return item.dim() > 1 and item.dtype == torch.long

    @property
    def num_graphs(self):
        return self.batch[-1].item() + 1
