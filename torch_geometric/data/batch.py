import torch
from torch_geometric.data import Data


class Batch(Data):
    def __init__(self, data_list):
        keys = data_list[0].keys
        assert 'batch' not in keys

        for key in keys:
            self[key] = []
        self.batch = []

        cumsum = 0
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            self.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in keys:
                item = data[key]
                item = item + cumsum if self.cumsum(key, item) else item
                self[key].append(item)
            cumsum += num_nodes

        for key in keys:
            self[key] = torch.cat(self[key], dim=data_list[0].cat_dim(key))
        self.batch = torch.cat(self.batch, dim=-1)
        self.contiguous()

    def cumsum(self, key, item):
        return item.dim() > 1 and item.dtype == torch.long

    @property
    def num_graphs(self):
        return self.batch[-1] + 1
