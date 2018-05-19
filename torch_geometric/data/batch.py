import torch

from .data import Data


class Batch(Data):
    def __init__(self, data_list):
        assert data_list[0].batch is None
        keys = data_list[0].keys

        for key in keys:
            self[key] = []
        self.batch = []

        cumsum = 0
        for i, d in enumerate(data_list):
            self.batch.append(torch.full((d.num_nodes, ), i, dtype=torch.long))
            for key in keys:
                self[key].append(d[key] + cumsum if d.cumsum(key) else d[key])
            cumsum += d.num_nodes

        for key, item in self:
            self[key] = torch.cat(item, dim=data_list[0].cat_dim(key))
        self.contiguous()

    def cumsum(self, key):
        return self[key].dim() > 1 and self[key].dtype == torch.long

    @property
    def num_graphs(self):
        return self.batch[-1] + 1
