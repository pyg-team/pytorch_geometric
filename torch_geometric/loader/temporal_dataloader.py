from typing import List, Union

import torch

from torch_geometric.data import TemporalData
from torch_geometric.nn.models.tgn import LastNeighborLoader


class TemporalDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges succesive events of a
    :class:`torch_geometric.data.TemporalData` to a mini-batch.

    Args:
        data (TemporalData): The :obj:`~torch_geometric.data.TemporalData`
            from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(self, data: TemporalData, batch_size: int = 1,
                 negative_sampling=False, negative_sampler=None, **kwargs):
        # Remove for PyTorch Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        kwargs.pop('shuffle', None)

        self.data = data
        self.events_per_batch = batch_size
        self.negative_sampling = negative_sampling
        if self.negative_sampling:
            if neighbor_sampler is None:
                # Ensure to only sample actual destination nodes as negatives.
                self.min_dst_idx, self.max_dst_idx = int(self.data.dst.min()), int(
                    self.data.dst.max())
                self.negative_sampler = None
            else:
                self.negative_sampler = negative_sampler

        if kwargs.get('drop_last', False) and len(data) % batch_size != 0:
            arange = range(0, len(data) - batch_size, batch_size)
        else:
            arange = range(0, len(data), batch_size)

        super().__init__(arange, 1, shuffle=False, collate_fn=self, **kwargs)

    def __call__(self, arange: List[int]) -> TemporalData:
        batch = self.data[arange[0]:arange[0] + self.events_per_batch]
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        list_to_make_n_ids = [batch.src, pos_dst]
        if self.negative_sampling:
            # Sample negative destination nodes.
            if self.negative_sampler is None:
                batch.neg_dst = torch.randint(self.min_dst_idx,
                                              self.max_dst_idx + 1,
                                              (src.size(0), ), dtype=torch.long,
                                              device=src.device)
            else:
                batch.neg_dst = self.negative_sampler(batch)
            list_to_make_n_ids += [batch.neg_dst]
        batch.n_id = torch.cat(list_to_make_n_ids).unique()

        return batch
