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
                 negative_sampling=False, neighbor_loader_size=10, **kwargs):
        # Remove for PyTorch Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        kwargs.pop('shuffle', None)

        self.data = data
        self.events_per_batch = batch_size
        self.negative_sampling = negative_sampling
        if self.negative_sampling:
            # Ensure to only sample actual destination nodes as negatives.
            self.min_dst_idx, self.max_dst_idx = int(self.data.dst.min()), int(
                self.data.dst.max())
        self.neighbor_loader = LastNeighborLoader(data.num_nodes,
                                                  size=neighbor_loader_size)
        # Helper vector to map global node indices to local ones.
        self.assoc = torch.empty(data.num_nodes, dtype=torch.long)
        if kwargs.get('drop_last', False) and len(data) % batch_size != 0:
            arange = range(0, len(data) - batch_size, batch_size)
        else:
            arange = range(0, len(data), batch_size)

        super().__init__(arange, 1, shuffle=False, collate_fn=self, **kwargs)

    def _get_iterator(self):
        self.neighbor_loader.reset_state()  # Start with an empty graph.
        return super()._get_iterator()

    def __call__(self, arange: List[int]) -> TemporalData:
        batch = self.data[arange[0]:arange[0] + self.events_per_batch]
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        list_to_make_n_ids = [batch.src, pos_dst]
        if self.negative_sampling:
            # Sample negative destination nodes.
            batch.neg_dst = torch.randint(self.min_dst_idx,
                                          self.max_dst_idx + 1,
                                          (src.size(0), ), dtype=torch.long,)
            list_to_make_n_ids += [batch.neg_dst]
        n_id = torch.cat(list_to_make_n_ids).unique()
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0))
        batch.assoc, batch.n_id, batch.edge_index, batch.e_id = self.assoc, n_id, edge_index, e_id
        self.neighbor_loader.insert(src, pos_dst)
        return batch
