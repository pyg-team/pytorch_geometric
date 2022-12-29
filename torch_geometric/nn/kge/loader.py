from typing import List, Tuple

import torch
from torch import Tensor


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, num_nodes: int, edge_index: Tensor, edge_type: Tensor,
                 add_negative_samples: bool = True, **kwargs):
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.add_negative_samples = add_negative_samples
        super().__init__(range(edge_index.size(1)), collate_fn=self.sample,
                         **kwargs)

    def sample(self, batch: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        numel = len(batch)
        device = self.edge_index.device
        batch = torch.tensor(batch, device=device)

        edge_index = torch.empty((2, 2 * numel), dtype=torch.long,
                                 device=device)
        edge_type = torch.empty(2 * numel, dtype=torch.long, device=device)
        pos_mask = torch.empty(2 * numel, dtype=torch.bool, device=device)

        edge_index[0, :numel] = self.edge_index[0, batch]
        edge_index[1, :numel] = self.edge_index[1, batch]
        edge_index[0, numel:] = edge_index[0, :numel]
        edge_index[1, numel:] = edge_index[1, :numel]

        rnd = torch.randint(self.num_nodes, (numel, ), dtype=torch.long,
                            device=device)
        src_mask = torch.rand(numel, device=device) < 0.5

        edge_index[0, numel:][src_mask] = rnd[src_mask]
        edge_index[1, numel:][~src_mask] = rnd[~src_mask]

        edge_type[:numel] = self.edge_type[batch]
        edge_type[numel:] = edge_type[:numel]

        pos_mask[:numel] = True
        pos_mask[numel:] = False

        return edge_index, edge_type, pos_mask
