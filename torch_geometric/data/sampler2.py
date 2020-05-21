from typing import List, Optional, Tuple

import torch
from torch_sparse import SparseTensor


class NeighborSampler(torch.utils.data.DataLoader):
    def __init__(self, edge_index: torch.Tensor, node_idx: torch.Tensor,
                 sizes: List[int], num_nodes: Optional[int] = None,
                 flow: str = 'source_to_target', **kwargs):

        N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        edge_attr = torch.arange(edge_index.size(1))
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (N, N),
                                           is_sorted=False)
        self.adj = adj.t() if flow == 'source_to_target' else adj

        self.sizes = sizes
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        super(NeighborSampler,
              self).__init__(node_idx.tolist(), collate_fn=self.__gen_batch__,
                             **kwargs)

    def __gen_batch__(self, n_id: torch.Tensor
                      ) -> Tuple[torch.Tensor, List[SparseTensor]]:

        subsets = [torch.tensor(n_id)]
        adjs = []
        for size in self.sizes:
            adj, n_id = self.adj.sample_adj(subsets[-1], size, replace=False)
            row, col, e_id = adj.coo()
            size = adj.sparse_sizes()
            if self.flow == 'source_to_target':
                edge_index = torch.stack([col, row], dim=0)
                size = size[::-1]
            else:
                edge_index = torch.stack([row, col], dim=0)

            subsets.append(n_id)
            adjs.append((edge_index, e_id, size))

        return n_id, adjs[::-1]
