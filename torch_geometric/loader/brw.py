import os.path as osp
from typing import Optional

import torch
from tqdm import tqdm

from torch_geometric.loader import \
    GraphSAINTRandomWalkSampler as GraphSAINTSampler
from torch_geometric.typing import SparseTensor


class BiasedRandomWalkSampler(GraphSAINTSampler):
    r"""The BiasedRandomWalkSampler random walk sampler class (see
    :class:`~torch_geometric.loader.GraphSAINTSampler`).

    Args:
        walk_length (int): The length of each random walk.
        target_indices (list): the list of task target node indices to start the sampling from.
        For example, the paper node indices in OGBN-MAG paper-venue node classification dataset.
        target_node_ratio (float) : the ratio of target nodes from the batch_size, the default is 1.
    """
    def __init__(self, data, batch_size: int, walk_length: int, target_indices,
                 num_steps: int = 1, sample_coverage: int = 0,
                 target_node_ratio=1, save_dir: Optional[str] = None,
                 log: bool = True, **kwargs):
        self.walk_length = walk_length
        self.target_indices = target_indices
        self.target_node_ratio = target_node_ratio
        self.index = 0
        super().__init__(data, batch_size, walk_length, num_steps,
                         sample_coverage, save_dir, log, **kwargs)

    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __sample_nodes__(self, batch_size):
        target_count = int(batch_size * self.target_node_ratio)
        start_target = torch.randint(min(self.target_indices),
                                     max(self.target_indices),
                                     (target_count, ), dtype=torch.long)
        start_non_target = torch.randint(0, self.N,
                                         (batch_size - target_count, ),
                                         dtype=torch.long)
        start = torch.cat((start_target, start_non_target)).unique()
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)
