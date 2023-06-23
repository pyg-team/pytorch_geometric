from torch_geometric.deprecation import deprecated

from .dataloader import DataLoader
from .node_loader import NodeLoader
from .link_loader import LinkLoader
from .neighbor_loader import NeighborLoader
from .link_neighbor_loader import LinkNeighborLoader
from .hgt_loader import HGTLoader
from .cluster import ClusterData, ClusterLoader
from .graph_saint import (GraphSAINTSampler, GraphSAINTNodeSampler,
                          GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler)
from .shadow import ShaDowKHopSampler
from .random_node_loader import RandomNodeLoader
from .zip_loader import ZipLoader
from .data_list_loader import DataListLoader
from .dense_data_loader import DenseDataLoader
from .temporal_dataloader import TemporalDataLoader
from .neighbor_sampler import NeighborSampler
from .imbalanced_sampler import ImbalancedSampler
from .dynamic_batch_sampler import DynamicBatchSampler
from .prefetch import PrefetchLoader
from .mixin import AffinityMixin

import os
import torch
import inspect

def hook_nvtx_collate_fn(class_to_hook):
    original_init = class_to_hook.__init__

    def post_hooked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if not hasattr(self, "collate_fn"):
            return

        # Checking if a subclass was already hooked: no need to hook again
        if hasattr(self, "collate_fn_hooked"):
            return
        original_collate_fn = self.collate_fn
        def hooked_collate_fn(*args, **kwargs):
            nvtx_handle = torch.cuda.nvtx.range_start(f"[{class_to_hook.__name__}]] collate_fn for {self}")
            ret = original_collate_fn(*args, **kwargs)
            torch.cuda.nvtx.range_end(nvtx_handle)
            return ret
        self.collate_fn = hooked_collate_fn
        self.collate_fn_hooked = True
    class_to_hook.__init__ = post_hooked_init

if os.environ.get('NVIDIA_NVTX_RANGES', "0") == "1" and torch.cuda.is_available():
    syms = list(locals().keys())
    for sym in syms:
        cl = locals()[sym]
        if inspect.isclass(cl) and issubclass(cl, torch.utils.data.DataLoader):
            hook_nvtx_collate_fn(cl)

__all__ = classes = [
    'DataLoader',
    'NodeLoader',
    'LinkLoader',
    'NeighborLoader',
    'LinkNeighborLoader',
    'HGTLoader',
    'ClusterData',
    'ClusterLoader',
    'GraphSAINTSampler',
    'GraphSAINTNodeSampler',
    'GraphSAINTEdgeSampler',
    'GraphSAINTRandomWalkSampler',
    'ShaDowKHopSampler',
    'RandomNodeLoader',
    'ZipLoader',
    'DataListLoader',
    'DenseDataLoader',
    'TemporalDataLoader',
    'NeighborSampler',
    'ImbalancedSampler',
    'DynamicBatchSampler',
    'PrefetchLoader',
    'AffinityMixin',
]

RandomNodeSampler = deprecated(
    details="use 'loader.RandomNodeLoader' instead",
    func_name='loader.RandomNodeSampler',
)(RandomNodeLoader)
