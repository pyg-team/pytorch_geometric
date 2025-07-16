from torch_geometric.deprecation import deprecated

from .cache import CachedLoader
from .cluster import ClusterData, ClusterLoader
from .data_list_loader import DataListLoader
from .dataloader import DataLoader
from .dense_data_loader import DenseDataLoader
from .dynamic_batch_sampler import DynamicBatchSampler
from .graph_saint import (GraphSAINTEdgeSampler, GraphSAINTNodeSampler,
                          GraphSAINTRandomWalkSampler, GraphSAINTSampler)
from .hgt_loader import HGTLoader
from .imbalanced_sampler import ImbalancedSampler
from .link_loader import LinkLoader
from .link_neighbor_loader import LinkNeighborLoader
from .mixin import AffinityMixin
from .neighbor_loader import NeighborLoader
from .neighbor_sampler import NeighborSampler
from .node_loader import NodeLoader
from .prefetch import PrefetchLoader
from .rag_loader import RAGFeatureStore, RAGGraphStore, RAGQueryLoader
from .random_node_loader import RandomNodeLoader
from .shadow import ShaDowKHopSampler
from .temporal_dataloader import TemporalDataLoader
# from .ibmb_loader import IBMBBatchLoader, IBMBNodeLoader
from .zip_loader import ZipLoader

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
    # 'IBMBBatchLoader',
    # 'IBMBNodeLoader',
    'ZipLoader',
    'DataListLoader',
    'DenseDataLoader',
    'TemporalDataLoader',
    'NeighborSampler',
    'ImbalancedSampler',
    'DynamicBatchSampler',
    'PrefetchLoader',
    'CachedLoader',
    'AffinityMixin',
    'RAGQueryLoader',
    'RAGFeatureStore',
    'RAGGraphStore'
]

RandomNodeSampler = deprecated(
    details="use 'loader.RandomNodeLoader' instead",
    func_name='loader.RandomNodeSampler',
)(RandomNodeLoader)
