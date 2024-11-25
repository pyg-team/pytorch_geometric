# flake8: noqa

import torch
import torch_geometric.typing

from .feature_store import FeatureStore, TensorAttr
from .graph_store import GraphStore, EdgeAttr, EdgeLayout
from .data import Data
from .hetero_data import HeteroData
from .batch import Batch
from .temporal import TemporalData
from .database import Database, SQLiteDatabase, RocksDatabase
from .dataset import Dataset
from .in_memory_dataset import InMemoryDataset
from .on_disk_dataset import OnDiskDataset
from .makedirs import makedirs
from .download import download_url, download_google_url
from .extract import extract_tar, extract_zip, extract_bz2, extract_gz
from .large_graph_indexer import LargeGraphIndexer, TripletLike, get_features_for_triplets, get_features_for_triplets_groups

from torch_geometric.lazy_loader import LazyLoader

data_classes = [
    'Data',
    'HeteroData',
    'Batch',
    'TemporalData',
    'Dataset',
    'InMemoryDataset',
    'OnDiskDataset',
    'LargeGraphIndexer',
    'TripletLike',
]

remote_backend_classes = [
    'FeatureStore',
    'GraphStore',
    'TensorAttr',
    'EdgeAttr',
]

database_classes = [
    'Database',
    'SQLiteDatabase',
    'RocksDatabase',
]

helper_functions = [
    'makedirs',
    'download_url',
    'download_google_url',
    'extract_tar',
    'extract_zip',
    'extract_bz2',
    'extract_gz',
    'get_features_for_triplets',
    "get_features_for_triplets_groups",
]

__all__ = data_classes + remote_backend_classes + helper_functions

lightning = LazyLoader('lightning', globals(),
                       'torch_geometric.data.lightning')

from torch_geometric.deprecation import deprecated
from torch_geometric.loader import NeighborSampler
from torch_geometric.loader import ClusterData
from torch_geometric.loader import ClusterLoader
from torch_geometric.loader import GraphSAINTSampler
from torch_geometric.loader import GraphSAINTNodeSampler
from torch_geometric.loader import GraphSAINTEdgeSampler
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataListLoader
from torch_geometric.loader import DenseDataLoader

# Serialization ###############################################################

if torch_geometric.typing.WITH_PT24:
    torch.serialization.add_safe_globals([
        Data,
        HeteroData,
        TemporalData,
        ClusterData,
        TensorAttr,
        EdgeAttr,
        EdgeLayout,
    ])

# Deprecations ################################################################

NeighborSampler = deprecated(  # type: ignore
    details="use 'loader.NeighborSampler' instead",
    func_name='data.NeighborSampler',
)(NeighborSampler)
ClusterData = deprecated(  # type: ignore
    details="use 'loader.ClusterData' instead",
    func_name='data.ClusterData',
)(ClusterData)
ClusterLoader = deprecated(  # type: ignore
    details="use 'loader.ClusterLoader' instead",
    func_name='data.ClusterLoader',
)(ClusterLoader)
GraphSAINTSampler = deprecated(  # type: ignore
    details="use 'loader.GraphSAINTSampler' instead",
    func_name='data.GraphSAINTSampler',
)(GraphSAINTSampler)
GraphSAINTNodeSampler = deprecated(  # type: ignore
    details="use 'loader.GraphSAINTNodeSampler' instead",
    func_name='data.GraphSAINTNodeSampler',
)(GraphSAINTNodeSampler)
GraphSAINTEdgeSampler = deprecated(  # type: ignore
    details="use 'loader.GraphSAINTEdgeSampler' instead",
    func_name='data.GraphSAINTEdgeSampler',
)(GraphSAINTEdgeSampler)
GraphSAINTRandomWalkSampler = deprecated(  # type: ignore
    details="use 'loader.GraphSAINTRandomWalkSampler' instead",
    func_name='data.GraphSAINTRandomWalkSampler',
)(GraphSAINTRandomWalkSampler)
ShaDowKHopSampler = deprecated(  # type: ignore
    details="use 'loader.ShaDowKHopSampler' instead",
    func_name='data.ShaDowKHopSampler',
)(ShaDowKHopSampler)
RandomNodeSampler = deprecated(
    details="use 'loader.RandomNodeLoader' instead",
    func_name='data.RandomNodeSampler',
)(RandomNodeLoader)
DataLoader = deprecated(  # type: ignore
    details="use 'loader.DataLoader' instead",
    func_name='data.DataLoader',
)(DataLoader)
DataListLoader = deprecated(  # type: ignore
    details="use 'loader.DataListLoader' instead",
    func_name='data.DataListLoader',
)(DataListLoader)
DenseDataLoader = deprecated(  # type: ignore
    details="use 'loader.DenseDataLoader' instead",
    func_name='data.DenseDataLoader',
)(DenseDataLoader)
