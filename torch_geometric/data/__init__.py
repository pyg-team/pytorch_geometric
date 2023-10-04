# flake8: noqa

from .feature_store import FeatureStore, TensorAttr
from .graph_store import GraphStore, EdgeAttr
from .data import Data
from .hetero_data import HeteroData
from .batch import Batch
from .temporal import TemporalData
from .database import Database, SQLiteDatabase, RocksDatabase
from .dataset import Dataset
from .in_memory_dataset import InMemoryDataset
from .on_disk_dataset import OnDiskDataset
from .makedirs import makedirs
from .download import download_url
from .extract import extract_tar, extract_zip, extract_bz2, extract_gz

from torch_geometric.lazy_loader import LazyLoader

data_classes = [
    'Data',
    'HeteroData',
    'Batch',
    'TemporalData',
    'Dataset',
    'InMemoryDataset',
    'OnDiskDataset',
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
    'extract_tar',
    'extract_zip',
    'extract_bz2',
    'extract_gz',
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

NeighborSampler = deprecated("use 'loader.NeighborSampler' instead",
                             'data.NeighborSampler')(NeighborSampler)
ClusterData = deprecated("use 'loader.ClusterData' instead",
                         'data.ClusterData')(ClusterData)
ClusterLoader = deprecated("use 'loader.ClusterLoader' instead",
                           'data.ClusterLoader')(ClusterLoader)
GraphSAINTSampler = deprecated("use 'loader.GraphSAINTSampler' instead",
                               'data.GraphSAINTSampler')(GraphSAINTSampler)
GraphSAINTNodeSampler = deprecated(
    "use 'loader.GraphSAINTNodeSampler' instead",
    'data.GraphSAINTNodeSampler')(GraphSAINTNodeSampler)
GraphSAINTEdgeSampler = deprecated(
    "use 'loader.GraphSAINTEdgeSampler' instead",
    'data.GraphSAINTEdgeSampler')(GraphSAINTEdgeSampler)
GraphSAINTRandomWalkSampler = deprecated(
    "use 'loader.GraphSAINTRandomWalkSampler' instead",
    'data.GraphSAINTRandomWalkSampler')(GraphSAINTRandomWalkSampler)
ShaDowKHopSampler = deprecated("use 'loader.ShaDowKHopSampler' instead",
                               'data.ShaDowKHopSampler')(ShaDowKHopSampler)
RandomNodeSampler = deprecated("use 'loader.RandomNodeLoader' instead",
                               'data.RandomNodeSampler')(RandomNodeLoader)
DataLoader = deprecated("use 'loader.DataLoader' instead",
                        'data.DataLoader')(DataLoader)
DataListLoader = deprecated("use 'loader.DataListLoader' instead",
                            'data.DataListLoader')(DataListLoader)
DenseDataLoader = deprecated("use 'loader.DenseDataLoader' instead",
                             'data.DenseDataLoader')(DenseDataLoader)
