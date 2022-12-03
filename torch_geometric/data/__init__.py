from .data import Data
from .hetero_data import HeteroData
from .batch import Batch
from .temporal import TemporalData
from .dataset import Dataset
from .in_memory_dataset import InMemoryDataset
from .lightning_datamodule import (
    LightningDataset,
    LightningLinkData,
    LightningNodeData,
)
from .feature_store import FeatureStore, TensorAttr
from .graph_store import GraphStore, EdgeAttr
from .makedirs import makedirs
from .download import download_url
from .extract import extract_tar, extract_zip, extract_bz2, extract_gz

__all__ = [
    'Data',
    'HeteroData',
    'Batch',
    'TemporalData',
    'Dataset',
    'InMemoryDataset',
    'LightningDataset',
    'LightningNodeData',
    'LightningLinkData',
    'FeatureStore',
    'TensorAttr',
    'GraphStore',
    'EdgeAttr',
    'makedirs',
    'download_url',
    'extract_tar',
    'extract_zip',
    'extract_bz2',
    'extract_gz',
]

classes = __all__

from torch_geometric.deprecation import deprecated  # noqa
from torch_geometric.loader import NeighborSampler  # noqa
from torch_geometric.loader import ClusterData  # noqa
from torch_geometric.loader import ClusterLoader  # noqa
from torch_geometric.loader import GraphSAINTSampler  # noqa
from torch_geometric.loader import GraphSAINTNodeSampler  # noqa
from torch_geometric.loader import GraphSAINTEdgeSampler  # noqa
from torch_geometric.loader import GraphSAINTRandomWalkSampler  # noqa
from torch_geometric.loader import ShaDowKHopSampler  # noqa
from torch_geometric.loader import RandomNodeLoader  # noqa
from torch_geometric.loader import DataLoader  # noqa
from torch_geometric.loader import DataListLoader  # noqa
from torch_geometric.loader import DenseDataLoader  # noqa

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
