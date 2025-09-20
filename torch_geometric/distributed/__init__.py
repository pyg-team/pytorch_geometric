from warnings import warn

from .dist_context import DistContext
from .local_feature_store import LocalFeatureStore
from .local_graph_store import LocalGraphStore
from .partition import Partitioner
from .dist_neighbor_sampler import DistNeighborSampler
from .dist_loader import DistLoader
from .dist_neighbor_loader import DistNeighborLoader
from .dist_link_neighbor_loader import DistLinkNeighborLoader

warn(
    "`torch_geometric.distributed` has been deprecated since 2.7.0 and will "
    "no longer be maintained. For distributed training, refer to our "
    "tutorials on distributed training at "
    "https://pytorch-geometric.readthedocs.io/en/latest/tutorial/distributed.html "  # noqa: E501
    "or cuGraph examples at "
    "https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg/cugraph_pyg/examples",  # noqa: E501
    stacklevel=2,
    category=DeprecationWarning,
)

__all__ = classes = [
    'DistContext',
    'LocalFeatureStore',
    'LocalGraphStore',
    'Partitioner',
    'DistNeighborSampler',
    'DistLoader',
    'DistNeighborLoader',
    'DistLinkNeighborLoader',
]
