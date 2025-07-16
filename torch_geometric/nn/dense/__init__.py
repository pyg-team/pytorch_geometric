r"""Dense neural network module package.

This package provides modules applicable for operating on dense tensor
representations.
"""

from .dense_gat_conv import DenseGATConv
from .dense_gcn_conv import DenseGCNConv
from .dense_gin_conv import DenseGINConv
from .dense_graph_conv import DenseGraphConv
from .dense_sage_conv import DenseSAGEConv
from .diff_pool import dense_diff_pool
from .dmon_pool import DMoNPooling
from .linear import HeteroDictLinear, HeteroLinear, Linear
from .mincut_pool import dense_mincut_pool

__all__ = [
    'Linear',
    'HeteroLinear',
    'HeteroDictLinear',
    'DenseGCNConv',
    'DenseGINConv',
    'DenseGraphConv',
    'DenseSAGEConv',
    'DenseGATConv',
    'dense_diff_pool',
    'dense_mincut_pool',
    'DMoNPooling',
]

lin_classes = __all__[:3]
conv_classes = __all__[3:8]
pool_classes = __all__[8:]
