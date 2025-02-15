r"""GNN utility package."""

from .cheatsheet import paper_title, paper_link
from .cheatsheet import supports_sparse_tensor
from .cheatsheet import supports_edge_weights
from .cheatsheet import supports_edge_features
from .cheatsheet import supports_bipartite_graphs
from .cheatsheet import supports_static_graphs
from .cheatsheet import supports_lazy_initialization
from .cheatsheet import processes_heterogeneous_graphs
from .cheatsheet import processes_hypergraphs
from .cheatsheet import processes_point_clouds

__all__ = [
    'paper_title',
    'paper_link',
    'supports_sparse_tensor',
    'supports_edge_weights',
    'supports_edge_features',
    'supports_bipartite_graphs',
    'supports_static_graphs',
    'supports_lazy_initialization',
    'processes_heterogeneous_graphs',
    'processes_hypergraphs',
    'processes_point_clouds',
]
