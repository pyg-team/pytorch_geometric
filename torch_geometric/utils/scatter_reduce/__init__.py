from torch_geometric.utils.torch_version import torch_version_minor

if torch_version_minor() < 12:
    from torch_scatter import scatter
else:
    from .scatter import scatter

from .composite import scatter_composite

__all__ = [
    'scatter',
    'scatter_composite',
]
