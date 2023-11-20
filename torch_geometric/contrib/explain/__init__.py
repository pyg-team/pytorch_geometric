from torch_geometric.deprecation import deprecated

from .pgm_explainer import PGMExplainer
from torch_geometric.explain.algorithm.graphmask_explainer import (
    GraphMaskExplainer as NewGraphMaskExplainer)

GraphMaskExplainer = deprecated(
    "use 'torch_geometric.explain.algorithm.GraphMaskExplainer' instead", )(
        NewGraphMaskExplainer)

__all__ = classes = [
    'PGMExplainer',
]
