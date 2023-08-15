from .pgm_explainer import PGMExplainer
from torch_geometric.explain.algorithm.graphmask_explainer import (
    GraphMaskExplainer as NewGraphMaskExplainer)
from torch_geometric.deprecation import deprecated

GraphMaskExplainer = deprecated(
    "use 'torch_geometric.explain.algorithm.GraphMaskExplainer' instead", )(
        NewGraphMaskExplainer)

__all__ = classes = [
    'GraphMaskExplainer',
    'PGMExplainer',
]
