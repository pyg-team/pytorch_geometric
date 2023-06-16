from .augmentor import Graph, Augmentor, Compose, RandomChoice
from .identity import Identity
from .rw_sampling import RWSampling
from .edge_removing import EdgeRemoving
from .node_dropping import NodeDropping
from .feature_masking import FeatureMasking


__all__ = [
    'Graph',
    'Augmentor',
    'Compose',
    'RandomChoice',
    'EdgeRemoving',
    'FeatureMasking',
    'Identity',
    'NodeDropping',
    'RWSampling'
]

classes = __all__
