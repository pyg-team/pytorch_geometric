from .faust import FAUST
from .faust_patch import FAUSTPatch
from .cora import Cora
from .pubmed import PubMed
from .citeseer import CiteSeer
from .mnist_superpixel_75 import MNISTSuperpixel75
from .shape_net import ShapeNet

__all__ = [
    'FAUST', 'Cora', 'PubMed', 'CiteSeer', 'MNISTSuperpixel75', 'ShapeNet'
]
