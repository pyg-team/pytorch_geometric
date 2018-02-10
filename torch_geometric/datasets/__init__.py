from .data import Data

from .cora import Cora
from .pubmed import PubMed
from .citeseer import CiteSeer
from .faust import FAUST
from .mnist_superpixels import MNISTSuperpixels
from .shapenet import ShapeNet
from .semantic_3d import Semantic3D
from .cuneiform import Cuneiform
from .qm9 import QM9

__all__ = [
    'Data', 'Cora', 'PubMed', 'CiteSeer', 'FAUST', 'MNISTSuperpixels',
    'ShapeNet', 'Semantic3D', 'Cuneiform', 'QM9'
]
