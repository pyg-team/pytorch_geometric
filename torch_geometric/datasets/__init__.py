from .data import Data

from .cora import Cora
from .pubmed import PubMed
from .citeseer import CiteSeer
from .faust import FAUST
from .mnist_superpixels import MNISTSuperpixels
from .mnist_superpixels2 import MNISTSuperpixels2
from .shapenet import ShapeNet
from .cuneiform import Cuneiform
from .cuneiform2 import Cuneiform2
from .qm9 import QM9

__all__ = [
    'Data', 'Cora', 'PubMed', 'CiteSeer', 'FAUST', 'MNISTSuperpixels',
    'MNISTSuperpixels2', 'ShapeNet', 'Cuneiform', 'Cuneiform2', 'QM9'
]
