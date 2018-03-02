from .data import Data

from .cora import Cora
from .pubmed import PubMed
from .citeseer import CiteSeer
from .faust import FAUST
from .mnist_superpixels import MNISTSuperpixels
from .mnist_superpixels2 import MNISTSuperpixels2
from .shapenet import ShapeNet
from .shapenet2 import ShapeNet2
from .cuneiform import Cuneiform
from .cuneiform2 import Cuneiform2
from .qm9 import QM9
from .kitti import KITTI
from .modelnet10 import ModelNet10
from .modelnet40 import ModelNet40
from .modelnet10_pc import ModelNet10PC
from .modelnet10_rand_pc import ModelNet10RandPC
from .modelnet40_rand_pc import ModelNet40RandPC

__all__ = [
    'Data', 'Cora', 'PubMed', 'CiteSeer', 'FAUST', 'MNISTSuperpixels',
    'MNISTSuperpixels2', 'ShapeNet', 'ShapeNet2', 'Cuneiform', 'Cuneiform2',
    'QM9', 'KITTI', 'ModelNet10', 'ModelNet40', 'ModelNet10PC',
    'ModelNet10RandPC', 'ModelNet40RandPC'
]
