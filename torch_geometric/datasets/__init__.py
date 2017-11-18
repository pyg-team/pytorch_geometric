from .data import Data

from .cora import Cora
from .pubmed import PubMed
from .citeseer import CiteSeer
from .faust import FAUST
from .mnist_superpixel_75 import MNISTSuperpixel75

__all__ = ['Data', 'Cora', 'PubMed', 'CiteSeer', 'FAUST', 'MNISTSuperpixel75']
