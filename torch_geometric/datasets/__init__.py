from .data import Data

from .cora import Cora
from .pubmed import PubMed
from .citeseer import CiteSeer
from .faust import FAUST
from .mnist_superpixels import MNISTSuperpixels

__all__ = ['Data', 'Cora', 'PubMed', 'CiteSeer', 'FAUST', 'MNISTSuperpixels']
