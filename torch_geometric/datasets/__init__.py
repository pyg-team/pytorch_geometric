from .karate import KarateClub
from .tu_dataset import TUDataset
from .gnn_benchmark_dataset import GNNBenchmarkDataset
from .planetoid import Planetoid
from .citation_full import CitationFull, CoraFull
from .coauthor import Coauthor
from .amazon import Amazon
from .ppi import PPI
from .reddit import Reddit
from .flickr import Flickr
from .yelp import Yelp
from .qm7 import QM7b
from .qm9 import QM9
from .zinc import ZINC
from .molecule_net import MoleculeNet
from .entities import Entities
from .ged_dataset import GEDDataset
from .mnist_superpixels import MNISTSuperpixels
from .faust import FAUST
from .dynamic_faust import DynamicFAUST
from .shapenet import ShapeNet
from .modelnet import ModelNet
from .coma import CoMA
from .shrec2016 import SHREC2016
from .tosca import TOSCA
from .pcpnet_dataset import PCPNetDataset
from .s3dis import S3DIS
from .geometry import GeometricShapes
from .bitcoin_otc import BitcoinOTC
from .icews import ICEWS18
from .gdelt import GDELT
from .willow_object_class import WILLOWObjectClass
from .dbp15k import DBP15K
from .pascal import PascalVOCKeypoints
from .pascal_pf import PascalPF
from .snap_dataset import SNAPDataset
from .suite_sparse import SuiteSparseMatrixCollection
from .particle import TrackMLParticleTrackingDataset
from .aminer import AMiner
from .word_net import WordNet18
from .wikics import WikiCS

__all__ = [
    'KarateClub',
    'TUDataset',
    'GNNBenchmarkDataset',
    'Planetoid',
    'CitationFull',
    'CoraFull',
    'Coauthor',
    'Amazon',
    'PPI',
    'Reddit',
    'Flickr',
    'Yelp',
    'QM7b',
    'QM9',
    'ZINC',
    'MoleculeNet',
    'Entities',
    'GEDDataset',
    'MNISTSuperpixels',
    'FAUST',
    'DynamicFAUST',
    'ShapeNet',
    'ModelNet',
    'CoMA',
    'SHREC2016',
    'TOSCA',
    'PCPNetDataset',
    'S3DIS',
    'GeometricShapes',
    'BitcoinOTC',
    'ICEWS18',
    'GDELT',
    'DBP15K',
    'WILLOWObjectClass',
    'PascalVOCKeypoints',
    'PascalPF',
    'SNAPDataset',
    'SuiteSparseMatrixCollection',
    'TrackMLParticleTrackingDataset',
    'AMiner',
    'WordNet18',
    'WikiCS',
]
