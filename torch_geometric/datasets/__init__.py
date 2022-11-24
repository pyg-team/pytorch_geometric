from .karate import KarateClub
from .tu_dataset import TUDataset
from .gnn_benchmark_dataset import GNNBenchmarkDataset
from .planetoid import Planetoid
from .fake import FakeDataset, FakeHeteroDataset
from .nell import NELL
from .citation_full import CitationFull, CoraFull
from .coauthor import Coauthor
from .amazon import Amazon
from .ppi import PPI
from .reddit import Reddit
from .reddit2 import Reddit2
from .flickr import Flickr
from .yelp import Yelp
from .amazon_products import AmazonProducts
from .qm7 import QM7b
from .qm9 import QM9
from .md17 import MD17
from .zinc import ZINC
from .aqsol import AQSOL
from .molecule_net import MoleculeNet
from .entities import Entities
from .rel_link_pred_dataset import RelLinkPredDataset
from .ged_dataset import GEDDataset
from .attributed_graph_dataset import AttributedGraphDataset
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
# from .particle import TrackMLParticleTrackingDataset
from .aminer import AMiner
from .word_net import WordNet18, WordNet18RR
from .wikics import WikiCS
from .webkb import WebKB
from .wikipedia_network import WikipediaNetwork
from .actor import Actor
from .ogb_mag import OGB_MAG
from .dblp import DBLP
from .movie_lens import MovieLens
from .imdb import IMDB
from .last_fm import LastFM
from .hgb_dataset import HGBDataset
from .jodie import JODIEDataset
from .mixhop_synthetic_dataset import MixHopSyntheticDataset
from .upfd import UPFD
from .github import GitHub
from .facebook import FacebookPagePage
from .lastfm_asia import LastFMAsia
from .deezer_europe import DeezerEurope
from .gemsec import GemsecDeezer
from .twitch import Twitch
from .airports import Airports
from .ba_shapes import BAShapes
from .lrgb import LRGBDataset
from .malnet_tiny import MalNetTiny
from .omdb import OMDB
from .polblogs import PolBlogs
from .email_eu_core import EmailEUCore
from .sbm_dataset import StochasticBlockModelDataset
from .sbm_dataset import RandomPartitionGraphDataset
from .linkx_dataset import LINKXDataset
from .elliptic import EllipticBitcoinDataset
from .dgraph import DGraphFin
from .hydro_net import HydroNet

import torch_geometric.datasets.utils  # noqa

__all__ = [
    'KarateClub',
    'TUDataset',
    'GNNBenchmarkDataset',
    'Planetoid',
    'FakeDataset',
    'FakeHeteroDataset',
    'NELL',
    'CitationFull',
    'CoraFull',
    'Coauthor',
    'Amazon',
    'PPI',
    'Reddit',
    'Reddit2',
    'Flickr',
    'Yelp',
    'AmazonProducts',
    'QM7b',
    'QM9',
    'MD17',
    'ZINC',
    'AQSOL',
    'MoleculeNet',
    'Entities',
    'RelLinkPredDataset',
    'GEDDataset',
    'AttributedGraphDataset',
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
    # 'TrackMLParticleTrackingDataset',
    'AMiner',
    'WordNet18',
    'WordNet18RR',
    'WikiCS',
    'WebKB',
    'WikipediaNetwork',
    'Actor',
    'OGB_MAG',
    'DBLP',
    'MovieLens',
    'IMDB',
    'LastFM',
    'HGBDataset',
    'JODIEDataset',
    'MixHopSyntheticDataset',
    'UPFD',
    'GitHub',
    'FacebookPagePage',
    'LastFMAsia',
    'DeezerEurope',
    'GemsecDeezer',
    'Twitch',
    'Airports',
    'BAShapes',
    'LRGBDataset',
    'MalNetTiny',
    'OMDB',
    'PolBlogs',
    'EmailEUCore',
    'StochasticBlockModelDataset',
    'RandomPartitionGraphDataset',
    'LINKXDataset',
    'EllipticBitcoinDataset',
    'DGraphFin',
    'HydroNet',
]

classes = __all__
