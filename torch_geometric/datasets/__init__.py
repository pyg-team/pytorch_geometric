# flake8: noqa

from .karate import KarateClub
from .tu_dataset import TUDataset
from .gnn_benchmark_dataset import GNNBenchmarkDataset
from .planetoid import Planetoid
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
from .pcqm4m import PCQM4Mv2
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
from .gdelt_lite import GDELTLite
from .icews import ICEWS18
from .gdelt import GDELT
from .willow_object_class import WILLOWObjectClass
from .pascal import PascalVOCKeypoints
from .pascal_pf import PascalPF
from .snap_dataset import SNAPDataset
from .suite_sparse import SuiteSparseMatrixCollection
from .word_net import WordNet18, WordNet18RR
from .freebase import FB15k_237
from .wikics import WikiCS
from .webkb import WebKB
from .wikipedia_network import WikipediaNetwork
from .heterophilous_graph_dataset import HeterophilousGraphDataset
from .actor import Actor
from .upfd import UPFD
from .github import GitHub
from .facebook import FacebookPagePage
from .lastfm_asia import LastFMAsia
from .deezer_europe import DeezerEurope
from .gemsec import GemsecDeezer
from .twitch import Twitch
from .airports import Airports
from .lrgb import LRGBDataset
from .neurograph import NeuroGraphDataset
from .malnet_tiny import MalNetTiny
from .omdb import OMDB
from .polblogs import PolBlogs
from .email_eu_core import EmailEUCore
from .linkx_dataset import LINKXDataset
from .elliptic import EllipticBitcoinDataset
from .elliptic_temporal import EllipticBitcoinTemporalDataset
from .dgraph import DGraphFin
from .hydro_net import HydroNet
from .airfrans import AirfRANS
from .jodie import JODIEDataset
from .wikidata import Wikidata5M
from .myket import MyketDataset
from .brca_tgca import BrcaTcga

from .dbp15k import DBP15K
from .aminer import AMiner
from .ogb_mag import OGB_MAG
from .dblp import DBLP
from .movie_lens import MovieLens
from .movie_lens_100k import MovieLens100K
from .movie_lens_1m import MovieLens1M
from .imdb import IMDB
from .last_fm import LastFM
from .hgb_dataset import HGBDataset
from .taobao import Taobao
from .igmc_dataset import IGMCDataset
from .amazon_book import AmazonBook
from .hm import HM
from .ose_gvcs import OSE_GVCS
from .rcdd import RCDD
from .opf import OPFDataset

from .cornell import CornellTemporalHyperGraphDataset

from .fake import FakeDataset, FakeHeteroDataset
from .sbm_dataset import StochasticBlockModelDataset
from .sbm_dataset import RandomPartitionGraphDataset
from .mixhop_synthetic_dataset import MixHopSyntheticDataset
from .explainer_dataset import ExplainerDataset
from .infection_dataset import InfectionDataset
from .ba2motif_dataset import BA2MotifDataset
from .ba_multi_shapes import BAMultiShapesDataset
from .ba_shapes import BAShapes

import torch_geometric.datasets.utils

homo_datasets = [
    'KarateClub',
    'TUDataset',
    'GNNBenchmarkDataset',
    'Planetoid',
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
    'PCQM4Mv2',
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
    'GDELTLite',
    'ICEWS18',
    'GDELT',
    'WILLOWObjectClass',
    'PascalVOCKeypoints',
    'PascalPF',
    'SNAPDataset',
    'SuiteSparseMatrixCollection',
    'WordNet18',
    'WordNet18RR',
    'FB15k_237',
    'WikiCS',
    'WebKB',
    'WikipediaNetwork',
    'HeterophilousGraphDataset',
    'Actor',
    'UPFD',
    'GitHub',
    'FacebookPagePage',
    'LastFMAsia',
    'DeezerEurope',
    'GemsecDeezer',
    'Twitch',
    'Airports',
    'LRGBDataset',
    'MalNetTiny',
    'OMDB',
    'PolBlogs',
    'EmailEUCore',
    'LINKXDataset',
    'EllipticBitcoinDataset',
    'EllipticBitcoinTemporalDataset',
    'DGraphFin',
    'HydroNet',
    'AirfRANS',
    'JODIEDataset',
    'Wikidata5M',
    'MyketDataset',
    'BrcaTcga',
    'NeuroGraphDataset',
]

hetero_datasets = [
    'DBP15K',
    'AMiner',
    'OGB_MAG',
    'DBLP',
    'MovieLens',
    'MovieLens100K',
    'MovieLens1M',
    'IMDB',
    'LastFM',
    'HGBDataset',
    'Taobao',
    'IGMCDataset',
    'AmazonBook',
    'HM',
    'OSE_GVCS',
    'RCDD',
    'OPFDataset',
]
hyper_datasets = [
    'CornellTemporalHyperGraphDataset',
]
synthetic_datasets = [
    'FakeDataset',
    'FakeHeteroDataset',
    'StochasticBlockModelDataset',
    'RandomPartitionGraphDataset',
    'MixHopSyntheticDataset',
    'ExplainerDataset',
    'InfectionDataset',
    'BA2MotifDataset',
    'BAMultiShapesDataset',
    'BAShapes',
]

__all__ = homo_datasets + hetero_datasets + hyper_datasets + synthetic_datasets
