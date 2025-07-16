# flake8: noqa

import torch_geometric.datasets.utils

from .actor import Actor
from .airfrans import AirfRANS
from .airports import Airports
from .amazon import Amazon
from .amazon_book import AmazonBook
from .amazon_products import AmazonProducts
from .aminer import AMiner
from .aqsol import AQSOL
from .attributed_graph_dataset import AttributedGraphDataset
from .ba2motif_dataset import BA2MotifDataset
from .ba_multi_shapes import BAMultiShapesDataset
from .ba_shapes import BAShapes
from .bitcoin_otc import BitcoinOTC
from .brca_tgca import BrcaTcga
from .citation_full import CitationFull, CoraFull
from .city import CityNetwork
from .coauthor import Coauthor
from .coma import CoMA
from .cornell import CornellTemporalHyperGraphDataset
from .dblp import DBLP
from .dbp15k import DBP15K
from .deezer_europe import DeezerEurope
from .dgraph import DGraphFin
from .dynamic_faust import DynamicFAUST
from .elliptic import EllipticBitcoinDataset
from .elliptic_temporal import EllipticBitcoinTemporalDataset
from .email_eu_core import EmailEUCore
from .entities import Entities
from .explainer_dataset import ExplainerDataset
from .facebook import FacebookPagePage
from .fake import FakeDataset, FakeHeteroDataset
from .faust import FAUST
from .flickr import Flickr
from .freebase import FB15k_237
from .gdelt import GDELT
from .gdelt_lite import GDELTLite
from .ged_dataset import GEDDataset
from .gemsec import GemsecDeezer
from .geometry import GeometricShapes
from .git_mol_dataset import GitMolDataset
from .github import GitHub
from .gnn_benchmark_dataset import GNNBenchmarkDataset
from .heterophilous_graph_dataset import HeterophilousGraphDataset
from .hgb_dataset import HGBDataset
from .hm import HM
from .hydro_net import HydroNet
from .icews import ICEWS18
from .igmc_dataset import IGMCDataset
from .imdb import IMDB
from .infection_dataset import InfectionDataset
from .instruct_mol_dataset import InstructMolDataset
from .jodie import JODIEDataset
from .karate import KarateClub
from .last_fm import LastFM
from .lastfm_asia import LastFMAsia
from .linkx_dataset import LINKXDataset
from .lrgb import LRGBDataset
from .malnet_tiny import MalNetTiny
from .md17 import MD17
from .medshapenet import MedShapeNet
from .mixhop_synthetic_dataset import MixHopSyntheticDataset
from .mnist_superpixels import MNISTSuperpixels
from .modelnet import ModelNet
from .molecule_gpt_dataset import MoleculeGPTDataset
from .molecule_net import MoleculeNet
from .movie_lens import MovieLens
from .movie_lens_1m import MovieLens1M
from .movie_lens_100k import MovieLens100K
from .myket import MyketDataset
from .nell import NELL
from .neurograph import NeuroGraphDataset
from .ogb_mag import OGB_MAG
from .omdb import OMDB
from .opf import OPFDataset
from .ose_gvcs import OSE_GVCS
from .pascal import PascalVOCKeypoints
from .pascal_pf import PascalPF
from .pcpnet_dataset import PCPNetDataset
from .pcqm4m import PCQM4Mv2
from .planetoid import Planetoid
from .polblogs import PolBlogs
from .ppi import PPI
from .protein_mpnn_dataset import ProteinMPNNDataset
from .qm7 import QM7b
from .qm9 import QM9
from .rcdd import RCDD
from .reddit import Reddit
from .reddit2 import Reddit2
from .rel_link_pred_dataset import RelLinkPredDataset
from .s3dis import S3DIS
from .sbm_dataset import (RandomPartitionGraphDataset,
                          StochasticBlockModelDataset)
from .shapenet import ShapeNet
from .shrec2016 import SHREC2016
from .snap_dataset import SNAPDataset
from .suite_sparse import SuiteSparseMatrixCollection
from .tag_dataset import TAGDataset
from .taobao import Taobao
from .teeth3ds import Teeth3DS
from .tosca import TOSCA
from .tu_dataset import TUDataset
from .twitch import Twitch
from .upfd import UPFD
from .web_qsp_dataset import CWQDataset, WebQSPDataset
from .webkb import WebKB
from .wikics import WikiCS
from .wikidata import Wikidata5M
from .wikipedia_network import WikipediaNetwork
from .willow_object_class import WILLOWObjectClass
from .word_net import WordNet18, WordNet18RR
from .yelp import Yelp
from .zinc import ZINC

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
    'MedShapeNet',
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
    'WebQSPDataset',
    'CWQDataset',
    'GitMolDataset',
    'MoleculeGPTDataset',
    'InstructMolDataset',
    'ProteinMPNNDataset',
    'TAGDataset',
    'CityNetwork',
    'Teeth3DS',
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
