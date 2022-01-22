from .attentive_fp import AttentiveFP
from .autoencoder import ARGA, ARGVA, GAE, VGAE, InnerProductDecoder
from .basic_gnn import GAT, GCN, GIN, PNA, GraphSAGE
from .correct_and_smooth import CorrectAndSmooth
from .deep_graph_infomax import DeepGraphInfomax
from .deepgcn import DeepGCNLayer
from .dimenet import DimeNet
from .gnn_explainer import GNNExplainer
from .graph_unet import GraphUNet
from .jumping_knowledge import JumpingKnowledge
from .label_prop import LabelPropagation
from .lightgcn import LightGCN
from .linkx import LINKX
from .metapath2vec import MetaPath2Vec
from .mlp import MLP
from .node2vec import Node2Vec
from .re_net import RENet
from .rect import RECT_L
from .schnet import SchNet
from .signed_gcn import SignedGCN
from .tgn import TGNMemory

__all__ = [
    'MLP',
    'GCN',
    'GraphSAGE',
    'GIN',
    'GAT',
    'PNA',
    'JumpingKnowledge',
    'Node2Vec',
    'DeepGraphInfomax',
    'InnerProductDecoder',
    'GAE',
    'VGAE',
    'ARGA',
    'ARGVA',
    'SignedGCN',
    'RENet',
    'GraphUNet',
    'SchNet',
    'DimeNet',
    'GNNExplainer',
    'MetaPath2Vec',
    'DeepGCNLayer',
    'TGNMemory',
    'LabelPropagation',
    'CorrectAndSmooth',
    'AttentiveFP',
    'RECT_L',
    'LINKX',
    'LightGCN',
]

classes = __all__
