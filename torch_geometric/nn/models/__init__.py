r"""Model package."""

from .mlp import MLP
from .basic_gnn import GCN, GraphSAGE, GIN, GAT, PNA, EdgeCNN
from .jumping_knowledge import JumpingKnowledge, HeteroJumpingKnowledge
from .meta import MetaLayer
from .node2vec import Node2Vec
from .deep_graph_infomax import DeepGraphInfomax
from .autoencoder import InnerProductDecoder, GAE, VGAE, ARGA, ARGVA
from .signed_gcn import SignedGCN
from .re_net import RENet
from .graph_unet import GraphUNet
from .schnet import SchNet
from .dimenet import DimeNet, DimeNetPlusPlus
from .captum import to_captum_model
from .metapath2vec import MetaPath2Vec
from .deepgcn import DeepGCNLayer
from .tgn import TGNMemory
from .label_prop import LabelPropagation
from .correct_and_smooth import CorrectAndSmooth
from .attentive_fp import AttentiveFP
from .rect import RECT_L
from .linkx import LINKX
from .lightgcn import LightGCN
from .mask_label import MaskLabel
from .rev_gnn import GroupAddRev
from .gnnff import GNNFF
from .pmlp import PMLP
from .neural_fingerprint import NeuralFingerprint
from .visnet import ViSNet
from .g_retriever import GRetriever

# Deprecated:
from torch_geometric.explain.algorithm.captum import (to_captum_input,
                                                      captum_output_to_dicts)

__all__ = classes = [
    'MLP',
    'GCN',
    'GraphSAGE',
    'GIN',
    'GAT',
    'PNA',
    'EdgeCNN',
    'JumpingKnowledge',
    'HeteroJumpingKnowledge',
    'MetaLayer',
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
    'DimeNetPlusPlus',
    'to_captum_model',
    'to_captum_input',
    'captum_output_to_dicts',
    'MetaPath2Vec',
    'DeepGCNLayer',
    'TGNMemory',
    'LabelPropagation',
    'CorrectAndSmooth',
    'AttentiveFP',
    'RECT_L',
    'LINKX',
    'LightGCN',
    'MaskLabel',
    'GroupAddRev',
    'GNNFF',
    'PMLP',
    'NeuralFingerprint',
    'ViSNet',
    'GRetriever',
]
