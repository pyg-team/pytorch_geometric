r"""Model package."""

# Deprecated:
from torch_geometric.explain.algorithm.captum import (captum_output_to_dicts,
                                                      to_captum_input)

from .attentive_fp import AttentiveFP
from .attract_repel import ARLinkPredictor
from .autoencoder import ARGA, ARGVA, GAE, VGAE, InnerProductDecoder
from .basic_gnn import GAT, GCN, GIN, PNA, EdgeCNN, GraphSAGE
from .captum import to_captum_model
from .correct_and_smooth import CorrectAndSmooth
from .deep_graph_infomax import DeepGraphInfomax
from .deepgcn import DeepGCNLayer
from .dimenet import DimeNet, DimeNetPlusPlus
from .g_retriever import GRetriever
from .git_mol import GITMol
from .glem import GLEM
from .gnnff import GNNFF
from .gpse import GPSE, GPSENodeEncoder
from .graph_unet import GraphUNet
from .jumping_knowledge import HeteroJumpingKnowledge, JumpingKnowledge
from .label_prop import LabelPropagation
from .lightgcn import LightGCN
from .linkx import LINKX
from .mask_label import MaskLabel
from .meta import MetaLayer
from .metapath2vec import MetaPath2Vec
from .mlp import MLP
from .molecule_gpt import MoleculeGPT
from .neural_fingerprint import NeuralFingerprint
from .node2vec import Node2Vec
from .pmlp import PMLP
from .polynormer import Polynormer
from .protein_mpnn import ProteinMPNN
from .re_net import RENet
from .rect import RECT_L
from .rev_gnn import GroupAddRev
from .schnet import SchNet
from .sgformer import SGFormer
from .signed_gcn import SignedGCN
from .tgn import TGNMemory
from .visnet import ViSNet

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
    'GPSE',
    'GPSENodeEncoder',
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
    'GITMol',
    'MoleculeGPT',
    'ProteinMPNN',
    'GLEM',
    'SGFormer',
    'Polynormer',
    'ARLinkPredictor',
]
