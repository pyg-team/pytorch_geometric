# flake8: noqa

from .add_metapaths import AddMetaPaths, AddRandomMetaPaths
from .add_positional_encoding import AddLaplacianEigenvectorPE, AddRandomWalkPE
from .add_self_loops import AddSelfLoops
from .base_transform import BaseTransform
from .cartesian import Cartesian
from .center import Center
from .compose import Compose
from .constant import Constant
from .delaunay import Delaunay
from .distance import Distance
from .face_to_edge import FaceToEdge
from .feature_propagation import FeaturePropagation
from .fixed_points import FixedPoints
from .gcn_norm import GCNNorm
from .gdc import GDC
from .generate_mesh_normals import GenerateMeshNormals
from .grid_sampling import GridSampling
from .knn_graph import KNNGraph
from .laplacian_lambda_max import LaplacianLambdaMax
from .largest_connected_components import LargestConnectedComponents
from .line_graph import LineGraph
from .linear_transformation import LinearTransformation
from .local_cartesian import LocalCartesian
from .local_degree_profile import LocalDegreeProfile
from .mask import IndexToMask, MaskToIndex
from .normalize_features import NormalizeFeatures
from .normalize_rotation import NormalizeRotation
from .normalize_scale import NormalizeScale
from .one_hot_degree import OneHotDegree
from .pad import Pad
from .point_pair_features import PointPairFeatures
from .polar import Polar
from .radius_graph import RadiusGraph
from .random_flip import RandomFlip
from .random_jitter import RandomJitter
from .random_link_split import RandomLinkSplit
from .random_node_split import RandomNodeSplit
from .random_rotate import RandomRotate
from .random_scale import RandomScale
from .random_shear import RandomShear
from .remove_duplicated_edges import RemoveDuplicatedEdges
from .remove_isolated_nodes import RemoveIsolatedNodes
from .remove_training_classes import RemoveTrainingClasses
from .rooted_subgraph import RootedEgoNets, RootedRWSubgraph
from .sample_points import SamplePoints
from .sign import SIGN
from .spherical import Spherical
from .svd_feature_reduction import SVDFeatureReduction
from .target_indegree import TargetIndegree
from .to_dense import ToDense
from .to_device import ToDevice
from .to_sparse_tensor import ToSparseTensor
from .to_superpixels import ToSLIC
from .to_undirected import ToUndirected
from .two_hop import TwoHop
from .virtual_node import VirtualNode

general_transforms = [
    'BaseTransform',
    'Compose',
    'ToDevice',
    'ToSparseTensor',
    'Constant',
    'NormalizeFeatures',
    'SVDFeatureReduction',
    'RemoveTrainingClasses',
    'RandomNodeSplit',
    'RandomLinkSplit',
    'IndexToMask',
    'MaskToIndex',
    'Pad',
]

graph_transforms = [
    'ToUndirected',
    'OneHotDegree',
    'TargetIndegree',
    'LocalDegreeProfile',
    'AddSelfLoops',
    'RemoveIsolatedNodes',
    'RemoveDuplicatedEdges',
    'KNNGraph',
    'RadiusGraph',
    'ToDense',
    'TwoHop',
    'LineGraph',
    'LaplacianLambdaMax',
    'GDC',
    'SIGN',
    'GCNNorm',
    'AddMetaPaths',
    'AddRandomMetaPaths',
    'RootedEgoNets',
    'RootedRWSubgraph',
    'LargestConnectedComponents',
    'VirtualNode',
    'AddLaplacianEigenvectorPE',
    'AddRandomWalkPE',
    'FeaturePropagation',
]

vision_transforms = [
    'Distance',
    'Cartesian',
    'LocalCartesian',
    'Polar',
    'Spherical',
    'PointPairFeatures',
    'Center',
    'NormalizeRotation',
    'NormalizeScale',
    'RandomJitter',
    'RandomFlip',
    'LinearTransformation',
    'RandomScale',
    'RandomRotate',
    'RandomShear',
    'FaceToEdge',
    'SamplePoints',
    'FixedPoints',
    'GenerateMeshNormals',
    'Delaunay',
    'ToSLIC',
    'GridSampling',
]

__all__ = general_transforms + graph_transforms + vision_transforms

from torch_geometric.deprecation import deprecated

RandomTranslate = deprecated("use 'transforms.RandomJitter' instead",
                             'transforms.RandomTranslate')(RandomJitter)
