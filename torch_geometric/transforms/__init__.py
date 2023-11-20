# flake8: noqa

from .base_transform import BaseTransform
from .compose import Compose, ComposeFilters
from .to_device import ToDevice
from .to_sparse_tensor import ToSparseTensor
from .constant import Constant
from .normalize_features import NormalizeFeatures
from .svd_feature_reduction import SVDFeatureReduction
from .remove_training_classes import RemoveTrainingClasses
from .random_node_split import RandomNodeSplit
from .random_link_split import RandomLinkSplit
from .node_property_split import NodePropertySplit
from .mask import IndexToMask, MaskToIndex
from .pad import Pad

from .to_undirected import ToUndirected
from .one_hot_degree import OneHotDegree
from .target_indegree import TargetIndegree
from .local_degree_profile import LocalDegreeProfile
from .add_self_loops import AddSelfLoops
from .add_remaining_self_loops import AddRemainingSelfLoops
from .remove_isolated_nodes import RemoveIsolatedNodes
from .remove_duplicated_edges import RemoveDuplicatedEdges
from .knn_graph import KNNGraph
from .radius_graph import RadiusGraph
from .to_dense import ToDense
from .two_hop import TwoHop
from .line_graph import LineGraph
from .laplacian_lambda_max import LaplacianLambdaMax
from .gdc import GDC
from .sign import SIGN
from .gcn_norm import GCNNorm
from .add_metapaths import AddMetaPaths, AddRandomMetaPaths
from .rooted_subgraph import RootedEgoNets, RootedRWSubgraph
from .largest_connected_components import LargestConnectedComponents
from .virtual_node import VirtualNode
from .add_positional_encoding import AddLaplacianEigenvectorPE, AddRandomWalkPE
from .feature_propagation import FeaturePropagation
from .half_hop import HalfHop

from .distance import Distance
from .cartesian import Cartesian
from .local_cartesian import LocalCartesian
from .polar import Polar
from .spherical import Spherical
from .point_pair_features import PointPairFeatures
from .center import Center
from .normalize_rotation import NormalizeRotation
from .normalize_scale import NormalizeScale
from .random_jitter import RandomJitter
from .random_flip import RandomFlip
from .linear_transformation import LinearTransformation
from .random_scale import RandomScale
from .random_rotate import RandomRotate
from .random_shear import RandomShear
from .face_to_edge import FaceToEdge
from .sample_points import SamplePoints
from .fixed_points import FixedPoints
from .generate_mesh_normals import GenerateMeshNormals
from .delaunay import Delaunay
from .to_superpixels import ToSLIC
from .grid_sampling import GridSampling

general_transforms = [
    'BaseTransform',
    'Compose',
    'ComposeFilters',
    'ToDevice',
    'ToSparseTensor',
    'Constant',
    'NormalizeFeatures',
    'SVDFeatureReduction',
    'RemoveTrainingClasses',
    'RandomNodeSplit',
    'RandomLinkSplit',
    'NodePropertySplit',
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
    'AddRemainingSelfLoops',
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
    'HalfHop',
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
