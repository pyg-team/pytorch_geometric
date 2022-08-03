from .base_transform import BaseTransform
from .compose import Compose
from .to_device import ToDevice
from .to_sparse_tensor import ToSparseTensor
from .to_undirected import ToUndirected
from .constant import Constant
from .distance import Distance
from .cartesian import Cartesian
from .local_cartesian import LocalCartesian
from .polar import Polar
from .spherical import Spherical
from .point_pair_features import PointPairFeatures
from .one_hot_degree import OneHotDegree
from .target_indegree import TargetIndegree
from .local_degree_profile import LocalDegreeProfile
from .center import Center
from .normalize_rotation import NormalizeRotation
from .normalize_scale import NormalizeScale
from .random_jitter import RandomJitter
from .random_flip import RandomFlip
from .linear_transformation import LinearTransformation
from .random_scale import RandomScale
from .random_rotate import RandomRotate
from .random_shear import RandomShear
from .normalize_features import NormalizeFeatures
from .add_self_loops import AddSelfLoops
from .remove_isolated_nodes import RemoveIsolatedNodes
from .knn_graph import KNNGraph
from .radius_graph import RadiusGraph
from .face_to_edge import FaceToEdge
from .sample_points import SamplePoints
from .fixed_points import FixedPoints
from .to_dense import ToDense
from .two_hop import TwoHop
from .line_graph import LineGraph
from .laplacian_lambda_max import LaplacianLambdaMax
from .generate_mesh_normals import GenerateMeshNormals
from .delaunay import Delaunay
from .to_superpixels import ToSLIC
from .gdc import GDC
from .sign import SIGN
from .grid_sampling import GridSampling
from .gcn_norm import GCNNorm
from .svd_feature_reduction import SVDFeatureReduction
from .remove_training_classes import RemoveTrainingClasses
from .random_node_split import RandomNodeSplit
from .random_link_split import RandomLinkSplit
from .add_metapaths import AddMetaPaths
from .rooted_subgraph import RootedEgoNets, RootedRWSubgraph
from .largest_connected_components import LargestConnectedComponents
from .virtual_node import VirtualNode
from .add_positional_encoding import AddLaplacianEigenvectorPE, AddRandomWalkPE

__all__ = [
    'BaseTransform',
    'Compose',
    'ToDevice',
    'ToSparseTensor',
    'ToUndirected',
    'Constant',
    'Distance',
    'Cartesian',
    'LocalCartesian',
    'Polar',
    'Spherical',
    'PointPairFeatures',
    'OneHotDegree',
    'TargetIndegree',
    'LocalDegreeProfile',
    'Center',
    'NormalizeRotation',
    'NormalizeScale',
    'RandomJitter',
    'RandomFlip',
    'LinearTransformation',
    'RandomScale',
    'RandomRotate',
    'RandomShear',
    'NormalizeFeatures',
    'AddSelfLoops',
    'RemoveIsolatedNodes',
    'KNNGraph',
    'RadiusGraph',
    'FaceToEdge',
    'SamplePoints',
    'FixedPoints',
    'ToDense',
    'TwoHop',
    'LineGraph',
    'LaplacianLambdaMax',
    'GenerateMeshNormals',
    'Delaunay',
    'ToSLIC',
    'GDC',
    'SIGN',
    'GridSampling',
    'GCNNorm',
    'SVDFeatureReduction',
    'RemoveTrainingClasses',
    'RandomNodeSplit',
    'RandomLinkSplit',
    'AddMetaPaths',
    'RootedEgoNets',
    'RootedRWSubgraph',
    'LargestConnectedComponents',
    'VirtualNode',
    'AddLaplacianEigenvectorPE',
    'AddRandomWalkPE',
]

classes = __all__

from torch_geometric.deprecation import deprecated  # noqa

RandomTranslate = deprecated("use 'transforms.RandomJitter' instead",
                             'transforms.RandomTranslate')(RandomJitter)
