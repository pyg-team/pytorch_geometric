# flake8: noqa

from .random_jitter import RandomJitter

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
