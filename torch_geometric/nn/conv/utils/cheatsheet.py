import importlib
import inspect
import re
from typing import Optional


def paper_title(cls: str) -> Optional[str]:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    match = re.search('`\".+?\"', inspect.getdoc(cls), flags=re.DOTALL)
    return None if match is None else match.group().replace('\n', ' ')[2:-1]


def paper_link(cls: str) -> Optional[str]:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    match = re.search('<.+?>', inspect.getdoc(cls), flags=re.DOTALL)
    return None if match is None else match.group().replace('\n', ' ')[1:-1]


def supports_sparse_tensor(cls: str) -> bool:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    signature = inspect.signature(cls.forward)
    return 'SparseTensor' in str(signature)


def supports_edge_weights(cls: str) -> bool:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    signature = inspect.signature(cls.forward)
    return 'edge_weight' in str(signature)


def supports_edge_features(cls: str) -> bool:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    signature = inspect.signature(cls.forward)
    return 'edge_attr' in str(signature)


def supports_bipartite_graphs(cls: str) -> bool:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    signature = inspect.signature(cls.forward)
    return 'Union[torch.Tensor, Tuple[torch.Tensor' in str(signature)


def supports_static_graphs(cls: str) -> bool:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    return 'node_dim=' not in inspect.getsource(cls.__init__)


def supports_lazy_initialization(cls: str) -> bool:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    doc = re.sub(' +', ' ', inspect.getdoc(cls).replace('\n', ' '))
    match = re.search('or :obj:`-1` to derive the size from the first', doc)
    return match is not None


def processes_heterogeneous_graphs(cls: str) -> bool:
    if 'hetero' in cls.lower():
        return True
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    signature = inspect.signature(cls.forward)
    return 'edge_index_dict' in str(signature) or 'edge_type' in str(signature)


def processes_hypergraphs(cls: str) -> bool:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    signature = inspect.signature(cls.forward)
    return 'hyperedge_index' in str(signature)


def processes_point_clouds(cls: str) -> bool:
    cls = importlib.import_module('torch_geometric.nn.conv').__dict__[cls]
    signature = inspect.signature(cls.forward)
    return (('edge_index' not in str(signature)
             and 'csc' not in str(signature)) or 'pos' in str(signature))
