import copy

from .assortativity import assortativity
from .augmentation import add_random_edge, mask_feature, shuffle_node
from .coalesce import coalesce
from .convert import (
    from_cugraph,
    from_dgl,
    from_networkit,
    from_networkx,
    from_scipy_sparse_matrix,
    from_trimesh,
    to_cugraph,
    to_dgl,
    to_networkit,
    to_networkx,
    to_scipy_sparse_matrix,
    to_trimesh,
)
from .degree import degree
from .dropout import dropout_adj, dropout_edge, dropout_node, dropout_path
from .embedding import get_embeddings
from .geodesic import geodesic_distance
from .get_laplacian import get_laplacian
from .get_mesh_laplacian import get_mesh_laplacian
from .grid import grid
from .homophily import homophily
from .isolated import contains_isolated_nodes, remove_isolated_nodes
from .loop import (
    add_remaining_self_loops,
    add_self_loops,
    contains_self_loops,
    get_self_loop_attr,
    remove_self_loops,
    segregate_self_loops,
)
from .mask import index_to_mask, mask_select, mask_to_index
from .negative_sampling import (
    batched_negative_sampling,
    negative_sampling,
    structured_negative_sampling,
    structured_negative_sampling_feasible,
)
from .nested import from_nested_tensor, to_nested_tensor
from .normalized_cut import normalized_cut
from .one_hot import one_hot
from .random import (
    barabasi_albert_graph,
    erdos_renyi_graph,
    stochastic_blockmodel_graph,
)
from .scatter import scatter
from .segment import segment
from .select import narrow, select
from .smiles import from_smiles, to_smiles
from .softmax import softmax
from .sort import index_sort
from .sort_edge_index import sort_edge_index
from .sparse import (
    dense_to_sparse,
    is_sparse,
    is_torch_sparse_tensor,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csc_tensor,
    to_torch_csr_tensor,
)
from .spmm import spmm
from .subgraph import bipartite_subgraph, get_num_hops, k_hop_subgraph, subgraph
from .to_dense_adj import to_dense_adj
from .to_dense_batch import to_dense_batch
from .train_test_split_edges import train_test_split_edges
from .tree_decomposition import tree_decomposition
from .trim_to_layer import trim_to_layer
from .unbatch import unbatch, unbatch_edge_index
from .undirected import is_undirected, to_undirected

__all__ = [
    'scatter',
    'segment',
    'index_sort',
    'degree',
    'softmax',
    'dropout_node',
    'dropout_edge',
    'dropout_path',
    'dropout_adj',
    'shuffle_node',
    'mask_feature',
    'add_random_edge',
    'sort_edge_index',
    'coalesce',
    'is_undirected',
    'to_undirected',
    'contains_self_loops',
    'remove_self_loops',
    'segregate_self_loops',
    'add_self_loops',
    'add_remaining_self_loops',
    'get_self_loop_attr',
    'contains_isolated_nodes',
    'remove_isolated_nodes',
    'get_num_hops',
    'subgraph',
    'bipartite_subgraph',
    'k_hop_subgraph',
    'homophily',
    'assortativity',
    'get_laplacian',
    'get_mesh_laplacian',
    'mask_select',
    'index_to_mask',
    'mask_to_index',
    'select',
    'narrow',
    'to_dense_batch',
    'to_dense_adj',
    'to_nested_tensor',
    'from_nested_tensor',
    'dense_to_sparse',
    'is_torch_sparse_tensor',
    'is_sparse',
    'to_torch_coo_tensor',
    'to_torch_csr_tensor',
    'to_torch_csc_tensor',
    'to_edge_index',
    'spmm',
    'unbatch',
    'unbatch_edge_index',
    'one_hot',
    'normalized_cut',
    'grid',
    'geodesic_distance',
    'to_scipy_sparse_matrix',
    'from_scipy_sparse_matrix',
    'to_networkx',
    'from_networkx',
    'to_networkit',
    'from_networkit',
    'to_trimesh',
    'from_trimesh',
    'to_cugraph',
    'from_cugraph',
    'from_smiles',
    'to_smiles',
    'erdos_renyi_graph',
    'stochastic_blockmodel_graph',
    'barabasi_albert_graph',
    'negative_sampling',
    'batched_negative_sampling',
    'structured_negative_sampling',
    'structured_negative_sampling_feasible',
    'tree_decomposition',
    'get_embeddings',
    'trim_to_layer',
    'train_test_split_edges',
    'from_dgl',
    'to_dgl',
]

# `structured_negative_sampling_feasible` is a long name and thus destroys the
# documentation rendering. We remove it for now from the documentation:
classes = copy.copy(__all__)
classes.remove('structured_negative_sampling_feasible')
