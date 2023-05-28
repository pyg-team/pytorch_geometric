import copy

from .scatter import scatter
from .segment import segment
from .sort import index_sort
from .degree import degree
from .softmax import softmax
from .dropout import dropout_adj, dropout_node, dropout_edge, dropout_path
from .augmentation import shuffle_node, mask_feature, add_random_edge
from .sort_edge_index import sort_edge_index
from .coalesce import coalesce
from .undirected import is_undirected, to_undirected
from .loop import (contains_self_loops, remove_self_loops,
                   segregate_self_loops, add_self_loops,
                   add_remaining_self_loops, get_self_loop_attr)
from .isolated import contains_isolated_nodes, remove_isolated_nodes
from .subgraph import (get_num_hops, subgraph, k_hop_subgraph,
                       bipartite_subgraph)
from .homophily import homophily
from .assortativity import assortativity
from .get_laplacian import get_laplacian
from .get_mesh_laplacian import get_mesh_laplacian
from .mask import mask_select, index_to_mask, mask_to_index
from .select import select, narrow
from .to_dense_batch import to_dense_batch
from .to_dense_adj import to_dense_adj
from .nested import to_nested_tensor, from_nested_tensor
from .sparse import (dense_to_sparse, is_sparse, is_torch_sparse_tensor,
                     to_torch_coo_tensor, to_torch_csr_tensor,
                     to_torch_csc_tensor, to_torch_sparse_tensor,
                     to_edge_index)
from .spmm import spmm
from .unbatch import unbatch, unbatch_edge_index
from .one_hot import one_hot
from .normalized_cut import normalized_cut
from .grid import grid
from .geodesic import geodesic_distance
from .convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from .convert import to_networkx, from_networkx
from .convert import to_networkit, from_networkit
from .convert import to_trimesh, from_trimesh
from .convert import to_cugraph, from_cugraph
from .convert import to_dgl, from_dgl
from .smiles import from_smiles, to_smiles
from .random import (erdos_renyi_graph, stochastic_blockmodel_graph,
                     barabasi_albert_graph)
from .negative_sampling import (negative_sampling, batched_negative_sampling,
                                structured_negative_sampling,
                                structured_negative_sampling_feasible)
from .tree_decomposition import tree_decomposition
from .embedding import get_embeddings
from .trim_to_layer import trim_to_layer
from .train_test_split_edges import train_test_split_edges

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
    'to_torch_sparse_tensor',
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
    'to_dgl',
    'from_dgl',
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
]

# `structured_negative_sampling_feasible` is a long name and thus destroys the
# documentation rendering. We remove it for now from the documentation:
classes = copy.copy(__all__)
classes.remove('structured_negative_sampling_feasible')
