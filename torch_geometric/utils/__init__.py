r"""Utility package."""

import copy

from ._scatter import scatter, group_argsort, group_cat
from ._segment import segment
from ._index_sort import index_sort
from .functions import cumsum
from ._degree import degree
from ._softmax import softmax
from ._lexsort import lexsort
from ._sort_edge_index import sort_edge_index
from ._coalesce import coalesce
from .undirected import is_undirected, to_undirected
from .loop import (contains_self_loops, remove_self_loops,
                   segregate_self_loops, add_self_loops,
                   add_remaining_self_loops, get_self_loop_attr)
from .isolated import contains_isolated_nodes, remove_isolated_nodes
from ._subgraph import (get_num_hops, subgraph, k_hop_subgraph,
                        bipartite_subgraph)
from .dropout import dropout_adj, dropout_node, dropout_edge, dropout_path
from ._homophily import homophily
from ._assortativity import assortativity
from ._normalize_edge_index import normalize_edge_index
from .laplacian import get_laplacian
from .mesh_laplacian import get_mesh_laplacian
from .mask import mask_select, index_to_mask, mask_to_index
from ._select import select, narrow
from ._to_dense_batch import to_dense_batch
from ._to_dense_adj import to_dense_adj
from .nested import to_nested_tensor, from_nested_tensor
from .sparse import (dense_to_sparse, is_sparse, is_torch_sparse_tensor,
                     to_torch_coo_tensor, to_torch_csr_tensor,
                     to_torch_csc_tensor, to_torch_sparse_tensor,
                     to_edge_index)
from ._spmm import spmm
from ._unbatch import unbatch, unbatch_edge_index
from ._one_hot import one_hot
from ._normalized_cut import normalized_cut
from ._grid import grid
from .geodesic import geodesic_distance
from .convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from .convert import to_networkx, from_networkx
from .convert import to_networkit, from_networkit
from .convert import to_trimesh, from_trimesh
from .convert import to_cugraph, from_cugraph
from .convert import to_dgl, from_dgl
from .smiles import from_rdmol, to_rdmol, from_smiles, to_smiles
from .random import (erdos_renyi_graph, stochastic_blockmodel_graph,
                     barabasi_albert_graph)
from ._negative_sampling import (negative_sampling, batched_negative_sampling,
                                 structured_negative_sampling,
                                 structured_negative_sampling_feasible)
from .augmentation import shuffle_node, mask_feature, add_random_edge
from ._tree_decomposition import tree_decomposition
from .embedding import get_embeddings
from ._trim_to_layer import trim_to_layer
from .ppr import get_ppr
from ._train_test_split_edges import train_test_split_edges

__all__ = [
    'scatter',
    'group_argsort',
    'group_cat',
    'segment',
    'index_sort',
    'cumsum',
    'degree',
    'softmax',
    'lexsort',
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
    'dropout_node',
    'dropout_edge',
    'dropout_path',
    'dropout_adj',
    'homophily',
    'assortativity',
    'normalize_edge_index',
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
    'from_rdmol',
    'to_rdmol',
    'from_smiles',
    'to_smiles',
    'erdos_renyi_graph',
    'stochastic_blockmodel_graph',
    'barabasi_albert_graph',
    'negative_sampling',
    'batched_negative_sampling',
    'structured_negative_sampling',
    'structured_negative_sampling_feasible',
    'shuffle_node',
    'mask_feature',
    'add_random_edge',
    'tree_decomposition',
    'get_embeddings',
    'trim_to_layer',
    'get_ppr',
    'train_test_split_edges',
]

# `structured_negative_sampling_feasible` is a long name and thus destroys the
# documentation rendering. We remove it for now from the documentation:
classes = copy.copy(__all__)
classes.remove('structured_negative_sampling_feasible')
