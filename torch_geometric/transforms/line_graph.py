import torch
from torch_sparse import coalesce
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops
import scipy as sp
import numpy as np
from torch_sparse import SparseTensor



class LineGraph(object):
    r"""Converts a graph to its corresponding line-graph:

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

    Line-graph node indices are equal to indices in the original graph's
    coalesced :obj:`edge_index`.

    New node features are given by old edge attributes.
    For undirected graphs, edge attributes for reciprocal edges
    :obj:`(row, col)` and :obj:`(col, row)` get summed together.

    Args:
        force_directed (bool, optional): If set to :obj:`True`, the graph will
            be always treated as a directed graph. (default: :obj:`False`)
    """
    def __init__(self, force_directed=False):
        self.force_directed = force_directed

    def __call__(self, data):
        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        (row, col), edge_attr = coalesce(edge_index, edge_attr, N, N)

        def sparse_edge_multiply(row, col, N):
            
            # 100x speed-up if using cuda with SparseTensor class
            if row.device.type == 'cuda':          
                e_in_coo = SparseTensor(row=row, col=torch.arange(row.size(0), dtype=torch.long, device=row.device), sparse_sizes=(N+1, row.size(0)))
                e_out_coo = SparseTensor(row=col, col=torch.arange(row.size(0), dtype=torch.long, device=row.device), sparse_sizes=(N+1, row.size(0)))
                
                e_line_graph = e_out_coo.t().matmul(e_in_coo)
            
            # 10x speed-up if using cpu with scipy sparse class
            else:
                e_in_coo = sp.sparse.coo_matrix(([1]*row.size(0), np.vstack([row, np.arange(row.size(0))])), shape=[N+1, row.size(0)])
                e_out_coo = sp.sparse.coo_matrix(([1]*row.size(0), np.vstack([col, np.arange(row.size(0))])), shape=[N+1, row.size(0)])
                e_in_csr, e_out_csr = e_in_coo.tocsr(), e_out_coo.tocsr()
                
                e_total = e_out_csr.T * e_in_csr
                e_line_graph = SparseTensor.from_scipy(e_total)
                
            return e_line_graph
        
        if self.force_directed or data.is_directed():
            
            # Convert to Line Graph
            e_line_graph = sparse_edge_multiply(row, col, N)
            
            row, col, _ = e_line_graph.coo()
            data.edge_index = torch.stack([row, col], dim=0)
            data.x = edge_attr
            data.num_nodes = edge_index.size(1)

        else:
            # Remove self-loops
            self_mask = row!=col
            row, col = row[self_mask], col[self_mask]
            invert_self_mask = torch.where(self_mask)[0] #Used to recover the original edge_list ordering
            
            e_nonzero = sparse_edge_multiply(row, col, N)
            
            trip_row, trip_col, _ = e_nonzero.coo()
            trip_self_mask = row[trip_row] != col[trip_col]
            masked_trip_row, masked_trip_col = trip_row[trip_self_mask], trip_col[trip_self_mask]

            invert_row, invert_col = invert_self_mask[masked_trip_row], invert_self_mask[masked_trip_col]
            
            data.edge_index = torch.stack([invert_row, invert_col], dim=0)
            
            if edge_attr is not None:
                data.x = 2*edge_attr
            data.num_nodes = edge_index.size(1)

        data.edge_attr = None
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
