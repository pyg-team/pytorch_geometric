import scipy.sparse as sp
import torch


def to_dgnn(edge_index):
    col, row = edge_index.cpu()
    numlist = torch.arange(col.size(0), dtype=torch.int32)
    num_nodes = torch.max(col) - torch.min(col) + 1

    adj_csr = sp.csr_matrix((numlist.numpy(), (row, col)),
                            shape=(num_nodes, num_nodes))

    row_ptr = torch.from_numpy(adj_csr.indptr)
    col_idx = torch.from_numpy(adj_csr.indices)

    adj_csc = adj_csr.tocsc()

    col_ptr = torch.from_numpy(adj_csc.indptr)
    row_idx = torch.from_numpy(adj_csc.indices)

    adj_csr_new = sp.csr_matrix(
        (numlist.numpy(), col_idx.cpu().numpy(), row_ptr.cpu().numpy()))
    adj_csc_new = adj_csr_new.tocsc()
    permute = torch.from_numpy(adj_csc_new.data)

    dgnn_adj = (row_ptr.cuda(), col_idx.cuda(), col_ptr.cuda(), row_idx.cuda(),
                permute.cuda())

    return dgnn_adj


def compute_all_graph_representations(edge_index):
    col, row = edge_index
    numlist = torch.arange(col.size(0), dtype=torch.int32, device=col.device)
    coo = torch.sparse_coo_tensor([row, col], numlist).coalesce()
    coo_T = coo.transpose()
    csr = coo.to_sparse_csr().coalesce()
    csc = coo_T.to_sparse_csr().coalesce()
    row_ptr = csr.crow_indices()
    col_idx = csr.col_indices()
    col_ptr = csc.crow_indices()
    row_idx = csc.col_indices()
    permute = coo_T.values()
    all_graph_representations = (row_ptr, col_idx, col_ptr, row_idx, permute)
    return all_graph_representations
