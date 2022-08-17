import scipy.sparse as sp
import torch


def to_dgnn(edge_index):
    device = edge_index.device

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

    dgnn_adj = (row_ptr.to(device), col_idx.to(device), col_ptr.to(device), row_idx.to(device),
                permute.to(device))

    return dgnn_adj
