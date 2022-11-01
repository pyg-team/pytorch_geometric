import torch


def to_csr_csc(edge_index):
    '''
    Generate csr and csc tensor based on the coordinates of 
    the non-zero numbers of the sparse matrix

    input: edge_index,the coordinates of the non-zero numbers of the sparse matrix
    return: (row_ptr, col_idx, col_ptr, row_idx, permute)
    row_ptr: crow_indices in csr
    col_idx: col_indices in csr
    col_ptr: crow_indices in csc
    row_idx: col_indices in csc
    premute: sort order / permute matrix
    '''
    # print(edge_index.device)
    numlist = torch.arange(edge_index.size(1), dtype=torch.int32,
                           device=edge_index.device)
    coo = torch.sparse_coo_tensor(edge_index, numlist).coalesce()
    coo_T = torch.transpose(coo, 0, 1)
    csr = coo.to_sparse_csr()  # .coalesce()
    csc = coo_T.to_sparse_csr()  # .coalesce()
    row_ptr = csr.crow_indices().int()
    col_idx = csr.col_indices().int()
    col_ptr = csc.crow_indices().int()
    row_idx = csc.col_indices().int()
    permute = coo_T.coalesce().values().int()
    csr_csc = (row_ptr, col_idx, col_ptr, row_idx, permute)
    return csr_csc
