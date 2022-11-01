import torch


def to_csr_csc(edge_index):
    r"""Generate csr and csc tensor based on the coordinates: obj: `edge_index` of the non-zero
    numbers of the sparse matrix

    The method returns the csr and csc tensor : obj: `csr_csr` (tuple)
        (1) row_ptr: crow_indices in csr
        (2) col_idx: col_indices in csr
        (3) col_ptr: crow_indices in csc
        (4) row_idx: col_indices in csc
        (5) premute: sort order / permute matrix

    Arg:
        edge_index(list or : obj:`torch.tensor`): The coordinates of the non-zero
        numbers of the sparse matrix

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
        >>> # (0,1),(1,0) (0,0) are the non-zero numbers
        >>> dgnn_adj = to_csr_csc(edge_index)
        >>> dgnn_adj
        ([0,2,3],[0, 1, 0],[0, 2, 3],[0, 1, 0],[2, 1, 0])
    """
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
