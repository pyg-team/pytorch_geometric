from torch_scatter import scatter_add


def matmul(index, value, tensor):
    tensor = tensor if tensor.dim() > 1 else tensor.unsqueeze(-1)
    assert value.dim() == 1 and tensor.dim() == 2

    row, col = index
    out_col = value.unsqueeze(-1) * tensor[col]
    out = scatter_add(out_col, row, dim=0, dim_size=tensor.size(0))

    return out
