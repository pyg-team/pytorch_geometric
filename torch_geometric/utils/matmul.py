from torch_scatter import scatter_add


def matmul(index, value, tensor):
    tensor = tensor if tensor.dim() > 1 else tensor.unsqueeze(-1)
    assert (value is None or value.dim() == 1) and tensor.dim() == 2

    row, col = index
    out_col = tensor[col]
    out_col = out_col if value is None else out_col * value.unsqueeze(-1)
    out = scatter_add(out_col, row, dim=0, dim_size=tensor.size(0))

    return out
