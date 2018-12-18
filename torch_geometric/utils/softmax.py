from torch_scatter import scatter_max, scatter_add

from .num_nodes import maybe_num_nodes


def softmax(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / scatter_add(out, index, dim=0, dim_size=num_nodes)[index]

    return out
