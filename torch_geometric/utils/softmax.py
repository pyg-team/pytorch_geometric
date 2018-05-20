from torch_scatter import scatter_add

from .num_nodes import maybe_num_nodes


def softmax(src, index, num_nodes=None):
    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src.exp()
    out = out / scatter_add(out, index, dim=0, dim_size=num_nodes)[index]

    return out
