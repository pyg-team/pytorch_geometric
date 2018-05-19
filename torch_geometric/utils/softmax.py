from torch_scatter import scatter_add
from torch_geometric.utils import get_num_nodes


def softmax(src, index, num_nodes=None):
    num_nodes = get_num_nodes(index, num_nodes)

    out = src.exp()
    out = out / scatter_add(out, index, dim=0, dim_size=num_nodes)[index]

    return out
