from torch_scatter import scatter_add


def softmax(src, index, num_nodes=None):
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = src.exp()
    out = out / scatter_add(out, index, dim=0, dim_size=num_nodes)[index]
    return out
