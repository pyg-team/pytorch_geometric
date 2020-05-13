from typing import Optional


def maybe_num_nodes(index, num_nodes: Optional[int] = None):
    out: int = 0
    if num_nodes is None:
        out = index.max().item() + 1
    else:
        assert num_nodes is not None
        out = num_nodes
    return out
