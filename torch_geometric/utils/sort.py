from .num_nodes import maybe_num_nodes


def sort_edge_index(edge_index, edge_attr=None, num_nodes=None):
    r"""Row-wise sorts edge indices :obj:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index[0] * num_nodes + edge_index[1]
    perm = idx.argsort()

    return edge_index[:, perm], None if edge_attr is None else edge_attr[perm]
