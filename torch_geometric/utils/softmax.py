from torch_scatter import scatter_max, scatter_add

from .num_nodes import maybe_num_nodes


def softmax(src, index, num_nodes=None):
    r"""Sparse softmax of all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out
