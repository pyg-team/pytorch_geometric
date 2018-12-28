from torch_scatter import scatter_max, scatter_add

from .num_nodes import maybe_num_nodes


def softmax(src, index, num_nodes=None):
    r"""Sparse softmax of all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): Automatically create output tensor with size
            :attr:`num_nodes` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_geometric.utils import softmax
        src = torch.Tensor([2, 3, -2, 1, 1])
        index = torch.tensor([0, 1, 0, 1, 2])
        out = softmax(src, index)
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / scatter_add(out, index, dim=0, dim_size=num_nodes)[index]

    return out
