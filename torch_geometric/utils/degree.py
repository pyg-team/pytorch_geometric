import torch

from .num_nodes import maybe_num_nodes


def degree(index, num_nodes=None, dtype=None, device=None):
    """Computes the degree of a given index tensor.

    Args:
        index (LongTensor): Source or target indices of edges.
        num_nodes (int, optional): The number of nodes in :attr:`index`.
            (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional). The desired data type of returned
            tensor.
        device (:obj:`torch.device`, optional): The desired device of returned
            tensor.

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_geometric.utils import degree
        index = torch.tensor([0, 1, 0, 2, 0])
        output = degree(index)
        print(output)

    .. testoutput::

       tensor([3., 1., 1.])
    """
    num_nodes = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((num_nodes), dtype=dtype, device=device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))
