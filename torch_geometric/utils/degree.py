import torch
from torch.autograd import Variable
from torch_geometric.utils import new


def degree(index, num_nodes=None, out=None):
    """Computes the degree of a given index tensor.

    Args:
        index (LongTensor): Source or target indices of edges
        num_nodes (int, optional): The number of nodes in :obj:`index`
        out (Tensor, optional): The result tensor

    :rtype: :class:`Tensor`

    .. testsetup::

        import torch

    .. testcode::

        from torch_geometric.utils import degree
        index = torch.LongTensor([0, 1, 0, 2, 0])
        output = degree(index)
        print(output)

    .. testoutput::

        3
        1
        1
       [torch.FloatTensor of size 3]
    """

    num_nodes = index.max() + 1 if num_nodes is None else num_nodes
    out = index.new().float() if out is None else out
    index = index if torch.is_tensor(out) else Variable(index)

    if torch.is_tensor(out):
        out.resize_(num_nodes)
    else:
        out.data.resize_(num_nodes)

    one = new(out, index.size(0)).fill_(1)
    return out.fill_(0).scatter_add_(0, index, one)
