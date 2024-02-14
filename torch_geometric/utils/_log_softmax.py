import torch

from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes


def log_softmax(
    src: torch.Tensor,
    index: torch.Optional[torch.Tensor] = None,
    ptr: torch.Optional[torch.Tensor] = None,
    num_nodes: torch.Optional[int] = None,
    dim: int = 0,
) -> torch.Tensor:
    r"""Computes a sparsely evaluated log_softmax.

    Given a value tensor :attr:`src`, this function first groups the values
    along the specified dimension based on the indices specified in :attr:`index`
    or sorted inputs in CSR representation given by :attr:`ptr`, and then proceeds
    to compute the log_softmax individually for each group.

    The log_softmax operation is defined as the logarithm of the softmax
    probabilities, which can provide numerical stability improvements over
    separately computing softmax followed by a logarithm.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            log_softmax. When specified, `src` values are grouped by `index` to
            compute the log_softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the log_softmax based on
            sorted inputs in CSR representation. This allows for efficient
            computation over contiguous ranges of nodes. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*, the maximum value
            + 1 of :attr:`index`. This is required when `index` is specified to
            determine the dimension size for scattering operations.
            (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize. For most use
            cases, this should be set to 0, as log_softmax is typically applied
            along the node dimension in graph neural networks. (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:
        >>> src = torch.tensor([1., 2., 3., 4.])
        >>> index = torch.tensor([0, 0, 1, 1])
        >>> ptr = torch.tensor([0, 2, 4])
        >>> log_softmax(src, index)
        tensor([-3.0486, -2.0486, -2.0486, -3.0486])

        >>> log_softmax(src, None, ptr)
        tensor([-3.0486, -2.0486, -2.0486, -3.0486])

        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> log_softmax(src, index, dim=-1)
        tensor([[-1.3130, -0.6931, -0.3130, -1.3130],
                [-1.0408, -0.0408, -0.0408, -1.0408],
                [-0.5514, -0.5514, -0.1542, -0.5514],
                [-0.7520, -0.7520, -0.1542, -0.7520]])
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce="max")
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = src - src_max
        out_exp = out.exp()
        out_sum = segment(out_exp, ptr, reduce="sum") + 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
        log_out_sum = out_sum.log()
        out = out - log_out_sum
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce="max")
        out = src - src_max.index_select(dim, index)
        out_exp = out.exp()
        out_sum = scatter(out_exp, index, dim, dim_size=N,
                          reduce="sum") + 1e-16
        out_sum = out_sum.index_select(dim, index)
        log_out_sum = out_sum.log()
        out = out - log_out_sum
    else:
        raise NotImplementedError

    return out
