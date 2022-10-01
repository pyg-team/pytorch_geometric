from typing import Optional

from torch import Tensor
from torch_scatter import gather_csr, scatter, segment_csr

from .num_nodes import maybe_num_nodes


def softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:

        >>> src = torch.tensor([1., 1., 1., 1.])
        >>> index = torch.tensor([0, 0, 1, 2])
        >>> ptr = torch.tensor([0, 2, 3, 4])
        >>> softmax(src, index)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> softmax(src, None, ptr)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> softmax(src, index, dim=-1)
        tensor([[0.7404, 0.2596, 1.0000, 1.0000],
                [0.1702, 0.8298, 1.0000, 1.0000],
                [0.7607, 0.2393, 1.0000, 1.0000],
                [0.8062, 0.1938, 1.0000, 1.0000]])
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src, index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)
