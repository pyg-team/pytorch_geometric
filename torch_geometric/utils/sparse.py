from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce


def dense_to_sparse(adj: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (Tensor): The dense adjacency matrix of shape
            :obj:`[num_nodes, num_nodes]` or
            :obj:`[batch_size, num_nodes, num_nodes]`.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:

        >>> # Forr a single adjacency matrix
        >>> adj = torch.tensor([[3, 1],
        ...                     [2, 0]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1],
                [0, 1, 0]]),
        tensor([3, 1, 2]))

        >>> # For two adjacency matrixes
        >>> adj = torch.tensor([[[3, 1],
        ...                      [2, 0]],
        ...                     [[0, 1],
        ...                      [0, 2]]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1, 2, 3],
                [0, 1, 0, 3, 3]]),
        tensor([3, 1, 2, 1, 2]))
    """
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be 2- or "
                         f"3-dimensional (got {adj.dim()} dimensions)")

    edge_index = adj.nonzero().t()

    if edge_index.size(0) == 2:
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        edge_attr = adj[edge_index[0], edge_index[1], edge_index[2]]
        row = edge_index[1] + adj.size(-2) * edge_index[0]
        col = edge_index[2] + adj.size(-1) * edge_index[0]
        return torch.stack([row, col], dim=0), edge_attr


def is_torch_sparse_tensor(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is a
    :class:`torch.sparse.Tensor` (in any sparse layout).

    Args:
        src (Any): The input object to be checked.
    """
    if isinstance(src, Tensor):
        if src.layout == torch.sparse_coo:
            return True
        if src.layout == torch.sparse_csr:
            return True
        if src.layout == torch.sparse_csc:
            return True
    return False


def is_sparse(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is of type
    :class:`torch.sparse.Tensor` (in any sparse layout) or of type
    :class:`torch_sparse.SparseTensor`.

    Args:
        src (Any): The input object to be checked.
    """
    return is_torch_sparse_tensor(src) or isinstance(src, SparseTensor)


def to_torch_coo_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    is_coalesced: bool = False,
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`torch.sparse.Tensor` with layout
    `torch.sparse_coo`.
    See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)

    :rtype: :class:`torch.sparse.Tensor`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_torch_coo_tensor(edge_index)
        tensor(indices=tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(4, 4), nnz=6, layout=torch.sparse_coo)

    """
    if size is None:
        size = int(edge_index.max()) + 1
    if not isinstance(size, (tuple, list)):
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size))

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_attr,
        size=tuple(size) + edge_attr.size()[1:],
        device=edge_index.device,
    )
    adj = adj._coalesced_(True)

    return adj


def to_torch_csr_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    is_coalesced: bool = False,
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`torch.sparse.Tensor` with layout
    `torch.sparse_csr`.
    See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)

    :rtype: :class:`torch.sparse.Tensor`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_torch_csr_tensor(edge_index)
        tensor(crow_indices=tensor([0, 1, 3, 5, 6]),
               col_indices=tensor([1, 0, 2, 1, 3, 2]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(4, 4), nnz=6, layout=torch.sparse_csr)

    """
    adj = to_torch_coo_tensor(edge_index, edge_attr, size, is_coalesced)
    return adj.to_sparse_csr()


def to_torch_csc_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    is_coalesced: bool = False,
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`torch.sparse.Tensor` with layout
    `torch.sparse_csc`.
    See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)

    :rtype: :class:`torch.sparse.Tensor`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_torch_csc_tensor(edge_index)
        tensor(ccol_indices=tensor([0, 1, 3, 5, 6]),
               row_indices=tensor([1, 0, 2, 1, 3, 2]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(4, 4), nnz=6, layout=torch.sparse_csc)

    """
    if size is None:
        size = int(edge_index.max()) + 1
    if not isinstance(size, (tuple, list)):
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size),
                                         sort_by_row=False)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    adj = torch.sparse_csc_tensor(
        ccol_indices=index2ptr(edge_index[1], size[1]),
        row_indices=edge_index[0],
        values=edge_attr,
        size=tuple(size) + edge_attr.size()[1:],
        device=edge_index.device,
    )

    return adj


def to_torch_sparse_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    is_coalesced: bool = False,
    layout: torch.layout = torch.sparse_coo,
):
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`torch.sparse.Tensor` with custom :obj:`layout`.
    See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)
        layout (torch.layout, optional): The layout of the output sparse tensor
            (:obj:`torch.sparse_coo`, :obj:`torch.sparse_csr`,
            :obj:`torch.sparse_csc`). (default: :obj:`torch.sparse_coo`)

    :rtype: :class:`torch.sparse.Tensor`
    """
    if layout == torch.sparse_coo:
        return to_torch_coo_tensor(edge_index, edge_attr, size, is_coalesced)
    if layout == torch.sparse_csr:
        return to_torch_csr_tensor(edge_index, edge_attr, size, is_coalesced)
    if layout == torch.sparse_csc:
        return to_torch_csc_tensor(edge_index, edge_attr, size, is_coalesced)

    raise ValueError(f"Unexpected sparse tensor layout (got '{layout}')")


def to_edge_index(adj: Union[Tensor, SparseTensor]) -> Tuple[Tensor, Tensor]:
    r"""Converts a :class:`torch.sparse.Tensor` or a
    :class:`torch_sparse.SparseTensor` to edge indices and edge attributes.

    Args:
        adj (torch.sparse.Tensor or SparseTensor): The adjacency matrix.

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> adj = to_torch_coo_tensor(edge_index)
        >>> to_edge_index(adj)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([1., 1., 1., 1., 1., 1.]))
    """
    if isinstance(adj, SparseTensor):
        row, col, value = adj.coo()
        if value is None:
            value = torch.ones(row.size(0), device=row.device)
        return torch.stack([row, col], dim=0).long(), value

    if adj.layout == torch.sparse_coo:
        return adj.indices().detach().long(), adj.values()

    if adj.layout == torch.sparse_csr:
        row = ptr2index(adj.crow_indices().detach())
        col = adj.col_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    if adj.layout == torch.sparse_csc:
        col = ptr2index(adj.ccol_indices().detach())
        row = adj.row_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    raise ValueError(f"Unexpected sparse tensor layout (got '{adj.layout}')")


# Helper functions ############################################################


def get_sparse_diag(
    size: int,
    fill_value: float = 1.0,
    layout: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    return torch.sparse.spdiags(
        torch.full((1, size), fill_value, dtype=dtype, device=device),
        offsets=torch.zeros(1, dtype=torch.long, device=device),
        shape=(size, size),
        layout=layout,
    )


def set_sparse_value(adj: Tensor, value: Tensor) -> Tensor:
    size = adj.size()

    if value.dim() > 1:
        size = size + value.size()[1:]

    if adj.layout == torch.sparse_coo:
        return torch.sparse_coo_tensor(
            indices=adj.indices(),
            values=value,
            size=size,
            device=value.device,
        ).coalesce()

    if adj.layout == torch.sparse_csr:
        return torch.sparse_csr_tensor(
            crow_indices=adj.crow_indices(),
            col_indices=adj.col_indices(),
            values=value,
            size=size,
            device=value.device,
        )

    if adj.layout == torch.sparse_csc:
        return torch.sparse_csc_tensor(
            ccol_indices=adj.ccol_indices(),
            row_indices=adj.row_indices(),
            values=value,
            size=size,
            device=value.device,
        )

    raise ValueError(f"Unexpected sparse tensor layout (got '{adj.layout}')")


def ptr2index(ptr: Tensor) -> Tensor:
    ind = torch.arange(ptr.numel() - 1, dtype=ptr.dtype, device=ptr.device)
    return ind.repeat_interleave(ptr[1:] - ptr[:-1])


def index2ptr(index: Tensor, size: int) -> Tensor:
    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype == torch.int32)


def cat(tensors: List[Tensor], dim: Union[int, Tuple[int, int]]) -> Tensor:
    # TODO (matthias) We can make this more efficient by directly operating on
    # the individual sparse tensor layouts.
    assert dim in {0, 1, (0, 1)}

    size = [0, 0]
    edge_indices = []
    edge_attrs = []
    for tensor in tensors:
        assert is_torch_sparse_tensor(tensor)
        edge_index, edge_attr = to_edge_index(tensor)
        edge_index = edge_index.clone()

        if dim == 0:
            edge_index[0] += size[0]
            size[0] += tensor.size(0)
            size[1] = max(size[1], tensor.size(1))
        elif dim == 1:
            edge_index[1] += size[1]
            size[0] = max(size[0], tensor.size(0))
            size[1] += tensor.size(1)
        else:
            edge_index[0] += size[0]
            edge_index[1] += size[1]
            size[0] += tensor.size(0)
            size[1] += tensor.size(1)

        edge_indices.append(edge_index)
        edge_attrs.append(edge_attr)

    return to_torch_sparse_tensor(
        edge_index=torch.cat(edge_indices, dim=1),
        edge_attr=torch.cat(edge_attrs, dim=0),
        size=size,
        is_coalesced=dim == (0, 1),
        layout=tensors[0].layout,
    )
