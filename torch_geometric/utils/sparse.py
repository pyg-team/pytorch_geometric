import typing
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.index import index2ptr, ptr2index
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import coalesce, cumsum


def dense_to_sparse(
    adj: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (torch.Tensor): The dense adjacency matrix of shape
            :obj:`[num_nodes, num_nodes]` or
            :obj:`[batch_size, num_nodes, num_nodes]`.
        mask (torch.Tensor, optional): A boolean tensor of shape
            :obj:`[batch_size, num_nodes]` holding information about which
            nodes are in each example are valid. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> # For a single adjacency matrix:
        >>> adj = torch.tensor([[3, 1],
        ...                     [2, 0]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1],
                [0, 1, 0]]),
        tensor([3, 1, 2]))

        >>> # For two adjacency matrixes:
        >>> adj = torch.tensor([[[3, 1],
        ...                      [2, 0]],
        ...                     [[0, 1],
        ...                      [0, 2]]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1, 2, 3],
                [0, 1, 0, 3, 3]]),
        tensor([3, 1, 2, 1, 2]))

        >>> # First graph with two nodes, second with three:
        >>> adj = torch.tensor([[
        ...         [3, 1, 0],
        ...         [2, 0, 0],
        ...         [0, 0, 0]
        ...     ], [
        ...         [0, 1, 0],
        ...         [0, 2, 3],
        ...         [0, 5, 0]
        ...     ]])
        >>> mask = torch.tensor([
        ...         [True, True, False],
        ...         [True, True, True]
        ...     ])
        >>> dense_to_sparse(adj, mask)
        (tensor([[0, 0, 1, 2, 3, 3, 4],
                [0, 1, 0, 3, 3, 4, 3]]),
        tensor([3, 1, 2, 1, 2, 3, 5]))
    """
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be two- or "
                         f"three-dimensional (got {adj.dim()} dimensions)")

    if mask is not None and adj.dim() == 2:
        warnings.warn("Mask should not be provided in case the dense "
                      "adjacency matrix is two-dimensional")
        mask = None

    if mask is not None and mask.dim() != 2:
        raise ValueError(f"Mask must be two-dimensional "
                         f"(got {mask.dim()} dimensions)")

    if mask is not None and adj.size(-2) != adj.size(-1):
        raise ValueError(f"Mask is only supported on quadratic adjacency "
                         f"matrices (got [*, {adj.size(-2)}, {adj.size(-1)}])")

    if adj.dim() == 2:
        edge_index = adj.nonzero().t()
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        flatten_adj = adj.view(-1, adj.size(-1))
        if mask is not None:
            flatten_adj = flatten_adj[mask.view(-1)]
        edge_index = flatten_adj.nonzero().t()
        edge_attr = flatten_adj[edge_index[0], edge_index[1]]

        if mask is None:
            offset = torch.arange(
                start=0,
                end=adj.size(0) * adj.size(2),
                step=adj.size(2),
                device=adj.device,
            )
            offset = offset.repeat_interleave(adj.size(1))
        else:
            count = mask.sum(dim=-1)
            offset = cumsum(count)[:-1]
            offset = offset.repeat_interleave(count)

        edge_index[1] += offset[edge_index[0]]

        return edge_index, edge_attr


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
        if (torch_geometric.typing.WITH_PT112
                and src.layout == torch.sparse_csc):
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
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
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

    if isinstance(size, (tuple, list)):
        num_src_nodes, num_dst_nodes = size
        if num_src_nodes is None:
            num_src_nodes = int(edge_index[0].max()) + 1
        if num_dst_nodes is None:
            num_dst_nodes = int(edge_index[1].max()) + 1
        size = (num_src_nodes, num_dst_nodes)
    else:
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size))

    if edge_attr is None:
        # Expanded tensors are not yet supported in all PyTorch code paths :(
        # edge_attr = torch.ones(1, device=edge_index.device)
        # edge_attr = edge_attr.expand(edge_index.size(1))
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    if not torch_geometric.typing.WITH_PT21:
        adj = torch.sparse_coo_tensor(
            indices=edge_index,
            values=edge_attr,
            size=tuple(size) + edge_attr.size()[1:],
            device=edge_index.device,
        )
        adj = adj._coalesced_(True)
        return adj

    return torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_attr,
        size=tuple(size) + edge_attr.size()[1:],
        device=edge_index.device,
        is_coalesced=True,
    )


def to_torch_csr_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
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
    if size is None:
        size = int(edge_index.max()) + 1

    if isinstance(size, (tuple, list)):
        num_src_nodes, num_dst_nodes = size
        if num_src_nodes is None:
            num_src_nodes = int(edge_index[0].max()) + 1
        if num_dst_nodes is None:
            num_dst_nodes = int(edge_index[1].max()) + 1
        size = (num_src_nodes, num_dst_nodes)
    else:
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size))

    if edge_attr is None:
        # Expanded tensors are not yet supported in all PyTorch code paths :(
        # edge_attr = torch.ones(1, device=edge_index.device)
        # edge_attr = edge_attr.expand(edge_index.size(1))
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    adj = torch.sparse_csr_tensor(
        crow_indices=index2ptr(edge_index[0], size[0]),
        col_indices=edge_index[1],
        values=edge_attr,
        size=tuple(size) + edge_attr.size()[1:],
        device=edge_index.device,
    )

    return adj


def to_torch_csc_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
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
    if not torch_geometric.typing.WITH_PT112:
        if typing.TYPE_CHECKING:
            raise NotImplementedError
        return torch_geometric.typing.MockTorchCSCTensor(
            edge_index, edge_attr, size)

    if size is None:
        size = int(edge_index.max()) + 1

    if isinstance(size, (tuple, list)):
        num_src_nodes, num_dst_nodes = size
        if num_src_nodes is None:
            num_src_nodes = int(edge_index[0].max()) + 1
        if num_dst_nodes is None:
            num_dst_nodes = int(edge_index[1].max()) + 1
        size = (num_src_nodes, num_dst_nodes)
    else:
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size),
                                         sort_by_row=False)

    if edge_attr is None:
        # Expanded tensors are not yet supported in all PyTorch code paths :(
        # edge_attr = torch.ones(1, device=edge_index.device)
        # edge_attr = edge_attr.expand(edge_index.size(1))
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
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    is_coalesced: bool = False,
    layout: torch.layout = torch.sparse_coo,
) -> Tensor:
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
    if torch_geometric.typing.WITH_PT112 and layout == torch.sparse_csc:
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
        adj = adj._coalesced_(True)
        return adj.indices().detach().long(), adj.values()

    if adj.layout == torch.sparse_csr:
        row = ptr2index(adj.crow_indices().detach())
        col = adj.col_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    if torch_geometric.typing.WITH_PT112 and adj.layout == torch.sparse_csc:
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
    if value.dim() > 1:
        size = adj.size() + value.size()[1:]
    else:
        size = adj.size()

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

    if torch_geometric.typing.WITH_PT112 and adj.layout == torch.sparse_csc:
        return torch.sparse_csc_tensor(
            ccol_indices=adj.ccol_indices(),
            row_indices=adj.row_indices(),
            values=value,
            size=size,
            device=value.device,
        )

    raise ValueError(f"Unexpected sparse tensor layout (got '{adj.layout}')")


def cat_coo(tensors: List[Tensor], dim: Union[int, Tuple[int, int]]) -> Tensor:
    assert dim in {0, 1, (0, 1)}
    assert tensors[0].layout == torch.sparse_coo

    indices, values = [], []
    num_rows = num_cols = 0
    is_coalesced = True

    if dim == 0:
        for i, tensor in enumerate(tensors):
            if i == 0:
                indices.append(tensor._indices())
            else:
                offset = torch.tensor([[num_rows], [0]], device=tensor.device)
                indices.append(tensor._indices() + offset)
            values.append(tensor._values())
            num_rows += tensor.size(0)
            num_cols = max(num_cols, tensor.size(1))
            if not tensor.is_coalesced():
                is_coalesced = False

    elif dim == 1:
        for i, tensor in enumerate(tensors):
            if i == 0:
                indices.append(tensor._indices())
            else:
                offset = torch.tensor([[0], [num_cols]], device=tensor.device)
                indices.append(tensor.indices() + offset)
            values.append(tensor._values())
            num_rows = max(num_rows, tensor.size(0))
            num_cols += tensor.size(1)
            is_coalesced = False

    else:
        for i, tensor in enumerate(tensors):
            if i == 0:
                indices.append(tensor._indices())
            else:
                offset = torch.tensor([[num_rows], [num_cols]],
                                      device=tensor.device)
                indices.append(tensor._indices() + offset)
            values.append(tensor._values())
            num_rows += tensor.size(0)
            num_cols += tensor.size(1)
            if not tensor.is_coalesced():
                is_coalesced = False

    if not torch_geometric.typing.WITH_PT21:
        out = torch.sparse_coo_tensor(
            indices=torch.cat(indices, dim=-1),
            values=torch.cat(values),
            size=(num_rows, num_cols) + values[-1].size()[1:],
            device=tensor.device,
        )
        if is_coalesced:
            out = out._coalesced_(True)
        return out

    return torch.sparse_coo_tensor(
        indices=torch.cat(indices, dim=-1),
        values=torch.cat(values),
        size=(num_rows, num_cols) + values[-1].size()[1:],
        device=tensor.device,
        is_coalesced=True if is_coalesced else None,
    )


def cat_csr(tensors: List[Tensor], dim: Union[int, Tuple[int, int]]) -> Tensor:
    assert dim in {0, 1, (0, 1)}
    assert tensors[0].layout == torch.sparse_csr

    rows, cols, values = [], [], []
    num_rows = num_cols = nnz = 0

    if dim == 0:
        for i, tensor in enumerate(tensors):
            if i == 0:
                rows.append(tensor.crow_indices())
            else:
                rows.append(tensor.crow_indices()[1:] + nnz)
            cols.append(tensor.col_indices())
            values.append(tensor.values())
            num_rows += tensor.size(0)
            num_cols = max(num_cols, tensor.size(1))
            nnz += cols[-1].numel()

        return torch.sparse_csr_tensor(
            crow_indices=torch.cat(rows),
            col_indices=torch.cat(cols),
            values=torch.cat(values),
            size=(num_rows, num_cols) + values[-1].size()[1:],
            device=tensor.device,
        )

    elif dim == 1:
        for i, tensor in enumerate(tensors):
            rows.append(ptr2index(tensor.crow_indices()))
            if i == 0:
                cols.append(tensor.col_indices())
            else:
                cols.append(tensor.col_indices() + num_cols)
            values.append(tensor.values())
            num_rows = max(num_rows, tensor.size(0))
            num_cols += tensor.size(1)

        return torch.sparse_coo_tensor(
            indices=torch.stack((torch.cat(rows), torch.cat(cols)), 0),
            values=torch.cat(values),
            size=(num_rows, num_cols) + values[-1].size()[1:],
            device=tensor.device,
        )

    else:
        for i, tensor in enumerate(tensors):
            if i == 0:
                rows.append(tensor.crow_indices())
                cols.append(tensor.col_indices())
            else:
                rows.append(tensor.crow_indices()[1:] + nnz)
                cols.append(tensor.col_indices() + num_cols)
            values.append(tensor.values())
            num_rows += tensor.size(0)
            num_cols += tensor.size(1)
            nnz += cols[-1].numel()

        return torch.sparse_csr_tensor(
            crow_indices=torch.cat(rows),
            col_indices=torch.cat(cols),
            values=torch.cat(values),
            size=(num_rows, num_cols) + values[-1].size()[1:],
            device=tensor.device,
        )


def cat_csc(tensors: List[Tensor], dim: Union[int, Tuple[int, int]]) -> Tensor:
    assert dim in {0, 1, (0, 1)}
    assert tensors[0].layout == torch.sparse_csc

    rows, cols, values = [], [], []
    num_rows = num_cols = nnz = 0

    if dim == 0:
        for i, tensor in enumerate(tensors):
            cols.append(ptr2index(tensor.ccol_indices()))
            if i == 0:
                rows.append(tensor.row_indices())
            else:
                rows.append(tensor.row_indices() + num_rows)
            values.append(tensor.values())
            num_rows += tensor.size(0)
            num_cols = max(num_cols, tensor.size(1))

        return torch.sparse_coo_tensor(
            indices=torch.stack((torch.cat(rows), torch.cat(cols)), 0),
            values=torch.cat(values),
            size=(num_rows, num_cols) + values[-1].size()[1:],
            device=tensor.device,
        )

    elif dim == 1:
        for i, tensor in enumerate(tensors):
            if i == 0:
                cols.append(tensor.ccol_indices())
            else:
                cols.append(tensor.ccol_indices()[1:] + nnz)
            rows.append(tensor.row_indices())
            values.append(tensor.values())
            num_rows = max(num_rows, tensor.size(0))
            num_cols += tensor.size(1)
            nnz += rows[-1].numel()

        return torch.sparse_csc_tensor(
            row_indices=torch.cat(rows),
            ccol_indices=torch.cat(cols),
            values=torch.cat(values),
            size=(num_rows, num_cols) + values[-1].size()[1:],
            device=tensor.device,
        )

    else:
        for i, tensor in enumerate(tensors):
            if i == 0:
                rows.append(tensor.row_indices())
                cols.append(tensor.ccol_indices())
            else:
                rows.append(tensor.row_indices() + num_rows)
                cols.append(tensor.ccol_indices()[1:] + nnz)
            values.append(tensor.values())
            num_rows += tensor.size(0)
            num_cols += tensor.size(1)
            nnz += rows[-1].numel()

        return torch.sparse_csc_tensor(
            row_indices=torch.cat(rows),
            ccol_indices=torch.cat(cols),
            values=torch.cat(values),
            size=(num_rows, num_cols) + values[-1].size()[1:],
            device=tensor.device,
        )


def cat(tensors: List[Tensor], dim: Union[int, Tuple[int, int]]) -> Tensor:
    assert is_torch_sparse_tensor(tensors[0])

    if tensors[0].layout == torch.sparse_coo:
        return cat_coo(tensors, dim)
    elif tensors[0].layout == torch.sparse_csr:
        return cat_csr(tensors, dim)
    else:
        return cat_csc(tensors, dim)
