import warnings
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.utils import (
    coalesce,
    cumsum,
    degree,
    index_sort,
    remove_self_loops,
    to_undirected,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes

_MAX_NUM_EDGES = 10**4


def negative_sampling(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    num_neg_samples: Optional[int] = None,
    method: str = "auto",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return.
            If set to :obj:`None`, will try to return a negative edge for every
            positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"`, :obj:`"dense"`, or :obj:`"auto"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, but it could retrieve a different number of negative samples
            :obj:`"dense"` will work only on small graphs since it enumerates all possible edges
            :obj:`"auto"` will automatically choose the best method
            (default: :obj:`"auto"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:
        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> negative_sampling(edge_index)
        tensor([[3, 0, 0, 3],
                [2, 3, 2, 1]])

        >>> # For bipartite graph
        >>> negative_sampling(edge_index, num_nodes=(3, 4))
        tensor([[0, 2, 2, 1],
                [2, 2, 1, 3]])
    """
    assert method in ['sparse', 'dense', 'auto']

    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if isinstance(num_nodes, int):
        size = (num_nodes, num_nodes)
        bipartite = False
    else:
        size = num_nodes
        bipartite = True
        force_undirected = False

    num_edges = edge_index.size(1)
    num_tot_edges = (size[0] * size[1])

    if num_neg_samples is None:
        num_neg_samples = num_edges

    if force_undirected:
        num_neg_samples = num_neg_samples // 2

    # transform a pair (u,v) in an edge id
    edge_id = edge_index_to_vector_id(edge_index, size)
    edge_id, _ = index_sort(edge_id, max_value=num_tot_edges)

    method = get_method(method, size)

    k = None
    prob = 1 - (num_edges / num_tot_edges)
    if method == 'sparse':
        if prob >= 0.3:
            # the probability of sampling non-existing edge is high, so the sparse method should be ok
            k = int(num_neg_samples / (prob - 0.1))
        else:
            # the probability is too low, but the graph is too big for the exact sampling.
            # we perform the sparse sampling but we raise a warning
            k = int(
                min(10 * num_neg_samples,
                    max(num_neg_samples / (prob - 0.1), 10**-3)))

            warnings.warn(
                'The probability of sampling a negative edge is too low! '
                'It could be that the number of sampled edges is smaller than the numbers you required!'
            )

    guess_edge_index, guess_edge_id = sample_almost_k_edges(
        size, k, force_undirected=force_undirected,
        remove_self_loops=not bipartite, method=method,
        device=edge_index.device)

    neg_edge_mask = get_neg_edge_mask(edge_id, guess_edge_id)

    # we fiter the guessed id to maintain only the negative ones
    neg_edge_index = guess_edge_index[:, neg_edge_mask]

    if neg_edge_index.shape[-1] > num_neg_samples:
        neg_edge_index = neg_edge_index[:, :num_neg_samples]

    assert neg_edge_index is not None

    #print(f'{prob} - {method} - {k} - {num_neg_samples} - {neg_edge_index.shape[-1]}')

    if force_undirected:
        neg_edge_index = to_undirected(neg_edge_index)

    return neg_edge_index


def batched_negative_sampling(
    edge_index: Tensor,
    batch: Union[Tensor, Tuple[Tensor, Tensor]],
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of multiple graphs given by
    :attr:`edge_index` and :attr:`batch`.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor or Tuple[LongTensor, LongTensor]): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph connecting two different node types.
        num_neg_samples (int, optional): The number of negative samples to
            return for each graph in the batch. If set to :obj:`None`, will try to return a negative edge
            for every positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"`, :obj:`"dense"`, or :obj:`"auto"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, but it could retrieve a different number of negative samples
            :obj:`"dense"` will work only on small graphs since it enumerates all possible edges
            :obj:`"auto"` will automatically choose the best method
            (default: :obj:`"auto"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:
        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
        >>> edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
        >>> edge_index
        tensor([[0, 0, 1, 2, 4, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> batched_negative_sampling(edge_index, batch)
        tensor([[3, 1, 3, 2, 7, 7, 6, 5],
                [2, 0, 1, 1, 5, 6, 4, 4]])

        >>> # For bipartite graph
        >>> edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
        >>> edge_index2 = edge_index1 + torch.tensor([[2], [4]])
        >>> edge_index3 = edge_index2 + torch.tensor([[2], [4]])
        >>> edge_index = torch.cat([edge_index1, edge_index2,
        ...                         edge_index3], dim=1)
        >>> edge_index
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
        >>> src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        >>> batched_negative_sampling(edge_index,
        ...                           (src_batch, dst_batch))
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9]])
    """
    if isinstance(batch, Tensor):
        src_batch, dst_batch = batch, batch
    else:
        src_batch, dst_batch = batch[0], batch[1]

    split = degree(src_batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)

    num_src = degree(src_batch, dtype=torch.long)
    cum_src = cumsum(num_src)[:-1]

    if isinstance(batch, Tensor):
        num_nodes = num_src.tolist()
        ptr = cum_src
    else:
        num_dst = degree(dst_batch, dtype=torch.long)
        cum_dst = cumsum(num_dst)[:-1]

        num_nodes = torch.stack([num_src, num_dst], dim=1).tolist()
        ptr = torch.stack([cum_src, cum_dst], dim=1).unsqueeze(-1)

    neg_edge_indices = []
    for i, edge_index in enumerate(edge_indices):
        edge_index = edge_index - ptr[i]
        neg_edge_index = negative_sampling(edge_index, num_nodes[i],
                                           num_neg_samples, method,
                                           force_undirected)
        neg_edge_index += ptr[i]
        neg_edge_indices.append(neg_edge_index)

    return torch.cat(neg_edge_indices, dim=1)


def structured_negative_sampling(
        edge_index: Tensor, num_nodes: Optional[int] = None,
        contains_neg_self_loops: bool = True,
        method: str = "auto") -> Tuple[Tensor, Tensor, Tensor]:
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"`, :obj:`"dense"`, or :obj:`"auto"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, but it could retrieve a different number of negative samples
            :obj:`"dense"` will work only on small graphs since it enumerates all possible edges
            :obj:`"auto"` will automatically choose the best method
            (default: :obj:`"auto"`)

    :rtype: (LongTensor, LongTensor, LongTensor)

    Example:
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
        ...                               [0, 1, 2, 3]])
        >>> structured_negative_sampling(edge_index)
        (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    size = (num_nodes, num_nodes)
    num_edges = edge_index.size(1)
    num_tot_edges = size[0] * size[1]
    device = edge_index.device
    deg = degree(edge_index[0], num_nodes, dtype=torch.long)

    # transform a pair (u,v) in an edge id
    edge_id = edge_index_to_vector_id(edge_index, size)
    edge_id, _ = index_sort(edge_id, max_value=num_tot_edges)

    # select the method
    method = get_method(method, size)

    k = None
    torch.min(1 - deg / num_nodes)
    if method == 'sparse':
        k = 10
        # TODO: can we compute the prob to find the k?

    guess_col, guess_edge_id = sample_k_structured_edges(
        edge_index, num_nodes, k, not contains_neg_self_loops, method, device)

    neg_edge_mask = get_neg_edge_mask(edge_id, guess_edge_id)

    if method == 'sparse':
        neg_col = -torch.ones_like(edge_index[0])
        neg_edge_mask = get_first_k_true_values_for_each_row(
            neg_edge_mask.view(num_edges, k), 1)
        ok_edges = torch.any(
            neg_edge_mask, dim=1
        )  # this is the mask of edges for which we have obtained a neg sample
        col_to_save = guess_col.view(
            num_edges, k)[neg_edge_mask]  # this the list of neg samples
        neg_col[ok_edges] = col_to_save.view(-1)

        if not torch.all(ok_edges):
            warnings.warn(
                'We were not able to sample all negative edges requested!')

    else:
        shape = (num_nodes,
                 num_nodes if contains_neg_self_loops else num_nodes - 1)
        neg_edge_mask = get_first_k_true_values_for_each_row(
            neg_edge_mask.view(*shape), deg)
        col_to_save = guess_col.view(
            *shape)[neg_edge_mask]  # this the list of neg samples
        neg_col = col_to_save.view(-1)

    assert neg_col.size(0) == edge_index.size(1)

    return edge_index[0], edge_index[1], neg_col


def structured_negative_sampling_feasible(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> bool:
    r"""Returns :obj:`True` if
    :meth:`~torch_geometric.utils.structured_negative_sampling` is feasible
    on the graph given by :obj:`edge_index`.
    :meth:`~torch_geometric.utils.structured_negative_sampling` is infeasible
    if at least one node is connected to all other nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: bool

    Examples:
        >>> edge_index = torch.LongTensor([[0, 0, 1, 1, 2, 2, 2],
        ...                                [1, 2, 0, 2, 0, 1, 1]])
        >>> structured_negative_sampling_feasible(edge_index, 3, False)
        False

        >>> structured_negative_sampling_feasible(edge_index, 3, True)
        True
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    max_num_neighbors = num_nodes

    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    if not contains_neg_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
        max_num_neighbors -= 1  # Reduce number of valid neighbors

    deg = degree(edge_index[0], num_nodes)
    # structured sample is feasible if, for each node, deg > max_neigh/2
    return bool(torch.all(2 * deg <= max_num_neighbors))


###############################################################################


def get_method(method, size):
    # select the method
    tot_num_edges = size[0] * size[1]
    auto_method = 'dense' if tot_num_edges < _MAX_NUM_EDGES else 'sparse'  # prefer dense method if the graph is small
    method = auto_method if method == 'auto' else method

    if method == 'dense' and tot_num_edges >= _MAX_NUM_EDGES:
        warnings.warn(
            f'You choose the dense method on a graph with {tot_num_edges} possible edges! '
            f'It could require a lot of memory!')

    return method


def sample_almost_k_edges(
    size: Tuple[int, int],
    k: int,
    force_undirected: bool,
    remove_self_loops: bool,
    method: str,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[Tensor, Tensor]:

    assert method in ['sparse', 'dense']
    N1, N2 = size

    if method == 'sparse':
        k = 2 * k if force_undirected else k
        p = torch.tensor([1.], device=device).expand(N1 * N2)
        if k > N1 * N2:
            k = N1 * N2
        new_edge_id = torch.multinomial(p, k, replacement=False)

    else:
        new_edge_id = torch.randperm(N1 * N2, device=device)

    new_edge_index = torch.stack(vector_id_to_edge_index(new_edge_id, size),
                                 dim=0)

    if remove_self_loops:
        not_in_diagonal = new_edge_index[0] != new_edge_index[1]
        new_edge_index = new_edge_index[:, not_in_diagonal]
        new_edge_id = new_edge_id[not_in_diagonal]

    if force_undirected:
        # we consider only the upper part, i.e. col_idx > row_idx
        in_upper_part = new_edge_index[1] > new_edge_index[0]
        new_edge_index = new_edge_index[:, in_upper_part]
        new_edge_id = new_edge_id[in_upper_part]

    return new_edge_index, new_edge_id


def sample_k_structured_edges(
    edge_index: Tensor,
    num_nodes: int,
    k: int,
    remove_self_loops: bool,
    method: str,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[Tensor, Tensor]:

    assert method in ['sparse', 'dense']
    row, col = edge_index
    num_edges = edge_index.size(1)
    size = (num_nodes, num_nodes)

    if method == 'sparse':
        new_row, new_col = row.view(-1, 1).expand(-1, k), torch.randint(
            num_nodes, (num_edges, k))
        new_edge_id = edge_index_to_vector_id((new_row, new_col), size)
    else:
        new_edge_id = torch.randperm(num_nodes * num_nodes, device=device)
        new_row, new_col = vector_id_to_edge_index(new_edge_id, size)
        new_row, idx_to_sort = index_sort(new_row)
        new_edge_id = new_edge_id[idx_to_sort]
        new_col = new_col[idx_to_sort]

    if remove_self_loops:
        # instead of filterin (which break the shapes), we just add +1
        in_diagonal = torch.eq(new_row, new_col)
        if method == 'dense':
            not_in_diagonal = torch.logical_not(in_diagonal)
            new_col = new_col[not_in_diagonal]
            new_edge_id = new_edge_id[not_in_diagonal]
        else:
            new_v = (new_col[in_diagonal] + 1) % num_nodes
            diff_v = new_v - new_col[in_diagonal]
            new_col[in_diagonal] = new_v
            new_edge_id[in_diagonal] += diff_v

    return new_col.view(-1), new_edge_id.view(-1)


def get_first_k_true_values_for_each_row(input_mask, k):
    if isinstance(k, Tensor):
        k = k.unsqueeze(1)
    cum_mask = torch.le(torch.cumsum(input_mask, dim=1), k)
    return torch.logical_and(input_mask, cum_mask)


def get_neg_edge_mask(edge_id, guess_edge_id):
    num_edges = edge_id.size(0)
    pos = torch.searchsorted(edge_id, guess_edge_id)
    # pos contains the position where to insert the guessed id to maintain the edge_id sort.
    # 1) if pos == num_edges, it means that we should add the guessed if at the end of the vector -> the id is new!
    # 2) if pos != num_edges but the id in position pos != from the guessed one -> the id is new!
    neg_edge_mask = torch.eq(pos, num_edges)  # negative edge from case 1)
    not_neg_edge_mask = torch.logical_not(neg_edge_mask)
    # negative edge from case 2)
    neg_edge_mask[not_neg_edge_mask] = edge_id[
        pos[not_neg_edge_mask]] != guess_edge_id[not_neg_edge_mask]
    return neg_edge_mask


def edge_index_to_vector_id(
    edge_index: Union[Tensor, Tuple[Tensor, Tensor]],
    size: Tuple[int, int],
) -> Tensor:

    row, col = edge_index
    return (row * size[1]).add_(col)


def vector_id_to_edge_index(
    vector_id: Tensor,
    size: Tuple[int, int],
) -> Tuple[Tensor, Tensor]:

    row, col = vector_id // size[1], vector_id % size[1]
    return row, col
