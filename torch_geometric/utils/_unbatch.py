from typing import List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import cumsum, degree


def unbatch(
    src: Tensor,
    batch: Tensor,
    dim: int = 0,
    batch_size: Optional[int] = None,
) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`List[Tensor]`

    Example:
        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, batch_size, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_edge_index(
    edge_index: Tensor,
    batch: Tensor,
    batch_size: Optional[int] = None,
) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`List[Tensor]`

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, batch_size, dtype=torch.long)
    ptr = cumsum(deg)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, batch_size, dtype=torch.long).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def to_batch_edge_index(
        edge_index_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
    r"""Merges a list of :obj:`edge_index` tensors into a single batched
    :obj:`edge_index` tensor and returns the corresponding :obj:`batch` vector.

    Args:
        edge_index_list (List[Tensor]): A list of :obj:`edge_index` tensors.

    :rtype: (:class:`Tensor`, :class:`Tensor`)

    Example:
        >>> edge_index_list = [
        ...     torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                   [1, 0, 2, 1, 3, 2]]),
        ...     torch.tensor([[0, 1, 1, 2],
        ...                   [1, 0, 2, 1]]),
        ... ]
        >>> edge_index, batch = to_batch_edge_index(edge_index_list)
        >>> edge_index
        tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
                [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch
        tensor([0, 0, 0, 0, 1, 1, 1])
    """
    if len(edge_index_list) == 0:
        return torch.empty((2, 0),
                           dtype=torch.long), torch.empty(0, dtype=torch.long)

    # Get the number of nodes in each graph
    num_nodes_list = []
    for edge_index in edge_index_list:
        if edge_index.numel() == 0:
            num_nodes_list.append(0)
        else:
            num_nodes_list.append(int(edge_index.max()) + 1)

    # Compute cumulative sum for node offsets
    cumsum_nodes = cumsum(torch.tensor(num_nodes_list, dtype=torch.long))

    # Offset each edge_index and concatenate
    batched_edge_indices = []
    for i, edge_index in enumerate(edge_index_list):
        if edge_index.numel() > 0:
            offset = cumsum_nodes[i]
            batched_edge_indices.append(edge_index + offset)
        else:
            batched_edge_indices.append(edge_index)

    # Concatenate all edge indices
    if all(ei.numel() == 0 for ei in batched_edge_indices):
        batched_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        batched_edge_index = torch.cat(
            [ei for ei in batched_edge_indices if ei.numel() > 0], dim=1)

    # Create batch vector
    batch = torch.cat([
        torch.full((num_nodes, ), i, dtype=torch.long)
        for i, num_nodes in enumerate(num_nodes_list)
    ])

    return batched_edge_index, batch
