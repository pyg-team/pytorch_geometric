from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def shuffle_node(x: Tensor, training: bool = True) -> Tuple[Tensor, Tensor]:
    if not training:
        perm = torch.arange(x.size(0))
        return x, perm
    perm = torch.randperm(x.size(0))
    return x[perm], perm


def mask_feature(x: Tensor, p: float = 0.5,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Mask ratio has to be between 0 and 1 '
                         f'(got {p}')
    if not training or p == 0.0:
        return x, torch.ones_like(x, dtype=torch.bool)

    mask = torch.rand(x.size(1), device=x.device) >= p
    mask = mask.tile(x.size(0), 1)
    # To avoid inplace operation
    x = x * (mask.float())
    return x, mask


def add_edge(edge_index, p: float = 0.2,
             num_nodes: Optional[Union[Tuple[int], int]] = None,
             training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Sampling ratio has to be between 0 and 1 '
                         f'(got {p}')

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([], device=device).view(2, 0)
        return edge_index, edge_index_to_add

    num_nodes = (num_nodes,
                 num_nodes) if not isinstance(num_nodes, tuple) else num_nodes
    num_src_nodes = num_nodes[0] or edge_index[0].max().item() + 1
    num_dst_nodes = num_nodes[1] or edge_index[1].max().item() + 1

    num_edges_to_add = round(edge_index.size(1) * p)
    row = torch.randint(0, num_src_nodes, size=(num_edges_to_add, ))
    col = torch.randint(0, num_dst_nodes, size=(num_edges_to_add, ))

    edge_index_to_add = torch.stack([row, col], dim=0).to(device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)
    return edge_index, edge_index_to_add
