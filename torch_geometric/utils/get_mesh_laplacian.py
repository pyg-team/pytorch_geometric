from typing import Tuple

import torch
from torch import Tensor
from torch_scatter import scatter_add

from torch_geometric.utils import add_self_loops, coalesce


def get_mesh_laplacian(pos: Tensor, face: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Computes the mesh Laplacian of a mesh given by :obj:`pos` and
    :obj:`face`.
    It is computed as

    .. math::
        \mathbf{L}_{ij} = \begin{cases}
            \frac{\cot \angle_{ikj} + \cot \angle_{ilj}}{2 a_{ij}} &
            \mbox{if } i, j \mbox{ is an edge} \\
            \sum_{j \in N(i)}{L_{ij}} &
            \mbox{if } i \mbox{ is in the diagonal} \\
            0 \mbox{ otherwise}
      \end{cases}

    where :math:`a_{ij}` is the local area element, *i.e.* one-third of the
    neighbouring triangle's area.

    Args:
        pos (Tensor): The node positions.
        face (LongTensor): The face indices.
    """
    assert pos.size(1) == 3 and face.size(0) == 3

    num_nodes = pos.shape[0]

    def add_angles(left, centre, right):
        left_pos, central_pos, right_pos = pos[left], pos[centre], pos[right]
        left_vec = left_pos - central_pos
        right_vec = right_pos - central_pos
        dot = torch.einsum('ij, ij -> i', left_vec, right_vec)
        cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
        cot = dot / cross  # cot = cos / sin
        area = cross / 6.0  # one-third of a triangle's area is cross / 6.0
        return cot / 2.0, area / 2.0

    # For each triangle face, add all three angles:
    cot_201, area_201 = add_angles(face[2], face[0], face[1])
    cot_012, area_012 = add_angles(face[0], face[1], face[2])
    cot_120, area_120 = add_angles(face[1], face[2], face[0])

    cot_weight = torch.cat(
        [cot_201, cot_201, cot_012, cot_012, cot_120, cot_120])
    area_weight = torch.cat(
        [area_201, area_201, area_012, area_012, area_120, area_120])

    edge_index = torch.cat([
        torch.stack([face[2], face[1]], dim=0),
        torch.stack([face[1], face[2]], dim=0),
        torch.stack([face[0], face[2]], dim=0),
        torch.stack([face[2], face[0]], dim=0),
        torch.stack([face[1], face[0]], dim=0),
        torch.stack([face[0], face[1]], dim=0)
    ], dim=1)

    edge_index, weight = coalesce(edge_index, [cot_weight, area_weight])
    cot_weight, area_weight = weight

    # Compute the diagonal part:
    row, col = edge_index
    cot_deg = scatter_add(cot_weight, row, dim=0, dim_size=num_nodes)
    area_deg = scatter_add(area_weight, row, dim=0, dim_size=num_nodes)
    deg = cot_deg / area_deg
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.cat([-cot_weight, deg], dim=0)

    return edge_index, edge_weight
