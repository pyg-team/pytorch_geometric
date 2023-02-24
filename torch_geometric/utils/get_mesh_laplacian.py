from typing import Tuple, Optional

import torch
from torch import Tensor

from torch_geometric.utils import add_self_loops, coalesce, to_undirected


def get_mesh_laplacian(
    pos: Tensor,
    face: Tensor,
    normalization: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the mesh Laplacian of a mesh given by :obj:`pos` and
    :obj:`face`. Computation is based on the cotangent matrix defined as

    .. math::
        \mathbf{C}_{ij} = \begin{cases}
            \frac{\cot \angle_{ikj} + \cot \angle_{ilj}}{2} &
            \mbox{if } i, j \mbox{ is an edge} \\
            -\sum_{j \in N(i)}{C_{ij}} &
            \mbox{if } i \mbox{ is in the diagonal} \\
            0 \mbox{ otherwise}
      \end{cases}

    Normalization depends on the mass matrix defined as

    .. math::
        \mathbf{M}_{ij} = \begin{cases}
            a(i) \mbox{ if } i \mbox{ is in the diagonal} \\
            0 \mbox{ otherwise}
      \end{cases}

    where :math:`a(i)` is obtained by joining the barycenters of the
    triangles around vertex :math:`i`.

    Args:
        pos (Tensor): The node positions.
        face (LongTensor): The face indices.
        normalization (str, optional): The normalization scheme for the mesh
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{C}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{M}^{-1/2} \mathbf{C}\mathbf{M}^{-1/2}`

            3. :obj:`"rw"`: Row-wise normalization
            :math:`\mathbf{L} = \mathbf{M}^{-1} \mathbf{C}`
    """
    assert pos.size(1) == 3 and face.size(0) == 3

    num_nodes = pos.shape[0]

    def get_cots(left, centre, right):
        left_pos, central_pos, right_pos = pos[left], pos[centre], pos[right]
        left_vec = left_pos - central_pos
        right_vec = right_pos - central_pos
        dot = torch.einsum('ij, ij -> i', left_vec, right_vec)
        cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
        cot = dot / cross  # cot = cos / sin
        return cot / 2.0  # by definition

    # For each triangle face, get all three cotangents:
    cot_021 = get_cots(face[0], face[2], face[1])
    cot_102 = get_cots(face[1], face[0], face[2])
    cot_012 = get_cots(face[0], face[1], face[2])
    cot_weight = torch.cat([cot_021, cot_102, cot_012])

    # Face to edge:
    cot_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    cot_index, cot_weight = to_undirected(*coalesce(cot_index, cot_weight))

    # Compute the diagonal part:
    row, col = cot_index
    cot_deg = torch.zeros(num_nodes).scatter_add_(0, row, cot_weight)
    edge_index, _ = add_self_loops(cot_index, num_nodes=num_nodes)
    edge_weight = torch.cat([cot_weight, -cot_deg], dim=0)

    if normalization is not None:

        def get_areas(left, centre, right):
            left_pos, central_pos, right_pos = pos[left], pos[centre], pos[right]
            left_vec = left_pos - central_pos
            right_vec = right_pos - central_pos
            cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
            area = cross / 6.0  # one-third of a triangle's area is cross / 6.0
            return area / 2.0  # since each corresponding area is counted twice

        # Like before, but here we only need the diagonal (i.e. the mass matrix):
        area_021 = get_areas(face[0], face[2], face[1])
        area_102 = get_areas(face[1], face[0], face[2])
        area_012 = get_areas(face[0], face[1], face[2])
        area_weight = torch.cat([area_021, area_102, area_012])
        area_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
        area_index, area_weight = to_undirected(*coalesce(area_index, area_weight))
        row, col = area_index
        area_deg = torch.zeros(num_nodes).scatter_add_(0, row, area_weight)

        if normalization == 'sym':
            area_deg_inv_sqrt = area_deg.pow_(-0.5)
            area_deg_inv_sqrt.masked_fill_(area_deg_inv_sqrt == float('inf'), 0)
            row, col = edge_index
            edge_weight = area_deg_inv_sqrt[row] * edge_weight * area_deg_inv_sqrt[col]
        elif normalization == 'rw':
            area_deg_inv = 1.0 / area_deg
            area_deg_inv.masked_fill_(area_deg_inv == float('inf'), 0)
            row, col = edge_index
            edge_weight = area_deg_inv[row] * edge_weight

    return edge_index, edge_weight
