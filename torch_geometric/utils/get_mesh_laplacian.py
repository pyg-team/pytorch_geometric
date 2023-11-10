from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import add_self_loops, scatter, to_undirected


def get_mesh_laplacian(
    pos: Tensor,
    face: Tensor,
    normalization: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the mesh Laplacian of a mesh given by :obj:`pos` and
    :obj:`face`.

    Computation is based on the cotangent matrix defined as

    .. math::
        \mathbf{C}_{ij} = \begin{cases}
            \frac{\cot \angle_{ikj}~+\cot \angle_{ilj}}{2} &
            \text{if } i, j \text{ is an edge} \\
            -\sum_{j \in N(i)}{C_{ij}} &
            \text{if } i \text{ is in the diagonal} \\
            0 & \text{otherwise}
      \end{cases}

    Normalization depends on the mass matrix defined as

    .. math::
        \mathbf{M}_{ij} = \begin{cases}
            a(i) & \text{if } i \text{ is in the diagonal} \\
            0 & \text{otherwise}
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
    cot_index, cot_weight = to_undirected(cot_index, cot_weight)

    # Compute the diagonal part:
    cot_deg = scatter(cot_weight, cot_index[0], 0, num_nodes, reduce='sum')
    edge_index, _ = add_self_loops(cot_index, num_nodes=num_nodes)
    edge_weight = torch.cat([cot_weight, -cot_deg], dim=0)

    if normalization is not None:

        def get_areas(left, centre, right):
            central_pos = pos[centre]
            left_vec = pos[left] - central_pos
            right_vec = pos[right] - central_pos
            cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
            area = cross / 6.0  # one-third of a triangle's area is cross / 6.0
            return area / 2.0  # since each corresponding area is counted twice

        # Like before, but here we only need the diagonal (the mass matrix):
        area_021 = get_areas(face[0], face[2], face[1])
        area_102 = get_areas(face[1], face[0], face[2])
        area_012 = get_areas(face[0], face[1], face[2])
        area_weight = torch.cat([area_021, area_102, area_012])
        area_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
        area_index, area_weight = to_undirected(area_index, area_weight)
        area_deg = scatter(area_weight, area_index[0], 0, num_nodes, 'sum')

        if normalization == 'sym':
            area_deg_inv_sqrt = area_deg.pow_(-0.5)
            area_deg_inv_sqrt[area_deg_inv_sqrt == float('inf')] = 0.0
            edge_weight = (area_deg_inv_sqrt[edge_index[0]] * edge_weight *
                           area_deg_inv_sqrt[edge_index[1]])
        elif normalization == 'rw':
            area_deg_inv = 1.0 / area_deg
            area_deg_inv[area_deg_inv == float('inf')] = 0.0
            edge_weight = area_deg_inv[edge_index[0]] * edge_weight

    return edge_index, edge_weight
