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

    # For each triangle face, get all three cotangents:
    # 1. Compute edge vectors of edges opposite to each vertex
    e_21 = pos[face[2]] - pos[face[1]]
    e_02 = pos[face[0]] - pos[face[2]]
    e_10 = -(e_21 + e_02)  # this is equal to pos[face[1]] - pos[face[0]]

    # 2. Compute areas of all triangles
    area_times_four = 2 * torch.norm(torch.cross(e_02, e_10, dim=1), dim=1)

    # 3. Compute cotangent weights
    #    cot(e_ik, e_kj) = -dot(e_ik, e_kj) / (4 * area)
    cot_0 = -torch.sum(e_10 * e_02, dim=1) / area_times_four
    cot_1 = -torch.sum(e_21 * e_10, dim=1) / area_times_four
    cot_2 = -torch.sum(e_02 * e_21, dim=1) / area_times_four
    cot_weight = torch.cat([cot_2, cot_0, cot_1])

    # Face to edge:
    cot_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    cot_index, cot_weight = to_undirected(cot_index, cot_weight)

    # Compute the diagonal part:
    cot_deg = scatter(cot_weight, cot_index[0], 0, num_nodes, reduce='sum')
    edge_index, _ = add_self_loops(cot_index, num_nodes=num_nodes)
    edge_weight = torch.cat([cot_weight, -cot_deg], dim=0)

    if normalization is not None:

        # Like before, but here we only need the diagonal (the mass matrix):
        # Attribute a third of the area from a face to each contained vertex
        # and accumulate
        area_weight = (area_times_four / 12).repeat(3)
        area_deg = scatter(area_weight, face.flatten(), 0, num_nodes, 'sum')

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
