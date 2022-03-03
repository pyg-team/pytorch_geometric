import torch
from torch_scatter import scatter_add

from torch_geometric.utils import add_self_loops


def get_mesh_laplacian(pos, face):
    r""" Computes the mesh Laplacian of the mesh given by :obj:`pos`
    and :obj:`face`. It is computed as :math:`\mathbf{L_{ij}} =
    \begin{cases} 
        \frac{\cot \angle_{ikj} + \cot \angle_{ilj}}{2 a_{ij}} &
        \mbox{if } \{i, j\} \mbox{ is an edge} \\
        \sum_{j \in N(i)}{L_{ij}} &
        \mbox{if } \{i\} \mbox{ is in the diagonal}
        \0 \mbox{ otherwise}
    \end{cases}`
    where :math:`a_{ij}` is the local area element,
    i.e. one-third of the neighbouring triangle's area.
    
    Args:
        pos (Tensor): The node positions.
        face (LongTensor): The face indices.
    """

    assert pos.shape[1] == 3
    assert face.shape[0] == 3

    device = pos.device
    dtype = pos.dtype
    num_nodes = pos.shape[0]
    cot_weight = torch.Tensor().to(dtype).to(device)
    area_weight = torch.Tensor().to(dtype).to(device)
    edge_index = torch.Tensor().long().to(device)

    def add_edge(left, centre, right):
        left_pos, central_pos, right_pos = pos[left], pos[centre], pos[right]
        left_vec = left_pos - central_pos
        right_vec = right_pos - central_pos
        dot = torch.einsum('ij, ij -> i', left_vec, right_vec)
        cross = torch.norm(torch.cross(left_vec, right_vec, dim=1), dim=1)
        cot = dot / cross  # cos / sin
        nonlocal cot_weight, area_weight, edge_index
        cot_weight = torch.cat([cot_weight, cot / 2.0, cot / 2.0])
        # one-third of a triangle's area is cross / 6.0
        # since each edge is accounted twice, we compute cross / 12.0 instead
        area_weight = torch.cat([area_weight, cross / 12.0, cross / 12.0])
        edge_index = torch.cat([
            edge_index,
            torch.stack([left, right], dim=1),
            torch.stack([right, left], dim=1)
        ])

    # add all 3 edges of the triangles
    add_edge(face[2], face[0], face[1])
    add_edge(face[0], face[1], face[2])
    add_edge(face[1], face[2], face[0])

    # eliminate duplicate matrix entries by adding them together
    index_linearizer = torch.Tensor([num_nodes, 1]).to(device)
    lin_index = torch.matmul(edge_index.float(), index_linearizer).long()
    y, idx = lin_index.unique(return_inverse=True)
    edge_index = torch.stack([y // num_nodes, y % num_nodes])
    cot_weight = scatter_add(cot_weight, idx, dim=0)
    area_weight = scatter_add(area_weight, idx, dim=0)

    # compute the diagonal part
    row, col = edge_index
    cot_deg = scatter_add(cot_weight, row, dim=0, dim_size=num_nodes)
    area_deg = scatter_add(area_weight, row, dim=0, dim_size=num_nodes)
    deg = cot_deg / area_deg
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.cat([-cot_weight, deg], dim=0)
    return edge_index, edge_weight
