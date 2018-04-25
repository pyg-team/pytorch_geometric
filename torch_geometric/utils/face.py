import torch
from torch_geometric.utils import coalesce


def face_to_edge_index(face, num_nodes=None):
    edge_index = torch.cat([face[:, :2], face[:, 1:], face[:, ::2]], dim=0)
    edge_index, _ = coalesce(edge_index.t(), num_nodes)

    return edge_index
