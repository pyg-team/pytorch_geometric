import torch

from .undirected import to_undirected


def face_to_edge_index(face, num_nodes=None):
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index
