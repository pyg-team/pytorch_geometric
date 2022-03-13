import torch


def get_meshes_edge_index(meshes):
    num_edges_len = max(
        [mesh.mesh_data.edge_index.shape[1] for mesh in meshes])
    edge_index = []
    for mesh in meshes:
        padding = -1*torch.ones(2,
                              num_edges_len - mesh.mesh_data.edge_index.shape[
                                  1])
        edge_index.append(
            torch.cat([mesh.mesh_data.edge_index, padding], dim=1))

    return torch.stack([e for e in edge_index]).long()
