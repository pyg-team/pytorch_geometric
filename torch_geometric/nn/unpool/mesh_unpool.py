import torch
import torch.nn as nn


class MeshUnpool(nn.Module):
    r"""The Mesh Un-pooling operator from the `"MeshCNN: A Network with an Edge"
    <https://arxiv.org/abs/1809.05910>`_ paper
    This class implements the un-pooling operation on a Mesh structure.
    We define mesh un-pooling as the partial inverse of the MeshPooling operation. This operation increase the number of
    mesh edges and, therefore, increase the resolution of the feature activations.
    Un-pooling operation does not have any learned parameters and it uses the Mesh history in order to recover the
    original resolution of the mesh. Each un-pooled edge feature is a weighted combination of the pooled edge features.

    Args:
        unroll_target (int): number of edges after un-pooling operation.
    """

    def __init__(self, unroll_target: int):
        super(MeshUnpool, self).__init__()
        self.unroll_target = unroll_target

    def __call__(self, features, meshes):
        return self.forward(features, meshes)

    def pad_groups(self, group, unroll_start):
        start, end = group.shape
        padding_rows = unroll_start - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols !=0:
            padding = nn.ConstantPad2d((0, padding_cols, 0, padding_rows), 0)
            group = padding(group)
        return group

    def pad_occurrences(self, occurrences):
        padding = self.unroll_target - occurrences.shape[0]
        if padding != 0:
            padding = nn.ConstantPad1d((0, padding), 1)
            occurrences = padding(occurrences)
        return occurrences

    def forward(self, features, meshes):
        batch_size, nf, edges = features.shape
        groups = [self.pad_groups(mesh.get_groups(), edges) for mesh in meshes]
        unroll_mat = torch.cat(groups, dim=0).view(batch_size, edges, -1)
        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes]
        occurrences = torch.cat(occurrences, dim=0).view(batch_size, 1, -1)
        occurrences = occurrences.expand(unroll_mat.shape)
        unroll_mat = unroll_mat / occurrences
        unroll_mat = unroll_mat.to(features.device)
        for mesh in meshes:
            mesh.unroll_edge_indexes()
        return torch.matmul(features, unroll_mat)
