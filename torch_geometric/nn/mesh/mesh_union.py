import torch
from torch.nn import ConstantPad2d


class MeshUnion:
    r"""This class is a utility class for Mesh pooling operation from
    `"MeshCNN: A Network with an Edge" <https://arxiv.org/abs/1809.05910>`_
    paper.
    This class implements Mesh Union operation which merges edge features by
    making an average feature vector:
    :math::
    `\mathbf{(p)_i} = \mathbf{avg}\left ( \mathbf(a)_i, \mathbf(b)_i,
                                          \mathbf(e)_i  \right )`
    While  math: `\mathbf(a), \mathbf(b), \mathbf(e)` are the three edges which
    are merged to the
    new edge math: `\mathbf(p)`. :math:`\mathbf{i}` represents the feature
    channel index.

    Args:
        n (int): total number of edges in the mesh structure.
        device (torch.device, optional):  the device on which a torch.Tensor is
        or will be allocated (default: `cpu`)
    """

    def __init__(self, n: int, device=torch.device('cpu')):
        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        self.groups = torch.eye(n, device=device)

    def union(self, source, target):
        self.groups[target, :] += self.groups[source, :]

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        return self.groups[edge_key, :]

    def get_occurrences(self):
        return torch.sum(self.groups, 0)

    def get_groups(self, tensor_mask):
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[tensor_mask, :]

    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)
        fe = torch.matmul(features.squeeze(-1), self.groups)
        occurrences = torch.sum(self.groups, 0).expand(fe.shape)
        fe = fe / occurrences
        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        return fe

    def prepare_groups(self, features, mask):
        tensor_mask = torch.from_numpy(mask)
        self.groups = torch.clamp(self.groups[tensor_mask, :], 0,
                                  1).transpose_(1, 0)
        padding_a = features.shape[1] - self.groups.shape[0]
        if padding_a > 0:
            padding_a = ConstantPad2d((0, 0, 0, padding_a), 0)
            self.groups = padding_a(self.groups)
