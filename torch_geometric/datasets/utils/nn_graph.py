import torch
import scipy.spatial


def nn_graph(pos, k=6):
    row = torch.arange(0, pos.size(0)).view(-1, 1).repeat(1, k).view(-1).long()
    tree = scipy.spatial.cKDTree(pos.numpy())
    _, col = tree.query(pos, k + 1)
    col = torch.LongTensor(col[:len(col)])[:, 1:].contiguous().view(-1)
    return torch.stack([row, col], dim=0)
