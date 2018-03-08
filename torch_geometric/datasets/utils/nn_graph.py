import torch
import scipy.spatial
import numpy as np


def nn_graph(pos, k=6):
    row = torch.arange(0, pos.size(0)).view(-1, 1).repeat(1, k).view(-1).long()
    tree = scipy.spatial.cKDTree(pos.numpy())
    _, col = tree.query(pos, k + 1)
    col = torch.LongTensor(col[:len(col)])[:, 1:].contiguous().view(-1)
    return torch.stack([row, col], dim=0)


def radius_graph(data, radius):
    # Get neigbhors in radius
    tree = scipy.spatial.cKDTree(data)
    indices = tree.query_ball_tree(tree, radius)

    length = len(sorted(indices, key=len, reverse=True)[0])
    length = max(length,1)
    indices = np.array([x + [None] * (length - len(x)) for x in indices])

    # Nodes with no neighbors
    no_neighbors = indices[indices[:, 1] == None]
    no_neighbor_idx = no_neighbors[:, 0]
    nns = np.array(tree.query(tree.data[indices[:, 1] == None], k=2))[1, :, 1]

    indices[np.int32(no_neighbor_idx), 1] = np.int32(nns)

    # Convert to sparse rep
    shape = indices.shape
    indices1 = np.reshape(
        np.transpose(np.tile(np.array(range(shape[0])), (shape[1], 1))),
        [shape[0] * shape[1]])
    indices2 = np.reshape(indices, [shape[0] * shape[1]])
    indices = np.stack([indices1, indices2], axis=1)

    # Eliminate Nones and self-edges
    indices = indices[indices[:, 1] != None]
    indices = indices[indices[:, 0] != indices[:, 1]]
    return torch.LongTensor(indices.transpose().astype(np.long))