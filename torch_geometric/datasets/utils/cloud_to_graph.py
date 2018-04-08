import scipy.spatial
import numpy as np
import torch


def compute_neighborhoods(data, radius):
    # Get neigbhors in radius
    tree = scipy.spatial.cKDTree(data)
    indices = tree.query_ball_tree(tree, radius)

    length = len(sorted(indices, key=len, reverse=True)[0])
    indices = np.array([x + [None] * (length - len(x)) for x in indices])

    # Nodes with no neighbors
    no_neighbors = indices[indices[:, 1] == None]  # noqa
    no_neighbor_idx = no_neighbors[:, 0]
    nns = np.array(tree.query(tree.data[indices[:, 1] == None], k=2))  # noqa
    nns = nns[1, :, 1]

    indices[np.int32(no_neighbor_idx), 1] = np.int32(nns)

    # Convert to sparse rep
    shape = indices.shape
    indices1 = np.reshape(
        np.transpose(np.tile(np.array(range(shape[0])), (shape[1], 1))),
        [shape[0] * shape[1]])
    indices2 = np.reshape(indices, [shape[0] * shape[1]])
    indices = np.stack([indices1, indices2], axis=1)

    # Eliminate Nones and self-edges
    indices = indices[indices[:, 1] != None]  # noqa
    indices = indices[indices[:, 0] != indices[:, 1]]
    return torch.LongTensor(indices.transpose().astype(np.long))
