import scipy.spatial
import numpy as np
import time


def compute_neighborhoods(data, radius):
    # Get neigbhors in radius
    tree = scipy.spatial.cKDTree(data)
    indices = tree.query_ball_tree(tree, radius)
    # indices = np.array(list(tree.sparse_distance_matrix(tree,max_distance=radius).keys()))
    length = len(sorted(indices, key=len, reverse=True)[0])
    indices = np.array([x + [None] * (length - len(x)) for x in indices])

    # Nodes with no neighbors
    no_neighbors = indices[indices[:, 1] is None]
    no_neighbor_idx = no_neighbors[:, 0]
    nns = np.array(tree.query(tree.data[indices[:, 1] is None], k=2))[1, :, 1]

    indices[np.int32(no_neighbor_idx), 1] = np.int32(nns)

    # Convert to sparse rep
    shape = indices.shape
    indices1 = np.reshape(np.transpose(np.tile(np.array(range(shape[0])), (shape[1], 1))), [shape[0] * shape[1]])
    indices2 = np.reshape(indices, [shape[0] * shape[1]])
    indices = np.stack([indices1, indices2], axis=1)

    # Eliminate Nones and self-edges
    indices = indices[indices[:, 1] is not None]
    indices = indices[indices[:, 0] != indices[:, 1]]
    return indices


test_data = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.1, 0.0],
                      [0.1, 0.0, 0.0],
                      [0.1, 0.1, 0.0],
                      [-0.1, 0.0, 0.0],
                      [-0.3, 0.0, 0.0],
                      [-0.1, 0.2, 0.0]])

test_file = '/media/jan/DataExt4/BenchmarkDataSets/Point Clouds/KITTI/City/2011_09_26/2011_09_26_drive_0001_extract' \
            '/velodyne_points/data/0000000000.txt'

data_from_file = []
F = open(test_file, 'r')
for line in F:
    splits = line.split()
    data_from_file.append((float(splits[0]), float(splits[1]), float(splits[2])))

data_from_file = np.array(data_from_file)

print('Number of Points:', data_from_file.shape[0])
starttime = time.clock()

print(compute_neighborhoods(data_from_file, 0.05))
print('Compute Edges in seconds:', time.clock()-starttime)
