import unittest
import torch

if torch.cuda.is_available():
    from .voxel_cluster_gpu import voxel_cluster_gpu


class VoxelClusterGPUTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), 'no GPU')
    def test_voxel_cluster(self):
        position = [[-2, -2], [2, 2], [-1, -1], [1, 1]]
        position = torch.cuda.FloatTensor(position)
        cluster_size = torch.cuda.LongTensor([2, 2])
        K = cluster_size.prod()
        cluster = voxel_cluster_gpu(position, cluster_size, K)
        print(cluster)
