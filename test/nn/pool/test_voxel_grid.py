# import torch
# from torch_geometric.nn.pool.voxel_grid import voxel_grid

# def test_voxel_grid():
#     pos = torch.Tensor([[0.5, 0.5], [1.2, 1.5], [0.2, 1.7], [0.3, 0.4],
#                         [0.5, 0.5], [1.2, 1.5], [0.2, 1.7], [0.3, 0.4]])
#     batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])

#     cluster, batch = voxel_grid(pos, size=1, start=0, end=2, batch=batch)
#     assert cluster.tolist() == [0, 2, 1, 0, 3, 5, 4, 3]
#     assert batch.tolist() == [0, 0, 0, 1, 1, 1]
