# import torch
# from torch_geometric.nn.pool import graclus
# from torch_geometric.data import Batch

# def test_graclus_pool():
#     x = torch.Tensor([[1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]])
#     row = torch.LongTensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
#     col = torch.LongTensor([1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4])
#     edge_index = torch.stack([row, col], dim=0)
#     batch = torch.LongTensor([0, 0, 0, 1, 1, 1])
#     data = Batch(x=x, edge_index=edge_index, batch=batch)

#     data = graclus_pool(data)

#     assert data.num_nodes == 4
