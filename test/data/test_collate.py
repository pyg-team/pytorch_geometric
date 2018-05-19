# def test_collate_to_set():
#     x1 = torch.tensor([1, 2, 3], dtype=torch.float)
#     e1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
#     x2 = torch.tensor([1, 2], dtype=torch.float)
#     e2 = torch.tensor([[0, 1], [1, 0]])

#     data, slices = collate_to_set([Data(x1, e1), Data(x2, e2)])

#     assert len(data) == 2
#     assert data.x.tolist() == [1, 2, 3, 1, 2]
#     data.edge_index.tolist() == [[0, 1, 1, 2, 0, 1], [1, 0, 2, 1, 1, 0]]
#     assert len(slices.keys()) == 2
#     assert slices['x'].tolist() == [0, 3, 5]
#     assert slices['edge_index'].tolist() == [0, 4, 6]
