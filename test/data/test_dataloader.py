import torch
from torch_geometric.data import Data, DataLoader


def test_dataloader():
    x = torch.Tensor([[1], [1], [1]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    face = torch.tensor([[0], [1], [2]])

    data = Data(x=x, edge_index=edge_index, y=2)
    assert data.__repr__() == 'Data(edge_index=[2, 4], x=[3, 1], y=[1])'
    data.face = face

    loader = DataLoader([data, data], batch_size=2, shuffle=False)

    for batch in loader:
        assert len(batch) == 5
        assert batch.batch.tolist() == [0, 0, 0, 1, 1, 1]
        assert batch.x.tolist() == [[1], [1], [1], [1], [1], [1]]
        assert batch.edge_index.tolist() == [[0, 1, 1, 2, 3, 4, 4, 5],
                                             [1, 0, 2, 1, 4, 3, 5, 4]]
        assert batch.y.tolist() == [2, 2]
        assert batch.face.tolist() == [[0, 3], [1, 4], [2, 5]]

    loader = DataLoader([data, data],
                        batch_size=2,
                        shuffle=False,
                        follow_batch=['edge_index'])

    for batch in loader:
        assert len(batch) == 6
        assert batch.edge_index_batch.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
