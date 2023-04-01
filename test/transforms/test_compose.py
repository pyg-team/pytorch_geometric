import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data


def test_compose():
    transform = T.Compose([T.Center(), T.AddSelfLoops()])
    assert str(transform) == ('Compose([\n'
                              '  Center(),\n'
                              '  AddSelfLoops()\n'
                              '])')

    pos = torch.Tensor([[0, 0], [2, 0], [4, 0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    data = Data(edge_index=edge_index, pos=pos)
    data = transform(data)
    assert len(data) == 2
    assert data.pos.tolist() == [[-2, 0], [0, 0], [2, 0]]
    assert data.edge_index.size() == (2, 7)

def test_compose_filters():
    filter = T.ComposeFilters([
        lambda d: d.num_nodes > 2,
        lambda d: d.num_edges > 2
    ])

    data = [
        Data(
            x=torch.tensor([[0.], [1.], [2.]])
        ),
        Data(
            x=torch.tensor([[0.], [1.]]),
            edge_index=torch.tensor([[0, 0, 1], [0, 1, 1]])
        ),
        Data(
            x=torch.tensor([[0.], [1.], [2.]]),
            edge_index=torch.tensor([[0, 0, 1], [0, 1, 1]])
        ),
    ]

    mask = filter(data[0])
    assert mask is False

    mask = filter(data)
    assert mask == [False, False, True]
