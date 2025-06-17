from torch_geometric.data import Data
from torch_geometric.datasets.teeth3ds import Teeth3DS


def test_teeth3ds():
    dataset = Teeth3DS(root="teeth3ds", split="Teeth3DS", is_train=True)

    assert len(dataset) > 0
    data = dataset[0]
    assert isinstance(data, Data)
    assert data.pos.shape[1] == 3
    assert data.x.shape[0] == data.pos.shape[0]
    assert data.y.shape[0] == data.pos.shape[0]
    assert isinstance(data.jaw, str)
