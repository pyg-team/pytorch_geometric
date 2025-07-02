from torch_geometric.data import Data
from torch_geometric.datasets import Teeth3DS
from torch_geometric.testing import withPackage


@withPackage('trimesh', 'fpsample')
def test_teeth3ds(tmp_path) -> None:
    dataset = Teeth3DS(root=tmp_path, split='sample', train=True)

    assert len(dataset) > 0
    data = dataset[0]
    assert isinstance(data, Data)
    assert data.pos.size(1) == 3
    assert data.x.size(0) == data.pos.size(0)
    assert data.y.size(0) == data.pos.size(0)
    assert isinstance(data.jaw, str)
