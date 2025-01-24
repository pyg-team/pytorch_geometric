from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform, Center


@functional_transform('normalize_scale')
class NormalizeScale(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """
    def __init__(self) -> None:
        self.center = Center()

    def forward(self, data: Data) -> Data:
        data = self.center(data)

        assert data.pos is not None
        scale = (1.0 / data.pos.abs().max()) * 0.999999
        data.pos = data.pos * scale

        return data
