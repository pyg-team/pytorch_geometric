import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, one_hot


@functional_transform('one_hot_degree')
class OneHotDegree(BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features
    (functional name: :obj:`one_hot_degree`).

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """
    def __init__(
        self,
        max_degree: int,
        in_degree: bool = False,
        cat: bool = True,
    ) -> None:
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg = one_hot(deg, num_classes=self.max_degree + 1)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_degree})'
