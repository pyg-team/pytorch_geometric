import inspect
from typing import Dict, Union

import torch
from torch import LongTensor, Tensor

from torch_geometric.nn.edge_updater import EdgeUpdater
from torch_geometric.utils import scatter


def copy_signature(frm, to):
    def wrapper(*args, **kwargs):
        return to(*args, **kwargs)
    wrapper.__signature__ = inspect.signature(frm)
    return wrapper


class Fill(EdgeUpdater):
    def __init__(self, value: Union[float, int]):
        super().__init__()
        self.value = value

    def forward(self, edge_weight) -> Dict:
        return {
            'edge_weight': torch.fill(edge_weight, self.value)
        }


class Cache(EdgeUpdater):
    def __init__(self, edge_updater: EdgeUpdater):
        super().__init__()
        self.edge_updater = edge_updater
        self._cached = None

    def reset_parameters(self) -> None:
        self.edge_updater.reset_parameters()
        self._cached = None

    def forward(self, *args, **kwargs) -> Dict:
        if self._cached is None:
            self._cached = self.edge_updater(*args, **kwargs)

        return self._cached

    def __getattribute__(self, item):
        value = object.__getattribute__(self, item)
        if item == 'forward':
            value = copy_signature(frm=self.edge_updater.forward, to=value)
        return value


class GCNNorm(EdgeUpdater):
    r"""TODO
    """
    def forward(self, edge_index_i: LongTensor, edge_index_j: LongTensor,  # TODO I might have mixed _i and _j
                edge_weight: Tensor, dim_size: int) -> Dict:
        deg = scatter(edge_weight, edge_index_i, dim=0, dim_size=dim_size, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[edge_index_i] * edge_weight * deg_inv_sqrt[edge_index_j]

        return {
            'edge_weight': edge_weight
        }


if __name__ == '__main__':
    # TODO REMOVE
    edge_weight = torch.randn(5)
    print(edge_weight)

    updater = Cache(Fill(2.))
    x = updater(edge_weight)
    print(x)
    x['edge_weight'][0] = 3
    print(updater(edge_weight))

    print(inspect.signature(updater.forward))

