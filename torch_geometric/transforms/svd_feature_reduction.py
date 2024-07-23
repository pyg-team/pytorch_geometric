import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('svd_feature_reduction')
class SVDFeatureReduction(BaseTransform):
    r"""Dimensionality reduction of node features via Singular Value
    Decomposition (SVD) (functional name: :obj:`svd_feature_reduction`).

    Args:
        out_channels (int): The dimensionlity of node features after
            reduction.
    """
    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def forward(self, data: Data) -> Data:
        assert data.x is not None

        if data.x.size(-1) > self.out_channels:
            U, S, _ = torch.linalg.svd(data.x)
            data.x = torch.mm(U[:, :self.out_channels],
                              torch.diag(S[:self.out_channels]))
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'
