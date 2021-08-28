import torch
from torch_geometric.transforms import BaseTransform


class FeatureReduction(BaseTransform):
    r''' Dimension reduction via Singular Value Decomposition (SVD)

     Args:
         d (int): The dimension after reduction. (default: :obj:`-1`)
     '''
    def __init__(self, d):
        self.d = d

    def __call__(self, data):
        if(data.x.shape[1] > self.d):
            U, S, Vh = torch.linalg.svd(data.x)
            data.x = torch.mm(U[:, 0:self.d], torch.diag(S[0:self.d]))
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.d)
