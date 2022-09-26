import os.path as osp

from torch_geometric.datasets import HydroNet

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'HydroNet')
dataset = HydroNet(path)
