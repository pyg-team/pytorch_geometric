

# import torch
import torch_geometric

# import torch_geometric
from torch_geometric.datasets import TUDataset
# from torch_geometric.datasets import TUDataset



dataset = TUDataset(root='./dataset/*', name ='REDDIT-BINARY')

print(dataset)