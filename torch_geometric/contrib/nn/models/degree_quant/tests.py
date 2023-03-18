import torch_geometric
from quantized_layers import MessagePassingQuant
from layers import GCNConvQuant, GATConvQuant, GINConvQuant
from torch_geometric.datasets import TUDataset, Amazon
# from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='./dataset/*', name ='REDDIT-BINARY')
print(dataset.data)


import unittest

