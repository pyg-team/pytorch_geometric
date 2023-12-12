import os

from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import copy

import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from torch_geometric.nn import GCNConv
from xgnn_model import GCN_Graph


# exrtact a mutagenic molecule from the dataset
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
data = dataset[0]

print(data)

# # plot the molecule
# import matplotlib.pyplot as plt
# from torch_geometric.utils import to_networkx
# G = to_networkx(data, to_undirected=True)
# plt.subplot(121)
# nx.draw(G, with_labels=True)
# plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = {'device': device,
        'dropout': 0.1,
        'epochs': 1000,
        'input_dim' : 7,
        'opt': 'adam',
        'opt_scheduler': 'none',
        'opt_restart': 0,
        'weight_decay': 5e-5,
        'lr': 0.001}
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        
args = objectview(args)
        
model = GCN_Graph(args.input_dim, output_dim=2, dropout=args.dropout).to(device)

# Assume 'model_to_freeze' is the model you want to freeze
for param in model.parameters():
    param.requires_grad = False
    
# depending on os change path
path = "examples/explain/xgnn/mutag_model.pth"
if os.name == 'nt':
    path = "examples\\explain\\xgnn\\mutag_model.pth"

model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.to(device)

# predict the class of the molecule
gnn_output = model(data)
print(gnn_output)