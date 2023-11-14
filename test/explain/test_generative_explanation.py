import os.path as osp
from typing import Optional, Union

import pytest
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, '/Users/blazpridgar/Documents/GitHub/pytorch_geometric')

from torch_geometric.data import Data
from torch_geometric.explain import GenerativeExplanation
from torch_geometric.explain.config import MaskType
from torch_geometric.testing import withPackage


from torch_geometric.explain import XGNNExplainer, XGNNGenerator, Explainer

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

print("GENERATIVE EXPLANATION TEST")

# TODO: Get our acyclic dataset
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset)
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class RLGraphGen(XGNNGenerator):
    def __init__(self):
        super().__init__()
    def train(self, model):
        pass
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 21):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

#########################################################################

explainer = Explainer(
    model=model,
    algorithm=XGNNExplainer(generative_model = XGNNGenerator(), epochs = 200),
    explanation_type='model',
    node_mask_type=None,
    edge_mask_type=None,
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

print("EXPLAINER DONE!")

class_index = 1
explanation = explainer(data.x, data.edge_index) # Generates explanations for all classes at once
print(explanation)