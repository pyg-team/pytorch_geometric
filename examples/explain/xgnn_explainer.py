import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, XGNNExplainer, XGNNGenerator
from torch_geometric.nn import GCNConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset)
data = dataset[0]

### REPLACE WITH BETTER DATASET EXAMPLE ################################

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

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

#########################################################################

explainer = Explainer(
    model=model,
    algorithm=XGNNExplainer(generative_model = RLGraphGen, epochs = 200),
    explanation_type='model',
    # node_mask_type='attributes',
    # edge_mask_type='object',
    model_config=dict(
        # ADD ATRIBUTES
        # mode='multiclass_classification',
        # task_level='node',
        # return_type='log_probs',
    ),
)

class_index = 1
explanation = explainer(data.x, data.edge_index) # explained_class=class_index
print(explanation)

# print(f'Generated explanations in {explanation.available_explanations}') # ??

# path = "explanation_graph.png"
# explanation.visualize_subgraph(path, )

# path = 'feature_importance.png'
# explanation.visualize_feature_importance(path, top_k=10)
# print(f"Feature importance plot has been saved to '{path}'")

# path = 'subgraph.pdf'
# explanation.visualize_graph(path)
# print(f"Subgraph visualization plot has been saved to '{path}'")
