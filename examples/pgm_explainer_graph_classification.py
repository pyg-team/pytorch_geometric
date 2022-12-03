import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, MNISTSuperpixels
from torch_geometric.explain import Explainer, ExplainerConfig, ModelConfig
from torch_geometric.explain.algorithm import PGMExplainer
from torch_geometric.nn import GCNConv,GATv2Conv
from torch_geometric.loader import DataLoader

dataset = 'PROTEINS'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TUDataset')
transform = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
dataset = TUDataset(path, dataset, transform=transform, use_edge_attr=True ,use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(dataset.num_features, 16)
        self.conv2 = GATv2Conv(16, dataset.num_classes, normalize=False)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    data = dataset.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x, edge_index, edge_weight, target = data.x, data.edge_index, data.edge_weight, data.y

    for epoch in range(1, 20):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index, edge_weight)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


    explainer = Explainer(
        model=model, algorithm=PGMExplainer(),
        explainer_config=ExplainerConfig(explanation_type="phenomenon",
                                         node_mask_type="attributes",
                                         edge_mask_type="object"),
        model_config=ModelConfig(mode="classification", task_level="node",
                                 return_type="raw"))
    node_idx = 10
    explanation = explainer(x=data.x, edge_index=edge_index, index=node_idx,
                            target=target, edge_weight=edge_weight)
    print(explanation.available_explanations)
