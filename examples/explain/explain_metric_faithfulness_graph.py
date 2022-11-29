import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool
from torch_geometric.explain.metrics.faithfulness import get_graph_faithfulness


dataset = "MUTAG"
batch_size = 128
lr = 0.01
epochs = 100
hidden_channels = 32
num_layers=5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

init_wandb(name=f'GIN-{dataset}', batch_size=batch_size, lr=lr,
           epochs=epochs, hidden_channels=hidden_channels,
           num_layers=num_layers, device=device)

path = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'data', 'TU')
dataset = TUDataset(path, name=dataset).shuffle()

train_dataset = dataset[len(dataset) // 10:]
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

test_dataset = dataset[:len(dataset) // 10]
test_loader = DataLoader(test_dataset, batch_size)


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


model = Net(dataset.num_features, hidden_channels, dataset.num_classes,
            num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


for epoch in range(1, epochs + 1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)


explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=300),
        explainer_config=dict(
            explanation_type="model",
            node_mask_type='object',
            edge_mask_type='object',
        ),
        model_config=dict(
            mode='classification',
            task_level='graph',
            return_type='raw',
        ),
    )


data = dataset.data.to(device)
add_args = {
    'batch': torch.zeros((1,)).long().to(device)
}

explanation = explainer(x=data.x, edge_index=data.edge_index,**add_args)

graph_idx = 10
gdata = dataset[graph_idx]
topk = 4
graph_faithfulness = get_graph_faithfulness(gdata, explanation, model, topk, device, **add_args)
print(f'Faithfulness for graph {graph_idx} faithfulness:{graph_faithfulness:.4f}')