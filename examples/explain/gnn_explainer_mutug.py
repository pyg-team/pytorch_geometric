import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import MLP, GINConv, global_add_pool
from torch_geometric.loader import DataLoader

dataset = "Mutagenicity"
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TUDataset')
dataset = TUDataset(path, dataset)
train_loader = DataLoader(dataset[:0.9], batch_size=128, shuffle=True)
test_loader = DataLoader(dataset[0.9:], batch_size=128)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch): # N nodes, batch=[0-127]^N, bs=128 graphs, graph index
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(
    in_channels=dataset.num_features,
    hidden_channels=32,
    out_channels=dataset.num_classes,
    num_layers=5,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        
def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)  
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
    return float(loss)

@torch.no_grad()
def test(loader):
    model.eval()
    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

pbar = tqdm(range(100))
for epoch in pbar:
    loss = train()
    if epoch == 1 or epoch % 200 == 0:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        pbar.set_description(f'Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                             f'Test: {test_acc:.4f}')
pbar.close()
model.eval()

'''
    0	C
	1	O
	2	Cl
	3	H
	4	N
	5	F
	6	Br
	7	S
	8	P
	9	I
	10	Na
	11	K
	12	Li
	13	Ca
'''
color_dict = {  0: 'blue',
                1: 'green',
                2: 'grey',
                3: 'orange',
                4: 'lightblue',
                5: 'grey',
                6: 'grey',
                7: 'grey',
                8: 'grey',
                9: 'grey',
                10: 'grey',
                11: 'grey',
                12: 'grey',
                13: 'grey',
                }

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=300),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',  # multiclass_classification, binary_classification
        task_level='graph',
        return_type='raw',
    ),
)

for graph_id in range(100):
    if dataset[graph_id].y == 0: # Mutagenicity Graph 
        explanation = explainer(dataset[graph_id].x.to(device), dataset[graph_id].edge_index.to(device), batch=dataset[graph_id].batch)
        node_label = torch.argmax(dataset[graph_id].x, axis=1)
        print(f'Generated explanations in {explanation.available_explanations}')
        explanation.visualize_graph(f"./output/GNNExplainer/Mutagenicity/{graph_id}.png", 
                                node_label = node_label,
                                color_dict = color_dict,
                                draw_node_idx = False,
                                )

