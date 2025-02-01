import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GAE, SAGEConv
from torch_geometric.utils import train_test_split_edges

path = osp.join( '.', 'data', 'Planetoid')
dataset = Planetoid(path, 'Cora', split='random', transform=T.NormalizeFeatures(),
					num_train_per_class=200, num_val=500, num_test=1000)

data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)

train_loader = NeighborSampler(data.train_pos_edge_index, node_idx = None,
							   sizes=[25, 10], batch_size=data.x.shape[0], shuffle = True,
							   num_workers=1)

class Encoder(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels):
		super(Encoder, self).__init__()

		self.num_layers = 2

		self.convs = torch.nn.ModuleList()
		self.convs.append(SAGEConv(in_channels, hidden_channels))
		self.convs.append(SAGEConv(hidden_channels, hidden_channels))

	def forward(self, x, adjs):
		for i, (edge_index, _, size) in enumerate(adjs):
			x_target = x[:size[1]]  # Target nodes are always placed first.
			x = self.convs[i]((x, x_target), edge_index)
			if i != self.num_layers - 1:
				x = F.relu(x)
				x = F.dropout(x, p=0.5, training=self.training)
		return x

def train():
	model.train()
	total_loss = 0
	for batch_size, n_id, adjs in train_loader:
		adjs = [adj.to(dev) for adj in adjs]
		optimizer.zero_grad()
		z = model.encode(x, adjs)
		loss = model.recon_loss(z, train_pos_edge_index)
		loss.backward()
		optimizer.step()
		total_loss += float(loss)
	loss = total_loss / len(train_loader)
	return loss

def test(pos_edge_index, neg_edge_index):
	model.eval()
	with torch.no_grad():
		for _, _, adjs in train_loader:
			adjs = [adj.to(dev) for adj in adjs]
			z = model.encode(x, adjs)
	return model.test(z, pos_edge_index, neg_edge_index)

latent_dim = 32 # hyperparameter
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GAE(Encoder(dataset.num_features, latent_dim)).to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)

for epoch in range(1, 151):
	loss = train()
	auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
	print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, Loss: {:.4f}'.format(epoch, auc, ap, loss))
