import torch

from torch_geometric.data import Data, Batch
from torch_geometric.loader.gnan_dataloader import (
	GNANCollater,
	GNANDataLoader,
)
from torch_geometric.nn.models import TensorGNAN


def _dummy_data(num_nodes: int = 5, num_feats: int = 4):
	x = torch.randn(num_nodes, num_feats)
	edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
	# full distance matrix:
	rand_dist = torch.rand(num_nodes, num_nodes)
	rand_dist = (rand_dist + rand_dist.t()) / 2  # symmetric
	# Use a simple normalisation matrix of ones to avoid division issues
	norm = torch.ones_like(rand_dist)
	data = Data(x=x, edge_index=edge_index)
	data.node_distances = rand_dist
	data.normalization_matrix = norm
	return data


def test_gnan_collater_block_diag_and_restore():
	g1 = _dummy_data(num_nodes=3, num_feats=4)
	g2 = _dummy_data(num_nodes=4, num_feats=4)
	g3 = _dummy_data(num_nodes=2, num_feats=4)

	# Keep copies to verify restoration on originals
	d1 = g1.node_distances.clone()
	n1 = g1.normalization_matrix.clone()
	d2 = g2.node_distances.clone()
	n2 = g2.normalization_matrix.clone()
	d3 = g3.node_distances.clone()
	n3 = g3.normalization_matrix.clone()

	collate = GNANCollater()
	batch = collate([g1, g2, g3])

	assert isinstance(batch, Batch)
	N = g1.num_nodes + g2.num_nodes + g3.num_nodes
	assert batch.node_distances.shape == (N, N)
	assert batch.normalization_matrix.shape == (N, N)

	expected_dist = torch.block_diag(d1, d2, d3)
	expected_norm = torch.block_diag(n1, n2, n3)
	assert torch.allclose(batch.node_distances, expected_dist)
	assert torch.allclose(batch.normalization_matrix, expected_norm)

	# Original Data objects should have attributes restored and unchanged
	assert torch.allclose(g1.node_distances, d1)
	assert torch.allclose(g1.normalization_matrix, n1)
	assert torch.allclose(g2.node_distances, d2)
	assert torch.allclose(g2.normalization_matrix, n2)
	assert torch.allclose(g3.node_distances, d3)
	assert torch.allclose(g3.normalization_matrix, n3)


def test_gnan_dataloader_batch_content():
	g1 = _dummy_data(num_nodes=3, num_feats=3)
	g2 = _dummy_data(num_nodes=5, num_feats=3)
	loader = GNANDataLoader([g1, g2], batch_size=2, shuffle=False)
	batch = next(iter(loader))

	assert isinstance(batch, Batch)
	assert batch.x.size(0) == g1.num_nodes + g2.num_nodes
	assert hasattr(batch, 'node_distances') and hasattr(batch, 'normalization_matrix')

	expected_dist = torch.block_diag(g1.node_distances, g2.node_distances)
	expected_norm = torch.block_diag(g1.normalization_matrix, g2.normalization_matrix)
	assert torch.allclose(batch.node_distances, expected_dist)
	assert torch.allclose(batch.normalization_matrix, expected_norm)


def test_gnan_dataloader_with_tensor_gnan():
	g1 = _dummy_data(num_nodes=3, num_feats=4)
	g2 = _dummy_data(num_nodes=4, num_feats=4)
	loader = GNANDataLoader([g1, g2], batch_size=2, shuffle=False)
	batch = next(iter(loader))

	model = TensorGNAN(
		in_channels=4,
		out_channels=3,
		n_layers=2,
		hidden_channels=8,
		graph_level=True,
	)
	model.eval()

	# Batched forward vs. separate forwards
	out_batched = model(batch)  # [2, 3]
	out_sep = torch.cat([model(g1), model(g2)], dim=0)
	assert out_batched.shape == (2, 3)
	assert torch.allclose(out_batched, out_sep, atol=1e-5)
