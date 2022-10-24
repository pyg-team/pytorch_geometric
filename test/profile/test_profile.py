import os.path

import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphSAGE
from torch_geometric.profile import (
    get_stats_summary,
    profileit,
    rename_profile_file,
    timeit,
)
from torch_geometric.profile.profile import torch_profile
from torch_geometric.testing import onlyFullTest, withCUDA


@withCUDA
@onlyFullTest
def test_profile(get_dataset):
    dataset = get_dataset(name='PubMed')
    data = dataset[0].cuda()
    model = GraphSAGE(dataset.num_features, hidden_channels=64, num_layers=3,
                      out_channels=dataset.num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    @profileit()
    def train(model, x, edge_index, y):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out, y)
        loss.backward()
        return float(loss)

    @timeit()
    @torch.no_grad()
    def test(model, x, edge_index, y):
        model.eval()
        out = model(x, edge_index).argmax(dim=-1)
        return int((out == y).sum()) / y.size(0)

    stats_list = []
    for epoch in range(5):
        _, stats = train(model, data.x, data.edge_index, data.y)
        assert len(stats) == 6
        assert stats.time > 0
        assert stats.max_allocated_cuda > 0
        assert stats.max_reserved_cuda > 0
        assert stats.max_active_cuda > 0
        assert stats.nvidia_smi_free_cuda > 0
        assert stats.nvidia_smi_used_cuda > 0

        _, time = test(model, data.x, data.edge_index, data.y)
        assert time > 0

        if epoch >= 2:  # Warm-up
            stats_list.append(stats)

    stats_summary = get_stats_summary(stats_list)
    assert len(stats_summary) == 7
    assert stats_summary.time_mean > 0
    assert stats_summary.time_std > 0
    assert stats_summary.max_allocated_cuda > 0
    assert stats_summary.max_reserved_cuda > 0
    assert stats_summary.max_active_cuda > 0
    assert stats_summary.min_nvidia_smi_free_cuda > 0
    assert stats_summary.max_nvidia_smi_used_cuda > 0


@onlyFullTest
def test_torch_profile(get_dataset):
    dataset = get_dataset(name='PubMed')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    model = GraphSAGE(dataset.num_features, hidden_channels=64, num_layers=3,
                      out_channels=dataset.num_classes).to(device)
    model.eval()

    @timeit()
    def inference_e2e(model, data):
        model(data.x, data.edge_index)

    @torch_profile()
    def inference_profile(model, data):
        model(data.x, data.edge_index)

    for epoch in range(3):
        inference_e2e(model, data)
        inference_profile(model, data)
    rename_profile_file('test_profile')
    assert os.path.exists('profile-test_profile.json')
