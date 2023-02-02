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
from torch_geometric.testing import onlyCUDA, onlyFullTest, onlyUnix, withCUDA


@withCUDA
@onlyUnix
def test_timeit(device):
    x = torch.randn(100, 16, device=device)
    lin = torch.nn.Linear(16, 32).to(device)

    with timeit(log=False) as t:
        assert not hasattr(t, 'duration')

        with torch.no_grad():
            lin(x)
        t.reset()
        assert t.duration > 0

        del t.duration
        assert not hasattr(t, 'duration')
    assert t.duration > 0


@onlyCUDA
@onlyFullTest
def test_profile(get_dataset):
    dataset = get_dataset(name='Cora')
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


@withCUDA
def test_torch_profile(get_dataset, device):
    dataset = get_dataset(name='Cora')
    data = dataset[0].to(device)
    model = GraphSAGE(dataset.num_features, hidden_channels=64, num_layers=3,
                      out_channels=dataset.num_classes).to(device)

    with torch_profile():
        model(data.x, data.edge_index)

    rename_profile_file('test_profile')
    assert os.path.exists('profile-test_profile.json')
    os.remove('profile-test_profile.json')
