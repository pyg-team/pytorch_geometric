import os
import os.path as osp
import warnings

import pytest
import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphSAGE
from torch_geometric.profile import (
    get_stats_summary,
    profileit,
    rename_profile_file,
    timeit,
)
from torch_geometric.profile.profile import torch_profile, xpu_profile
from torch_geometric.testing import (
    onlyCUDA,
    onlyLinux,
    onlyOnline,
    onlyXPU,
    withDevice,
    withPackage,
)


@withDevice
@onlyLinux
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
@onlyOnline
@withPackage('pytorch_memlab')
def test_profileit_cuda(get_dataset):
    warnings.filterwarnings('ignore', '.*arguments of DataFrame.drop.*')

    dataset = get_dataset(name='Cora')
    data = dataset[0].cuda()
    model = GraphSAGE(dataset.num_features, hidden_channels=64, num_layers=3,
                      out_channels=dataset.num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    @profileit('cuda')
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
        assert stats.time > 0
        assert stats.max_allocated_gpu > 0
        assert stats.max_reserved_gpu > 0
        assert stats.max_active_gpu > 0
        assert stats.nvidia_smi_free_cuda > 0
        assert stats.nvidia_smi_used_cuda > 0

        if epoch >= 2:  # Warm-up
            stats_list.append(stats)

    stats_summary = get_stats_summary(stats_list)
    assert stats_summary.time_mean > 0
    assert stats_summary.time_std > 0
    assert stats_summary.max_allocated_gpu > 0
    assert stats_summary.max_reserved_gpu > 0
    assert stats_summary.max_active_gpu > 0
    assert stats_summary.min_nvidia_smi_free_cuda > 0
    assert stats_summary.max_nvidia_smi_used_cuda > 0


@onlyXPU
def test_profileit_xpu(get_dataset):
    warnings.filterwarnings('ignore', '.*arguments of DataFrame.drop.*')

    dataset = get_dataset(name='Cora')
    device = torch.device('xpu')
    data = dataset[0].to(device)
    model = GraphSAGE(dataset.num_features, hidden_channels=64, num_layers=3,
                      out_channels=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    @profileit('xpu')
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
        assert stats.time > 0
        assert stats.max_allocated_gpu > 0
        assert stats.max_reserved_gpu > 0
        assert stats.max_active_gpu > 0
        assert not hasattr(stats, 'nvidia_smi_free_cuda')
        assert not hasattr(stats, 'nvidia_smi_used_cuda')

        if epoch >= 2:  # Warm-up
            stats_list.append(stats)

    stats_summary = get_stats_summary(stats_list)
    assert stats_summary.time_mean > 0
    assert stats_summary.time_std > 0
    assert stats_summary.max_allocated_gpu > 0
    assert stats_summary.max_reserved_gpu > 0
    assert stats_summary.max_active_gpu > 0
    assert not hasattr(stats_summary, 'min_nvidia_smi_free_cuda')
    assert not hasattr(stats_summary, 'max_nvidia_smi_used_cuda')


@withDevice
@onlyOnline
def test_torch_profile(capfd, get_dataset, device):
    dataset = get_dataset(name='Cora')
    data = dataset[0].to(device)
    model = GraphSAGE(dataset.num_features, hidden_channels=64, num_layers=3,
                      out_channels=dataset.num_classes).to(device)

    with torch_profile():
        model(data.x, data.edge_index)

    out, _ = capfd.readouterr()
    assert 'Self CPU time total' in out
    if data.x.is_cuda:
        assert 'Self CUDA time total' in out

    rename_profile_file('test_profile')
    assert osp.exists('profile-test_profile.json')
    os.remove('profile-test_profile.json')


@onlyXPU
@onlyOnline
@pytest.mark.parametrize('export_chrome_trace', [False, True])
def test_xpu_profile(capfd, get_dataset, export_chrome_trace):
    dataset = get_dataset(name='Cora')
    device = torch.device('xpu')
    data = dataset[0].to(device)
    model = GraphSAGE(dataset.num_features, hidden_channels=64, num_layers=3,
                      out_channels=dataset.num_classes).to(device)

    with xpu_profile(export_chrome_trace):
        model(data.x, data.edge_index)

    out, _ = capfd.readouterr()
    assert 'Self CPU' in out
    if data.x.is_xpu:
        assert 'Self XPU' in out

    f_name = 'timeline.json'
    f_exists = osp.exists(f_name)
    if not export_chrome_trace:
        assert not f_exists
    else:
        assert f_exists
        os.remove(f_name)
