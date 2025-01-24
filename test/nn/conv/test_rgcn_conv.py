import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import FastRGCNConv, RGCNConv
from torch_geometric.testing import is_full_test, withCUDA, withDevice
from torch_geometric.typing import SparseTensor

classes = [RGCNConv, FastRGCNConv]
confs = [(None, None), (2, None), (None, 2)]


@withDevice
@pytest.mark.parametrize('conf', confs)
def test_rgcn_conv_equality(conf, device):
    num_bases, num_blocks = conf

    x1 = torch.randn(4, 4, device=device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [0, 0, 1, 0, 1, 1],
    ], device=device)
    edge_type = torch.tensor([0, 1, 1, 0, 0, 1], device=device)

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    ], device=device)
    edge_type = torch.tensor([0, 1, 1, 0, 0, 1, 2, 3, 3, 2, 2, 3],
                             device=device)

    torch.manual_seed(12345)
    conv1 = RGCNConv(4, 32, 4, num_bases, num_blocks, aggr='sum').to(device)

    torch.manual_seed(12345)
    conv2 = FastRGCNConv(4, 32, 4, num_bases, num_blocks,
                         aggr='sum').to(device)

    out1 = conv1(x1, edge_index, edge_type)
    out2 = conv2(x1, edge_index, edge_type)
    assert torch.allclose(out1, out2, atol=1e-2)

    if num_blocks is None:
        out1 = conv1(None, edge_index, edge_type)
        out2 = conv2(None, edge_index, edge_type)
        assert torch.allclose(out1, out2, atol=1e-2)


@withCUDA
@pytest.mark.parametrize('cls', classes)
@pytest.mark.parametrize('conf', confs)
def test_rgcn_conv_basic(cls, conf, device):
    num_bases, num_blocks = conf

    x1 = torch.randn(4, 4, device=device)
    x2 = torch.randn(2, 16, device=device)
    idx1 = torch.arange(4, device=device)
    idx2 = torch.arange(2, device=device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [0, 0, 1, 0, 1, 1],
    ], device=device)
    edge_type = torch.tensor([0, 1, 1, 0, 0, 1], device=device)

    conv = cls(4, 32, 2, num_bases, num_blocks, aggr='sum').to(device)
    assert str(conv) == f'{cls.__name__}(4, 32, num_relations=2)'

    out1 = conv(x1, edge_index, edge_type)
    assert out1.size() == (4, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_type, (4, 4))
        assert torch.allclose(conv(x1, adj.t()), out1, atol=1e-3)

    if num_blocks is None:
        out2 = conv(None, edge_index, edge_type)
        assert torch.allclose(conv(idx1, edge_index, edge_type), out2, 1e-3)
        assert out2.size() == (4, 32)
        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(conv(None, adj.t()), out2, atol=1e-3)
            assert torch.allclose(conv(idx1, adj.t()), out2, atol=1e-3)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x1, edge_index, edge_type), out1, atol=1e-3)
        if num_blocks is None:
            assert torch.allclose(jit(idx1, edge_index, edge_type), out2,
                                  atol=1e-3)
            assert torch.allclose(jit(None, edge_index, edge_type), out2,
                                  atol=1e-3)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, adj.t()), out1)
            if num_blocks is None:
                assert torch.allclose(jit(idx1, adj.t()), out2, atol=1e-3)
                assert torch.allclose(jit(None, adj.t()), out2, atol=1e-3)

    # Test bipartite message passing:
    conv = cls((4, 16), 32, 2, num_bases, num_blocks, aggr='sum').to(device)
    assert str(conv) == f'{cls.__name__}((4, 16), 32, num_relations=2)'

    out1 = conv((x1, x2), edge_index, edge_type)
    assert out1.size() == (2, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_type, (4, 2))
        assert torch.allclose(conv((x1, x2), adj.t()), out1, atol=1e-3)

    if num_blocks is None:
        out2 = conv((None, idx2), edge_index, edge_type)
        assert out2.size() == (2, 32)
        assert torch.allclose(conv((idx1, idx2), edge_index, edge_type), out2,
                              atol=1e-3)
        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(conv((None, idx2), adj.t()), out2, atol=1e-3)
            assert torch.allclose(conv((idx1, idx2), adj.t()), out2, atol=1e-3)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit((x1, x2), edge_index, edge_type), out1,
                              atol=1e-3)
        if num_blocks is None:
            assert torch.allclose(jit((None, idx2), edge_index, edge_type),
                                  out2, atol=1e-3)
            assert torch.allclose(jit((idx1, idx2), edge_index, edge_type),
                                  out2, atol=1e-3)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit((x1, x2), adj.t()), out1, atol=1e-3)
            if num_blocks is None:
                assert torch.allclose(jit((None, idx2), adj.t()), out2,
                                      atol=1e-3)
                assert torch.allclose(jit((idx1, idx2), adj.t()), out2,
                                      atol=1e-3)
