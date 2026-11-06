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


@pytest.mark.parametrize('dtype', [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
])
@pytest.mark.parametrize('conf, input_kind', [
    ((None, None), 'features'),
    ((2, None), 'features'),
    ((None, 2), 'features'),
    ((None, None), 'indices'),
    ((2, None), 'indices'),
    ((None, None), 'none'),
    ((2, None), 'none'),
])
@pytest.mark.parametrize('aggr', ['sum', 'mean'])
def test_rgcn_conv_output_dtype(dtype, conf, input_kind, aggr, monkeypatch):
    monkeypatch.setattr(torch_geometric.backend, 'use_segment_matmul', False)
    conv = RGCNConv(4, 4, 3, *conf, aggr=aggr).to(dtype)
    with torch.no_grad():
        for param in conv.parameters():
            param.copy_(
                torch.arange(param.numel()).reshape(
                    param.shape).remainder(5).to(dtype) / 8)
    features = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=dtype)
    if input_kind == 'features':
        features = torch.tensor([
            [0.5, 1, -0.25, 0],
            [0, -0.5, 1, 0.25],
            [1, 0.25, 0, -0.5],
            [-0.25, 0, 0.5, 1],
        ], dtype=dtype, requires_grad=True)
    x = (features if input_kind == 'features' else
         torch.arange(4) if input_kind == 'indices' else None)
    edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 1, 2, 0, 2]])
    edge_type = torch.tensor([0, 0, 1, 1, 0])
    out = conv(x, edge_index, edge_type)
    assert out.dtype == dtype
    assert torch.nn.Linear(4, 1).to(dtype)(out).dtype == dtype

    # Dense adjacency multiplication supplies a message-passing-free oracle.
    expected = features @ conv.root + conv.bias
    weights = conv.weight
    if conv.num_bases is not None:
        weights = (conv.comp @ weights.flatten(1)).reshape(3, 4, 4)
    for relation in range(3):
        adjacency = torch.zeros(4, 4, dtype=dtype)
        for src, dst in edge_index[:, edge_type == relation].t():
            adjacency[dst, src] += 1
        if aggr == 'mean':
            adjacency /= adjacency.sum(1, keepdim=True).clamp(min=1)
        weight = weights[relation]
        if conv.num_blocks is not None:
            weight = torch.block_diag(*weight.unbind())
        expected = expected + adjacency @ features @ weight
    torch.testing.assert_close(out, expected)
    parameters = tuple(conv.parameters())
    if input_kind == 'features':
        parameters = parameters + (features, )
    actual_grad = torch.autograd.grad(out.square().sum(), parameters,
                                      retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference)


def test_rgcn_conv_explicit_dtype_with_different_default(monkeypatch):
    monkeypatch.setattr(torch_geometric.backend, 'use_segment_matmul', False)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        conv = RGCNConv(2, 2, 1, root_weight=False, bias=False).float()
        x = torch.eye(2, dtype=torch.float32)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        out = conv(x, edge_index, torch.zeros(2, dtype=torch.long))
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, x.flip(0) @ conv.weight[0])
    finally:
        torch.set_default_dtype(original_dtype)
