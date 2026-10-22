import copy

import pytest
import torch
from torch.nn import ReLU

from torch_geometric.nn import (
    DeepGCNLayer,
    GCNConv,
    GENConv,
    GraphNorm,
    InstanceNorm,
    LayerNorm,
)


class BatchAwareNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, x, *, batch=None):
        self.batches.append(batch)
        if batch is None:
            return x
        return x + batch.to(x.dtype).unsqueeze(-1)


class RecordingConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received = None

    def forward(self, x, edge_index, batch=None, marker=None):
        self.received = (edge_index, batch, marker)
        return x


class LinearConv(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lin = torch.nn.Linear(channels, channels)

    def forward(self, x, edge_index):
        return self.lin(x)


@pytest.mark.parametrize(
    'block_tuple',
    [('res+', 1), ('res', 1), ('dense', 2), ('plain', 1)],
)
@pytest.mark.parametrize('ckpt_grad', [True, False])
def test_deepgcn(block_tuple, ckpt_grad):
    block, expansion = block_tuple
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    conv = GENConv(8, 8)
    norm = LayerNorm(8)
    act = ReLU()
    layer = DeepGCNLayer(conv, norm, act, block=block, ckpt_grad=ckpt_grad)
    assert str(layer) == f'DeepGCNLayer(block={block})'

    out = layer(x, edge_index)
    assert out.size() == (3, 8 * expansion)


@pytest.mark.parametrize('block', ['res+', 'res', 'dense', 'plain'])
@pytest.mark.parametrize('norm_cls', [InstanceNorm, GraphNorm])
def test_deepgcn_norm_batch_graph_independence(block, norm_cls):
    torch.manual_seed(17)
    x = torch.tensor([
        [-2.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0],
        [0.0, 2.0, -1.0],
        [1.0, 3.0, 2.0],
    ])
    companion = torch.tensor([
        [20.0, 30.0, 10.0],
        [22.0, 28.0, 14.0],
        [18.0, 35.0, 7.0],
    ])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    companion_edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])

    norm = norm_cls(
        3,
        **({
            'track_running_stats': False
        } if norm_cls is InstanceNorm else {}))
    layer = DeepGCNLayer(
        GCNConv(3, 3),
        norm,
        torch.nn.Identity(),
        block=block,
    )
    layer.eval()

    out_solo = layer(
        x,
        edge_index,
        norm_batch=torch.zeros(x.size(0), dtype=torch.long),
    )

    node_offset = companion.size(0)
    batched_x = torch.cat([companion, x])
    batched_edges = torch.cat([
        companion_edge_index,
        edge_index + node_offset,
    ], dim=1)
    batch = torch.cat([
        torch.zeros(node_offset, dtype=torch.long),
        torch.ones(x.size(0), dtype=torch.long),
    ])
    out_batch = layer(batched_x, batched_edges, norm_batch=batch)
    out_batch = out_batch[node_offset:]

    torch.testing.assert_close(out_solo, out_batch, rtol=1e-5, atol=1e-6)


def test_deepgcn_norm_batch_routing():
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    conv_batch = torch.tensor([3, 3, 8, 8])
    norm_batch = torch.tensor([0, 0, 1, 1])
    marker = object()
    conv = RecordingConv()
    norm = BatchAwareNorm()
    layer = DeepGCNLayer(conv, norm, block='plain')

    layer(
        x,
        edge_index,
        batch=conv_batch,
        marker=marker,
        norm_batch=norm_batch,
    )

    assert conv.received == (edge_index, conv_batch, marker)
    assert norm.batches == [norm_batch]


def test_deepgcn_norm_batch_compatibility():
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    norm_batch = torch.tensor([0, 0, 1, 1])

    aware_norm = BatchAwareNorm()
    aware_layer = DeepGCNLayer(RecordingConv(), aware_norm, block='plain')
    out_without_batch = aware_layer(x, edge_index)
    assert aware_norm.batches == [None]
    out_with_batch = aware_layer(x, edge_index, norm_batch=norm_batch)
    assert aware_norm.batches[-1] is norm_batch
    torch.testing.assert_close(out_without_batch, x)
    torch.testing.assert_close(
        out_with_batch,
        x + norm_batch.float().unsqueeze(-1),
    )

    layer_norm = torch.nn.LayerNorm(3)
    layer = DeepGCNLayer(RecordingConv(), layer_norm, block='plain')
    torch.testing.assert_close(
        layer(x, edge_index),
        layer(x, edge_index, norm_batch=norm_batch),
    )

    no_norm = DeepGCNLayer(RecordingConv(), None, block='plain')
    torch.testing.assert_close(no_norm(x, edge_index), x)
    torch.testing.assert_close(
        no_norm(x, edge_index, norm_batch=norm_batch),
        x,
    )


def test_deepgcn_scripted_norm():
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    norm_batch = torch.tensor([0, 0, 1, 1])
    norm = torch.jit.script(torch.nn.LayerNorm(3))
    layer = DeepGCNLayer(RecordingConv(), norm, block='plain')

    out = layer(x, edge_index, norm_batch=norm_batch)

    torch.testing.assert_close(out, norm(x))


def test_deepgcn_traced_norm():
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    norm_batch = torch.tensor([0, 0, 1, 1])
    norm = torch.jit.trace(torch.nn.LayerNorm(3), torch.randn(2, 3))
    layer = DeepGCNLayer(RecordingConv(), norm, block='plain')

    out_without_batch = layer(x, edge_index)
    out_with_batch = layer(x, edge_index, norm_batch=norm_batch)

    torch.testing.assert_close(out_without_batch, norm(x))
    torch.testing.assert_close(out_with_batch, norm(x))


@pytest.mark.parametrize('block', ['res+', 'res', 'dense', 'plain'])
def test_deepgcn_checkpoint_gradient_parity(block):
    torch.manual_seed(23)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    batch = torch.tensor([0, 0, 0, 1, 1])
    base = DeepGCNLayer(
        LinearConv(4),
        GraphNorm(4),
        ReLU(),
        block=block,
        dropout=0.0,
    )
    checkpointed = copy.deepcopy(base)
    checkpointed.ckpt_grad = True

    x_base = x.clone().requires_grad_()
    x_checkpointed = x.clone().requires_grad_()
    out_base = base(x_base, edge_index, norm_batch=batch)
    out_checkpointed = checkpointed(
        x_checkpointed,
        edge_index,
        norm_batch=batch,
    )
    weights = torch.randn_like(out_base)
    loss_base = (out_base * weights).sum() + 0.1 * out_base.square().sum()
    loss_checkpointed = ((out_checkpointed * weights).sum() +
                         0.1 * out_checkpointed.square().sum())
    loss_base.backward()
    loss_checkpointed.backward()

    torch.testing.assert_close(
        out_base,
        out_checkpointed,
        rtol=1e-5,
        atol=1e-6,
    )
    assert x_base.grad is not None
    assert x_checkpointed.grad is not None
    torch.testing.assert_close(
        x_base.grad,
        x_checkpointed.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    for param_base, param_checkpointed in zip(
            base.parameters(),
            checkpointed.parameters(),
    ):
        assert param_base.grad is not None
        assert param_checkpointed.grad is not None
        torch.testing.assert_close(
            param_base.grad,
            param_checkpointed.grad,
            rtol=1e-5,
            atol=1e-6,
        )
