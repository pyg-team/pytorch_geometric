import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import GPSConv, SAGEConv
from torch_geometric.testing import withCUDA
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_dense_batch, to_torch_csc_tensor


@pytest.mark.parametrize('attn_type', ['multihead', 'performer'])
@pytest.mark.parametrize('norm', [None, 'batch_norm', 'layer_norm'])
def test_gps_conv(norm, attn_type):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    batch = torch.tensor([0, 0, 1, 1])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = GPSConv(16, conv=SAGEConv(16, 16), heads=4, norm=norm,
                   attn_type=attn_type)
    conv.reset_parameters()
    assert str(conv) == (f'GPSConv(16, conv=SAGEConv(16, 16, aggr=mean), '
                         f'heads=4, attn_type={attn_type})')

    out = conv(x, edge_index)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj2.t()), out, atol=1e-6)

    out = conv(x, edge_index, batch)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t(), batch), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert torch.allclose(conv(x, adj2.t(), batch), out, atol=1e-6)


@withCUDA
def test_gps_conv_mha_fastpath(device: torch.device, monkeypatch):
    # `merge_masks` is only called on the native MHA fastpath, so record
    # whether a padding mask is present whenever the fastpath is taken:
    calls = []
    merge_masks = torch.nn.MultiheadAttention.merge_masks

    def spy(self, attn_mask, key_padding_mask, query):
        calls.append(key_padding_mask is not None)
        return merge_masks(self, attn_mask, key_padding_mask, query)

    monkeypatch.setattr(torch.nn.MultiheadAttention, 'merge_masks', spy)

    x = torch.randn(6, 16, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], device=device)
    batch = torch.tensor([0, 0, 1, 1, 1, 1], device=device)

    conv = GPSConv(16, conv=None, heads=2).to(device)
    conv.eval()

    # Positive control: a boolean key padding mask takes the fastpath, so the
    # assertions below cannot pass vacuously with the fastpath unavailable:
    h, mask = to_dense_batch(x, batch)
    with torch.no_grad():
        conv.attn(h, h, h, ~mask, need_weights=False)
    assert calls == [True]
    calls.clear()

    with torch.no_grad():
        out = conv(x, edge_index, batch)
    assert out.size() == (6, 16)
    assert out.isfinite().all()

    # Masked attention must avoid the fastpath on CUDA:
    if device.type == 'cuda':
        assert calls == []
    else:
        assert calls == [True]
