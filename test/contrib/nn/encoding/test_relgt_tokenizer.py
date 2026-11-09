import pytest
import torch

from torch_geometric.contrib.nn.encoding import RelGTTokenizer, SubgraphGNNPE
from torch_geometric.testing import withDevice

NODE_TYPES = ['user', 'item']


class DictEncoder(torch.nn.Module):
    # Stand-in for a per-table encoder (e.g., RelBench's `HeteroEncoder`)
    # that maps raw per-node-type features to `channels`-dimensional ones:
    def __init__(self, in_channels, channels):
        super().__init__()
        self.lins = torch.nn.ModuleDict({
            key:
            torch.nn.Linear(in_channels, channels)
            for key in NODE_TYPES
        })

    def forward(self, feat_dict):
        return {key: self.lins[key](x) for key, x in feat_dict.items()}


class IdentityEncoder(torch.nn.Module):
    def forward(self, feat_dict):
        return feat_dict


class ZeroPE(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.batch = None

    def forward(self, edge_index, num_nodes, batch=None, batch_size=None):
        self.batch = batch
        self.batch_size = batch_size
        return torch.zeros(num_nodes, self.channels)


def get_inputs(device=None):
    # Two seed nodes with four tokens each:
    feat_dict = {
        'user': torch.randn(3, 4, device=device),
        'item': torch.randn(5, 4, device=device),
    }
    node_idx = torch.tensor([[0, 2, 4, 1], [1, 0, 3, 3]], device=device)
    node_type = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 1]], device=device)
    hop = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 2]], device=device)
    rel_time = torch.tensor([[0., 3600., 86400., 1e7], [0., 0., 60., 1e5]],
                            device=device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 4, 5, 4, 6, 6, 7],
        [1, 0, 2, 1, 5, 4, 6, 4, 7, 6],
    ], device=device)
    return feat_dict, node_idx, node_type, hop, rel_time, edge_index


def select(model, slot):
    # Let the final projection output only the `slot`-th of the five
    # concatenated token elements:
    C = model.channels
    with torch.no_grad():
        model.lin.weight.zero_()
        model.lin.weight[:, slot * C:(slot + 1) * C] = torch.eye(C)
        model.lin.bias.zero_()


@withDevice
def test_relgt_tokenizer(device):
    model = RelGTTokenizer(16, NODE_TYPES, DictEncoder(4, 16)).to(device)
    assert str(model) == 'RelGTTokenizer(16, num_node_types=2, max_hops=2)'
    assert isinstance(model.pe_encoder, SubgraphGNNPE)
    assert model.hop_emb.num_embeddings == 4  # Hops 0, 1, 2 plus filler.

    out = model(*get_inputs(device))
    assert out.size() == (2, 4, 16)
    assert out.device.type == torch.device(device).type


def test_relgt_tokenizer_backward():
    model = RelGTTokenizer(16, NODE_TYPES, DictEncoder(4, 16))
    model(*get_inputs()).sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name


def test_relgt_tokenizer_feature_gathering():
    feat_dict, node_idx, node_type, hop, rel_time, edge_index = get_inputs()
    feat_dict = {
        key: torch.randn(x.size(0), 8)
        for key, x in feat_dict.items()
    }

    model = RelGTTokenizer(8, NODE_TYPES, IdentityEncoder())
    select(model, slot=0)
    out = model(feat_dict, node_idx, node_type, hop, rel_time, edge_index)

    for b in range(node_idx.size(0)):
        for k in range(node_idx.size(1)):
            key = NODE_TYPES[int(node_type[b, k])]
            expected = feat_dict[key][node_idx[b, k]]
            assert torch.allclose(out[b, k], expected)


def test_relgt_tokenizer_hop_embedding():
    feat_dict, node_idx, node_type, hop, rel_time, edge_index = get_inputs()

    model = RelGTTokenizer(8, NODE_TYPES, DictEncoder(4, 8))
    select(model, slot=2)
    out = model(feat_dict, node_idx, node_type, hop, rel_time, edge_index)

    # Tokens with equal hops share the same embedding:
    assert torch.allclose(out[0, 1], out[1, 1])  # Hop 1.
    assert torch.allclose(out[0, 2], out[1, 3])  # Hop 2.
    # Filler nodes get their own embedding, distinct from 2-hop neighbors:
    assert torch.allclose(out[0, 3], model.hop_emb.weight[3])
    assert not torch.allclose(out[0, 3], out[0, 2])


def test_relgt_tokenizer_time_encoding():
    feat_dict, node_idx, node_type, hop, rel_time, edge_index = get_inputs()

    model = RelGTTokenizer(8, NODE_TYPES, DictEncoder(4, 8))
    select(model, slot=3)
    out = model(feat_dict, node_idx, node_type, hop, rel_time, edge_index)

    # Equal relative times share the same encoding:
    assert torch.allclose(out[0, 0], out[1, 0])
    assert torch.allclose(out[0, 0], out[1, 1])
    # Relative time is measured in days:
    days = torch.tensor([1.0])
    expected = model.time_lin(model.time_enc(days))[0]
    assert torch.allclose(out[0, 2], expected, atol=1e-6)


def test_relgt_tokenizer_elements_affect_output():
    feat_dict, node_idx, node_type, hop, rel_time, edge_index = get_inputs()

    model = RelGTTokenizer(16, NODE_TYPES, DictEncoder(4, 16),
                           pe_encoder=ZeroPE(16))
    out = model(feat_dict, node_idx, node_type, hop, rel_time, edge_index)
    assert torch.allclose(
        out, model(feat_dict, node_idx, node_type, hop, rel_time, edge_index))

    hop2 = hop.clone()
    hop2[0, 1] = 2
    out2 = model(feat_dict, node_idx, node_type, hop2, rel_time, edge_index)
    assert not torch.allclose(out2[0, 1], out[0, 1])
    assert torch.allclose(out2[1], out[1])

    rel_time2 = rel_time.clone()
    rel_time2[1, 2] = 1e6
    out2 = model(feat_dict, node_idx, node_type, hop, rel_time2, edge_index)
    assert not torch.allclose(out2[1, 2], out[1, 2])
    assert torch.allclose(out2[0], out[0])

    # Swapping the type also swaps which features are gathered:
    node_type2 = node_type.clone()
    node_type2[0, 0] = 1
    out2 = model(feat_dict, node_idx, node_type2, hop, rel_time, edge_index)
    assert not torch.allclose(out2[0, 0], out[0, 0])


def test_relgt_tokenizer_pe_encoder_inputs():
    pe_encoder = ZeroPE(16)
    model = RelGTTokenizer(16, NODE_TYPES, DictEncoder(4, 16),
                           pe_encoder=pe_encoder)
    model(*get_inputs())

    assert pe_encoder.batch.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert pe_encoder.batch_size == 2


def test_relgt_tokenizer_missing_node_type():
    feat_dict, node_idx, node_type, hop, rel_time, edge_index = get_inputs()

    model = RelGTTokenizer(16, NODE_TYPES + ['order'], DictEncoder(4, 16))
    # Node types without tokens may be missing from the encoder output:
    out = model(feat_dict, node_idx, node_type, hop, rel_time, edge_index)
    assert out.size() == (2, 4, 16)

    with pytest.raises(ValueError, match="features for node type 'item'"):
        model({'user': feat_dict['user']}, node_idx, node_type, hop, rel_time,
              edge_index)


def test_relgt_tokenizer_invalid_inputs():
    feat_dict, node_idx, node_type, hop, rel_time, edge_index = get_inputs()
    model = RelGTTokenizer(16, NODE_TYPES, DictEncoder(4, 16))

    with pytest.raises(ValueError, match="Expected 'hop' to have shape"):
        model(feat_dict, node_idx, node_type, hop[:, :3], rel_time, edge_index)

    with pytest.raises(ValueError, match="to be two-dimensional"):
        model(feat_dict, node_idx.view(-1), node_type, hop, rel_time,
              edge_index)

    bad_model = RelGTTokenizer(16, NODE_TYPES, DictEncoder(4, 8))
    with pytest.raises(ValueError, match=r"features of shape \[\*, 16\]"):
        bad_model(feat_dict, node_idx, node_type, hop, rel_time, edge_index)

    with pytest.raises(ValueError, match="duplicate entries"):
        RelGTTokenizer(16, ['user', 'user'], DictEncoder(4, 16))

    with pytest.raises(ValueError, match="at least one entry"):
        RelGTTokenizer(16, [], DictEncoder(4, 16))
