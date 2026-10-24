import torch

from torch_geometric.llm.models import ProteinMPNN
from torch_geometric.llm.models.protein_mpnn import Encoder
from torch_geometric.testing import withPackage


@withPackage('pyg_lib')
def test_protein_mpnn():
    num_nodes = 10
    vocab_size = 21

    model = ProteinMPNN(vocab_size=vocab_size)
    x = torch.randn(num_nodes, 4, 3)
    chain_seq_label = torch.randint(0, vocab_size, (num_nodes, ))
    mask = torch.ones(num_nodes)
    chain_mask_all = torch.ones(num_nodes)
    residue_idx = torch.randint(0, 10, (num_nodes, ))
    chain_encoding_all = torch.ones(num_nodes)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    logits = model(x, chain_seq_label, mask, chain_mask_all, residue_idx,
                   chain_encoding_all, batch)
    assert logits.size() == (num_nodes, vocab_size)


def test_protein_mpnn_encoder_uses_out_v_for_messages():
    # The encoder keeps separate heads for the node message (`out_v`) and the
    # edge update (`out_e`); both must be trained.
    hidden_channels = 16
    encoder = Encoder(in_channels=3 * hidden_channels,
                      hidden_channels=hidden_channels)

    num_nodes, num_edges = 6, 12
    x = torch.randn(num_nodes, hidden_channels)
    edge_attr = torch.randn(num_edges, hidden_channels)
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges, )),
        torch.randint(0, num_nodes, (num_edges, )),
    ])

    x_out, edge_attr_out = encoder(x, edge_index, edge_attr)
    (x_out.sum() + edge_attr_out.sum()).backward()

    assert all(param.grad is not None for param in encoder.out_v.parameters())
    assert all(param.grad is not None for param in encoder.out_e.parameters())
