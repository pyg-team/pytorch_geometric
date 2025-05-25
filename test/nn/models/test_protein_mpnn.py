import torch

from torch_geometric.nn.models import ProteinMPNN


def test_protein_mpnn():
    model = ProteinMPNN()
    x = torch.randn(10, 4, 3)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ])
    num_nodes = x.size(0)
    num_edges = edge_index.size(1)
    vocab_size = 21
    edge_attr = torch.randn(num_edges, 400)
    chain_seq_label = torch.randint(0, vocab_size, (num_edges, ))
    mask = torch.ones(num_nodes)
    chain_mask_all = torch.ones(num_nodes)
    residue_idx = torch.randint(0, 10, (num_nodes, ))
    chain_encoding_all = torch.ones(num_nodes)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    logits = model(x, edge_index, edge_attr, chain_seq_label, mask,
                   chain_mask_all, residue_idx, chain_encoding_all, batch)
    assert logits.size() == (num_nodes, vocab_size)
