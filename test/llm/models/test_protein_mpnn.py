import torch

from torch_geometric.llm.models import ProteinMPNN
from torch_geometric.testing import withPackage


@withPackage('torch_cluster')
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
