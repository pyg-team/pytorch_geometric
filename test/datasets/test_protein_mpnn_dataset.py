from torch_geometric.datasets import ProteinMPNNDataset
from torch_geometric.testing import onlyOnline, withPackage


@onlyOnline
@withPackage('pandas')
def test_protein_mpnn_dataset():
    dataset = ProteinMPNNDataset(root='./data/ProteinMPNN')

    assert len(dataset) == 150
    assert dataset[0].x.size() == (229, 4, 3)
    assert dataset[0].chain_seq_label.size() == (229, )
    assert dataset[0].mask.size() == (229, )
    assert dataset[0].chain_mask_all.size() == (229, )
    assert dataset[0].residue_idx.size() == (229, )
    assert dataset[0].chain_encoding_all.size() == (229, )
