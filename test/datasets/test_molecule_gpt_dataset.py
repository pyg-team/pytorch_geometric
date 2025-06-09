from torch_geometric.datasets import MoleculeGPTDataset
from torch_geometric.testing import withPackage
import torch.distributed as dist

@withPackage('transformers', 'sentencepiece', 'accelerate', 'rdkit')
def test_molecule_gpt_dataset():
    dataset = MoleculeGPTDataset(root='./data/MoleculeGPT')
    assert str(dataset) == f'MoleculeGPTDataset({len(dataset)})'
    assert dataset.num_edge_features == 4
    assert dataset.num_node_features == 6

    # Optional cleanup if distributed is initialized
    if dist.is_initialized():
        dist.destroy_process_group()
