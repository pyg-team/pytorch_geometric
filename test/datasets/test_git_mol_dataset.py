from typing import Tuple

import pytest

from torch_geometric.datasets import GitMolDataset
from torch_geometric.testing import onlyFullTest, withPackage


@onlyFullTest
@withPackage('torchvision', 'rdkit', 'PIL')
@pytest.mark.parametrize('split', [
    (0, 3610),
    (1, 451),
    (2, 451),
])
def test_git_mol_dataset(split: Tuple[int, int]) -> None:
    dataset = GitMolDataset(root='./data/GITMol', split=split[0])

    assert len(dataset) == split[1]
    assert dataset[0].image.size() == (1, 3, 224, 224)
    assert dataset[0].num_node_features == 9
    assert dataset[0].num_edge_features == 3
