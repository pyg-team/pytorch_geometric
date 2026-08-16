import os
import shutil
import pytest
from torch_geometric.datasets import SynthFinDataset

def test_synthfin_dataset(tmp_path):
    # We will test the initialization. To prevent a full download in CI, 
    # we would usually mock the `download` method or just let it download if it's small.
    # The raw.zip is about 35MB. For a simple local test, we can just let it run.
    dataset = SynthFinDataset(root=str(tmp_path))
    
    assert len(dataset) == 1
    assert dataset.__class__.__name__ == 'SynthFinDataset'
    assert dataset.url.endswith('raw.zip')
    
    data = dataset[0]
    assert data.num_nodes == 100000
    assert data.num_features == 10
    assert data.y.size(0) == 100000
