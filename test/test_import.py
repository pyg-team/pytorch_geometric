import subprocess
import sys
from pathlib import Path


def test_import_does_not_load_llm_dependencies():
    code = """
import sys

import torch_geometric
from torch_geometric.datasets import (MoleculeGPTDataset, TAGDataset,
                                      WebQSPDataset)

assert 'torch_geometric.llm' not in sys.modules
assert 'pandas' not in sys.modules
assert MoleculeGPTDataset.__name__ == 'MoleculeGPTDataset'
assert TAGDataset.__name__ == 'TAGDataset'
assert WebQSPDataset.__name__ == 'WebQSPDataset'
"""
    subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
