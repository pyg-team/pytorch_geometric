import os.path as osp
import random
import shutil
import sys

import pytest

from torch_geometric.datasets import SNAPDataset


@pytest.mark.skipif(True, reason="'https://snap.stanford.edu' not available")
def test_snap_dataset():
    root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))

    for name in ['ego-facebook', 'soc-Slashdot0811', 'wiki-vote']:
        SNAPDataset(root, name)

    shutil.rmtree(root)
