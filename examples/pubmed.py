from __future__ import print_function

import sys

sys.path.insert(0, '.')
sys.path.insert(0, '..')

from torch_geometric.datasets import PubMed  # noqa: E402

path = '~/PubMed'
train_dataset = PubMed(path)
