# In this example, you will find a data loading implementation using PyTorch
# DataPipes (https://pytorch.org/data/) for molecular graph datasets.
# Here, we make use of two PyG-related DataPipes
# (implemented in `torch_geometric.data.datapipes`):
# * parse_smiles: Converts a SMILES string (and an optional target) into a
#   molecular graph representation of PyG.
# * batch_graphs: Batches multiple PyG data objects together.

import os.path as osp

import torchdata

# Register functional datapipes:
import torch_geometric.data.datapipes  # noqa
from torch_geometric.data import download_url

# Download HIV dataset from 'https://moleculenet.org':
url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv'
root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
path = download_url(url, root)

datapipe = torchdata.datapipes.iter.FileOpener([path])
datapipe = datapipe.parse_csv_as_dict()
datapipe = datapipe.parse_smiles(target_key='HIV_active')
datapipe = datapipe.shuffle()
datapipe = datapipe.batch_graphs(batch_size=128)

for batch in datapipe:
    print(batch)
