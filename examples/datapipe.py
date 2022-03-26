# In this example, you will find data loading implementations using PyTorch
# DataPipes (https://pytorch.org/data/) across various tasks:

# (1) molecular graph data loading pipe
# (2) mesh/point cloud data loading pipe

# In particular, we make use of PyG built-in DataPipes, e.g., for batching
# multiple PyG data objects together, for converting SMILES strings into
# molecular graphs, or for converting meshes into point cloud graphs.

import argparse
import os.path as osp
import time

import torchdata

import torch_geometric.data.datapipes  # noqa: Register functional datapipes.
from torch_geometric.data import download_url


def molecule_datapipe(batch_size: int = 128):
    # Download HIV dataset from 'https://moleculenet.org':
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv'
    root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    path = download_url(url, root)

    datapipe = torchdata.datapipes.iter.FileOpener([path])
    datapipe = datapipe.parse_csv_as_dict()
    datapipe = datapipe.parse_smiles(target_key='HIV_active')
    datapipe = datapipe.cache()  # Cache graph instances in-memory.
    datapipe = datapipe.shuffle()
    datapipe = datapipe.batch_graphs(batch_size=batch_size)

    return datapipe


def pointcloud_datapipe(batch_size: int = 128):
    pass


DATAPIPES = {
    'molecule': molecule_datapipe,
    'pointcloud': pointcloud_datapipe,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=DATAPIPES.keys())
    args = parser.parse_args()

    datapipe = DATAPIPES[args.task]()

    print('Example output:')
    print(next(iter(datapipe)))

    print('Iterating over all data...')
    t = time.perf_counter()
    for batch in datapipe:
        pass
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    print('Iterating over all data a second time...')
    t = time.perf_counter()
    for batch in datapipe:
        pass
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
