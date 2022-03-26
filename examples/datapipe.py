# In this example, you will find data loading implementations using PyTorch
# DataPipes (https://pytorch.org/data/) across various tasks:

# (1) molecular graph data loading pipe
# (2) mesh/point cloud data loading pipe

# In particular, we make use of PyG's built-in DataPipes, e.g., for batching
# multiple PyG data objects together or for converting SMILES strings into
# molecular graph representations. We also showcase how to write your own
# DataPipe (i.e. for loading and parsing mesh data into PyG data objects).

import argparse
import os.path as osp
import time

import torch
import torchdata
from torchdata.datapipes.iter import FileLister, FileOpener, IterDataPipe

import torch_geometric.data.datapipes  # noqa: Register functional datapipes.
import torch_geometric.transforms as T
from torch_geometric.data import Data, download_url, extract_zip


def molecule_datapipe() -> IterDataPipe:
    # Download HIV dataset from MoleculeNet:
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets'
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    path = download_url(f'{url}/HIV.csv', root_dir)

    datapipe = FileOpener([path])
    datapipe = datapipe.parse_csv_as_dict()
    datapipe = datapipe.parse_smiles(target_key='HIV_active')
    datapipe = datapipe.in_memory_cache()  # Cache graph instances in-memory.

    return datapipe


@torchdata.datapipes.functional_datapipe('read_mesh')
class MeshOpener(IterDataPipe):
    # A custom DataPipe to load and parse mesh data into PyG data objects.
    def __init__(self, dp: IterDataPipe):
        super().__init__()
        self.dp = dp

    def __iter__(self):
        import meshio

        for path in self.dp:
            category = osp.basename(path).split('_')[0]

            mesh = meshio.read(path)
            pos = torch.from_numpy(mesh.points).to(torch.float)
            face = torch.from_numpy(mesh.cells[0].data).t().contiguous()

            yield Data(pos=pos, face=face, category=category)


def mesh_datapipe() -> IterDataPipe:
    # Download ModelNet10 dataset from Princeton:
    url = 'http://vision.princeton.edu/projects/2014/3DShapeNets'
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    path = download_url(f'{url}/ModelNet10.zip', root_dir)
    root_dir = osp.join(root_dir, 'ModelNet10')
    if not osp.exists(root_dir):
        extract_zip(path, root_dir)

    def is_train(path: str) -> bool:
        return 'train' in path

    datapipe = FileLister([root_dir], masks='*.off', recursive=True)
    datapipe = datapipe.filter(is_train)
    datapipe = datapipe.read_mesh()
    datapipe = datapipe.in_memory_cache()  # Cache graph instances in-memory.
    datapipe = datapipe.copy()  # Copy in order to use in-place transforms.
    datapipe = datapipe.map(T.SamplePoints(num=1024))  # Use PyG transforms.

    return datapipe


DATAPIPES = {
    'molecule': molecule_datapipe,
    'mesh': mesh_datapipe,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=DATAPIPES.keys())
    args = parser.parse_args()

    datapipe = DATAPIPES[args.task]()

    print('Example output:')
    print(next(iter(datapipe)))

    # Shuffling + Batching support:
    datapipe = datapipe.shuffle()
    datapipe = datapipe.batch_graphs(batch_size=32)

    # The first epoch will take longer than the remaining ones...
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
