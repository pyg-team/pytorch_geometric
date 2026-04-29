# In this example, you will find data loading implementations using PyTorch
# DataPipes (https://pytorch.org/data/) across various tasks:

# (1) molecular graph data loading pipe
# (2) mesh/point cloud data loading pipe

# In particular, we make use of PyG's built-in DataPipes, e.g., for batching
# multiple PyG data objects together or for converting SMILES strings into
# molecular graph representations. We also showcase how to write your own
# DataPipe (i.e. for loading and parsing mesh data into PyG data objects).

import argparse
import csv
import os.path as osp
import time
from itertools import chain, tee

import torch
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import (
    FileLister,
    FileOpener,
    IterableWrapper,
)

from torch_geometric.data import Data, download_url, extract_zip


def molecule_datapipe() -> IterDataPipe:
    # Download HIV dataset from MoleculeNet:
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets'
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    path = download_url(f'{url}/HIV.csv', root_dir)

    datapipe = FileOpener([path], mode="rt")
    # Convert CSV rows into dictionaries, skipping the header row
    datapipe = datapipe.map(lambda file: (
        dict(zip(["smiles", "activity", "HIV_active"], row))
        for i, row in enumerate(csv.reader(file[1])) if i > 0 and row))

    datapipe = IterableWrapper(chain.from_iterable(datapipe))
    datapipe = datapipe.parse_smiles(target_key='HIV_active')
    datapipe, = tee(datapipe, 1)
    return IterableWrapper(datapipe)


@torch.utils.data.functional_datapipe('read_mesh')
class MeshOpener(IterDataPipe):
    # A custom DataPipe to load and parse mesh data into PyG data objects.
    def __init__(self, dp: IterDataPipe) -> None:
        try:
            import meshio  # noqa: F401
            import torch_cluster  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "To run this example, please install required packages:\n"
                "pip install meshio torch-cluster") from e

        super().__init__()
        self.dp = dp

    def __iter__(self):
        import meshio

        for path in self.dp:
            category = osp.basename(path).split('_')[0]
            try:
                mesh = meshio.read(path)
            except UnicodeDecodeError:
                # Failed to read the file because it is not in the expected OFF
                # format.
                continue

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
    datapipe, = tee(datapipe, 1)
    datapipe = IterableWrapper(datapipe)
    datapipe = datapipe.sample_points(1024)  # Use PyG transforms from here.
    datapipe = datapipe.knn_graph(k=8)
    return datapipe


DATAPIPES = {
    'molecule': molecule_datapipe,
    'mesh': mesh_datapipe,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='molecule', choices=DATAPIPES.keys())

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
    for _ in datapipe:
        pass
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    print('Iterating over all data a second time...')
    t = time.perf_counter()
    for _ in datapipe:
        pass
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
