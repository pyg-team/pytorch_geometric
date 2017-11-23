import os

import torch
from torch.utils.data import Dataset

from .data import Data
from ..sparse import SparseTensor
from ..graph.geometry import edges_from_faces
from .utils.dir import make_dirs
from .utils.ply import read_ply


class FAUST(Dataset):
    """`MPI FAUST <http://faust.is.tue.mpg.de>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``raw/MPI-FAUST`` exist.
            Dataset needs to be downloaded manually.
        train (bool, optional): If :obj:`True`, creates dataset from
            ``training.pt``, otherwise from ``test.pt``. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in a
            :class:`Data` object and returns a transformed version.
            (default: :obj:`None`)
    """

    url = 'http://faust.is.tue.mpg.de/'
    distance_url = 'http://ls7-www.cs.uni-dortmund.de/cvpr_geometric_dl/' \
                   'faust_geodesic_distance.tar.gz'

    def __init__(self, root, train=True, transform=None):
        super(FAUST, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw', 'MPI-FAUST')
        self.processed_folder = os.path.join(self.root, 'processed')

        self.training_file = os.path.join(self.processed_folder, 'training.pt')
        self.test_file = os.path.join(self.processed_folder, 'test.pt')

        self.train = train
        self.transform = transform

        # Download and process data.
        self.download()
        self.process()

        # Load processed data.
        data_file = self.training_file if train else self.test_file
        self.index, self.position = torch.load(data_file)

    def __getitem__(self, i):
        index = self.index[i]
        weight = torch.ones(index.size(1))
        position = self.position[i]
        n = position.size(0)
        adj = SparseTensor(index, weight, torch.Size([n, n]))
        data = Data(None, adj, position, None)

        if self.transform is not None:
            data = self.transform(data)

        return data.all()

    def __len__(self):
        return self.index.size(0)

    @property
    def _raw_exists(self):
        return os.path.exists(self.raw_folder)

    @property
    def _processed_exists(self):
        train_exists = os.path.exists(self.training_file)
        test_exists = os.path.exists(self.training_file)
        return train_exists and test_exists

    def _read_example(self, index):
        path = os.path.join(self.raw_folder, 'training', 'registrations',
                            'tr_reg_{0:03d}.ply'.format(index))

        position, face = read_ply(path)
        index = edges_from_faces(face)

        return index, position

    def _save_examples(self, indices, path):
        data = [self._read_example(i) for i in indices]

        index = torch.stack([example[0] for example in data], dim=0)
        position = torch.stack([example[1] for example in data], dim=0)

        torch.save((index, position), path)

    def download(self):
        if not self._raw_exists:
            raise RuntimeError('Dataset not found. Please download it from ' +
                               '{}'.format(self.url))

    def process(self):
        if self._processed_exists:
            return

        print('Processing...')

        make_dirs(os.path.join(self.processed_folder))

        train_indices = range(0, 80)
        self._save_examples(train_indices, self.training_file)

        test_indices = range(20, 100)
        self._save_examples(test_indices, self.test_file)

        print('Done!')
