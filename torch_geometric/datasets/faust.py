import os

import torch
from torch.utils.data import Dataset

from ..graph.geometry import edges_from_faces
from .utils.dir import make_dirs
from .utils.ply import read_ply
from .utils.download import download_url
from .utils.extract import extract_tar


class FAUST(Dataset):
    """`MPI FAUST <http://faust.is.tue.mpg.de/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where
            ``processed/training.pt`` and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``. (default: `True`)
        shot (bool, optionall): If True, loads 544-dimensional SHOT-descriptors
            to each node. (default: `False`)
        correspondence (bool, optional): Whether to return shape correspondence
            label are pose classification label. (default: `False`)
        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``. (default: `None`)
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it. (default: `None`)
    """

    url = 'http://faust.is.tue.mpg.de/'
    shot_url = 'http://www.roemisch-drei.de/faust_shot.tar.gz'
    n_training = 80
    n_test = 20

    def __init__(self,
                 root,
                 train=True,
                 shot=False,
                 correspondence=False,
                 transform=None,
                 target_transform=None):

        super(FAUST, self).__init__()

        self.root = os.path.expanduser(root)
        self.shot_folder = os.path.join(self.root, 'shot')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.training_file = os.path.join(self.processed_folder, 'training.pt')
        self.test_file = os.path.join(self.processed_folder, 'test.pt')

        self.train = train
        self.correspondence = correspondence
        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()

        data_file = self.training_file if train else self.test_file
        self.position, self.index = torch.load(data_file)

        if shot is True:
            data_file = os.path.join(
                self.shot_folder,
                'training_shot.pt') if train else os.path.join(
                    self.shot_folder, 'test_shot.pt')
            self.shot = torch.load(data_file)

    def __getitem__(self, i):
        position = self.position[i]
        index = self.index[i]
        weight = torch.FloatTensor(index.size(1)).fill_(1)
        if self.shot is None:
            input = torch.FloatTensor(position.size(0)).fill_(1)
        else:
            input = self.shot[i]
        adj = torch.sparse.FloatTensor(index, weight, torch.Size([6890, 6890]))
        data = (input, adj, position)

        if self.correspondence:
            target = torch.arange(0, input.size(0)).long()
        else:
            target = index % 10  # Every subject is representated by 10 poses.

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return self.n_training if self.train else self.n_test

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(self.training_file) and \
               os.path.exists(self.test_file)

    def _read_example(self, index):
        path = os.path.join(self.root, 'training', 'registrations',
                            'tr_reg_{0:03d}.ply'.format(index))

        vertices, faces = read_ply(path)
        edges = edges_from_faces(faces)

        return vertices, edges

    def _save_examples(self, indices, path):
        data = [self._read_example(i) for i in indices]

        vertices = torch.stack([example[0] for example in data], dim=0)
        edges = torch.stack([example[1] for example in data], dim=0)

        torch.save((vertices, edges), path)

    def download(self):
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Please download it from ' +
                               '{}'.format(self.url))

        if not os.path.exists(self.shot_folder):
            print('Downloading {}'.format(self.shot_url))

            file_path = download_url(self.shot_url, self.shot_folder)
            extract_tar(file_path, self.shot_folder)
            os.unlink(file_path)

    def process(self):
        if self._check_processed():
            return

        print('Processing...')

        make_dirs(os.path.join(self.processed_folder))

        train_indices = range(0, self.n_training)
        self._save_examples(train_indices, self.training_file)

        test_indices = range(self.n_training, self.n_training + self.n_test)
        self._save_examples(test_indices, self.test_file)

        print('Done!')
