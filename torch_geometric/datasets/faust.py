from os.path import join

import torch

from .dataset import Dataset
from .utils.progress import Progress
from .utils.ply import read_ply


class FAUST(Dataset):
    """`MPI FAUST <http://faust.is.tue.mpg.de>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``raw/MPI-FAUST`` exist.
            Dataset needs to be downloaded manually.
        train (bool, optional): If :obj:`True`, returns training shapes,
            otherwise test shapes. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in a
            :class:`Data` object and returns a transformed version.
            (default: :obj:`None`)
    """

    url = 'http://faust.is.tue.mpg.de/'
    num_nodes = 6890

    def __init__(self, root, train=True, transform=None):
        super(FAUST, self).__init__(root, transform)

        name = self._processed_files[0] if train else self._processed_files[1]
        self.set = torch.load(name)

    @property
    def raw_files(self):
        path = join('MPI-FAUST', 'training', 'registrations')
        return [join(path, 'tr_reg_{0:03d}.ply'.format(i)) for i in range(100)]

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from ' +
            '{} and extract it to {}'.format(self.url, self.raw_folder))

    def process(self):
        progress = Progress('Processing', end=100, type='')

        set = []
        for i, filename in enumerate(self._raw_files):
            data = read_ply(filename)
            data.target = i % 10
            set.append(data)
            progress.inc()

        progress.success()
        return set[:80], set[80:]
