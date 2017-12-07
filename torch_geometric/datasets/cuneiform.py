import os

import torch
from torch.utils.data import Dataset

from .data import Data
from ..sparse import SparseTensor
from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.dir import make_dirs
from .utils.spinner import Spinner


class Cuneiform(Dataset):
    url = 'http://www.roemisch-drei.de/cuneiform.tar.gz'
    training_raw = ['BU1', 'BU2', 'HA1', 'HA2', 'HU1', 'MAKI1', 'MAKI2']
    test_raw = ['BU3', 'HU2']

    def __init__(self, root, train=True, transform=None):
        super(Cuneiform, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.training_file = os.path.join(self.processed_folder, 'training.pt')
        self.test_file = os.path.join(self.processed_folder, 'test.pt')

        self.train = train
        self.transform = transform

        self.download()
        self.process()

        # Load processed data.
        data_file = self.training_file if train else self.test_file
        data = torch.load(data_file)
        self.index, self.position, target = data[:3]
        self.index_slice, self.slice = data[3:]
        self.target = target.long()

    def __getitem__(self, i):
        index = self.index[:, self.index_slice[i]:self.index_slice[i + 1]]
        weight = torch.ones(index.size(1))
        position = self.position[self.slice[i]:self.slice[i + 1]]
        n = position.size(0)
        adj = SparseTensor(index, weight, torch.Size([n, n]))
        target = self.target[i]
        data = Data(None, adj, position, target)

        if self.transform is not None:
            data = self.transform(data)

        return data.all()

    def __len__(self):
        return self.target.size(0)

    @property
    def _raw_exists(self):
        files = ['{}.txt'.format(f) for f in self.training_raw + self.test_raw]
        files = [os.path.join(self.raw_folder, f) for f in files]
        return all([os.path.exists(f) for f in files])

    @property
    def _processed_exists(self):
        train_exists = os.path.exists(self.training_file)
        test_exists = os.path.exists(self.test_file)
        return train_exists and test_exists

    def download(self):
        if self._raw_exists:
            return

        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder, mode='r')
        os.unlink(file_path)

    def process(self):
        if self._processed_exists:
            return

        spinner = Spinner('Processing').start()
        make_dirs(self.processed_folder)

        data = self._process_file_names(self.training_raw)
        torch.save(data, self.training_file)

        data = self._process_file_names(self.test_raw)
        torch.save(data, self.test_file)

        spinner.success()

    def _process_file_names(self, file_names):
        # Read cuneiform wedges into memory and concat.
        positions = []
        targets = [0]
        slices = [0]
        wedge_count = 0
        for name in file_names:
            path = os.path.join(self.raw_folder, '{}.txt'.format(name))
            with open(path, 'r') as f:
                content = f.read().split('\n')[1:-1]
            for wedge in content:
                wedge_count += 1
                data = [float(x) for x in wedge.split(',')[:-1]]
                target = int(data[-1])
                if targets[-1] != target:
                    targets.append(target)
                    slices.append(slices[-1] + wedge_count - 1)
                    wedge_count = 1
                positions.append(data[1:13])
        slices.append(len(positions))

        # Convert to tensors.
        position = torch.FloatTensor(positions).view(-1, 3)
        target = torch.ByteTensor(targets)
        slice = torch.LongTensor(slices)

        # Create adjacency for each cuneiform sign.
        rows, cols = [], []
        index_slices = [0]
        for i in range(target.size(0)):
            end = slice[i + 1] - slice[i]

            for i in range(end):
                r = [4 * i] * (end - 1)
                c = list(range(0, end * 4, 4))
                del c[i]
                r += [4 * i] * 3
                c += list(range(4 * i + 1, 4 * (i + 1)))
                rows += r
                cols += c
            index_slices.append(index_slices[-1] + end * 3 + end * (end - 1))

        index = torch.LongTensor([rows, cols])
        index_slice = torch.LongTensor(index_slices)
        slice *= 4

        return index, position, target, index_slice, slice
