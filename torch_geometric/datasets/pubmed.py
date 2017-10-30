from __future__ import print_function

import os
from itertools import islice

import torch

from .utils.dir import make_dirs
from .utils.download import download_url
from .utils.extract import extract_tar


class PubMed(object):
    url = "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz"

    def __init__(self, root, transform=None, target_transform=None):
        super(PubMed, self).__init__()

        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.data_file = os.path.join(self.processed_folder, 'data.pt')

        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()

        # Load processed data.
        data = torch.load(self.data_file)
        self.input, (r, c), self.target = data
        index = torch.stack([torch.cat([r, c]), torch.cat([c, r])], dim=0)

        # Create unweighted sparse adjacency matrix.
        weight = torch.FloatTensor(index.size(1)).fill_(1)
        n = self.input.size(0)
        self.adj = torch.sparse.FloatTensor(index, weight, torch.Size([n, n]))

    def __getitem__(self, index):
        data = (self.input, self.adj)
        target = self.target

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (data, target)

    def __len__(self):
        return 1

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(self.data_file)

    def _read(self):
        raw_folder = os.path.join(self.raw_folder, 'Pubmed-Diabetes', 'data')

        key = []
        input = []
        target = []

        path = os.path.join(raw_folder, 'Pubmed-Diabetes.NODE.paper.tab')
        with open(path, 'r') as f:
            feature_name = next(islice(f, 1, 2))[:-1].split('\t')[1:-1]
            feature_name = [f[8:-4] for f in feature_name]

            for line in f:
                s = line[:-1].split('\t')
                key.append(s[0])
                target.append(int(s[1][-1:]))
                feature = s[2:-1]
                index = [feature_name.index(f.split('=')[0]) for f in feature]
                index = torch.LongTensor(index)
                value = [float(f.split('=')[1]) for f in feature]
                value = torch.FloatTensor(value)
                feature = torch.zeros(500)
                feature[index] = value
                input.append(feature)

            input = torch.stack(input, dim=0)
            target = torch.LongTensor(target)

        path = os.path.join(raw_folder, 'Pubmed-Diabetes.DIRECTED.cites.tab')
        with open(path, 'r') as f:
            index = []
            for line in islice(f, 2, 44340):
                s = line[:-1].split('\t')
                x = key.index(s[1][6:])
                y = key.index(s[3][6:])
                index.append([x, y])

            index = torch.LongTensor(index).t()

        return input, index, target

    def download(self):
        if self._check_exists():
            return

        print('Downloading {}'.format(self.url))

        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder)
        os.unlink(file_path)

    def process(self):
        if self._check_processed():
            return

        print('Processing...')

        make_dirs(os.path.join(self.processed_folder))
        data = self._read()
        torch.save(data, self.data_file)

        print('Done!')
