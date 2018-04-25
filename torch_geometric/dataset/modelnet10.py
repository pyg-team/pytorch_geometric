import os
import os.path as osp
import glob

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.read import read_off
from torch_geometric.datasets.utils.download import download_url
from torch_geometric.datasets.utils.extract import extract_zip
from torch_geometric.datasets.utils.progress import Progress


class ModelNet10(InMemoryDataset):
    num_graphs = 3991 + 908
    url = 'http://vision.princeton.edu/projects/2014/3DShapeNets/' \
          'ModelNet10.zip'
    categories = [
        'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand',
        'sofa', 'table', 'toilet'
    ]

    def __init__(self, root, train, transform=None):
        super(ModelNet10, self).__init__(root, transform)

    @property
    def raw_files(self):
        return self.categories

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        filepath = download_url(self.url, self.root)
        extract_zip(filepath, self.root)
        os.unlink(filepath)
        os.rename(osp.join(self.root, ModelNet10.__name__), self.raw_dir)

    def process(self):
        progress = Progress('Processing', end=self.num_graphs, type='')

        train_data_list, test_data_list = [], []

        for y, category in enumerate(self.categories):
            path = osp.join(self.raw_dir, category)

            for filename in glob.glob('{}/train/*.off'.format(path)):
                data = read_off(filename)
                data.y = torch.LongTensor([y])
                train_data_list.append(data)
                progress.inc()

            for filename in glob.glob('{}/test/*.off'.format(path)):
                data = read_off(filename)
                data.y = torch.LongTensor([y])
                test_data_list.append(data)
                progress.inc()

        progress.success()
