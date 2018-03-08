import os
import os.path as osp
import glob

import torch

from .dataset import Dataset
from .utils.download import download_url
from .utils.extract import extract_zip
from .utils.progress import Progress
from .utils.off import read_off
from .utils.mesh_to_cloud import load_triangles


class ModelNet10RandAugPC(Dataset):
    url = 'http://vision.princeton.edu/projects/2014/3DShapeNets/' \
          'ModelNet10.zip'
    categories = [
        'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand',
        'sofa', 'table', 'toilet'
    ]
    num_examples = 3991 + 908  # train + test

    def __init__(self, root, train=True, transform=None):
        super(ModelNet10RandAugPC, self).__init__(root, transform)

        name = self._processed_files[0] if train else self._processed_files[1]
        self.set = torch.load(name)

    @property
    def raw_files(self):
        return ['ModelNet10/{}'.format(c) for c in self.categories]

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        file_path = download_url(self.url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)

    def process(self):
        p = Progress('Processing', end=self.num_examples, type='')

        train, test = [], []
        for target, category in enumerate(self.categories):
            dir = osp.join(self.raw_folder, 'ModelNet10', category)
            train_files = glob.glob('{}/train/*.off'.format(dir))
            test_files = glob.glob('{}/test/*.off'.format(dir))
            train += [self.process_example(f, target, p) for f in train_files]
            test += [self.process_example(f, target, p) for f in test_files]

        p.success()
        return train, test

    def process_example(self, filename, target, progress):
        data = load_triangles(filename)
        data.target = target
        progress.inc()
        return data
