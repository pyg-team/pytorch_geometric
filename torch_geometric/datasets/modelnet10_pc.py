import os
import os.path as osp
import glob
import shutil

import torch

from .dataset import Dataset
from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.progress import Progress
from .utils.pcd import read_pcd


class ModelNet10PC(Dataset):
    url = 'http://imagine.enpc.fr/~simonovm/ModelNetsPcd.tar.gz'
    categories = [
        'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand',
        'sofa', 'table', 'toilet'
    ]
    num_examples = 3991 + 908  # train + test

    def __init__(self, root, train=True, transform=None):
        super(ModelNet10PC, self).__init__(root, transform)

        name = self._processed_files[0] if train else self._processed_files[1]
        self.set = torch.load(name)

    @property
    def raw_files(self):
        names = [osp.join('pc_sampled1k', c) for c in self.categories]
        return [osp.join('modelnet10', n) for n in names]

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder)
        os.unlink(file_path)
        shutil.rmtree(osp.join(self.raw_folder, 'modelnet40'))

    def process(self):
        p = Progress('Processing', end=self.num_examples, type='')

        train, test = [], []
        for target, c in enumerate(self.categories):
            dir = osp.join(self.raw_folder, 'modelnet10', 'pc_sampled1k', c)
            train_files = glob.glob('{}/train/*.pcd'.format(dir))
            test_files = glob.glob('{}/test/*.pcd'.format(dir))
            train += [self.process_example(f, target, p) for f in train_files]
            test += [self.process_example(f, target, p) for f in test_files]

        p.success()
        return train, test

    def process_example(self, filename, target, progress):
        data = read_pcd(filename)
        data.target = target
        progress.inc()
        return data
