import os
import os.path as osp
import glob

import torch

from .dataset import Dataset
from .utils.download import download_url
from .utils.extract import extract_zip
from .utils.progress import Progress
from .utils.off import read_off


class ModelNet40(Dataset):
    url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    categories = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
        'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
        'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
        'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
        'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table',
        'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    num_examples = 30991 + 9008  # train + test

    def __init__(self, root, train=True, transform=None):
        super(ModelNet40, self).__init__(root, transform)

        name = self._processed_files[0] if train else self._processed_files[1]
        self.set = torch.load(name)

    @property
    def raw_files(self):
        return ['ModelNet40/{}'.format(c) for c in self.categories]

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        file_path = download_url(self.url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)

    def process(self):
        train, test = [], []
        p = Progress('Processing', end=self.num_examples, type='')
        for target, category in enumerate(self.categories):
            dir = osp.join(self.raw_folder, 'ModelNet40', category)
            train_files = glob.glob('{}/train/*.off'.format(dir))
            test_files = glob.glob('{}/test/*.off'.format(dir))
            train += [self.process_example(f, target, p) for f in train_files]
            test += [self.process_example(f, target, p) for f in test_files]
        p.success()
        return train, test

    def process_example(self, filename, target, progress):
        data = read_off(filename)
        data.target = target
        progress.inc()
        return data
