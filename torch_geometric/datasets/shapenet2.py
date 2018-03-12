import os
import os.path as osp
import glob

import torch
import scipy.io as spio

from .dataset import Dataset, Set
from .utils.download import download_url
from .utils.extract import extract_zip
from .utils.progress import Spinner
from .utils.nn_graph import nn_graph


class ShapeNet2(Dataset):
    url = 'https://shapenet.cs.stanford.edu/ericyi/scnn_data.zip'

    categories = [
        'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar',
        'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket',
        'Skateboard', 'Table'
    ]
    sets = ['train', 'val', 'test']

    def __init__(self, root, category, set='train', transform=None):
        assert category in self.categories, 'No valid category'
        assert set in self.sets, 'No valid set name'

        self.category = category
        super(ShapeNet2, self).__init__(root, transform)

        name = self._processed_files[self.sets.index(set)]
        self.set = torch.load(name)

        self.set.input = torch.ones(self.set.input.size(0))

    @property
    def raw_files(self):
        return 'Data/Categories/{}'.format(self.category)

    @property
    def processed_files(self):
        return [
            '{}_training.pt'.format(self.category),
            '{}_val.pt'.format(self.category),
            '{}_test.pt'.format(self.category),
        ]

    def download(self):
        file_path = download_url(self.url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)

    def process(self):
        dir = osp.join(self.raw_folder, 'Data', 'Categories', self.category)

        spinner = Spinner('Processing').start()
        train_set = self.process_set(glob.glob(osp.join(dir, 'train*')))
        val_set = self.process_set(glob.glob(osp.join(dir, 'val*')))
        test_set = self.process_set(glob.glob(osp.join(dir, 'test*')))
        spinner = spinner.success()

        return train_set, val_set, test_set

    def process_set(self, filenames):
        input, pos, target = [], [], []

        for f in filenames:
            data = spio.loadmat(f)
            input += list(data['v'])
            pos += list(data['pts'])
            target += list(data['label'])

        slice = [0]
        for t in target:
            slice.append(slice[-1] + len(t[0]))
        slice = torch.LongTensor(slice)
        index_slice = slice * 6

        pos = [torch.FloatTensor(p[0]) for p in pos]
        index = [nn_graph(p, k=6) for p in pos]
        pos = torch.cat(pos, dim=0)
        index = torch.cat(index, dim=1)
        input = torch.cat([torch.FloatTensor(i[0]) for i in input], dim=0)
        target = torch.cat([torch.LongTensor(t[0]) for t in target], dim=0)
        target = target.squeeze()

        return Set(input, pos, index, None, target, None, slice, index_slice)
