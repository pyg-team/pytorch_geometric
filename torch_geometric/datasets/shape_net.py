import os

# import torch
from torch.utils.data import Dataset

from .utils.dir import make_dirs
from .utils.download import download_url
from .utils.extract import extract_zip


class ShapeNet(Dataset):
    url = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip'

    categories = {
        '02691156': 'airplaine',
        '02773838': 'bag',
        '02954340': 'cap',
        '02958343': 'car',
        '03001627': 'chair',
        '03261776': 'earphone',
        '03467517': 'guitar',
        '03624134': 'knife',
        '03636649': 'lamp',
        '03642806': 'laptop',
        '03790512': 'bike',
        '03797390': 'mug',
        '03948459': 'pistol',
        '04099429': 'rocket',
        '04225987': 'skateboard',
        '04379243': 'table',
    }

    def __init__(self,
                 root,
                 train=True,
                 category='airplaine',
                 transform=None,
                 target_transform=None):

        super(ShapeNet, self).__init__()

        if category not in self.categories.values():
            raise ValueError

        self.root = os.path.expanduser(root)
        self.processed_folder = os.path.join(self.root, 'processed')

        self.category = category
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(self.processed_folder)

    def download(self):
        if not self._check_exists():
            return

        print('Downloading {}'.format(self.url))

        file_path = download_url(self.url, self.root)
        extract_zip(file_path, self.root)
        os.unlink(file_path)

    def process(self):
        if self._check_processed():
            return

        print('Processing...')

        make_dirs(os.path.join(self.processed_folder))

        print('Done!')
