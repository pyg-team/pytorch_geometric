import os

import torch
from torch.utils.data import Dataset

from .utils.dir import make_dirs
from .utils.download import download_url
from .utils.extract import extract_tar


class MNISTSuperpixel75(Dataset):

    url = "http://geometricdeeplearning.com/code/MoNet/MoNet_code.tar.gz"

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):

        super(MNISTSuperpixel75, self).__init__()

        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.data_file = os.path.join(self.processed_folder, 'data.pt')

        self.transform = transform
        self.target_transform = target_transform

        self.download()
        self.process()

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_processed(self):
        return os.path.exists(self.data_file)

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
        dir = os.path.join(self.raw_folder, 'MoNet_code', 'MNIST', 'datasets')
        read_mnist_monet(dir)
        # torch.save(data, self.data_file)

        print('Done!')
