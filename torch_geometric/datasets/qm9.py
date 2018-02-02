import os
import os.path as osp
import re

import torch
from torch.utils.data import Dataset

from .utils.download import download_url
from .utils.extract import extract_tar


class QM9(Dataset):
    """`QM9 <http://quantum-machine.org/datasets>`_ Molecule Dataset from the
    `"Quantum Chemistry Structures and Properties of 134 Kilo Molecules"
    <https://www.nature.com/articles/sdata201422>`_ paper. QM9 consists of
    130k molecules with 13 properties for each molecule.

    Args:
        root (string): Root directory of dataset. Downloads data automatically
            if dataset doesn't exist.
        train (bool, optional): If :obj:`True`, returns training molecules,
            otherwise test molecules. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in a
            ``Data`` object and returns a transformed version.
            (default: ``None``)
    """

    data_url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
               'datasets/gdb9.tar.gz'
    mask_url = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root, train=True, transform=None):
        super(QM9, self).__init__()

        # Set dataset properites.
        self.root = osp.expanduser(root)
        self.raw_folder = osp.join(self.root, 'raw')
        self.processed_folder = osp.join(self.root, 'processed')

        # Download and process data.
        self.download()
        self.process()

    @property
    def _raw_exists(self):
        r = self.raw_folder
        input_exists = osp.exists(osp.join(r, 'input.sdf'))
        target_exists = osp.exists(osp.join(r, 'target.txt'))
        mask_exists = osp.exists(osp.join(r, 'mask.txt'))
        return input_exists and target_exists and mask_exists

    @property
    def _processed_exists(self):
        return False

    def download(self):
        if self._raw_exists:
            return

        file_path = download_url(self.data_url, self.raw_folder)
        extract_tar(file_path, osp.join(self.raw_folder), mode='r')
        os.unlink(file_path)
        download_url(self.mask_url, self.raw_folder)

        r = self.raw_folder
        os.rename(osp.join(r, 'gdb9.sdf'), osp.join(r, 'input.sdf'))
        os.rename(osp.join(r, 'gdb9.sdf.csv'), osp.join(r, 'target.txt'))
        os.rename(osp.join(r, '3195404'), osp.join(r, 'mask.txt'))

    def process(self):
        if self._processed_exists:
            return

        with open(osp.join(self.raw_folder, 'target.txt'), 'r') as f:
            target = [float(x) for x in re.split(',|\n', f.read())[21:-1]]
            target = torch.FloatTensor(target).view(-1, 21)[:, 5:17]

        with open(osp.join(self.raw_folder, 'mask.txt'), 'r') as f:
            tmp = f.read().split('\n')[9:-2]
            tmp = [int(x.split()[0]) for x in tmp]
            tmp = torch.LongTensor(tmp)
            mask = torch.ByteTensor(target.size(0)).fill_(1)
            mask[tmp] = 0
            target = target.masked_select(mask.view(-1, 1)).view(-1, 12)

        with open(osp.join(self.raw_folder, 'input.sdf'), 'r') as f:
            for idx, sdf in enumerate(f.read().split('$$$$\n')[:-1]):
                if mask[idx] == 0:
                    continue
