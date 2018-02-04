import os
import os.path as osp

import torch

from .dataset import Dataset

from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.progress import Progress
from .utils.sdf import read_sdf


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
        super(QM9, self).__init__(root, transform)
        name = self.processed_files[0] if train else self.processed_files[0]
        self.set = torch.load(osp.join(self.processed_folder, name))

    @property
    def raw_files(self):
        return ['input.sdf', 'mask.txt']

    def _rename(self, a, b):
        os.rename(osp.join(self.raw_folder, a), osp.join(self.raw_folder, b))

    def download(self):
        file_path = download_url(self.data_url, self.raw_folder)
        extract_tar(file_path, osp.join(self.raw_folder), mode='r')
        os.unlink(file_path)
        os.unlink(osp.join(self.raw_folder, 'gdb9.sdf.csv'))
        download_url(self.mask_url, self.raw_folder)
        self._rename('gdb9.sdf', 'input.sdf')
        self._rename('3195404', 'mask.txt')

    def process(self):
        with open(osp.join(self.raw_folder, 'mask.txt'), 'r') as f:
            tmp = f.read().split('\n')[9:-2]
            tmp = [int(x.split()[0]) for x in tmp]
            tmp = torch.LongTensor(tmp)
            mask = torch.ByteTensor(133885).fill_(1)
            mask[tmp] = 0

        examples = []
        with open(osp.join(self.raw_folder, 'input.sdf'), 'r') as f:
            sdfs = f.read().split('$$$$\n')[:-1]
            progress = Progress('Processing', end=len(sdfs), type='')
            for idx, sdf in enumerate(sdfs):
                if mask[idx] == 1:
                    examples.append(read_sdf(sdf))
                progress.update(idx + 1)
            progress.success()

        return examples[:-10000], examples[-10000:]
