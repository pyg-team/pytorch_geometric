import os
import os.path as osp

import torch
from torch.utils.data import Dataset

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
        input_exists = osp.exists(osp.join(self.raw_folder, 'input.sdf'))
        mask_exists = osp.exists(osp.join(self.raw_folder, 'mask.txt'))
        return input_exists and mask_exists

    @property
    def _processed_exists(self):
        return False

    def download(self):
        if self._raw_exists:
            return

        file_path = download_url(self.data_url, self.raw_folder)
        extract_tar(file_path, osp.join(self.raw_folder), mode='r')
        os.unlink(file_path)
        os.unlink(osp.join(self.raw_folder, 'gdb9.sdf.csv'))
        download_url(self.mask_url, self.raw_folder)

        r = self.raw_folder
        os.rename(osp.join(r, 'gdb9.sdf'), osp.join(r, 'input.sdf'))
        os.rename(osp.join(r, '3195404'), osp.join(r, 'mask.txt'))

    def process(self):
        if self._processed_exists:
            return

        with open(osp.join(self.raw_folder, 'mask.txt'), 'r') as f:
            tmp = f.read().split('\n')[9:-2]
            tmp = [int(x.split()[0]) for x in tmp]
            tmp = torch.LongTensor(tmp)
            mask = torch.ByteTensor(133885).fill_(1)
            mask[tmp] = 0

        input, index, weight, position, target = [], [], [], [], []
        slice, index_slice = [0], [0]

        with open(osp.join(self.raw_folder, 'input.sdf'), 'r') as f:
            sdfs = f.read().split('$$$$\n')[:-1]
            progress = Progress('Processing', end=len(sdfs), type='')
            for idx, sdf in enumerate(sdfs):
                if mask[idx] == 0:
                    continue
                f, i, w, p, t = read_sdf(sdf)
                input.append(f)
                index.append(i)
                weight.append(w)
                position.append(p)
                target.append(t)

                slice.append(slice[-1] + f.size(0))
                index_slice.append(index_slice[-1] + w.size(0))

                progress.update(idx + 1)
            progress.success()

        input = torch.cat(input, dim=0)
        index = torch.cat(index, dim=0).t()
        weight = torch.cat(weight, dim=0)
        position = torch.cat(position, dim=0)
        target = torch.cat(target, dim=0)
        slice = torch.LongTensor(slice)
        index_slice = torch.LongTensor(index_slice)
