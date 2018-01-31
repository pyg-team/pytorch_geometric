import os
import os.path as osp

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

    url = 'https://ndownloader.figshare.com/files/{}'
    data_id = '3195389'
    filter_id = '3195404'

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
        data_file = osp.join(self.raw_folder, 'data')
        filter_file = osp.join(self.raw_folder, self.filter_id)
        return osp.exists(data_file) and osp.exists(filter_file)

    def download(self):
        if self._raw_exists:
            return

        data_url = self.url.format(self.data_id)
        file_path = download_url(data_url, self.raw_folder)
        extract_tar(file_path, osp.join(self.raw_folder, 'data'), mode='r')
        os.unlink(file_path)

        filter_url = self.url.format(self.filter_id)
        download_url(filter_url, self.raw_folder)

    def process(self):
        pass
