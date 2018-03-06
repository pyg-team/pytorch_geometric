import torch

from .dataset import Dataset
from .utils.download import download_url
from .utils.planetoid import read_planetoid
from .utils.spinner import Spinner


class Cora(Dataset):
    """`Cora <https://linqs.soe.ucsc.edu/node/236>`_ Dataset.

    Args:
        root (string): Root directory of dataset. Downloads and processes data
            automatically if dataset doesn't exist.
        transform (callable, optional): A function/transform that takes in a
            :class:`Data` object and returns a transformed version.
            (default: :obj:`None`)
    """

    url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    extensions = ['tx', 'ty', 'allx', 'ally', 'graph', 'test.index']

    def __init__(self, root, transform=None):
        super(Cora, self).__init__(root, transform)
        self.set = torch.load(self._processed_files[0])

    @property
    def raw_files(self):
        return ['ind.cora.{}'.format(ext) for ext in self.extensions]

    @property
    def processed_files(self):
        return 'data.pt'

    def download(self):
        urls = ['{}/{}'.format(self.url, name) for name in self.raw_files]
        [download_url(url, self.raw_folder) for url in urls]

    def process(self):
        spinner = Spinner('Processing').start()
        data = read_planetoid(self._raw_files)
        spinner.success()
        return [data]
