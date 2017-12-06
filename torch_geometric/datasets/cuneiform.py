import os

from torch.utils.data import Dataset

from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.dir import make_dirs


class Cuneiform(Dataset):
    url = 'http://www.roemisch-drei.de/cuneiform.tar.gz'
    names = ['BU1', 'BU2', 'BU3', 'HA1', 'HA2', 'HU1', 'HU2', 'MAKI1', 'MAKI2']

    def __init__(self, root, train=True, transform=None):
        super(Cuneiform, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.training_file = os.path.join(self.processed_folder, 'training.pt')
        self.test_file = os.path.join(self.processed_folder, 'test.pt')

        self.train = train
        self.transform = transform

        self.download()
        self.process()

    @property
    def _raw_exists(self):
        files = ['{}.txt'.format(f) for f in self.names]
        files = [os.path.join(self.raw_folder, f) for f in files]
        return all([os.path.exists(f) for f in files])

    @property
    def _processed_exists(self):
        train_exists = os.path.exists(self.training_file)
        test_exists = os.path.exists(self.test_file)
        return train_exists and test_exists

    def download(self):
        if self._raw_exists:
            return

        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder, mode='r')
        os.unlink(file_path)

    def process(self):
        if self._processed_exists:
            return

        make_dirs(os.path.join(self.processed_folder))
