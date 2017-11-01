import os

from torch.utils.data import Dataset

from .utils.download import download_url
from .utils.extract import extract_tar


class MNISTSuperpixel75(Dataset):

    url = "http://www.roemisch-drei.de/mnist.tar.gz"

    def __init__(self, root, train=True, transform=None,
                 target_transform=None):

        super(MNISTSuperpixel75, self).__init__()

        self.root = os.path.expanduser(root)
        self.data_file = os.path.join(self.root, 'mnist.pt')

        self.transform = transform
        self.target_transform = target_transform

        self.download()

    def _check_exists(self):
        return os.path.exists(self.root)

    def download(self):
        if self._check_exists():
            return

        print('Downloading {}'.format(self.url))

        file_path = download_url(self.url, self.root)
        extract_tar(file_path, self.root)
        os.unlink(file_path)
