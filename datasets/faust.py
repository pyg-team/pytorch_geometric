import os

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class FAUST(Dataset):
    """`MPI FAUST <http://faust.is.tue.mpg.de/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    """

    url = 'http://faust.is.tue.mpg.de/main/download'
    filename = 'MPI-FAUST.zip'
    md5 = 'c58f30108f718f92721af3b95e74349a'

    def __init__(self,
                 root,
                 train=True,
                 download=False,
                 transform=None,
                 target_transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted. You can use' +
                               ' download=True to download it')
        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            quadruple: (adj, points, features, target) where target is index
            of the target class.
        """
        pass

    def __len__(self):
        return 200
        pass

    def _check_exists(self):
        return False

    def download(self):
        if self._check_exists():
            return

        download_url(self.url, self.root, self.filename, self.md5)
        print('ihihi')


FAUST('.', download=True)
