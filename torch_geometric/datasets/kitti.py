import os

from .dataset import Dataset
from .utils.download import download_url
from .utils.extract import extract_zip


class KITTI(Dataset):
    """`KITTI Vision Benchmark <http://www.cvlibs.net/datasets/kitti>`_
    Velodyne Point Cloud Dataset from the
    `"Are we ready for Autonous Driving? The KITTI Vision Benchmark Suite"
    <http://www.cvlibs.net/publications/Geiger2012CVPR.pdf>`_ paper. KITTI
    consists of 7,481 training and 7,518 test point clouds, covering three
    categories: Car, Pedestrian, and Cyclist.

    Args:
        root (string): Root directory of dataset. Downloads data automatically
            if dataset doesn't exist.
        train (bool, optional): If :obj:`True`, returns training data,
            otherwise test data. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in a
            ``Data`` object and returns a transformed version.
            (default: ``None``)
    """

    data_url = 'http://kitti.is.tue.mpg.de/kitti/data_object_velodyne.zip'
    target_url = 'http://kitti.is.tue.mpg.de/kitti/data_object_det_2.zip'

    def __init__(self, root, train=True, transform=None):
        super(KITTI, self).__init__(root, transform)

    @property
    def raw_files(self):
        return ['input.sdf', 'mask.txt']

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        file_path = download_url(self.data_url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)

        file_path = download_url(self.target_url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)

    def process(self):
        pass
