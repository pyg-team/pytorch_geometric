import os

from torch.utils.data import Dataset

from .utils.download import download_url
from .utils.extract import extract_7z


class Semantic3D(Dataset):
    """`Semantic3D <http://www.semantic3d.net/>`_ Dataset."""

    url = 'http://www.semantic3d.net/data/point-clouds/training1/' \
          '{}_intensity_rgb.7z'
    c_files = [
        'bildstein_station1_xyz_intensity_rgb.7z',
        'bildstein_station3_xyz_intensity_rgb.7z',
        'bildstein_station5_xyz_intensity_rgb.7z',
        'domfountain_station1_xyz_intensity_rgb.7z',
        'domfountain_station2_xyz_intensity_rgb.7z',
        'domfountain_station3_xyz_intensity_rgb.7z',
        'neugasse_station1_xyz_intensity_rgb.7z',
        'sg27_station1_intensity_rgb.7z',
        'sg27_station2_intensity_rgb.7z',
        'sg27_station4_intensity_rgb.7z',
        'sg27_station5_intensity_rgb.7z',
        'sg27_station9_intensity_rgb.7z',
        'sg28_station4_intensity_rgb.7z',
        'untermaederbrunnen_station1_xyz_intensity_rgb.7z',
        'untermaederbrunnen_station3_xyz_intensity_rgb.7z',
    ]
    e_files = [
        'bildstein_station1_xyz_intensity_rgb.txt',
        'bildstein_station3_xyz_intensity_rgb.txt',
        'bildstein_station5_xyz_intensity_rgb.txt',
        'domfountain_station1_xyz_intensity_rgb.txt',
        'domfountain_station2_xyz_intensity_rgb.txt',
        'domfountain_station3_xyz_intensity_rgb.txt',
        'station1_xyz_intensity_rgb.txt',
        'sg27_station1_intensity_rgb.txt',
        'sg27_station2_intensity_rgb.txt',
        'sg27_station4_intensity_rgb.txt',
        'sg27_station5_intensity_rgb.txt',
        'sg27_station9_intensity_rgb.txt',
        'sg28_station4_intensity_rgb.txt',
        'untermaederbrunnen_station1_xyz_intensity_rgb.txt',
        'untermaederbrunnen_station3_xyz_intensity_rgb.txt',
    ]

    def __init__(self, root, train=True, transform=None):
        super(Semantic3D, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.processed_folder = os.path.join(self.root, 'processed')
        self.train = train
        self.transform = transform

        self.download()

    @property
    def _processed_exists(self):
        return os.path.exists(self.processed_folder)

    def download(self):
        if self._processed_exists:
            return

        for i in range(len(self.compressed_files)):
            c_file = os.path.join(self.raw_folder, self.c_files[i])
            e_file = os.path.join(self.raw_folder, self.e_files[i])

            if not os.path.exists(e_file):
                if not os.path.exists(c_file):
                    url = '{}{}'.format(self.url, c_file)
                    download_url(url, self.raw_folder)
                extract_7z(c_file, self.raw_folder)
                os.unlink(c_file)
