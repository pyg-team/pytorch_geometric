import os

from torch.utils.data import Dataset

from .utils.download import download_url
from .utils.extract import extract_7z


class Semantic3D(Dataset):
    """`Semantic3D <http://www.semantic3d.net/>`_ Dataset."""

    url = 'http://www.semantic3d.net/data/point-clouds/training1/' \
          '{}_intensity_rgb.7z'
    files = [
        'bildstein_station1_xyz_intensity_rgb',
        'bildstein_station3_xyz_intensity_rgb',
        'bildstein_station5_xyz_intensity_rgb',
        'domfountain_station1_xyz_intensity_rgb',
        'domfountain_station2_xyz_intensity_rgb',
        'domfountain_station3_xyz_intensity_rgb',
        'neugasse_station1_xyz_intensity_rgb',
        'sg27_station1_intensity_rgb',
        'sg27_station2_intensity_rgb',
        'sg27_station4_intensity_rgb',
        'sg27_station5_intensity_rgb',
        'sg27_station9_intensity_rgb',
        'sg28_station4_intensity_rgb',
        'untermaederbrunnen_station1_xyz_intensity_rgb',
        'untermaederbrunnen_station3_xyz_intensity_rgb',
    ]

    def __init__(self, root, train=True, transform=None):
        super(Semantic3D, self).__init__()

        # Set dataset properites.
        self.root = os.path.expanduser(root)
        self.raw_folder = os.path.join(self.root, 'raw')
        self.train = train
        self.transform = transform

        self.download()

    def download(self):
        for f in self.files:
            compressed_file = os.path.join(self.raw_folder, '{}.7z'.format(f))
            extracted_file = os.path.join(self.raw_folder, '{}.txt'.format(f))

            if not os.path.exists(extracted_file):
                if not os.path.exists(compressed_file):
                    url = '{}{}.7z'.format(self.url, f)
                    download_url(url, self.raw_folder)
                extract_7z(compressed_file, self.raw_folder)
                os.unlink(compressed_file)
