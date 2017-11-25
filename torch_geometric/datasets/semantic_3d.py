import os

from torch.utils.data import Dataset

from .utils.download import download_url
from .utils.extract import extract_7z


class Semantic3D(Dataset):
    """`Semantic3D <http://www.semantic3d.net/>`_ Dataset."""

    url = 'http://www.semantic3d.net/data/point-clouds/training1/'
    target_url = 'http://www.semantic3d.net/data/'

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
    c_target_file = 'sem8_labels_training.7z'
    e_target_files = [
        'bildstein_station1_xyz_intensity_rgb.labels',
        'bildstein_station3_xyz_intensity_rgb.labels',
        'bildstein_station5_xyz_intensity_rgb.labels',
        'domfountain_station1_xyz_intensity_rgb.labels',
        'domfountain_station2_xyz_intensity_rgb.labels',
        'domfountain_station3_xyz_intensity_rgb.labels',
        'neugasse_station1_xyz_intensity_rgb.labels',
        'sg27_station1_intensity_rgb.labels',
        'sg27_station2_intensity_rgb.labels',
        'sg27_station4_intensity_rgb.labels',
        'sg27_station5_intensity_rgb.labels',
        'sg27_station9_intensity_rgb.labels',
        'sg28_station4_intensity_rgb.labels',
        'untermaederbrunnen_station1_xyz_intensity_rgb.labels',
        'untermaederbrunnen_station3_xyz_intensity_rgb.labels',
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

        for i in range(len(self.c_files)):
            c = os.path.join(self.raw_folder, self.c_files[i])
            e = os.path.join(self.raw_folder, self.e_files[i])

            if not os.path.exists(e):
                if not os.path.exists(c):
                    url = '{}{}'.format(self.url, self.c_files[i])
                    download_url(url, self.raw_folder)
                extract_7z(c, self.raw_folder)
                os.unlink(c)

        c = os.path.join(self.raw_folder, self.c_target_file)
        es = [os.path.join(self.raw_folder, f) for f in self.e_target_files]
        exists = all(os.path.exists(e) for e in es)

        if not exists:
            if not os.path.exists(c):
                url = '{}{}'.format(self.target_url, self.c_target_file)
                download_url(url, self.raw_folder)
            extract_7z(c, self.raw_folder)
            os.unlink(c)
