import os

from .dataset import Dataset
from .utils.download import download_url
from .utils.extract import extract_tar
from .utils.tu_format import tu_files


class Cuneiform2(Dataset):
    url = 'http://www.roemisch-drei.de/cuneiform.tar.gz'
    prefix = 'CuneiformArrangement'

    def __init__(self, root, split=None, transform=None):
        super(Cuneiform2, self).__init__(root, transform)

    @property
    def raw_files(self):
        return tu_files(
            self.prefix,
            A=True,
            graph_indicator=True,
            graph_labels=True,
            node_labels=True,
            node_attributes=True)

    @property
    def processed_files(self):
        return 'data.pt'

    def download(self):
        file_path = download_url(self.url, self.raw_folder)
        extract_tar(file_path, self.raw_folder, mode='r')
        os.unlink(file_path)

    def process(self):
        print('huhu')
        return 'awd'
