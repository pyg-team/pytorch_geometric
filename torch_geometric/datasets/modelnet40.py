import os
import glob

from .dataset import Dataset
from .utils.download import download_url
from .utils.extract import extract_zip


class ModelNet40(Dataset):
    url = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'

    def __init__(self, root, train=True, transform=None):
        super(ModelNet40, self).__init__(root, transform)

    @property
    def raw_files(self):
        return ['awd']

    @property
    def processed_files(self):
        return ['training.pt', 'test.pt']

    def download(self):
        file_path = download_url(self.url, self.raw_folder)
        extract_zip(file_path, self.raw_folder)
        os.unlink(file_path)


#     def process(self):
#         train, test = [], []
#         for c, target in enumerate(self.categories):
#             dir = osp.join(self.raw_folder, 'ModelNet10')
#             train_files = glob
#             test_files
#             pass

#         raise NotImplementedError

#     def process_example(self, filename, target):
#         pass
