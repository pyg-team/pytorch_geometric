import os
import os.path as osp
import glob
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.read import read_off


class GeometricShapes(InMemoryDataset):
    url = "https://drive.google.com/open?id=1VE-EMIPKWsxJTvFRacrY_YZHB1-fH9Mr"

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GeometricShapes, self).__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['cone', 'cube']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])
        return

    def process_set(self, dataset):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split('/')[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/*.off'.format(folder))
            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)
