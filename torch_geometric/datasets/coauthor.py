import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.read import read_npz


class Coauthor(InMemoryDataset):
    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(self, root, name, transform=None, pre_transform=None):
        assert name.lower() in ['cs', 'physics']
        self.name = 'CS' if name.lower() == 'cs' else 'Physics'
        super(Coauthor, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'ms_academic_{}.npz'.format(self.name[:3].lower())

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url(self.url + self.raw_file_names, self.raw_dir)

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}{}()'.format(self.__class__.__name__, self.name)
