import os

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.read import read_txt_array


class MC3(InMemoryDataset):
    url = 'http://vacommunity.org/tiki-download_file.php?fileId=577'

    def __init__(self, root, transform=None, pre_transform=None):
        super(MC3, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['calls.csv', 'emails.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        # call = read_txt_array(self.raw_paths[0], sep=',', dtype=torch.long)
        # call = call[:, torch.tensor([0, 2])]

        # mail = read_txt_array(self.raw_paths[1], sep=',', dtype=torch.long)
        # mail = mail[:, torch.tensor([0, 2])]

        call = torch.load('/tmp/call.pt')
        mail = torch.load('/tmp/mail.pt')

        nodes = torch.cat([call, mail], dim=0).unique(sorted=True)
        print(nodes)
        print(nodes.size())

        raise NotImplementedError

        print(mail.size())
        print(call.size())

        node_idx, count = mail[:, 0].unique(return_counts=True)
        print(node_idx.size())

        node_idx, count = mail[:, 1].unique(return_counts=True)
        print(node_idx.size())
        node_idx, count = call[:, 0].unique(return_counts=True)
        print(node_idx.size())
        node_idx, count = call[:, 1].unique(return_counts=True)
        print(node_idx.size())

        raise NotImplementedError

    def __repr__(self):
        return '{}()'.format(self.name)
