import os

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_tar
from torch_geometric.read import parse_sdf, parse_txt_array


class QM9(InMemoryDataset):
    data_url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
               'datasets/gdb9.tar.gz'
    mask_url = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root, transform=None, pre_transform=None):
        super(QM9, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', '3195404']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        file_path = download_url(self.data_url, self.raw_dir)
        extract_tar(file_path, self.raw_dir, mode='r')
        os.unlink(file_path)
        download_url(self.mask_url, self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data_list = [parse_sdf(x) for x in f.read().split('$$$$\n')[:-1]]
        print(len(data_list))

        with open(self.raw_paths[1], 'r') as f:
            y = parse_txt_array(f.read().split('\n')[1:-1], ',', 4, 16)
            print(y.size())

        for i, data in enumerate(data_list):
            data.y = y[i].view(1, -1)

        # Remove invalid data.
        with open(self.raw_paths[2], 'r') as f:
            invalid = f.read().split('\n')[9:-2]
            invalid = parse_txt_array(invalid, end=1, dtype=torch.long)
            valid = torch.ones(len(data_list), dtype=torch.uint8)
            valid[invalid] = 0
            valid = valid.nonzero()
        data_list = [data_list[idx] for idx in valid]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
