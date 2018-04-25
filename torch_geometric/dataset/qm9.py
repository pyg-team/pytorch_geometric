import os

import torch
from torch import LongTensor as Long
from torch_geometric.data import InMemoryDataset, collate_to_set, split_set
from torch_geometric.read import parse_txt, parse_sdf
from torch_geometric.datasets.utils.download import download_url
from torch_geometric.datasets.utils.extract import extract_tar
from torch_geometric.datasets.utils.spinner import Spinner


class QM9(InMemoryDataset):
    num_graphs = 130831
    data_url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
               'datasets/gdb9.tar.gz'
    mask_url = 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root, split, transform=None):
        super(QM9, self).__init__(root, transform)

        self.split = split
        self.dataset, self.slices = torch.load(self._processed_files[0])
        print(self.dataset.edge_index.size())
        print(self.dataset.edge_index.is_contiguous())

    def __len__(self):
        return self.split.size(0)

    def get_data(self, i):
        return super(QM9, self).get_data(self.split[i])

    @property
    def raw_files(self):
        return 'gdb9.sdf', 'gdb9.sdf.csv', '3195404'

    @property
    def processed_files(self):
        return 'data.pt'

    def download(self):
        file_path = download_url(self.data_url, self.raw_dir)
        extract_tar(file_path, self.raw_dir, mode='r')
        os.unlink(file_path)
        download_url(self.mask_url, self.raw_dir)

    def process(self):
        spinner = Spinner('Processing').start()

        # Parse *.sdf to data list.
        with open(self._raw_files[0], 'r') as f:
            src = [x.split('\n')[3:-2] for x in f.read().split('$$$$\n')[:-1]]
            data_list = [parse_sdf(x) for x in src]

        dataset, slices = collate_to_set(data_list)

        # Add targets to dataset.
        with open(self._raw_files[1], 'r') as f:
            src = f.read().split('\n')[1:-1]
            dataset.y = parse_txt(src, sep=',', start=4, end=16)
            slices['y'] = torch.arange(dataset.y.size(0) + 1, out=Long())

        # Remove invalid data.
        with open(self._raw_files[2], 'r') as f:
            src = parse_txt(f.read().split('\n')[9:-2], end=1, out=Long())
            split = torch.arange(dataset.y.size(0), out=Long())
            split = split[split == split.clone().index_fill_(0, src, -1)]

        dataset, slices = split_set(dataset, slices, split)
        torch.save((dataset, slices), self._processed_files[0])

        spinner.success()
