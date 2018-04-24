import os
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, split_set
from torch_geometric.read import get_tu_filenames, read_tu_files
from torch_geometric.datasets.utils.download import download_url
from torch_geometric.datasets.utils.extract import extract_zip
from torch_geometric.datasets.utils.spinner import Spinner


class ENZYMES(InMemoryDataset):
    num_graphs = 600
    url = 'https://ls11-www.cs.uni-dortmund.de/people/morris/' \
          'graphkerneldatasets/ENZYMES.zip'

    def __init__(self, root, split, transform=None):
        super(ENZYMES, self).__init__(root, transform)

        filename = self._processed_files[0]
        dataset, slices = torch.load(filename)
        self.dataset, self.slices = split_set(dataset, slices, split)

    @property
    def raw_files(self):
        return get_tu_filenames(
            ENZYMES.__name__,
            graph_indicator=True,
            graph_labels=True,
            node_labels=True)

    @property
    def processed_files(self):
        return 'data.pt'

    def download(self):
        filepath = download_url(self.url, self.root)
        extract_zip(filepath, self.root)
        os.unlink(filepath)
        os.rename(osp.join(self.root, ENZYMES.__name__), self.raw_dir)

    def process(self):
        spinner = Spinner('Processing').start()

        dataset, slices = read_tu_files(
            self.raw_dir,
            ENZYMES.__name__,
            graph_indicator=True,
            graph_labels=True,
            node_labels=True)

        torch.save((dataset, slices), self._processed_files[0])

        spinner.success()
