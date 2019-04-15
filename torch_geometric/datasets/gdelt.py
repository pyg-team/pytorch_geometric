import torch
from torch_geometric.data import download_url
from torch_geometric.read import read_txt_array

from .icews import EventDataset


class GDELT(EventDataset):
    r"""The Global Database of Events, Language, and Tone (GDELT) dataset used
    in the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal
    Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of
    events collected from 4/1/2015 to 3/31/2016 (15 minutes time granularity).

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://github.com/INK-USC/RENet/raw/master/data/GDELT'

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(GDELT, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def num_nodes(self):
        return 7691

    @property
    def num_rels(self):
        return 240

    @property
    def splits(self):
        return [0, 1734399, 1973164, 2278405]  # Train/Val/Test splits.

    @property
    def raw_file_names(self):
        return ['{}.txt'.format(name) for name in ['train', 'valid', 'test']]

    def download(self):
        for filename in self.raw_file_names:
            download_url('{}/{}'.format(self.url, filename), self.raw_dir)

    def process_events(self):
        events = []
        for path in self.raw_paths:
            data = read_txt_array(path, sep='\t', end=4, dtype=torch.long)
            data[:, 3] = data[:, 3] / 15
            events += [data]
        return torch.cat(events, dim=0)
