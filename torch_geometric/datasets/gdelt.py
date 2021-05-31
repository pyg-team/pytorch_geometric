from typing import Optional, Callable, List

import torch
from torch_geometric.data import download_url
from torch_geometric.io import read_txt_array

from .icews import EventDataset


class GDELT(EventDataset):
    r"""The Global Database of Events, Language, and Tone (GDELT) dataset used
    in the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal
    Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of
    events collected from 1/1/2018 to 1/31/2018 (15 minutes time granularity).

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
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
    splits = [0, 1734399, 1973164, 2278405]  # Train/Val/Test splits.

    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index(f'{split}.pt')
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def num_nodes(self) -> int:
        return 7691

    @property
    def num_rels(self) -> int:
        return 240

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{name}.txt' for name in ['train', 'valid', 'test']]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process_events(self) -> torch.Tensor:
        events = []
        for path in self.raw_paths:
            data = read_txt_array(path, sep='\t', end=4, dtype=torch.long)
            data[:, 3] = data[:, 3] // 15
            events += [data]
        return torch.cat(events, dim=0)

    def process(self):
        s = self.splits
        data_list = super(GDELT, self).process()
        torch.save(self.collate(data_list[s[0]:s[1]]), self.processed_paths[0])
        torch.save(self.collate(data_list[s[1]:s[2]]), self.processed_paths[1])
        torch.save(self.collate(data_list[s[2]:s[3]]), self.processed_paths[2])
