from typing import Callable, List, Optional

import torch
from torch import Tensor

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.io import read_txt_array


class EventDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

    @property
    def num_nodes(self) -> int:
        raise NotImplementedError

    @property
    def num_rels(self) -> int:
        raise NotImplementedError

    def process_events(self) -> Tensor:
        raise NotImplementedError

    def _process_data_list(self) -> List[Data]:
        events = self.process_events()
        events = events - events.min(dim=0, keepdim=True)[0]

        data_list = []
        for (sub, rel, obj, t) in events.tolist():
            data = Data(sub=sub, rel=rel, obj=obj, t=t)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list


class ICEWS18(EventDataset):
    r"""The Integrated Crisis Early Warning System (ICEWS) dataset used in
    the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal
    Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of
    events collected from 1/1/2018 to 10/31/2018 (24 hours time granularity).

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
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
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://github.com/INK-USC/RE-Net/raw/master/data/ICEWS18'
    splits = [0, 373018, 419013, 468558]  # Train/Val/Test splits.

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        idx = self.processed_file_names.index(f'{split}.pt')
        self.load(self.processed_paths[idx])

    @property
    def num_nodes(self) -> int:
        return 23033

    @property
    def num_rels(self) -> int:
        return 256

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{name}.txt' for name in ['train', 'valid', 'test']]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process_events(self) -> Tensor:
        events = []
        for path in self.raw_paths:
            data = read_txt_array(path, sep='\t', end=4, dtype=torch.long)
            data[:, 3] = data[:, 3] // 24
            events += [data]
        return torch.cat(events, dim=0)

    def process(self) -> None:
        s = self.splits
        data_list = self._process_data_list()
        self.save(data_list[s[0]:s[1]], self.processed_paths[0])
        self.save(data_list[s[1]:s[2]], self.processed_paths[1])
        self.save(data_list[s[2]:s[3]], self.processed_paths[2])
