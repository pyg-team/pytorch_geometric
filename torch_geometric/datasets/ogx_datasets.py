import os
import pickle as pkl
from typing import Callable, Optional

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.utils import from_networkx


class OGXBenchmark(InMemoryDataset):
    r"""The OpenGraphXAI benchmark datasets from the
    `"A method for the systematic generation of graph XAI benchmarks via Weisfeiler–Leman coloring"
    <https://doi.org/10.1007/s10618-026-01212-z>`_ paper.
    The benchmark collection comprises 15 binary graph classification datasets.
    Graphs are molecules with a single categorical input feature (atom type), edges are bonds.
    Each graph has a ground truth explanation annotated in the :code:`mask` attribute.

    Args:
        root: Root directory where the dataset should be saved.
        name: The name of the dataset (e.g. :obj:`"alpha"`).
        split: Dataset split, optional (:obj:`"train"`, :obj:`"val"`, :obj:`"test"`).
            If not specified, the full dataset is loaded.
        transform: A function/transform that takes in a
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform: A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk.
        force_reload: Whether to re-process the dataset.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 12
        :header-rows: 1

        * - Name
          - #graphs
          - #class 0
          - #class 1
          - ground truth
        * - alfa
          - 1,103
          - 505
          - 598
          - both
        * - bravo
          - 1,124
          - 573
          - 551
          - both
        * - charlie
          - 1,006
          - 463
          - 543
          - both
        * - delta
          - 1,078
          - 515
          - 563
          - both
        * - echo
          - 1,156
          - 598
          - 558
          - both
        * - foxtrot
          - 2,182
          - 1,195
          - 987
          - class 1
        * - golf
          - 1,795
          - 833
          - 962
          - class 0
        * - hotel
          - 901
          - 444
          - 457
          - class 0
        * - india
          - 1,248
          - 658
          - 590
          - class 0
        * - juliett
          - 1,018
          - 563
          - 455
          - class 0
        * - kilo
          - 3,985
          - 2,007
          - 1,978
          - class 0
        * - lima
          - 5,872
          - 3,086
          - 2,786
          - class 0
        * - mike
          - 4,669
          - 2,444
          - 2,225
          - class 0
        * - november
          - 3,140
          - 1,609
          - 1,531
          - class 0
        * - oscar
          - 4,613
          - 2,340
          - 2,273
          - class 0
    """

    url = r'https://github.com/OpenGraphXAI/benchmarks/raw/refs/heads/main/data/raw/'

    def __init__(self,
                 root: str,
                 name: str,
                 split: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 force_reload: bool = False,
                 ) -> None:

        assert name in ['alfa', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf',
                        'hotel', 'india', 'juliet', 'kilo'], f'Wrong dataset name: "{name}"'
        assert split in ['train', 'val', 'test'] or split is None, f'Unknown split: "{split}"'

        self.name_id = name
        self.name = f'OGX_{self.name_id.capitalize()}'

        super().__init__(root=root, transform=transform, pre_transform=pre_transform,
                         force_reload=force_reload)

        if split == 'train':
            self.load(self.processed_paths[1])
        elif split == 'val':
            self.load(self.processed_paths[2])
        elif split == 'test':
            self.load(self.processed_paths[3])
        else:
            self.load(self.processed_paths[0])

    def download(self):
        for raw_file in self.raw_file_names:
            download_url(f'{self.url}{raw_file}', self.raw_dir)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.name_id}.pkl', f'{self.name_id}_splits.pkl']

    @property
    def processed_file_names(self):
        return [f'{self.name}.pt'] + [f'{self.name}_{split}.pt' for split in ['train', 'val', 'test']]

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            graphs = pkl.load(f)

        with open(self.raw_paths[1], 'rb') as f:
            splits = pkl.load(f)

        data_list = []

        for class_idx in (0, 1):
            for graph in graphs[f'class{class_idx}']:
                data = from_networkx(graph)
                data_list.append(Data(x=data.x,
                                      edge_index=data.edge_index,
                                      mask=data.mask if hasattr(data, 'mask') else torch.zeros_like(data.x).bool(),
                                      mask_root=data.mask_root if hasattr(data, 'mask_root') else torch.zeros_like(
                                          data.x).bool(),
                                      y=torch.tensor([class_idx])))

        for i, split in enumerate(['train', 'val', 'test']):
            split_data = [data_list[idx] for idx in splits[0][split]]
            if self.pre_filter is not None:
                split_data = [data for data in split_data if self.pre_filter(data)]
            if self.pre_transform is not None:
                split_data = [self.pre_transform(data) for data in split_data]
            self.save(split_data, self.processed_paths[i + 1])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)})'
