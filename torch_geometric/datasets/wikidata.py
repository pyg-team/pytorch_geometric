import os
import os.path as osp
from typing import Callable, Dict, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)


class Wikidata5M(InMemoryDataset):
    r"""The Wikidata-5M dataset from the `"KEPLER: A Unified Model for
    Knowledge Embedding and Pre-trained Language Representation"
    <https://arxiv.org/pdf/1911.06136.pdf>`_ paper,
    containing 4,594,485 entities, 822 relations,
    20,614,279 train triples, 5,163 validation triples, and 5,133 test triples.

    `Wikidata-5M <https://deepgraphlearning.github.io/project/wikidata5m>`_
    is a large-scale knowledge graph dataset with aligned corpus
    extracted form Wikidata.

    Args:
        root (str): Root directory where the dataset should be saved.
        setting (str, optional):
            If :obj:`"transductive"`, loads the transductive dataset.
            If :obj:`"inductive"`, loads the inductive dataset.
            (default: :obj:`"transductive"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(
        self,
        root: str,
        setting: str = 'transductive',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        if setting not in {'transductive', 'inductive'}:
            raise ValueError(f"Invalid 'setting' argument (got '{setting}')")

        self.setting = setting

        self.urls = [
            ('https://www.dropbox.com/s/7jp4ib8zo3i6m10/'
             'wikidata5m_text.txt.gz?dl=1'),
            'https://uni-bielefeld.sciebo.de/s/yuBKzBxsEc9j3hy/download',
        ]
        if self.setting == 'inductive':
            self.urls.append('https://www.dropbox.com/s/csed3cgal3m7rzo/'
                             'wikidata5m_inductive.tar.gz?dl=1')
        else:
            self.urls.append('https://www.dropbox.com/s/6sbhm0rwo4l73jq/'
                             'wikidata5m_transductive.tar.gz?dl=1')

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'wikidata5m_text.txt.gz',
            'download',
            f'wikidata5m_{self.setting}_train.txt',
            f'wikidata5m_{self.setting}_valid.txt',
            f'wikidata5m_{self.setting}_test.txt',
        ]

    @property
    def processed_file_names(self) -> str:
        return f'{self.setting}_data.pt'

    def download(self):
        for url in self.urls:
            download_url(url, self.raw_dir)
        path = osp.join(self.raw_dir, f'wikidata5m_{self.setting}.tar.gz')
        extract_tar(path, self.raw_dir)
        os.remove(path)

    def process(self):
        import gzip

        entity_to_id: Dict[str, int] = {}
        with gzip.open(self.raw_paths[0], 'rt') as f:
            for i, line in enumerate(f):
                values = line.strip().split('\t')
                entity_to_id[values[0]] = i

        x = torch.load(self.raw_paths[1])

        edge_index = []
        edge_type = []
        split_index = []

        rel_to_id: Dict[str, int] = {}
        for split, path in enumerate(self.raw_paths[2:]):
            with open(path, 'r') as f:
                for line in f:
                    head, rel, tail = line[:-1].split('\t')
                    edge_index.append([entity_to_id[head], entity_to_id[tail]])
                    if rel not in rel_to_id:
                        rel_to_id[rel] = len(rel_to_id)
                    edge_type.append(rel_to_id[rel])
                    split_index.append(split)

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_type = torch.tensor(edge_type)
        split_index = torch.tensor(split_index)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            train_mask=split_index == 0,
            val_mask=split_index == 1,
            test_mask=split_index == 2,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
