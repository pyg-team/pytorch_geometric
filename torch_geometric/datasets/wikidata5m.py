import gzip
import json
import os
import tarfile
from typing import Callable, List, Optional
import os.path as osp

import torch

from torch_geometric.data import Data, InMemoryDataset, download_url


class Wikidata5m_trans(InMemoryDataset):
    r"""The transductive Wikidata5m dataset from the `"KEPLER: A Unified Model for Knowledge Embedding and Pre-trained
    Language Representation"
    <https://arxiv.org/pdf/1911.06136.pdf>`_ paper,
    containing 4,594,485 entities, 822 relations and 20,614,279 train triples, 5,163 validation triples,
    and 5,133 test triples.

    .. note::

        "Wikidata5m"<https://deepgraphlearning.github.io/project/wikidata5m> is a large-scale knowledge graph dataset
        with aligned corpus extracted form Wikidata.

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
    """
    urls = ['https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1',
            'https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1']

    urls_features = {
        'text_emb_bert': 'https://uni-bielefeld.sciebo.de/s/yuBKzBxsEc9j3hy/download'
    }

    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)

        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt', 'wikidata5m_text.txt.gz']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt', 'entity_to_id.json', 'relation_to_id.json']

    def load_features(self, feature_type='text_emb_bert') -> torch.TensorType:
        """
        Features are separated from the Data objects in order to save memory as all splits use the same nodes with the
        same features.
        Args:
            feature_type: Available pre-computed features: text_emb_bert (384 dim bert embeddings)

        Returns: Requested feature tensor indexed according to entity_to_id.json.

        """
        if not osp.exists(osp.join(self.processed_dir, f'{feature_type}.pt')):
            download_url(self.urls_features[feature_type], self.processed_dir, filename=f'{feature_type}.pt')
        return torch.load(osp.join(self.processed_dir, f'{feature_type}.pt'))

    def get_uri_to_id_dics(self):
        id_dicts = {}
        for d in ['entity_to_id', 'relation_to_id']:
            id_dicts[d] = json.load(open(osp.join(self.processed_dir, f'{d}.json')))

    def download(self):
        for url in self.urls:
            download_url(url, self.raw_dir)
        compressed_dataset_path = osp.join(self.raw_dir, 'wikidata5m_transductive.tar.gz')
        with tarfile.open(compressed_dataset_path) as compressed_in:
            compressed_in.extractall(self.raw_dir)

        os.remove(compressed_dataset_path)

        for raw_file_name in ['train.txt', 'valid.txt', 'test.txt']:
            os.rename(osp.join(self.raw_dir, f'wikidata5m_transductive_{raw_file_name}'),
                      osp.join(self.raw_dir, raw_file_name))

    def process(self):

        # entity IDs are assigned according to the corpus s.t. indexing follows a common schema
        entity_to_id = {}
        with gzip.open(osp.join(self.raw_dir, 'wikidata5m_text.txt.gz'), 'rt') as descriptions_file_in:
            for i, line in enumerate(descriptions_file_in.readlines()):
                values = line.strip().split('\t')
                uri = values[0]
                entity_to_id[uri] = i

        with open(osp.join(self.processed_dir, 'entity_to_id.json'), 'w') as entity_2_id_out:
            json.dump(entity_to_id, entity_2_id_out)

        # relation IDs are assigned on the fly
        relation_to_id = {}
        data_list = []

        for raw_file_name in ['train.txt', 'valid.txt', 'test.txt']:
            path = osp.join(self.raw_dir, raw_file_name)
            print('raw path:', path)

            edge_index = []
            edge_type = []
            with open(path) as triples_in:
                for line in triples_in:
                    head, relation, tail = line[:-1].split('\t')
                    edge_index.append([entity_to_id[head], entity_to_id[tail]])
                    if relation not in relation_to_id:
                        relation_to_id[relation] = len(relation_to_id)
                    edge_type.append(relation_to_id[relation])

            data = Data(edge_index=torch.tensor(edge_index), edge_type=torch.tensor(edge_type))
            data_list.append(data)

        with open(osp.join(self.processed_dir, 'relation_to_id.json'), 'w') as relation_2_id_out:
            json.dump(relation_to_id, relation_2_id_out)

        for data, path in zip(data_list, self.processed_paths):
            data.num_nodes = len(entity_to_id)
            torch.save(self.collate([data]), path)
