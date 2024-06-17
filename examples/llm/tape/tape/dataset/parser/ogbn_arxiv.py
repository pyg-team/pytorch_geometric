import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import gdown
import pandas as pd
import requests
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from tape.dataset.parser.base import Article, Parser

import torch_geometric.transforms as T
from torch_geometric.data import Data


class OgbnArxivParser(Parser):
    """Parser for [OGB arXiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) dataset."""  # noqa

    urls = {
        'original':
        'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz',  # noqa
        'llm_responses':
        'https://drive.google.com/file/d/1A6mZSFzDIhJU795497R6mAAM2Y9qutI5/view?usp=sharing',  # noqa
    }

    def __init__(self, seed: int = 0, cache_dir: str = '.cache') -> None:
        super().__init__(seed, cache_dir)
        self._dtype_to_path = self.download_data()
        self.graph = OgbArxivGraph(dir_path=self._dtype_to_path['original'],
                                   cache_dir=self.cache_dir)
        self.split = None

    def parse(self) -> None:
        self.graph.load()

    def download_data(self) -> Dict[str, Path]:
        dtype_to_path = {}
        for dtype, url in OgbnArxivParser.urls.items():
            save_dir = self.cache_dir / dtype
            save_dir.mkdir(exist_ok=True, parents=True)
            dtype_to_path[dtype] = save_dir / (
                'ogbn-arxiv' + ('_orig' if dtype == 'original' else ''))
            if url.endswith('.tsv.gz'):
                file_name = url.split('/')[-1]
                dtype_to_path[dtype] /= file_name

            if not dtype_to_path[dtype].exists():
                if 'drive.google.com' in url:
                    zip_file_path = save_dir / 'ogbn-arxiv.zip'
                    file_id = url.split('/d/')[1].split('/')[0]
                    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'  # noqa
                    gdown.download(download_url, str(zip_file_path),
                                   quiet=False)
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(str(save_dir))
                    zip_file_path.unlink()
                else:
                    response = requests.get(url, stream=True)
                    dtype_to_path[dtype].parent.mkdir(exist_ok=True)
                    with open(dtype_to_path[dtype], 'wb') as f:
                        for chunk in response.iter_content(32_768):
                            if chunk:
                                f.write(chunk)

        return dtype_to_path


class OgbArxivGraph:
    def __init__(self, dir_path: Path, cache_dir: Path) -> None:

        self.dir_path = dir_path
        self.cache_dir = cache_dir

        self.dataset: Optional[Data] = None
        self.n_classes = 40
        self.n_nodes = 169_343
        self.n_features = 128
        self.class_id_to_label: Optional[Dict] = None
        self.articles: Optional[List[Article]] = None
        # Split containing train/val/test node ids
        self.split: Optional[Dict] = None

    def load(self):
        self._load_ogb_dataset()
        self._load_articles()
        self.class_id_to_label = self._load_class_label_mapping()

    def _load_ogb_dataset(self):
        print('Loading OGB dataset...')

        dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                         root=self.cache_dir,
                                         transform=T.ToSparseTensor())
        self.split = dataset.get_idx_split()

        data = dataset[0]

        train_mask = torch.zeros(data.num_nodes).bool()
        train_mask[self.split['train']] = True
        data.train_mask = train_mask

        val_mask = torch.zeros(data.num_nodes).bool()
        val_mask[self.split['valid']] = True
        data.val_mask = val_mask

        test_mask = torch.zeros(data.num_nodes).bool()
        test_mask[self.split['test']] = True
        data.test_mask = test_mask

        data.edge_index = data.adj_t.to_symmetric()

        self.dataset = data

    def _load_articles(self):

        mapping_df = pd.read_csv(
            self.cache_dir / 'ogbn_arxiv/mapping/nodeidx2paperid.csv.gz',
            skiprows=1, names=['node_idx', 'paper_id'], compression='gzip')
        title_abstract_df = pd.read_table(
            self.dir_path, header=None,
            names=['paper_id', 'title', 'abstract'], compression='gzip')
        df = mapping_df.astype(dict(paper_id=str)).join(
            title_abstract_df.set_index('paper_id'), on='paper_id')
        self.articles = []
        for row in df.itertuples(index=False):
            self.articles.append(
                Article(paper_id=row.paper_id, title=row.title,
                        abstract=row.abstract))

    def _load_class_label_mapping(self):
        mapping_df = pd.read_csv(
            self.cache_dir /
            'ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz', skiprows=1,
            names=['label_id', 'label'], compression='gzip')
        class_id_to_label = {}
        categories = OgbArxivGraph.fetch_arxiv_category_taxonomy()
        arxiv_cs_categories_df = pd.DataFrame(categories)
        for row in mapping_df.itertuples(index=False):
            class_id_to_label[row.label_id] = arxiv_cs_categories_df.query(
                'label == @label').iloc[0].to_dict()
        return class_id_to_label

    @staticmethod
    def fetch_arxiv_category_taxonomy(
            category: str = 'cs') -> List[Dict[str, str]]:
        text = requests.get(
            'https://r.jina.ai/https://arxiv.org/category_taxonomy').text
        sections = re.split(r'#### ', text)[1:]
        data_list = []
        for section in sections:
            match = re.match(rf'({category}\.\w+) \(([^)]+)\)\n\n', section)
            if match:
                label = match.group(1)
                category_name = match.group(2)
                description = section[match.end():].strip()
                data_list.append({
                    'label': label,
                    'category': category_name,
                    'description': description
                })
        return data_list
