import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import gdown
import torch
from tape.dataset.parser.base import Article, Parser

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


class PubmedParser(Parser):
    """Parser for [PubMed Diabetes](https://linqs.org/datasets/#pubmed-diabetes) dataset."""  # noqa

    urls = {
        'original':
        'https://drive.google.com/file/d/1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W/view?usp=sharing',  # noqa
        'llm_responses':
        'https://drive.google.com/file/d/166waPAjUwu7EWEvMJ0heflfp0-4EvrZS/view?usp=sharing',  # noqa
    }

    def __init__(self, seed: int = 0, cache_dir: str = '.cache') -> None:
        super().__init__(seed, cache_dir)
        self._dtype_to_path = self.download_data()
        self.graph = PubmedGraph(dir_path=self._dtype_to_path['original'],
                                 cache_dir=self.cache_dir)

    def parse(self) -> None:
        self.graph.load()

    def download_data(self) -> Dict[str, Path]:
        dtype_to_path = {}
        for dtype, url in PubmedParser.urls.items():
            save_dir = self.cache_dir / dtype
            save_dir.mkdir(exist_ok=True, parents=True)
            zip_file_path = save_dir / 'PubMed.zip'
            dtype_to_path[dtype] = save_dir / (
                'PubMed' + ('_orig' if dtype == 'original' else ''))

            if not dtype_to_path[dtype].exists():
                file_id = url.split('/d/')[1].split('/')[0]
                download_url = f'https://drive.google.com/uc?export=download&id={file_id}'  # noqa
                gdown.download(download_url, str(zip_file_path), quiet=False)

                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(str(save_dir))
                zip_file_path.unlink()

        return dtype_to_path


class PubmedGraph:
    def __init__(self, dir_path: Path, cache_dir: Path) -> None:

        self.dir_path = dir_path
        self.cache_dir = cache_dir

        self.dataset: Optional[Data] = None
        self.n_classes = 3
        self.class_id_to_label = {
            0:
            dict(
                label='Experimental Diabetes',
                description=('Studies investigating diabetes in controlled '
                             'experimental settings.'),
            ),
            1:
            dict(
                label='Type 1 Diabetes',
                description=(
                    'An autoimmune disease where the body attacks and '
                    'destroys insulin-producing cells in the pancreas.'),
            ),
            2:
            dict(
                label='Type 2 Diabetes', description=(
                    'A metabolic disorder characterized by high blood '
                    "sugar levels due to the body's inability to "
                    'effectively use insulin.')),
        }
        self.n_nodes = 19_717
        self.n_features = 500
        # PubMed Articles
        self.articles: Optional[List[Article]] = None
        self.pubmed_id_to_node_id = {}
        # Nodes
        self.node_features: Optional[torch.tensor] = None
        self.node_labels: Optional[List] = None
        self.node_feature_to_idx: Optional[Dict] = None
        # Edges
        self.edge_index: Optional[torch.tensor] = None
        self.adj_matrix: Optional[torch.tensor] = None
        # Split containing train/val/test node ids
        self.split: Optional[Dict] = None

    def load(self) -> None:

        self._load_articles()
        self._load_nodes()
        self._load_edges()
        self._load_pyg_dataset()

    def _load_pyg_dataset(self):

        print('Loading PyG dataset...')

        self.dataset = Planetoid(self.cache_dir, 'PubMed')[0]
        # Replace dataset matrices with the PubMed-Diabetes data,
        # for which we have the original PubMed IDs
        self.dataset.x = self.node_features
        self.dataset.y = self.node_labels
        self.dataset.edge_index = self.edge_index

        # Split dataset nodes into train/val/test and update
        # the train/val/test masks
        n_nodes = self.dataset.num_nodes
        node_ids = torch.randperm(n_nodes)
        self.split = {}
        for split_name in ('train', 'val', 'test'):
            if split_name == 'train':
                subset = slice(0, int(n_nodes * 0.6))
            elif split_name == 'val':
                subset = slice(int(n_nodes * 0.6), int(n_nodes * 0.8))
            else:
                subset = slice(int(n_nodes * 0.8), n_nodes)

            ids = node_ids[subset].sort()[0]
            setattr(self.dataset, f'{split_name}_id', ids)
            mask = torch.zeros(n_nodes, dtype=bool)
            mask[ids] = True
            setattr(self.dataset, f'{split_name}_mask', mask)
            self.split[split_name] = ids.tolist()

    def _load_articles(self):

        print('Loading articles...')

        self.articles = []
        path = self.dir_path / 'pubmed.json'
        data = json.loads(path.read_text())
        node_id = 0
        for article in data:
            if (pubmed_id := article.get('PMID')) and (
                    title := article.get('TI')) and (abstract :=
                                                     article.get('AB')):
                self.articles.append(
                    Article(paper_id=pubmed_id, title=title,
                            abstract=abstract))
                self.pubmed_id_to_node_id[pubmed_id] = node_id
                node_id += 1
            else:
                print(f'Ignoring PubMed article with node id "{node_id}" '
                      'due to missing PMID/Abstract/Title.')

        print('No. of PubMed articles with title and '
              f'abstract: {len(self.articles):,}')
        print(f'Updating no. of nodes from {self.n_nodes:,} '
              f'to {len(self.articles):,}')
        self.n_nodes = len(self.articles)

    def _load_nodes(self):

        print('Loading nodes...')

        self.node_features = torch.zeros((self.n_nodes, self.n_features),
                                         dtype=torch.float32)
        self.node_labels = torch.empty(self.n_nodes, dtype=torch.long)
        self.node_feature_to_idx = {}

        with open(self.dir_path / 'data/Pubmed-Diabetes.NODE.paper.tab',
                  'r') as node_file:
            node_file.readline()  # Ignore header
            node_file.readline()  # Ignore header
            k = 0

            for line in node_file.readlines():
                items = line.strip().split('\t')
                pubmed_id = items[0]
                if (node_id :=
                        self.pubmed_id_to_node_id.get(pubmed_id)) is None:
                    print(f'Ignoring PubMed article "{pubmed_id}" due to '
                          'missing PMID/Abstract/Title.')
                    continue

                label = int(items[1].split('=')[-1]) - 1
                self.node_labels[node_id] = label
                features = items[2:-1]
                for feature in features:
                    parts = feature.split('=')
                    fname = parts[0]
                    fvalue = float(parts[1])
                    if fname not in self.node_feature_to_idx:
                        self.node_feature_to_idx[fname] = k
                        k += 1
                    self.node_features[
                        node_id, self.node_feature_to_idx[fname]] = fvalue

    def _load_edges(self):

        print('Loading edges...')

        edges = []
        self.adj_matrix = torch.zeros((self.n_nodes, self.n_nodes),
                                      dtype=torch.float32)

        with open(self.dir_path / 'data/Pubmed-Diabetes.DIRECTED.cites.tab',
                  'r') as edge_file:
            edge_file.readline()  # Ignore header
            edge_file.readline()  # Ignore header

            for line in edge_file.readlines():
                items = line.strip().split('\t')
                tail = items[1].split(':')[-1]
                head = items[3].split(':')[-1]
                if ((head_node_id :=
                     self.pubmed_id_to_node_id.get(head)) is None
                        or (tail_node_id :=
                            self.pubmed_id_to_node_id.get(tail)) is None):
                    print(f'Ignoring edge ({head}, {tail}) due to either of '
                          'the PubMed articles being discarded.')
                    continue

                self.adj_matrix[tail_node_id, head_node_id] = 1.0
                self.adj_matrix[head_node_id, tail_node_id] = 1.0
                if head != tail:
                    edges.append((head_node_id, tail_node_id))
                    edges.append((tail_node_id, head_node_id))

        edges = torch.tensor(edges, dtype=torch.long)
        self.edge_index = torch.unique(edges, dim=0).T
