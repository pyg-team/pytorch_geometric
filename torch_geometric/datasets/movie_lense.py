import os.path as osp

import torch
import torch.nn.functional as F

import pandas as pd
from sentence_transformers import SentenceTransformer

from torch_geometric.data import HeteroData, download_url, extract_zip

from torch_geometric.data import InMemoryDataset, download_url
from typing import Optional, Callable, List


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class SequenceEncoder(object):
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


class GenresEncoder(object):
    # The 'GenreEncoder' splits the raw column strings by 'sep' and converts
    # individual elements to categorical labels.
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)



class MovieLense(InMemoryDataset):
    r"""A heterogeneous rating dataset, assembled by GroupLens Research, from 
    the `MovieLens web site <https://movielens.org>`_, consisting of nodes of
    type :obj:`"movie"` and :obj:`"user"`. User ratings for movies are available 
    as ground truth labels for the edges  between the users and the 
    movies ('user', 'rates', 'movie').

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    def __init__(self, root, transform: Optional[Callable] = None,pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [osp.join('ml-latest-small', 'movies.csv'), osp.join('ml-latest-small', 'ratings.csv')]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        #os.remove(path)
        

    def process(self):
        user_x, user_mapping = load_node_csv(osp.join(self.raw_dir, self.raw_file_names[1]), index_col='userId')

        movie_x, movie_mapping = load_node_csv(
            osp.join(self.raw_dir, self.raw_file_names[0]), index_col='movieId', encoders={
                'title': SequenceEncoder(),
                'genres': GenresEncoder()
            })

        edge_index, edge_label = load_edge_csv(
            osp.join(self.raw_dir, self.raw_file_names[1]),
            src_index_col='userId',
            src_mapping=user_mapping,
            dst_index_col='movieId',
            dst_mapping=movie_mapping,
            encoders={'rating': IdentityEncoder(dtype=torch.long)},
        )

        data = HeteroData()
        data['user'].num_nodes = len(user_mapping)  # Users do not have any features.
        data['movie'].x = movie_x
        
        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = edge_label

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])