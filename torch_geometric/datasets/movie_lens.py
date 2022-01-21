import os
import os.path as osp
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData, InMemoryDataset, download_url, extract_zip
)


class MovieLens(InMemoryDataset):
    r"""A heterogeneous rating dataset, assembled by GroupLens Research from
    the `MovieLens web site <https://movielens.org>`_, consisting of nodes of
    type :obj:`"movie"` and :obj:`"user"`.
    User ratings for movies are available as ground truth labels for the edges
    between the users and the movies :obj:`("user", "rates", "movie")`.

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
        model_name (str): Name of model used to transform movie titles to node
            features. The model comes from the`Huggingface SentenceTransformer
            <https://huggingface.co/sentence-transformers>`_.
    """

    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    def __init__(self, root, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 model_name: Optional[str] = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('ml-latest-small', 'movies.csv'),
            osp.join('ml-latest-small', 'ratings.csv'),
        ]

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        data = HeteroData()

        df = pd.read_csv(self.raw_paths[0], index_col='movieId')
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = df['genres'].str.get_dummies('|').values
        genres = torch.from_numpy(genres).to(torch.float)

        model = SentenceTransformer(self.model_name)
        with torch.no_grad():
            emb = model.encode(df['title'].values, show_progress_bar=True,
                               convert_to_tensor=True).cpu()

        data['movie'].x = torch.cat([emb, genres], dim=-1)

        df = pd.read_csv(self.raw_paths[1])
        user_mapping = {idx: i for i, idx in enumerate(df['userId'].unique())}
        data['user'].num_nodes = len(user_mapping)

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = rating

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
