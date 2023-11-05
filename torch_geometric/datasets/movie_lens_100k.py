import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

MOVIE_HEADERS = [
    "movieId", "title", "releaseDate", "videoReleaseDate", "IMDb URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
USER_HEADERS = ["userId", "age", "gender", "occupation", "zipCode"]
RATING_HEADERS = ["userId", "movieId", "rating", "timestamp"]


class MovieLens100K(InMemoryDataset):
    r"""The MovieLens 100K heterogeneous rating dataset, assembled by GroupLens
    Research from the `MovieLens web site <https://movielens.org>`__,
    consisting of movies (1,682 nodes) and users (943 nodes) with 100K
    ratings between them.
    User ratings for movies are available as ground truth labels.
    Features of users and movies are encoded according to the `"Inductive
    Matrix Completion Based on Graph Neural Networks"
    <https://arxiv.org/abs/1904.12058>`__ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 20 10 10 10
        :header-rows: 1

        * - Node/Edge Type
          - #nodes/#edges
          - #features
          - #tasks
        * - Movie
          - 1,682
          - 18
          -
        * - User
          - 943
          - 24
          -
        * - User-Movie
          - 80,000
          - 1
          - 1
    """
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return ['u.item', 'u.user', 'u1.base', 'u1.test']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-100k')
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self):
        import pandas as pd

        data = HeteroData()

        # Process movie data:
        df = pd.read_csv(
            self.raw_paths[0],
            sep='|',
            header=None,
            names=MOVIE_HEADERS,
            index_col='movieId',
            encoding='ISO-8859-1',
        )
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        x = df[MOVIE_HEADERS[6:]].values
        data['movie'].x = torch.from_numpy(x).to(torch.float)

        # Process user data:
        df = pd.read_csv(
            self.raw_paths[1],
            sep='|',
            header=None,
            names=USER_HEADERS,
            index_col='userId',
            encoding='ISO-8859-1',
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = df['age'].values / df['age'].values.max()
        age = torch.from_numpy(age).to(torch.float).view(-1, 1)

        gender = df['gender'].str.get_dummies().values
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values
        occupation = torch.from_numpy(occupation).to(torch.float)

        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        # Process rating data for training:
        df = pd.read_csv(
            self.raw_paths[2],
            sep='\t',
            header=None,
            names=RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_index = edge_index

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'movie'].rating = rating

        time = torch.from_numpy(df['timestamp'].values)
        data['user', 'rates', 'movie'].time = time

        data['movie', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = rating
        data['movie', 'rated_by', 'user'].time = time

        # Process rating data for testing:
        df = pd.read_csv(
            self.raw_paths[3],
            sep='\t',
            header=None,
            names=RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_label_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_label_index = edge_label_index

        edge_label = torch.from_numpy(df['rating'].values).to(torch.float)
        data['user', 'rates', 'movie'].edge_label = edge_label

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
