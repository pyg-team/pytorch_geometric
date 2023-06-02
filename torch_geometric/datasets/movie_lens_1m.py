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

MOVIE_HEADERS = ["movieId", "title", "genres"]
USER_HEADERS = ["userId", "gender", "age", "occupation", "zipCode"]
RATING_HEADERS = ['userId', 'movieId', 'rating', 'timestamp']


class MovieLens1M(InMemoryDataset):
    r"""The MovieLens 1M heterogeneous rating dataset, assembled by GroupLens
    Research from the `MovieLens web site <https://movielens.org>`__,
    consisting of movies (3,883 nodes) and users (6,040 nodes) with
    approximately 1 million ratings between them.
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
          - 3,883
          - 18
          -
        * - User
          - 6,040
          - 30
          -
        * - User-Movie
          - 1,000,209
          - 1
          - 1
    """
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'

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
        return ['movies.dat', 'users.dat', 'ratings.dat']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-1m')
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self):
        import pandas as pd

        data = HeteroData()

        # Process movie data:
        df = pd.read_csv(
            self.raw_paths[0],
            sep='::',
            header=None,
            index_col='movieId',
            names=MOVIE_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
        )
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = df['genres'].str.get_dummies('|').values
        genres = torch.from_numpy(genres).to(torch.float)

        data['movie'].x = genres

        # Process user data:
        df = pd.read_csv(
            self.raw_paths[1],
            sep='::',
            header=None,
            index_col='userId',
            names=USER_HEADERS,
            dtype='str',
            encoding='ISO-8859-1',
            engine='python',
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = df['age'].str.get_dummies().values
        age = torch.from_numpy(age).to(torch.float)

        gender = df['gender'].str.get_dummies().values
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values
        occupation = torch.from_numpy(occupation).to(torch.float)

        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        # Process rating data:
        df = pd.read_csv(
            self.raw_paths[2],
            sep='::',
            header=None,
            names=RATING_HEADERS,
            encoding='ISO-8859-1',
            engine='python',
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

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
