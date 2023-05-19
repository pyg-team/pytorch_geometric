import os
import os.path as osp
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


class MovieLens100K(InMemoryDataset):
    r"""The MovieLens 100K heterogeneous rating dataset, assembled by GroupLens
    Research from the `MovieLens web site <https://movielens.org>`_, consisting
    of entities of type :obj:`"movie"` (1700 nodes) and :obj:`"user"`
    (1000 nodes). User ratings for movies are available as ground truth labels
    for the edges between the users and the movies
    :obj:`("user", "rates", "movie")`.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

    def __init__(self, root, split="train",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)

        if split not in {'train', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'test'].index(split)]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('ml-100k', 'u.item'),
            osp.join('ml-100k', 'u.user'),
            osp.join('ml-100k', 'u1.base'),
            osp.join('ml-100k', 'u1.test'),
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'test_data.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        import pandas as pd

        # Process movie data
        df = pd.read_csv(self.raw_paths[0], sep="|", header=None,
                         names=MOVIE_HEADERS, index_col='movieId',
                         encoding="ISO-8859-1")
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genre_headers = MOVIE_HEADERS[6:]
        movie_features = df[genre_headers].values
        movie_features = torch.from_numpy(movie_features).to(torch.float)

        # Process user data
        df = pd.read_csv(
            self.raw_paths[1], sep="|", header=None,
            names=["userId", "age", "gender", "occupation",
                   "zipCode"], index_col='userId', encoding="ISO-8859-1")
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        max_age = df["age"].values.max()
        age = df["age"].values / max_age
        age = torch.from_numpy(age).to(torch.float).unsqueeze(dim=1)

        gender = df['gender'].str.get_dummies().values
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values
        occupation = torch.from_numpy(occupation).to(torch.float)

        user_features = torch.cat([age, gender, occupation], dim=-1)

        data_list = []

        for path in self.raw_paths[-2:]:
            data = HeteroData()
            data['movie'].x = movie_features
            data['user'].x = user_features

            # Process user ratings data
            df = pd.read_csv(
                path, sep="\t", header=None,
                names=['userId', 'movieId', 'rating', 'timestamp'])

            src = [user_mapping[idx] for idx in df['userId']]
            dst = [movie_mapping[idx] for idx in df['movieId']]
            edge_index = torch.tensor([src, dst])

            rating = torch.from_numpy(df['rating'].values).to(torch.long)
            data['user', 'rates', 'movie'].edge_index = edge_index
            data['user', 'rates', 'movie'].edge_label = rating

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        for data, path in zip(data_list, self.processed_paths):
            torch.save(self.collate([data]), path)
