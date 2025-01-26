import os
import zipfile

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import HeteroData, InMemoryDataset, download_url


class MovieLens25(InMemoryDataset):
    """The movielens dataset https://files.grouplens.org/datasets/movielens/.
    A bipartite graph where the edges indicate movie which is rated by a user.
    The observational causal information for the movies includes treatments
    based on the number of ratings and the outcome based on the average rating.
    Node features are embeddings from a SentenceTransformer on title and genre.


    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        local_filename = self.url.split("/")[-1]

        download_url(f"{self.url}", self.raw_dir)
        # Unzip the file
        with zipfile.ZipFile(f"{self.raw_dir}/{local_filename}",
                             "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        # Remove the downloaded zip file
        os.remove(f"{self.raw_dir}/{local_filename}")

    def process(
        self,
        ratings_dataset: str = "ml-25m/ratings.csv",
        movies_dataset: str = "ml-25m/movies.csv",
        user_threshold: int = 200,
    ):

        edge_index_df = pd.read_csv(f"{self.raw_dir}/{ratings_dataset}")
        edge_index_df = edge_index_df[["movieId", "userId", "rating"]]
        edge_index_df.columns = ["movie", "user", "weight"]

        gx = edge_index_df.groupby(["user"])["movie"].count()
        chosen_users = gx[gx > user_threshold].reset_index()[
            "user"]  # gx.mean()
        edge_index_df = edge_index_df[edge_index_df["user"].isin(chosen_users)]

        # define treated and untreated
        rating_count = edge_index_df.groupby(
            "movie")["weight"].count().reset_index()
        movie_map = {j: i for i, j in enumerate(rating_count.movie.unique())}
        rating_count["t"] = rating_count.weight >= rating_count.weight.median()

        edge_index_df["T"] = 1

        # derive the mappings
        user_map = {j: i for i, j in enumerate(edge_index_df["user"].unique())}

        edge_index_df["movie"] = edge_index_df["movie"].map(movie_map)
        rating_count["movie"] = rating_count["movie"].map(movie_map)
        edge_index_df["user"] = edge_index_df["user"].map(user_map)

        edge_index_df.to_csv(f"{self.processed_dir}/movielens_graph.csv",
                             index=False)

        movies = pd.read_csv(f"{self.raw_dir}/{movies_dataset}")

        movies["movieId"] = movies["movieId"].map(movie_map)

        movies = movies[movies["movieId"].isin(edge_index_df.movie.unique())]

        rating_count["t"] = rating_count["t"].astype(int)

        dict_treatment = dict(zip(rating_count["movie"], rating_count["t"]))
        movies["t"] = movies["movieId"].map(dict_treatment)

        movie_ratings = edge_index_df.groupby("movie")["weight"].mean()

        # ===== features
        moviesd = np.expand_dims(movies["movieId"].astype(int).values,
                                 axis=0).T
        treatmentd = np.expand_dims(movies["t"].values, axis=0).T
        outcome = np.expand_dims(
            movie_ratings[movies["movieId"].astype(int)].values, axis=0).T

        movies["sentence"] = (" title: " + movies["title"] + " genres:" +
                              movies["genres"])
        model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", device="cuda"
        )  # use multilingual models for texts with non-english characters
        embeddings_lite = model.encode(movies["sentence"].values.tolist())

        pca = PCA(n_components=16)
        embeddings_lite = pca.fit_transform(embeddings_lite)
        features = np.hstack([moviesd, treatmentd, outcome, embeddings_lite])
        features = pd.DataFrame(features).sort_values(0)
        features = pd.DataFrame(features.values[:, 1:])

        features.to_csv(f"{self.processed_dir}/movielens_features.csv",
                        index=False)

        normalized_data = StandardScaler().fit_transform(
            features.iloc[:, 2:].values)

        data = HeteroData()
        data["movie", "ratedby", "user"] = {
            "edge_index":
            torch.tensor(edge_index_df[["movie", "user"
                                        ]].values).type(torch.LongTensor).T,
            "treatment":
            torch.tensor(edge_index_df["T"].values).type(torch.BoolTensor),
        }

        data["movie"] = {
            "x": torch.tensor(normalized_data).type(torch.FloatTensor),
            "t": torch.tensor(features.iloc[:,
                                            0].values).type(torch.LongTensor),
            "y": torch.tensor(features.iloc[:,
                                            1].values).type(torch.FloatTensor),
        }
        data["users"] = {"num_users": len(edge_index_df["user"].unique())}

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
