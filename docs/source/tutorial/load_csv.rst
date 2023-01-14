Loading Graphs from CSV
=======================

In this example, we will show how to load a set of :obj:`*.csv` files as input and construct a **heterogeneous graph** from it, which can be used as input to a `heterogenous graph model <heterogeneous.html>`__.
This tutorial is also available as an executable `example script <https://github.com/pyg-team/pytorch_geometric/tree/master/examples/hetero/load_csv.py>`_ in the :obj:`examples/hetero` directory.

We are going to use the `MovieLens dataset <https://grouplens.org/datasets/movielens/>`_ collected by the GroupLens research group.
This toy dataset describes 5-star rating and tagging activity from MovieLens.
The dataset contains approximately 100k ratings across more than 9k movies from more than 600 users.
We are going to use this dataset to generate two node types holding data for **movies** and **users**, respectively, and one edge type connecting **users and movies**, representing the relation of how a user has rated a specific movie.

First, we download the dataset to an arbitrary folder (in this case, the current directory):

.. code-block:: python

    from torch_geometric.data import download_url, extract_zip

    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    extract_zip(download_url(url, '.'), '.')

    movie_path = './ml-latest-small/movies.csv'
    rating_path = './ml-latest-small/ratings.csv'

Before we create the heterogeneous graph, let's take a look at the data.

.. code-block:: python

    import pandas as pd

    print(pd.read_csv(movie_path).head())
    print(pd.read_csv(rating_path).head())

.. list-table:: Head of :obj:`movies.csv`
    :widths: 5 40 60
    :header-rows: 1

    * - movieId
      - title
      - genres
    * - 1
      - Toy Story (1995)
      - Adventure|Animation|Children|Comedy|Fantasy
    * - 2
      - Jumanji (1995)
      - Adventure|Children|Fantasy
    * - 3
      - Grumpier Old Men (1995)
      - Comedy|Romance
    * - 4
      - Waiting to Exhale (1995)
      - Comedy|Drama|Romance
    * - 5
      - Father of the Bride Part II (1995)
      - Comedy

We see that the :obj:`movies.csv` file provides three columns: :obj:`movieId` assigns a unique identifier to each movie, while the :obj:`title` and :obj:`genres` columns represent title and genres of the given movie.
We can make use of those two columns to define a feature representation that can be easily interpreted by machine learning models.

.. list-table:: Head of :obj:`ratings.csv`
    :widths: 5 5 10 30
    :header-rows: 1

    * - userId
      - movieId
      - rating
      - timestamp
    * - 1
      - 1
      - 4.0
      - 964982703
    * - 1
      - 3
      - 4.0
      - 964981247
    * - 1
      - 6
      - 4.0
      - 964982224
    * - 1
      - 47
      - 5.0
      - 964983815
    * - 1
      - 50
      - 5.0
      - 964982931

The :obj:`ratings.csv` data connects users (as given by :obj:`userId`) and movies (as given by :obj:`movieId`), and defines how a given user has rated a specific movie (:obj:`rating`).
Due to simplicity, we do not make use of the additional :obj:`timestamp` information.

For representing this data in the :pyg:`PyG` data format, we first define a method :meth:`load_node_csv` that reads in a :obj:`*.csv` file and returns a node-level feature representation :obj:`x` of shape :obj:`[num_nodes, num_features]`:

.. code-block:: python

    import torch

    def load_node_csv(path, index_col, encoders=None, **kwargs):
        df = pd.read_csv(path, index_col=index_col, **kwargs)
        mapping = {index: i for i, index in enumerate(df.index.unique())}

        x = None
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)

        return x, mapping

Here, :meth:`load_node_csv` reads the :obj:`*.csv` file from :obj:`path`, and creates a dictionary :obj:`mapping` that maps its index column to a consecutive value in the range :obj:`{ 0, ..., num_rows - 1 }`.
This is needed as we want our final data representation to be as compact as possible, *e.g.*, the representation of a movie in the first row should be accessible via :obj:`x[0]`.

We further utilize the concept of encoders, which define how the values of specific columns should be encoded into a numerical feature representation.
For example, we can define a sentence encoder that encodes raw column strings into low-dimensional embeddings.
For this, we make use of the excellent `sentence-transformers <https://www.sbert.net/>`_ library which provides a large number of state-of-the-art pretrained NLP embedding models:

.. code-block:: bash

    pip install sentence-transformers

.. code-block:: python

    class SequenceEncoder(object):
        def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
            self.device = device
            self.model = SentenceTransformer(model_name, device=device)

        @torch.no_grad()
        def __call__(self, df):
            x = self.model.encode(df.values, show_progress_bar=True,
                                  convert_to_tensor=True, device=self.device)
            return x.cpu()

The :class:`SequenceEncoder` class loads a pre-trained NLP model as given by :obj:`model_name`, and uses it to encode a list of strings into a :pytorch:`PyTorch` tensor of shape :obj:`[num_strings, embedding_dim]`.
We can use this :class:`SequenceEncoder` to encode the :obj:`title` of the :obj:`movies.csv` file.

In a similar fashion, we can create another encoder that converts the genres of movies, *e.g.*, :obj:`Adventure|Children|Fantasy`, into categorical labels.
For this, we first need to find all existing genres present in the data, create a feature representation :obj:`x` of shape :obj:`[num_movies, num_genres]`, and assign a :obj:`1` to :obj:`x[i, j]` in case the genre :obj:`j` is present in movie :obj:`i`:

.. code-block:: python

    class GenresEncoder(object):
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

With this, we can obtain our final representation of movies via:

.. code-block:: python

    movie_x, movie_mapping = load_node_csv(
        movie_path, index_col='movieId', encoders={
            'title': SequenceEncoder(),
            'genres': GenresEncoder()
        })

Similarly, we can utilize :meth:`load_node_csv` for obtaining a user mapping from :obj:`userId` to consecutive values as well.
However, there is no additional feature information for users present in this dataset.
As such, we do not define any encoders:

.. code-block:: python

    _, user_mapping = load_node_csv(rating_path, index_col='userId')

With this, we are ready to initialize our :class:`~torch_geometric.data.HeteroData` object and pass two node types into it:

.. code-block:: python

    from torch_geometric.data import HeteroData

    data = HeteroData()

    data['user'].num_nodes = len(user_mapping)  # Users do not have any features.
    data['movie'].x = movie_x

    print(data)
    HeteroData(
      user={ num_nodes=610 },
      movie={ x[9742, 404] }
    )

As users do not have any node-level information, we solely define its number of nodes.
As a result, we likely need to learn distinct user embeddings via :class:`torch.nn.Embedding` in an end-to-end fashion during training of a heterogeneous graph model.

Next, we take a look at connecting users with movies as defined by their ratings.
For this, we define a method :meth:`load_edge_csv` that returns the final :obj:`edge_index` representation of shape :obj:`[2, num_ratings]` from :obj:`ratings.csv`, as well as any additional features present in the raw :obj:`*.csv` file:

.. code-block:: python

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

Here, :obj:`src_index_col` and :obj:`dst_index_col` define the index columns of source and destination nodes, respectively.
We further make use of the node-level mappings :obj:`src_mapping` and :obj:`dst_mapping` to ensure that raw indices are mapped to the correct consecutive indices in our final representation.
For every edge defined in the file, it looks up the forward indices in :obj:`src_mapping` and :obj:`dst_mapping`, and moves the data appropriately.

Similarly to :meth:`load_node_csv`, encoders are used to return additional edge-level feature information.
For example, for loading the ratings from the :obj:`rating` column in :obj:`ratings.csv`, we can define an :class:`IdentityEncoder` that simply converts a list of floating-point values into a :pytorch:`PyTorch` tensor:

.. code-block:: python

    class IdentityEncoder(object):
        def __init__(self, dtype=None):
            self.dtype = dtype

        def __call__(self, df):
            return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

With this, we are ready to finalize our :class:`~torch_geometric.data.HeteroData` object:

.. code-block:: python

    edge_index, edge_label = load_edge_csv(
        rating_path,
        src_index_col='userId',
        src_mapping=user_mapping,
        dst_index_col='movieId',
        dst_mapping=movie_mapping,
        encoders={'rating': IdentityEncoder(dtype=torch.long)},
    )

    data['user', 'rates', 'movie'].edge_index = edge_index
    data['user', 'rates', 'movie'].edge_label = edge_label

    print(data)
    HeteroData(
      user={ num_nodes=610 },
      movie={ x=[9742, 404] },
      (user, rates, movie)={
        edge_index=[2, 100836],
        edge_label=[100836, 1]
      }
    )

This :class:`~torch_geometric.data.HeteroData` object is the native format of heterogenous graphs in :pyg:`PyG` and can be used as input for `heterogenous graph models <heterogeneous.html>`__.

.. note::

    Click `here <https://github.com/pyg-team/pytorch_geometric/tree/master/examples/hetero/load_csv.py>`_ to see the final example script.
