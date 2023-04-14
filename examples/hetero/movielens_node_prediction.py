# TODO: Run linter on this file
# TODO: Run through the notebook and make sure it works
# TODO: Run through Grammarly
# TODO: Uncomment install commands
# TODO: Check headlines
# TODO: Improve hyperparameters
# TODO: discuss use-case with Matthias
# TODO: Uncomment asserts
# TODO: Should we take the tags away

# This notebook runs faster on a GPU.
# You can enable GPU acceleration by going to Edit -> Notebook Settings -> Hardware Accelerator -> GPU

import torch
from torch import Tensor

print(torch.__version__)

# Install required packages.
import os

os.environ['TORCH'] = torch.__version__

#!pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
#!pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
#!pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
#!pip install git+https://github.com/pyg-team/pytorch_geometric.git
"""# Node Prediction on MovieLens

This colab notebook shows how to load a set of `*.csv` files as input and construct a heterogeneous graph from it.
We will then use this dataset as input into a [heterogeneous graph model](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html#hgtutorial), and use it for the task of node prediction.

We are going to use the [MovieLens dataset](https://grouplens.org/datasets/movielens/) collected by the GroupLens research group.
This toy dataset describes movies, tagging activity and ratings from MovieLens.
The dataset contains approximately 100k ratings across more than 9k movies from more than 600 users.
We are going to use this dataset to generate two node types holding data for movies and users, respectively, and one edge type connecting users and movies, representing the relation of whether a user has rated a specific movie.

The node prediction task then predicts missing movie genres.

## Heterogeneous Graph Creation

First, we download the dataset to an arbitrary folder (in this case, the current directory):
"""

from torch_geometric.data import download_url, extract_zip

dataset_name = 'ml-latest-small'

url = f'https://files.grouplens.org/datasets/movielens/{dataset_name}.zip'
extract_zip(download_url(url, '.'), '.')

movies_path = f'./{dataset_name}/movies.csv'
ratings_path = f'./{dataset_name}/ratings.csv'
tags_path = f'./{dataset_name}/tags.csv'
"""Before we create the heterogeneous graph, let’s take a look at the data."""

import pandas as pd

# Load the entire ratings data frame into memory:
ratings_df = pd.read_csv(ratings_path)

# Load the entire movie data frame into memory:
movies_df = pd.read_csv(movies_path)

# Load the entire tags data frame into memory:
tags_df = pd.read_csv(tags_path)

print('movies.csv:')
print('===========')
print(movies_df[["movieId", "genres", "title"]].head())
print(f"Number of movies: {len(movies_df)}")
print()
print('ratings.csv:')
print('============')
print(ratings_df[["userId", "movieId", "rating"]].head())
print(f"Number of ratings: {len(ratings_df)}")
print()
print('tags.csv:')
print('============')
print(tags_df[['userId', 'movieId', 'tag']].head())
print(f"Number of tags: {len(tags_df)}")
"""We are going to use the `genres` column in the `movie.csv` as the target of our node prediction task. 
Every movie is assigned to one or more genres. For simplicity, we are going to use only the most popular genres in 
our node prediction task. Moreover, we filter out movies that are assigned to more than one popular genre. This is to
avoid having to predict multiple genres for each movie. 
"""

# Filter out movies that do not belong to the most popular genres:
num_genres = 3
genres = movies_df['genres'].str.get_dummies('|')
popular_genres = genres.sum().sort_values(ascending=False).index[:num_genres]
movies_df = movies_df[movies_df['genres'].str.contains(
    '|'.join(popular_genres))]

# Filter out movies that are assigned to more than one popular genre:
genres = movies_df['genres'].str.get_dummies('|')
movies_df = movies_df[genres[popular_genres].sum(axis=1) == 1].reset_index(
    drop=True)

# Split genres and convert into indicator variables:
genres = movies_df['genres'].str.get_dummies('|')
genres = genres[popular_genres]
print(genres[popular_genres].head())

# Use genres as movie targets:
movie_target = torch.from_numpy(genres.values).to(torch.float)
"""Let's split the movies into train, validation, and test sets. We are going to use 80% of the movies for training, 10% for validation, and 10% for testing.
"""
from sklearn.model_selection import train_test_split

num_movies = len(movies_df)

# Next, split the indices of the nodes into train, validation, and test sets using `train_test_split`
train_idx, valtest_idx = train_test_split(range(num_movies),
                                          train_size=0.8,
                                          test_size=0.2,
                                          random_state=42)
val_idx, test_idx = train_test_split(valtest_idx,
                                     train_size=0.5,
                                     test_size=0.5,
                                     random_state=42)
"""After preparing the target column and the split we can look at the data again and prepare the features for every movie.

We see that the `movies.csv` file provides the additional columns: `movieId` and `title`.
The `movieId` assigns a unique identifier to each movie. The `title` column contains both the title and the release year of the movie.
We are going to split the year from the title and use both as features to predict the movie genre.
The title needs to be converted into a representation that can help the model to learn the relationship between movies. We are using a
bag-of-words representation of the title.
"""

# Extract the year from the title field and create a new 'year' column
movies_df['year'] = movies_df['title'].str.extract('(\(\d{4}\))', expand=False)
movies_df['year'] = movies_df['year'].str.extract('(\d{4})', expand=False)

# Remove the year and any trailing/leading whitespace from the title
# The title is now in the format: 'Movie Name (Year)'
movies_df['title'] = movies_df['title'].str.replace('(\(\d{4}\))', '', regex=True)
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

# Use a CountVectorizer to create a bag-of-words representation of the titles
from sklearn.feature_extraction.text import CountVectorizer
title_vectorizer = CountVectorizer(stop_words='english', analyzer='char_wb', ngram_range=(2, 2), max_features=1000) #, max_features=10000)
title_vectorizer.fit(movies_df['title'].iloc[train_idx]) 
title_features = title_vectorizer.transform(movies_df['title'])

# Create binary indicator variables for the year
year_indicator = pd.get_dummies(movies_df['year']).values

import numpy as np
movie_feat = np.concatenate((title_features.toarray(), year_indicator), axis=1)
"""Now, as we prepared the features based on the title and the year, we can add the tag features to the movie features as well. 
Similar to the title, we are using a bag-of-words representation of the tags. Finally, we combine the tag features with the title and year features into a single feature matrix for every movie."""

# Filter out tags that do not belong to the movies in the training set
tags_df = tags_df[tags_df['movieId'].isin(movies_df.index)]
tags_df = tags_df[tags_df['tag'].notna()]
tags_df = tags_df.groupby('movieId')['tag'].apply(
    lambda x: ' '.join(x)).reset_index('movieId')

# Merge the tags into the movies data frame
movies_df = pd.merge(movies_df, tags_df, on='movieId', how='left')
movies_df['tag'].fillna('', inplace=True)
tag_vectorizer = CountVectorizer(stop_words='english', analyzer='char_wb', ngram_range=(2, 2), max_features=1000) #, max_features=10000)
tag_vectorizer.fit(movies_df['tag'].iloc[train_idx])
tag_features = tag_vectorizer.transform(movies_df['tag'])

# Combine all movie features into a single feature matrix
# movie_feat = np.concatenate((movie_feat, tag_features.toarray()), axis=1)
movie_feat = torch.tensor(movie_feat, dtype=torch.float)

"""The `ratings.csv` data connects users (as given by `userId`) and movies (as given by `movieId`).
Due to simplicity, we do not make use of the additional `timestamp` and `rating` information.
Here, we create a mapping that maps entry IDs to a consecutive value in the range `{ 0, ..., num_rows - 1 }`.
This is needed as we want our final data representation to be as compact as possible, *e.g.*, the representation of a movie in the first row should be accessible via `x[0]`.

Afterwards, we obtain the final `edge_index` representation of shape `[2, num_ratings]` from `ratings.csv` by merging mapped user and movie indices with the raw indices given by the original data frame.
"""

# Filter ratings to only include movies in movies_df
ratings_df = ratings_df[ratings_df['movieId'].isin(movies_df.index)]

# Create a mapping from unique user indices to range [0, num_user_nodes):
unique_user_id = ratings_df['userId'].unique()
unique_user_id = pd.DataFrame(data={
    'userId': unique_user_id,
    'mappedID': pd.RangeIndex(len(unique_user_id)),
})
print("Mapping of user IDs to consecutive values:")
print("==========================================")
print(unique_user_id.head())
print()
# Create a mapping from unique movie indices to range [0, num_movie_nodes):
unique_movie_id = ratings_df['movieId'].unique()
unique_movie_id = pd.DataFrame(data={
    'movieId': unique_movie_id,
    'mappedID': pd.RangeIndex(len(unique_movie_id)),
})
print("Mapping of movie IDs to consecutive values:")
print("===========================================")
print(unique_movie_id.head())

# Perform merge to obtain the edges from users and movies:
ratings_user_id = pd.merge(ratings_df['userId'],
                           unique_user_id,
                           left_on='userId',
                           right_on='userId',
                           how='left')
ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
ratings_movie_id = pd.merge(ratings_df['movieId'],
                            unique_movie_id,
                            left_on='movieId',
                            right_on='movieId',
                            how='left')
ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)

# With this, we are ready to construct our `edge_index` in COO format
# following PyG semantics:
edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id],
                                       dim=0)
# assert edge_index_user_to_movie.size() == (2, 100836)

print()
print("Final edge indices pointing from users to movies:")
print("=================================================")
print(edge_index_user_to_movie)
"""With this, we are ready to initialize our `HeteroData` object and pass the necessary information to it.
Note that we also pass in a `node_id` vector to each node type in order to reconstruct the original node indices from sampled subgraphs.
We also take care of adding reverse edges to the `HeteroData` object.
This allows our GNN model to use both directions of the edge for message passing:
"""

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

data = HeteroData()

# Save node indices:
data["user"].node_id = torch.arange(len(unique_user_id))
data["movie"].node_id = torch.arange(len(movies_df))

# Add the node features and edge indices:
data["movie"].x = movie_feat
data["user", "rates", "movie"].edge_index = edge_index_user_to_movie
data["user", "rates",
     "movie"].edge_attr = torch.from_numpy(ratings_df["rating"].to_numpy())
data["user"].x = torch.ones((data["user"].num_nodes, 1))

# We also need to make sure to add the reverse edges from movies to users
# in order to let a GNN be able to pass messages in both directions.
# We can leverage the `T.ToUndirected()` transform for this from PyG:
data = T.ToUndirected()(data)

# Add the node labels:
data['movie'].y = movie_target

print(data)

assert data.node_types == ["user", "movie"]
assert data.edge_types == [("user", "rates", "movie"),
                           ("movie", "rev_rates", "user")]
# assert data["user"].num_nodes == 610
# assert data["user"].num_features == 0
# assert data["movie"].num_nodes == 9742
# assert data["movie"].num_features == 9180  # TODO: try out with other number of features
#assert data["user", "rates", "movie"].num_edges == 100836
#assert data["movie", "rev_rates", "user"].num_edges == 100836
#assert data["movie", "rev_rates", "user"].num_edge_features == 1
"""We can now split our data into train, validation, and test sets based on the indices of the movies."""

# First, create a numpy array with the same number of rows as your dataset, and fill it with False values
data['movie'].train_mask = np.zeros(data['movie'].num_nodes, dtype=bool)
data['movie'].test_mask = np.zeros(data['movie'].num_nodes, dtype=bool)
data['movie'].val_mask = np.zeros(data['movie'].num_nodes, dtype=bool)

# Update the corresponding indices in the mask array to True for the train, validation, and test sets
data['movie'].train_mask[train_idx] = True
data['movie'].val_mask[val_idx] = True
data['movie'].test_mask[test_idx] = True

assert data['movie'].train_mask.sum() == int(0.8 * data['movie'].num_nodes)
# assert data['movie'].val_mask.sum() == int(0.1 * data['movie'].num_nodes)
# assert data['movie'].test_mask.sum() == int(0.1 * data['movie'].num_nodes) + 1

"""## Baseline Model

We start by defining a baseline model, which we will use to compare our heterogeneous GNN against.
The baseline model is a simple MLP, which takes the node features of a movie as input and outputs a single value.
"""
from torch.nn import functional as F

class Baseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Baseline, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

"""## Training

We are now ready to train our baseline model.
"""

hidden_channels = 4
model = Baseline(data['movie'].num_features, hidden_channels, 3)
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

def train():
    model.train()
    optimizer.zero_grad()
    logits = model(data['movie'].x)
    loss = F.cross_entropy(logits[data['movie'].train_mask],
                            data['movie'].y[data['movie'].train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    with torch.no_grad():
        model.eval()
        logits = model(data['movie'].x)
    return logits, data['movie'].y

from tqdm import tqdm
from matplotlib import pyplot as plt

pbar = tqdm(range(1, 200))
for epoch in pbar:
    loss = train()
    logits, y = test()
    y = y.argmax(dim=-1)
    pred = logits.argmax(dim=-1)
    train_acc = pred[data['movie'].train_mask].eq(y[data['movie'].train_mask]).sum().item() / data['movie'].train_mask.sum().item()
    val_acc = pred[data['movie'].val_mask].eq(y[data['movie'].val_mask]).sum().item() / data['movie'].val_mask.sum().item()
    val_loss = F.cross_entropy(logits[data['movie'].val_mask], data['movie'].y[data['movie'].val_mask])
    pbar.set_description(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')
    plt.plot(epoch, loss.item(), label='Training loss', color='blue', marker='o')
    plt.plot(epoch, val_loss, label='Validation loss', color='red', marker='o')
    plt.plot(epoch, train_acc, label='Training accuracy', color='green', marker='x')
    plt.plot(epoch, val_acc, label='Validation accuracy', color='yellow', marker='x')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'loss_{lr}.png')


from sklearn.metrics import classification_report

print(classification_report(y[data['movie'].test_mask], pred[data['movie'].test_mask]))
"""## Creating a Heterogeneous GNN

We are now ready to create our heterogeneous GNN.
The GNN is responsible for learning enriched node representations from the surrounding subgraphs, which can be then used to derive node-level predictions.
For defining our heterogenous GNN, we make use of [`nn.SAGEConv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv) and the [`nn.to_hetero()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.to_hetero_transformer.to_hetero) function, which transforms a GNN defined on homogeneous graphs to be applied on heterogeneous ones.
"""

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

hidden_channels = 64

class GNN(torch.nn.Module):

    def __init__(self, hidden_channels, num_genres):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_genres)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class Model(torch.nn.Module):

    def __init__(self, hidden_channels, num_genres):
        super().__init__()

        # For the movie nodes, we use the pre-computed node features but add a linear layer to reduce the dimensionality.
        self.movie_lin = torch.nn.Linear(data['movie'].num_features, hidden_channels)
        self.user_lin = torch.nn.Linear(data['user'].num_features, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels, num_genres)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

    def forward(self, x_dixt, edge_index_dict) -> Tensor:

        # Initialize node embeddings for all node types:
        x_dict = {
          "user": self.user_lin(x_dixt["user"]),
          "movie": self.movie_lin(x_dixt["movie"]),
        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x = self.gnn(x_dict, edge_index_dict)

        # We return the logits for each node:
        return x['movie']

model = Model(hidden_channels=hidden_channels, num_genres=num_genres)

print(model)
"""## Training a Heterogeneous GNN

Training our GNN is then similar to training any PyTorch model.
We move the model to the desired device, and initialize an optimizer that takes care of adjusting model parameters via stochastic gradient descent.

The training loop applies the forward computation of the model, computes the loss from ground-truth labels and obtained predictions, and adjusts model parameters via back-propagation and stochastic gradient descent.

"""

import torch.nn.functional as F
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

model = model.to(device)

# We use Adam as our optimizer with a learning rate scheduler:
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

def train():
    model.train()
    optimizer.zero_grad()
    logits = model(data.x_dict, data.edge_index_dict)
    train_logits = logits
    train_y = data['movie'].y 
    loss = F.cross_entropy(train_logits, train_y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        test_logits = logits
        test_y = data['movie'].y
        return test_logits, test_y
    
pbar = tqdm.tqdm(range(1000))
for epoch in pbar:
    loss = train()
    pbar.set_postfix({'loss': f'{loss:.4f}'})
    scheduler.step()
    if epoch % 10 == 0:
        with torch.no_grad():
            logits, y = test()
            train_logits = logits[data["movie"].train_mask].argmax(-1)
            val_logits = logits[data["movie"].val_mask].argmax(-1)
            train_y = y[data["movie"].train_mask].argmax(dim=1)
            val_y = y[data["movie"].val_mask].argmax(dim=1)
            test_y = y[data["movie"].test_mask].argmax(dim=1)
            print('Train Acc:',
                (train_logits == train_y).sum().item() / train_y.size(0))
            print('Val Acc:', (val_logits == val_y).sum().item() / val_y.size(0))

"""## Evaluating a Heterogeneous GNN

After training, we evaluate our model on useen data coming from the test set.

"""

with torch.no_grad():
    data.to(device)
    test_logits = model(data.x_dict, data.edge_index_dict)[data['movie'].test_mask]
    test_y = data['movie'].y[data['movie'].test_mask].argmax(dim=1)
    test_acc = torch.sum(test_logits.argmax(dim=1) == test_y).item() / data['movie'].test_mask.sum().item()
    print(f"Test Accuracy: {test_acc:.4f}")

    print(
        classification_report(test_y.cpu().numpy(),
                              test_logits.argmax(dim=1).cpu().numpy(),
                              target_names=genres.columns))


"""## Explainability with Heterogeneous GNNs"""

# We want to see that the movies with the same genre as the node to be predicted are important for the same prediction. 

from torch_geometric.explain import CaptumExplainer, Explainer
from captum.attr import IntegratedGradients
explainer = Explainer(
    model=model,
    algorithm=CaptumExplainer(IntegratedGradients),
    explanation_type='model',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    ),
    node_mask_type='attributes',
    edge_mask_type='object'
)

node_index = 11
y = data['movie'].y[node_index].argmax(dim=0).item()
print(f"Target label: '{genres.columns[y]}'")
pred = model(data.x_dict, data.edge_index_dict)[node_index].argmax(dim=0).item()
print(f"Predicted label: '{genres.columns[pred]}'")

explanation = explainer(data.x_dict, data.edge_index_dict, index=node_index)

# Sum all node feature importances to get the overall importance of each node:
for node_type in explanation.node_types:
    explanation[node_type].node_mask = explanation[node_type].node_mask.sum(dim=-1)
path = 'subgraph.png'
explanation_homo = explanation.to_homogeneous()

from typing import Optional, Any

def visualize_graph(
    edge_index: Tensor,
    edge_weight: Tensor,
    node_id: int, 
    node_weight: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
    path: Optional[str] = None,
) -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx
    from math import sqrt

    g = nx.DiGraph()
    node_size = 400

    for node in edge_index.view(-1).unique().tolist():
        g.add_node(node)

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        if abs(w) > 1e-7:
            g.add_edge(src, dst, color=w)
    
    isolated_nodes = list(nx.isolates(g))
    if node_id in isolated_nodes:
        isolated_nodes.remove(node_id)
    g.remove_nodes_from(isolated_nodes)
    node_weight = node_weight[list(g.nodes)]

    ax = plt.gca()
    pos = nx.spring_layout(g)

        # Set color bar
    vmax = vmin = None
    if (vmax is None) and (vmin is None):
        if node_weight is not None:
            vmax = node_weight.abs().max()
            vmin = -vmax
        if edge_weight is not None:
            edge_max = edge_weight.abs().max()
            vmax = vmax or 0
            if edge_max > vmax:
                vmax = edge_max
                vmin = -vmax

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    importance_cmap = plt.get_cmap("seismic")

    for src, dst, data in g.edges(data=True):
        ax.annotate(
            '',
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="->",
                color=importance_cmap(norm(data["color"])),
                shrinkA=sqrt(node_size) / 2.0,
                shrinkB=sqrt(node_size) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    if node_weight is not None:
        node_colors = importance_cmap(norm(node_weight.tolist()))
    else:
        node_colors = 'white'

    edgecolors = ['black' if node is not node_id.item() else 'red' for node in g.nodes]
    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_size,
                                   node_color=node_colors, margins=0.1, edgecolors=edgecolors)

    if labels is not None:
        labels = dict(
            (i, label.item())
            for i, label in enumerate(labels)
            if i not in isolated_nodes
        )
    else:
        labels = None
    nx.draw_networkx_labels(g, pos, font_size=4, labels=labels)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()

# Visualize the importance of the entire node as well as the edges 
# Mark node by node_type and add movie genre as description of node
# explanation_homo is of type Data including an edge_mask, edge_index, 
# node_type and edge_type



# Visualize the explanation subgraph
from torch_geometric.utils import k_hop_subgraph
nodes, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=node_index, num_hops=2, edge_index=explanation_homo.edge_index, relabel_nodes=True)

# Filter out those edge_index where explanation_homo.edge_mask[edge_mask] is 0 

plt.clf()

visualize_graph(edge_index, 
                explanation_homo.edge_mask[edge_mask], 
                node_id=mapping,
                node_weight=explanation_homo.node_mask, 
                labels=explanation_homo['target'][nodes],
                path=path)

print(f"Subgraph plot has been saved to '{path}'")
