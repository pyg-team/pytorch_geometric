# Examples for Heterogeneous Data

| Example                                                | Description                                                                                                                            |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| [`to_hetero_mag.py`](./to_hetero_mag.py)               | Shows how to use the `to_hetero(...)` functionality; Trains it for node classification on the `ogb-mag` dataset.                       |
| [`hetero_link_pred.py`](./hetero_link_pred.py)         | Shows how to use the `to_hetero(...)` functionality; Trains it for link prediction on the `MovieLens` dataset.                         |
| [`hierarchical_sage.py`](./hierarchical_sage.py)       | Shows how to perform hierarchical sampling; Trains a heterogeneous `GraphSAGE` model for node classification on the `ogb-mag` dataset. |
| [`load_csv.py`](./load_csv.py)                         | Shows how to create heterogeneous graphs from raw `*.csv` data.                                                                        |
| [`metapath2vec.py`](./metapath2vec.py)                 | Train an unsupervised `MetaPath2Vec` model; Tests embeddings for node classification on the `AMiner` dataset.                          |
| [`temporal_link_pred.py`](./temporal_link_pred.py)     | Trains a heterogeneous `GraphSAGE` model for temporal link prediction on the `MovieLens` dataset.                                      |
| [`bipartite_sage.py`](./bipartite_sage.py)             | Trains a GNN via metapaths for link prediction on the `MovieLens` dataset.                                                             |
| [`bipartite_sage_unsup.py`](./bipartite_sage_unsup.py) | Trains a GNN via metapaths for link prediction on the large-scale `TaoBao` dataset.                                                    |
| [`recommender_system.py`](./recommender_system.py)     | Shows how to train a temporal GNN-based recommender system on the `MovieLens` dataset.                                                 |
