# `papers_100M` Example

This example shows how to use the remote backend feature of [Kùzu](https://kuzudb.com) to work with a large graph of papers and citations on a single machine.
The data used in this example is `ogbn-papers100M` from the [Open Graph Benchmark](https://ogb.stanford.edu/).
The dataset contains approximately 100 million nodes and 1.6 billion edges.

## Prepare the data

1. Download the dataset from [`http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip`](http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip) and put the `*.zip` file into this directory.
2. Run `python prepare_data.py`.
   The script will automatically extract the data and convert it to the format that Kùzu can read.
   A Kùzu database instance is then created under `papers_100M` and the data is loaded into the it.

## Train a Model

Afterwards, run `python train.py` to train a three-layer [`GraphSAGE`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html) model on this dataset.
